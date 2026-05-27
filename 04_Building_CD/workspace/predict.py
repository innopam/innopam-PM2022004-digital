"""
건물 변화탐지 추론 스크립트 (urban_cd_v1 / DINOv3 + UPerNet 기반).

기존 04_Building_CD 인터페이스 유지:
  - 입력: <dataset_path>/T1/*.tif, <dataset_path>/T2/*.tif
  - 출력: <output_path>/<img>.json, status.json
  - 내부 산출물: <output_path>/results/<img>/<img>.tif, *_conf.tif, *_prob_*.tif

내부 추론 엔진은 기존 Mamba/Ray/PNG 패치 방식에서 DINOv3 ViT-L/16
+ UPerNet sliding-window 직접 추론 방식으로 교체.

클래스:
  1 = 신축
  2 = 소멸
  3 = 갱신
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
import torch
import yaml
from rasterio.features import rasterize, shapes, sieve
from rasterio.windows import Window, from_bounds
from shapely.geometry import mapping, shape
from tqdm import tqdm


URBAN_CD_DIR = os.environ.get("URBAN_CD_DIR", "/root/urban_cd_v1")
for path in (
    f"{URBAN_CD_DIR}/common/src",
    f"{URBAN_CD_DIR}/building_cd/phase3_semantic_cd/src",
    f"{URBAN_CD_DIR}/road_cd/scripts",
):
    if path not in sys.path:
        sys.path.insert(0, path)

from infer_real_ortho import (  # noqa: E402
    PairInfo,
    build_tile_specs,
    preprocess_tiles,
    repair_geometry,
    resolve_clip_window,
    trim_tile,
)
from phase3_semantic_cd import DirectionalSemanticChangeDetector  # noqa: E402


DEFAULT_CONFIG = (
    f"{URBAN_CD_DIR}/building_cd/phase3_semantic_cd/configs/"
    "phase3_update14_phase2init_lvd_lora_last4_rankaux_v1.yaml"
)
DINOV3_BACKBONE_DIRNAME = "dinov3-vitl16-pretrain-lvd1689m"

CLASS_NAME_MAP = {
    1: "신축",
    2: "소멸",
    3: "갱신",
}

PERCENTILE_LOW = 2.0
PERCENTILE_HIGH = 98.0
SAMPLE_SIZE_FOR_SCALE = 2048


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Building change detection (urban_cd_v1)")
    parser.add_argument("--model_path", type=str, default="/workspace/model/")
    parser.add_argument("--dataset_path", type=str, default="/workspace/input/")
    parser.add_argument("--output_path", type=str, default="/workspace/output/")
    parser.add_argument("--patch_size", type=int, default=1024)
    parser.add_argument(
        "--overlap_ratio",
        type=str,
        default="25",
        help="겹침 비율. 'min' 또는 0~100 사이의 퍼센트 값 (기본 25)",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.7,
        help="변화 클래스로 인정할 softmax confidence 하한",
    )
    parser.add_argument(
        "--min_component_pixels",
        type=int,
        default=200,
        help="클래스별 connected component 최소 픽셀 수. 0이면 비활성화",
    )
    parser.add_argument("--min_area_m2", type=float, default=20.0)
    parser.add_argument("--simplify_tolerance", type=float, default=0.2)
    parser.add_argument("--status_file", type=str, default="status.json")
    return parser.parse_args()


def overlap_to_pixels(overlap_ratio: str, patch_size: int) -> int:
    if overlap_ratio == "min":
        return 0
    pct = float(overlap_ratio)
    if not (0.0 <= pct < 100.0):
        raise ValueError(f"overlap_ratio percent must be in [0, 100), got {pct}")
    return int(round(patch_size * pct / 100.0))


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed

    return wrapper


def init_status(output_path: str, status_file: str, total_step: int) -> None:
    path = os.path.join(output_path, status_file)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "Total_step": total_step,
                "Current_step": 0,
                "Process": 0,
                "Status": "pending",
                "ElapsedTime": {},
            },
            f,
            indent=4,
            ensure_ascii=False,
        )


def update_status(output_path: str, status_file: str, **updates) -> dict:
    path = os.path.join(output_path, status_file)
    with open(path, "r", encoding="utf-8") as f:
        status = json.load(f)
    elapsed_update = updates.pop("_elapsed", None)
    status.update(updates)
    if elapsed_update:
        status.setdefault("ElapsedTime", {}).update(elapsed_update)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=4, ensure_ascii=False)
    return status


def log_error(output_path: str, message: str) -> None:
    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_path, "error_log.txt"), "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")


def load_model_for_inference(model_path: Path, device: torch.device):
    checkpoint_path = model_path / "best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"학습 체크포인트가 없습니다: {checkpoint_path}\n"
            "workspace/model/best.pth 파일을 실제 Building DINOv3 가중치로 교체하세요."
        )
    if checkpoint_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"체크포인트가 더미 파일입니다: {checkpoint_path}\n"
            "사용 전에 최신 Building best.pth로 덮어쓰세요."
        )

    ckpt_size_mb = checkpoint_path.stat().st_size / (1024 ** 2)
    print(f"[checkpoint] best.pth: size={ckpt_size_mb:.1f} MB", flush=True)
    state = torch.load(str(checkpoint_path), map_location="cpu")

    epoch = state.get("epoch", "<missing>")
    best_metric_name = state.get("best_metric_name", state.get("best_metric", "<missing>"))
    best_metric_value = state.get("best_metric_value", state.get("best_score", "<missing>"))
    print(
        f"[checkpoint] epoch={epoch}, best_metric={best_metric_name}, "
        f"best_value={best_metric_value}",
        flush=True,
    )

    cfg = state.get("config")
    if cfg is None:
        with open(DEFAULT_CONFIG, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        print("[checkpoint] config NOT in .pth -> fallback to bundled YAML", flush=True)
    else:
        exp_name = cfg.get("experiment", {}).get("name", "<missing>")
        num_classes = cfg.get("model", {}).get("decoder", {}).get("num_classes", "<missing>")
        print(
            f"[checkpoint] embedded config: experiment={exp_name}, "
            f"decoder.num_classes={num_classes}",
            flush=True,
        )

    dino_dir = model_path / DINOV3_BACKBONE_DIRNAME
    safetensors_path = dino_dir / "model.safetensors"
    if not safetensors_path.exists() or safetensors_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"DINOv3 backbone 가중치를 찾을 수 없습니다: {safetensors_path}\n"
            f"workspace/model/{DINOV3_BACKBONE_DIRNAME}/ 디렉토리에 "
            "config.json, preprocessor_config.json, model.safetensors 를 배치하세요."
        )
    dino_size_mb = safetensors_path.stat().st_size / (1024 ** 2)
    print(f"[checkpoint] DINOv3 model.safetensors: size={dino_size_mb:.1f} MB", flush=True)

    cfg["model"]["backbone"]["checkpoint_path"] = str(dino_dir)

    model_kwargs = {
        "checkpoint_path": cfg["model"]["backbone"]["checkpoint_path"],
        "output_layers": cfg["model"]["backbone"].get("output_layers", (6, 12, 18, 24)),
        "freeze": cfg["model"]["backbone"].get("freeze", True),
        "drop_path_rate": cfg["model"]["backbone"].get("drop_path_rate", 0.0),
        "unfreeze_last_n_blocks": cfg["model"]["backbone"].get("unfreeze_last_n_blocks", 0),
        "lora": cfg["model"]["backbone"].get("lora"),
        "fusion_out_channels": cfg["model"]["neck"].get("out_channels", 256),
        "scale_factors": cfg["model"]["neck"].get("scale_factors", (4.0, 2.0, 1.0, 0.5)),
        "ppm_bins": cfg["model"]["decoder"].get("ppm_bins", (1, 2, 3, 6)),
        "fpn_channels": cfg["model"]["decoder"].get("fpn_channels", 256),
        "num_classes": cfg["model"]["decoder"].get("num_classes", 4),
        "aux_head": cfg["model"].get("aux_head"),
        "building_head": cfg["model"].get("building_head"),
        "structural_branch": cfg["model"].get("structural_branch"),
    }
    supports_rank_aux = "rank_aux" in inspect.signature(DirectionalSemanticChangeDetector).parameters
    if supports_rank_aux:
        model_kwargs["rank_aux"] = cfg["model"].get("rank_aux")

    model = DirectionalSemanticChangeDetector(**model_kwargs)
    incompatible = model.load_state_dict(state["model_state"], strict=supports_rank_aux)
    if not supports_rank_aux:
        unexpected = [key for key in incompatible.unexpected_keys if key.startswith("rank_aux_")]
        other_unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("rank_aux_")]
        if incompatible.missing_keys or other_unexpected:
            raise RuntimeError(
                "Checkpoint/model mismatch outside rank_aux fallback: "
                f"missing={incompatible.missing_keys}, unexpected={other_unexpected[:10]}"
            )
        print(
            f"[checkpoint] rank_aux fallback: ignored {len(unexpected)} auxiliary head keys "
            "because installed urban_cd_v1 model.py does not define rank_aux. "
            "Main inference logits are unaffected.",
            flush=True,
        )
    model.to(device)
    model.eval()
    return model, cfg


def build_output_profile(t1_path: Path, clip_window):
    clip_x, clip_y, clip_w, clip_h = clip_window
    with rasterio.open(str(t1_path), IGNORE_COG_LAYOUT_BREAK="YES") as src:
        transform = src.window_transform(Window(clip_x, clip_y, clip_w, clip_h))
        profile = src.profile.copy()
    block_size = max(16, min(512, clip_w, clip_h))
    block_size -= block_size % 16
    block_size = max(16, block_size)
    profile.update(
        driver="GTiff",
        height=clip_h,
        width=clip_w,
        count=1,
        dtype="uint8",
        compress="lzw",
        tiled=True,
        blockxsize=block_size,
        blockysize=block_size,
        transform=transform,
    )
    return profile


def compute_band_scale(band_sample: np.ndarray, dtype_str: str) -> tuple[float, float]:
    if dtype_str == "uint8":
        return 0.0, 255.0
    arr = band_sample.astype(np.float32)
    nonzero = arr[arr > 0]
    if nonzero.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(nonzero, PERCENTILE_LOW))
    hi = float(np.percentile(nonzero, PERCENTILE_HIGH))
    if hi <= lo:
        return lo, lo + 1.0
    return lo, hi


def compute_image_scales(src) -> list[tuple[float, float]]:
    sw = min(SAMPLE_SIZE_FOR_SCALE, src.width)
    sh = min(SAMPLE_SIZE_FOR_SCALE, src.height)
    cx = max(0, (src.width - sw) // 2)
    cy = max(0, (src.height - sh) // 2)
    sample = src.read(indexes=(1, 2, 3), window=Window(cx, cy, sw, sh))
    return [compute_band_scale(sample[i], src.dtypes[i]) for i in range(3)]


def apply_band_scale(band_arr: np.ndarray, scale: tuple[float, float], dtype_str: str) -> np.ndarray:
    if dtype_str == "uint8":
        return band_arr.astype(np.uint8, copy=False)
    lo, hi = scale
    arr = band_arr.astype(np.float32)
    clipped = np.clip(arr, lo, hi)
    scaled = (clipped - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def read_tile_dtype_safe(src1, src2, clip_x: int, clip_y: int, spec, scales1: list, scales2: list):
    read_window = Window(clip_x + spec.read_x, clip_y + spec.read_y, spec.read_width, spec.read_height)
    raw1 = src1.read(indexes=(1, 2, 3), window=read_window)
    raw2 = src2.read(indexes=(1, 2, 3), window=read_window)
    t1_bands = [apply_band_scale(raw1[i], scales1[i], src1.dtypes[i]) for i in range(3)]
    t2_bands = [apply_band_scale(raw2[i], scales2[i], src2.dtypes[i]) for i in range(3)]
    t1 = np.moveaxis(np.stack(t1_bands, axis=0), 0, -1)
    t2 = np.moveaxis(np.stack(t2_bands, axis=0), 0, -1)
    valid = np.ones((spec.read_height, spec.read_width), dtype=bool)
    return t1, t2, valid


def log_input_properties(t1_path: Path, t2_path: Path) -> None:
    for label, path in (("T1", t1_path), ("T2", t2_path)):
        with rasterio.open(str(path), IGNORE_COG_LAYOUT_BREAK="YES") as src:
            color_interp = tuple(c.name for c in src.colorinterp)
            print(
                f"[input] {label} {path.name}: size={src.width}x{src.height}, "
                f"bands={src.count}, dtypes={src.dtypes}, color_interp={color_interp}",
                flush=True,
            )
            tr = src.transform
            print(
                f"[input] {label} transform: a={tr.a:.4f}, b={tr.b:.4f}, "
                f"c={tr.c:.2f}, d={tr.d:.4f}, e={tr.e:.4f}, f={tr.f:.2f}",
                flush=True,
            )


def validate_pair_relaxed(t1_path: Path, t2_path: Path) -> PairInfo:
    with rasterio.open(t1_path, IGNORE_COG_LAYOUT_BREAK="YES") as src1, \
         rasterio.open(t2_path, IGNORE_COG_LAYOUT_BREAK="YES") as src2:
        for attr in ("width", "height"):
            if getattr(src1, attr) != getattr(src2, attr):
                raise ValueError(
                    f"Raster mismatch for {attr}: {getattr(src1, attr)} != {getattr(src2, attr)}"
                )
        if src1.count < 3 or src2.count < 3:
            raise ValueError(f"Both rasters must contain at least 3 bands (T1={src1.count}, T2={src2.count})")
        if (src1.crs is None) ^ (src2.crs is None) or (
            src1.crs is not None and src2.crs is not None and src1.crs != src2.crs
        ):
            raise ValueError(f"CRS mismatch: {src1.crs} != {src2.crs}")

        t1, t2 = src1.transform, src2.transform
        max_drift = max(
            abs(t1.a - t2.a),
            abs(t1.b - t2.b),
            abs(t1.d - t2.d),
            abs(t1.e - t2.e),
            abs(t1.c - t2.c) / max(abs(t1.a), 1e-9),
            abs(t1.f - t2.f) / max(abs(t1.e), 1e-9),
        )
        if max_drift > 0.5:
            print(f"[warn] transform drift max={max_drift:.4f} pixel; proceeding", flush=True)

        return PairInfo(
            t1_path=str(t1_path),
            t2_path=str(t2_path),
            width=src1.width,
            height=src1.height,
            count=src1.count,
            crs=src1.crs.to_string() if src1.crs else None,
            res_x=float(src1.res[0]),
            res_y=float(src1.res[1]),
            pixel_area=float(abs(src1.transform.a * src1.transform.e)),
        )


@measure_time
def step_patch(t1_path: Path, t2_path: Path, patch_size: int, overlap_px: int):
    log_input_properties(t1_path, t2_path)
    pair_info = validate_pair_relaxed(t1_path, t2_path)
    clip_window = resolve_clip_window(pair_info, None)
    clip_x, clip_y, clip_w, clip_h = clip_window
    specs = build_tile_specs(clip_w, clip_h, patch_size, overlap_px)
    print(
        f"[tiles] patch_size={patch_size}, overlap_px={overlap_px}, "
        f"tiles={len(specs)}, clip={clip_window}",
        flush=True,
    )
    return pair_info, clip_window, specs


@measure_time
def step_inference(
    model,
    cfg: dict,
    t1_path: Path,
    t2_path: Path,
    clip_window,
    specs: Sequence,
    result_dir: Path,
    final_name: str,
    batch_size: int,
    confidence_threshold: float,
    device: torch.device,
):
    clip_x, clip_y, clip_w, clip_h = clip_window
    mask_path = result_dir / f"{final_name}.tif"
    conf_path = result_dir / f"{final_name}_conf.tif"
    valid_path = result_dir / f"{final_name}_valid.tif"
    prob_paths = {
        class_id: result_dir / f"{final_name}_prob_{class_id}.tif"
        for class_id in CLASS_NAME_MAP
    }

    profile = build_output_profile(t1_path, clip_window)
    mean = cfg["data"]["aug"]["normalize_mean"]
    std = cfg["data"]["aug"]["normalize_std"]
    num_classes = int(cfg["model"]["decoder"].get("num_classes", 4))
    autocast_enabled = device.type == "cuda"
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    total_pixels = 0
    pred_pixels = {class_id: 0 for class_id in CLASS_NAME_MAP}
    max_conf_seen = 0.0

    with rasterio.open(str(t1_path), IGNORE_COG_LAYOUT_BREAK="YES") as src1, \
         rasterio.open(str(t2_path), IGNORE_COG_LAYOUT_BREAK="YES") as src2, \
         rasterio.open(str(mask_path), "w", BIGTIFF="YES", **profile) as mask_dst, \
         rasterio.open(str(conf_path), "w", BIGTIFF="YES", **profile) as conf_dst, \
         rasterio.open(str(valid_path), "w", BIGTIFF="YES", **profile) as valid_dst, \
         rasterio.open(str(prob_paths[1]), "w", BIGTIFF="YES", **profile) as prob1_dst, \
         rasterio.open(str(prob_paths[2]), "w", BIGTIFF="YES", **profile) as prob2_dst, \
         rasterio.open(str(prob_paths[3]), "w", BIGTIFF="YES", **profile) as prob3_dst:

        prob_dsts = {1: prob1_dst, 2: prob2_dst, 3: prob3_dst}
        scales1 = compute_image_scales(src1)
        scales2 = compute_image_scales(src2)
        print(
            f"[normalize] {final_name}: T1 dtypes={src1.dtypes} scales={scales1}, "
            f"T2 dtypes={src2.dtypes} scales={scales2}",
            flush=True,
        )

        pbar = tqdm(total=len(specs), desc=f"Inferring {final_name}")
        for start in range(0, len(specs), batch_size):
            batch_specs = list(specs[start:start + batch_size])
            t1_tiles, t2_tiles, valid_tiles = [], [], []
            for spec in batch_specs:
                t1, t2, valid = read_tile_dtype_safe(src1, src2, clip_x, clip_y, spec, scales1, scales2)
                t1_tiles.append(t1)
                t2_tiles.append(t2)
                valid_tiles.append(valid)

            t1_tensor, t2_tensor, pad = preprocess_tiles(t1_tiles, t2_tiles, mean, std, device)
            with torch.inference_mode():
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                    outputs = model(t1_tensor, t2_tensor)
                    logits = outputs["main"] if isinstance(outputs, dict) else outputs
                    probs = torch.softmax(logits.float(), dim=1).cpu().numpy()

            pad_h, pad_w = pad
            if pad_h or pad_w:
                probs = probs[:, :, : probs.shape[2] - pad_h, : probs.shape[3] - pad_w]

            for idx, spec in enumerate(batch_specs):
                prob = probs[idx]
                pred = prob.argmax(axis=0).astype(np.uint8)
                conf = prob.max(axis=0)
                pred = np.where((pred > 0) & (conf >= confidence_threshold), pred, 0).astype(np.uint8)
                valid = valid_tiles[idx].astype(np.uint8)
                conf_u8 = np.clip(np.rint(conf * 255.0), 0, 255).astype(np.uint8)

                pred_trim = trim_tile(pred, spec)
                conf_trim = trim_tile(conf_u8, spec)
                valid_trim = trim_tile(valid, spec)
                pred_trim = np.where(valid_trim > 0, pred_trim, 0).astype(np.uint8)
                conf_trim = np.where(valid_trim > 0, conf_trim, 0).astype(np.uint8)

                prob_trims = {}
                for class_id in CLASS_NAME_MAP:
                    if class_id < num_classes:
                        prob_u8 = np.clip(np.rint(prob[class_id] * 255.0), 0, 255).astype(np.uint8)
                    else:
                        prob_u8 = np.zeros_like(conf_u8)
                    prob_trim = trim_tile(prob_u8, spec)
                    prob_trims[class_id] = np.where(valid_trim > 0, prob_trim, 0).astype(np.uint8)

                total_pixels += pred_trim.size
                for class_id in CLASS_NAME_MAP:
                    pred_pixels[class_id] += int((pred_trim == class_id).sum())
                max_conf_seen = max(max_conf_seen, float(conf.max()))

                window = Window(spec.write_x, spec.write_y, spec.write_width, spec.write_height)
                mask_dst.write(pred_trim, 1, window=window)
                conf_dst.write(conf_trim, 1, window=window)
                valid_dst.write(valid_trim, 1, window=window)
                for class_id, prob_trim in prob_trims.items():
                    prob_dsts[class_id].write(prob_trim, 1, window=window)
            pbar.update(len(batch_specs))
        pbar.close()

    print(
        f"[inference] {final_name}: total_pixels={total_pixels}, "
        f"class1(신축)={pred_pixels[1]}, class2(소멸)={pred_pixels[2]}, "
        f"class3(갱신)={pred_pixels[3]}, max_conf={max_conf_seen:.3f}, "
        f"threshold={confidence_threshold}",
        flush=True,
    )
    return mask_path, conf_path, valid_path, prob_paths


@measure_time
def step_reconstruct():
    return "no-op"


def apply_component_filter(mask_path: Path, min_pixels: int) -> None:
    if min_pixels <= 0:
        return
    with rasterio.open(str(mask_path)) as src:
        profile = src.profile.copy()
        mask = src.read(1)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for class_id in CLASS_NAME_MAP:
        class_mask = (mask == class_id).astype(np.uint8)
        if class_mask.any():
            filtered = sieve(class_mask, size=min_pixels, connectivity=8)
            cleaned[filtered == 1] = class_id
    with rasterio.open(str(mask_path), "w", **profile) as dst:
        dst.write(cleaned, 1)


def polygon_mean_probability(prob_src, polygon) -> float:
    window = from_bounds(*polygon.bounds, transform=prob_src.transform)
    window = window.round_offsets().round_lengths()
    col_off = max(0, int(window.col_off))
    row_off = max(0, int(window.row_off))
    width = min(prob_src.width - col_off, max(1, int(window.width)))
    height = min(prob_src.height - row_off, max(1, int(window.height)))
    window = Window(col_off, row_off, width, height)
    data = prob_src.read(1, window=window)
    transform = prob_src.window_transform(window)
    mask = rasterize([(polygon, 1)], out_shape=data.shape, fill=0, transform=transform, dtype=np.uint8)
    valid_values = data[mask == 1]
    if valid_values.size == 0:
        return 0.0
    return float(valid_values.mean() / 255.0)


def iter_polygons(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return [p for p in geom.geoms if not p.is_empty]
    return []


@measure_time
def step_vectorize(
    mask_path: Path,
    valid_path: Path,
    prob_paths: dict[int, Path],
    output_json_path: Path,
    img_name: str,
    min_area_m2: float,
    simplify_tolerance: float,
):
    raw_features = []
    n_total_shapes = 0
    n_class_match = 0
    n_after_area = 0

    with rasterio.open(str(mask_path)) as mask_src, \
         rasterio.open(str(valid_path)) as valid_src, \
         rasterio.open(str(prob_paths[1])) as prob1_src, \
         rasterio.open(str(prob_paths[2])) as prob2_src, \
         rasterio.open(str(prob_paths[3])) as prob3_src:

        prob_srcs = {1: prob1_src, 2: prob2_src, 3: prob3_src}
        crs = mask_src.crs
        crs_str = f"EPSG:{crs.to_epsg()}" if crs and crs.to_epsg() else str(crs) if crs else None

        iterator = shapes(
            rasterio.band(mask_src, 1),
            mask=rasterio.band(valid_src, 1),
            transform=mask_src.transform,
            connectivity=8,
        )
        for geom, value in iterator:
            n_total_shapes += 1
            class_id = int(value)
            if class_id not in CLASS_NAME_MAP:
                continue
            n_class_match += 1
            polygon = repair_geometry(shape(geom))
            if polygon is None:
                continue
            if simplify_tolerance > 0:
                polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)
                polygon = repair_geometry(polygon)
            if polygon is None:
                continue
            for sub in iter_polygons(polygon):
                if sub.area < min_area_m2:
                    continue
                n_after_area += 1
                conf_val = polygon_mean_probability(prob_srcs[class_id], sub)
                centroid = sub.centroid
                raw_features.append(
                    {
                        "polygon": sub,
                        "class_id": class_id,
                        "area": float(sub.area),
                        "conf": conf_val,
                        "cx": centroid.x,
                        "cy": centroid.y,
                    }
                )

    print(
        f"[vectorize] {img_name}: shapes={n_total_shapes}, class_match={n_class_match}, "
        f"after_area>={min_area_m2}={n_after_area}, final={len(raw_features)}",
        flush=True,
    )

    raw_features.sort(key=lambda f: (f["cy"], f["cx"]), reverse=True)
    geojson_features = []
    for idx, feature in enumerate(raw_features, start=1):
        geojson_features.append(
            {
                "type": "Feature",
                "properties": {
                    "CLS_ID": feature["class_id"],
                    "CLS_NAME": CLASS_NAME_MAP[feature["class_id"]],
                    "CONF": round(feature["conf"] * 100.0, 2),
                    "AREA": round(feature["area"], 2),
                    "ID": idx,
                },
                "geometry": mapping(feature["polygon"]),
            }
        )

    geojson_data = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": f"{crs_str}"}},
        "map_number": img_name,
        "features": geojson_features,
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=4, ensure_ascii=False)

    return f"Vectorization done: {len(geojson_features)} features"


def main() -> None:
    args = make_args()

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    init_t1_path = Path(args.dataset_path) / "T1"
    init_t2_path = Path(args.dataset_path) / "T2"
    t1_files = sorted(p.name for p in init_t1_path.glob("*.tif"))
    t2_files = sorted(p.name for p in init_t2_path.glob("*.tif"))
    if t1_files != t2_files:
        raise RuntimeError("=> Please match before/after image filenames")

    output_root = Path(args.output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    init_status(args.output_path, args.status_file, total_step=len(t1_files))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model_for_inference(Path(args.model_path), device)
    overlap_px = overlap_to_pixels(args.overlap_ratio, args.patch_size)

    results_root = output_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    final_name = None
    try:
        for img_filename in t1_files:
            final_name = img_filename.rsplit(".", 1)[0]
            t1_path = init_t1_path / img_filename
            t2_path = init_t2_path / img_filename
            result_dir = results_root / final_name
            result_dir.mkdir(parents=True, exist_ok=True)
            output_json_path = result_dir / f"{final_name}.json"

            update_status(
                args.output_path,
                args.status_file,
                Status="in progress",
                final_name=final_name,
            )

            (_, clip_window, specs), elapsed_patch = step_patch(
                t1_path, t2_path, args.patch_size, overlap_px
            )
            update_status(
                args.output_path,
                args.status_file,
                Process=20,
                CurrentTask="patch done",
                _elapsed={"패치 생성": f"{round(elapsed_patch, 2)}s"},
            )

            (mask_path, _conf_path, valid_path, prob_paths), elapsed_inf = step_inference(
                model,
                cfg,
                t1_path,
                t2_path,
                clip_window,
                specs,
                result_dir,
                final_name,
                args.batch_size,
                args.confidence_threshold,
                device,
            )
            if args.min_component_pixels > 0:
                apply_component_filter(mask_path, args.min_component_pixels)
            update_status(
                args.output_path,
                args.status_file,
                Process=70,
                CurrentTask="inference done",
                _elapsed={"추론": f"{round(elapsed_inf, 2)}s"},
            )

            _, elapsed_rec = step_reconstruct()
            update_status(
                args.output_path,
                args.status_file,
                Process=90,
                CurrentTask="reconstruct done",
                _elapsed={"재구성": f"{round(elapsed_rec, 2)}s"},
            )

            _, elapsed_vec = step_vectorize(
                mask_path,
                valid_path,
                prob_paths,
                output_json_path,
                final_name,
                args.min_area_m2,
                args.simplify_tolerance,
            )
            shutil.copy2(output_json_path, output_root / output_json_path.name)

            status = update_status(
                args.output_path,
                args.status_file,
                Process=100,
                Status="done",
                CurrentTask="process completed",
                _elapsed={"벡터화": f"{round(elapsed_vec, 2)}s"},
            )
            status["Current_step"] += 1
            with open(output_root / args.status_file, "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)

            print(f"{final_name}.tif process done", flush=True)

        print("=== All tasks completed ===", flush=True)

    except Exception as ex:
        msg = f"Error during processing for {final_name}: {ex}"
        print(msg, flush=True)
        log_error(args.output_path, msg)
        update_status(args.output_path, args.status_file, Status="failed")
        raise


if __name__ == "__main__":
    main()
