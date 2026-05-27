"""
도로 변화탐지 추론 스크립트 (urban_cd_v1 / DINOv3 + UPerNet 기반).

기존 02_Road_CD 인터페이스를 유지:
  - 입력: <dataset_path>/T1/*.tif, <dataset_path>/T2/*.tif (쌍 단위 처리)
  - 출력: <output_path>/<img>.json (FeatureCollection), status.json (진행률)
  - status.json 단계: 패치 생성 / 추론 / 재구성 / 벡터화

내부 추론 엔진은 ChangeMamba → DINOv3 ViT-L/16 + UPerNet (Phase 3 class-2) 으로 교체.
클래스: 1 = 신설/확장, 2 = 철거/축소.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml
from rasterio.features import rasterize, shapes
from rasterio.windows import Window, from_bounds
from shapely.geometry import mapping, shape
from shapely.ops import linemerge, unary_union
from tqdm import tqdm

# urban_cd_v1 import 경로 (dockerfile에서 PYTHONPATH로도 잡지만 안전하게 sys.path에도 추가)
URBAN_CD_DIR = "/root/urban_cd_v1"
for path in (
    f"{URBAN_CD_DIR}/common/src",
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
from postprocess_road_display_v2 import (  # noqa: E402
    component_mask_from_geometry,
    iter_lines,
    minimum_rotated_dimensions,
    skeleton_segments,
    smooth_polygon,
)

DEFAULT_CONFIG = (
    f"{URBAN_CD_DIR}/road_cd/configs/"
    "road_cd_phase3_dataset10_class2_lvd_lora_upernet.yaml"
)
DINOV3_BACKBONE_DIRNAME = "dinov3-vitl16-pretrain-lvd1689m"

# 클래스 ID → 한글 라벨
CLASS_NAME_MAP = {
    1: "신설/확장",
    2: "철거/축소",
}

# v2 후처리 파라미터 — urban_cd_v1/road_cd/scripts/postprocess_road_display_v2.py 의 기본값과 동일.
# 선형(elongated) 객체는 skeleton 추출 → buffer로 재구성하고, 그 외(또는 coverage 부족시)는 smooth만 적용.
LINEAR_ELONGATION = 1.9
LINEAR_LENGTH_MIN_M = 10.0
LINEAR_WIDTH_MAX_M = 32.0
CENTERLINE_SIMPLIFY_M = 1.2
BRANCH_MIN_LENGTH_M = 10.0
BUFFER_MIN_WIDTH_M = 5.0
BUFFER_MAX_WIDTH_M = 14.0
BUFFER_WIDTH_SCALE = 1.18
BUFFER_PADDING_M = 0.8
MIN_SOURCE_COVER = 0.85
POLYGON_SMOOTH_M = 1.0


def make_args():
    parser = argparse.ArgumentParser(description="Road change detection (urban_cd_v1)")
    parser.add_argument("--model_path", type=str, default="/workspace/model/")
    parser.add_argument("--dataset_path", type=str, default="/workspace/input/")
    parser.add_argument("--output_path", type=str, default="/workspace/output/")
    parser.add_argument("--patch_size", type=int, default=1024)
    parser.add_argument(
        "--overlap_ratio", type=str, default="25",
        help="겹침 비율 — 'min' 또는 0~100 사이의 퍼센트 값 (기본 25)",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--confidence_threshold", type=float, default=0.45)
    parser.add_argument("--min_area_m2", type=float, default=30.0)
    parser.add_argument("--simplify_tolerance", type=float, default=0.8)
    parser.add_argument("--status_file", type=str, default="status.json")
    return parser.parse_args()


def overlap_to_pixels(overlap_ratio: str, patch_size: int) -> int:
    """기존 인자 호환:
      - 'min' → 0 픽셀 (겹침 없음)
      - 0~100 퍼센트 문자열 → patch_size * pct/100 픽셀
    """
    if overlap_ratio == "min":
        return 0
    pct = float(overlap_ratio)
    if not (0.0 <= pct <= 100.0):
        raise ValueError(f"overlap_ratio percent must be 0~100, got {pct}")
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
    with open(path, "w") as f:
        json.dump(
            {
                "Total_step": total_step,
                "Current_step": 0,
                "Process": 0,
                "Status": "pending",
                "ElapsedTime": {},
            },
            f, indent=4, ensure_ascii=False,
        )


def update_status(output_path: str, status_file: str, **updates) -> dict:
    path = os.path.join(output_path, status_file)
    with open(path, "r") as f:
        st = json.load(f)
    elapsed_update = updates.pop("_elapsed", None)
    st.update(updates)
    if elapsed_update:
        st.setdefault("ElapsedTime", {}).update(elapsed_update)
    with open(path, "w") as f:
        json.dump(st, f, indent=4, ensure_ascii=False)
    return st


def log_error(output_path: str, message: str) -> None:
    with open(os.path.join(output_path, "error_log.txt"), "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")


def load_model_for_inference(model_path: Path, device: torch.device):
    """학습 가중치(*.pth 첫 파일) + DINOv3 backbone(workspace/model 하위)을 로드.

    workspace/model/
      ├── *.pth
      └── dinov3-vitl16-pretrain-lvd1689m/
          ├── config.json
          ├── preprocessor_config.json
          └── model.safetensors
    """
    from urban_cd.models import build_dino_upernet_change_detector

    checkpoint_candidates = sorted(
        p for p in model_path.glob("*.pth")
        if p.is_file() and p.stat().st_size > 0
    )
    if not checkpoint_candidates:
        raise FileNotFoundError(
            f"학습 체크포인트가 없습니다: {model_path}\n"
            f"workspace/model/ 경로에 실제 .pth 파일을 배치하세요."
        )
    checkpoint_path = checkpoint_candidates[0]
    print(f"[checkpoint] selected: {checkpoint_path.name}", flush=True)

    state = torch.load(str(checkpoint_path), map_location="cpu")

    cfg = state.get("config")
    if cfg is None:
        with open(DEFAULT_CONFIG, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    dino_dir = model_path / DINOV3_BACKBONE_DIRNAME
    safetensors_path = dino_dir / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(
            f"DINOv3 backbone 가중치를 찾을 수 없습니다: {safetensors_path}\n"
            f"workspace/model/{DINOV3_BACKBONE_DIRNAME}/ 디렉토리에 "
            f"config.json, preprocessor_config.json, model.safetensors 를 배치하세요."
        )
    cfg["model"]["backbone"]["checkpoint_path"] = str(dino_dir)

    model = build_dino_upernet_change_detector(cfg["model"])
    model.load_state_dict(state["model_state"])
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
        driver="GTiff", height=clip_h, width=clip_w,
        count=1, dtype="uint8", compress="lzw",
        tiled=True, blockxsize=block_size, blockysize=block_size,
        transform=transform,
    )
    return profile


PERCENTILE_LOW = 2.0
PERCENTILE_HIGH = 98.0
SAMPLE_SIZE_FOR_SCALE = 2048


def compute_band_scale(band_sample: np.ndarray, dtype_str: str) -> tuple:
    """샘플 영역에서 per-band (lo, hi) 컷오프 값 계산.
    - uint8: (0, 255) identity (smoke test 동작 유지)
    - 그 외 (uint16/int16/float 등): nonzero 픽셀의 2~98 percentile
      (no-data/검은 padding 영향 배제, outlier 둔감화)
    """
    if dtype_str == "uint8":
        return (0.0, 255.0)
    arr = band_sample.astype(np.float32)
    nonzero = arr[arr > 0]
    if nonzero.size == 0:
        return (0.0, 1.0)
    lo = float(np.percentile(nonzero, PERCENTILE_LOW))
    hi = float(np.percentile(nonzero, PERCENTILE_HIGH))
    if hi <= lo:
        return (lo, lo + 1.0)
    return (lo, hi)


def compute_image_scales(src) -> list:
    """이미지 중앙에서 sample 영역을 읽어 per-band (lo, hi) 컷오프 계산.
    Tile 별이 아니라 image 전체에 동일한 스케일 적용 → 타일 경계 일관성 보장.
    """
    sw = min(SAMPLE_SIZE_FOR_SCALE, src.width)
    sh = min(SAMPLE_SIZE_FOR_SCALE, src.height)
    cx = max(0, (src.width - sw) // 2)
    cy = max(0, (src.height - sh) // 2)
    sample = src.read(indexes=(1, 2, 3), window=Window(cx, cy, sw, sh))
    return [compute_band_scale(sample[i], src.dtypes[i]) for i in range(3)]


def apply_band_scale(band_arr: np.ndarray, scale: tuple, dtype_str: str) -> np.ndarray:
    """주어진 (lo, hi) 로 0~255 uint8 변환. uint8 은 그대로."""
    if dtype_str == "uint8":
        return band_arr.astype(np.uint8, copy=False)
    lo, hi = scale
    arr = band_arr.astype(np.float32)
    clipped = np.clip(arr, lo, hi)
    scaled = (clipped - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def read_tile_dtype_safe(src1, src2, clip_x: int, clip_y: int, spec,
                         scales1: list, scales2: list) -> tuple:
    """read_tile 대체 — image-level percentile 스케일 적용 + valid 항상 all-true."""
    read_window = Window(clip_x + spec.read_x, clip_y + spec.read_y,
                         spec.read_width, spec.read_height)
    raw1 = src1.read(indexes=(1, 2, 3), window=read_window)
    raw2 = src2.read(indexes=(1, 2, 3), window=read_window)
    t1_bands = [apply_band_scale(raw1[i], scales1[i], src1.dtypes[i]) for i in range(3)]
    t2_bands = [apply_band_scale(raw2[i], scales2[i], src2.dtypes[i]) for i in range(3)]
    t1 = np.moveaxis(np.stack(t1_bands, axis=0), 0, -1)
    t2 = np.moveaxis(np.stack(t2_bands, axis=0), 0, -1)
    valid = np.ones((spec.read_height, spec.read_width), dtype=bool)
    return t1, t2, valid


def log_input_properties(t1_path: Path, t2_path: Path) -> None:
    """진단용: 입력 이미지 dtype/사이즈/샘플 값 로그."""
    for label, path in (("T1", t1_path), ("T2", t2_path)):
        with rasterio.open(str(path), IGNORE_COG_LAYOUT_BREAK="YES") as src:
            sw = min(500, src.width)
            sh = min(500, src.height)
            sample = src.read(1, window=Window(0, 0, sw, sh))
            print(
                f"[input] {label} {path.name}: "
                f"size={src.width}x{src.height}, bands={src.count}, "
                f"dtypes={src.dtypes}, "
                f"band1 sample min/max/mean="
                f"{int(sample.min())}/{int(sample.max())}/{float(sample.mean()):.1f}",
                flush=True,
            )


def validate_pair_relaxed(t1_path: Path, t2_path: Path) -> PairInfo:
    """size/band/CRS 일치만 강제. transform 은 검사하지 않음 (기존 02_Road_CD 와 동일).

    실제 정사보정 산출물은 픽셀 크기와 origin 이 부동소수 노이즈 수준으로 다를 수 있어
    urban_cd_v1 의 strict equality 체크는 너무 엄격함.
    """
    with rasterio.open(t1_path, IGNORE_COG_LAYOUT_BREAK="YES") as src1, \
         rasterio.open(t2_path, IGNORE_COG_LAYOUT_BREAK="YES") as src2:
        for attr in ("width", "height", "count", "crs"):
            if getattr(src1, attr) != getattr(src2, attr):
                raise ValueError(f"Raster mismatch for {attr}: "
                                 f"{getattr(src1, attr)} != {getattr(src2, attr)}")
        if src1.count < 3 or src2.count < 3:
            raise ValueError("Both rasters must contain at least 3 bands")

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
    """패치 윈도우 명세 생성 (실제 파일은 자르지 않고 좌표만 계산)."""
    log_input_properties(t1_path, t2_path)
    pair_info = validate_pair_relaxed(t1_path, t2_path)
    clip_window = resolve_clip_window(pair_info, None)
    clip_x, clip_y, clip_w, clip_h = clip_window
    specs = build_tile_specs(clip_w, clip_h, patch_size, overlap_px)
    return pair_info, clip_window, specs


@measure_time
def step_inference(
    model, cfg, t1_path: Path, t2_path: Path, clip_window, specs,
    result_dir: Path, final_name: str, batch_size: int,
    confidence_threshold: float, device: torch.device,
):
    """타일 단위 추론. 클래스 마스크/confidence/valid 마스크를 GeoTIFF로 저장."""
    clip_x, clip_y, clip_w, clip_h = clip_window
    mask_path = result_dir / f"{final_name}.tif"
    conf_path = result_dir / f"{final_name}_conf.tif"
    valid_path = result_dir / f"{final_name}_valid.tif"

    profile = build_output_profile(t1_path, clip_window)
    mean = cfg["data"]["aug"]["normalize_mean"]
    std = cfg["data"]["aug"]["normalize_std"]
    autocast_enabled = device.type == "cuda"
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # band 4 (alpha/NIR)를 valid mask로 쓰지 않고 전부 유효 처리.
    # urban_cd_v1 의 read_tile 은 4밴드 입력 시 band4>0 만 유효로 보지만,
    # 운영 입력의 band 4가 alpha 의미가 아니거나 0인 경우 추론 결과가 다 무효 처리됨.
    # 기존 02_Road_CD 동작과 맞추기 위해 항상 all-true.
    total_pixels = 0
    pred_pixels_class1 = 0
    pred_pixels_class2 = 0
    max_conf_seen = 0.0

    with rasterio.open(str(t1_path), IGNORE_COG_LAYOUT_BREAK="YES") as src1, \
         rasterio.open(str(t2_path), IGNORE_COG_LAYOUT_BREAK="YES") as src2, \
         rasterio.open(str(mask_path), "w", BIGTIFF="YES", **profile) as mask_dst, \
         rasterio.open(str(conf_path), "w", BIGTIFF="YES", **profile) as conf_dst, \
         rasterio.open(str(valid_path), "w", BIGTIFF="YES", **profile) as valid_dst:

        # image-level per-band 스케일 1번만 계산 → 모든 타일에 동일 적용 (uint8 은 identity).
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
                # urban_cd_v1.read_tile 의 단순 uint8 캐스팅 회피용 wrapper.
                # image-level percentile 스케일 + band 4 valid_mask 무시.
                t1, t2, v = read_tile_dtype_safe(
                    src1, src2, clip_x, clip_y, spec, scales1, scales2
                )
                t1_tiles.append(t1)
                t2_tiles.append(t2)
                valid_tiles.append(v)

            t1_tensor, t2_tensor, pad = preprocess_tiles(t1_tiles, t2_tiles, mean, std, device)
            with torch.inference_mode():
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                    logits = model(t1_tensor, t2_tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()

            pad_h, pad_w = pad
            if pad_h or pad_w:
                probs = probs[:, :, : probs.shape[2] - pad_h, : probs.shape[3] - pad_w]

            for idx, spec in enumerate(batch_specs):
                prob = probs[idx]
                pred = prob.argmax(axis=0).astype(np.uint8)
                conf = prob.max(axis=0)
                pred = np.where((pred > 0) & (conf >= confidence_threshold), pred, 0).astype(np.uint8)
                conf_u8 = np.clip(np.rint(conf * 255.0), 0, 255).astype(np.uint8)
                valid = valid_tiles[idx].astype(np.uint8)

                pred_trim = trim_tile(pred, spec)
                conf_trim = trim_tile(conf_u8, spec)
                valid_trim = trim_tile(valid, spec)
                pred_trim = np.where(valid_trim > 0, pred_trim, 0).astype(np.uint8)
                conf_trim = np.where(valid_trim > 0, conf_trim, 0).astype(np.uint8)

                # 진단 통계
                total_pixels += pred_trim.size
                pred_pixels_class1 += int((pred_trim == 1).sum())
                pred_pixels_class2 += int((pred_trim == 2).sum())
                max_conf_seen = max(max_conf_seen, float(conf.max()))

                w = Window(spec.write_x, spec.write_y, spec.write_width, spec.write_height)
                mask_dst.write(pred_trim, 1, window=w)
                conf_dst.write(conf_trim, 1, window=w)
                valid_dst.write(valid_trim, 1, window=w)
            pbar.update(len(batch_specs))
        pbar.close()

    print(
        f"[inference] {final_name}: total_pixels={total_pixels}, "
        f"class1(신설/확장)={pred_pixels_class1}, class2(철거/축소)={pred_pixels_class2}, "
        f"max_conf={max_conf_seen:.3f}, threshold={confidence_threshold}",
        flush=True,
    )
    return mask_path, conf_path, valid_path


@measure_time
def step_reconstruct():
    """신규 추론 엔진은 mosaic을 추론 단계에서 직접 작성하므로 별도 작업 없음."""
    return "no-op"


def polygon_mean_conf(conf_src, polygon) -> float:
    window = from_bounds(*polygon.bounds, transform=conf_src.transform)
    window = window.round_offsets().round_lengths()
    col_off = max(0, int(window.col_off))
    row_off = max(0, int(window.row_off))
    width = min(conf_src.width - col_off, max(1, int(window.width)))
    height = min(conf_src.height - row_off, max(1, int(window.height)))
    window = Window(col_off, row_off, width, height)
    data = conf_src.read(1, window=window)
    transform = conf_src.window_transform(window)
    mask = rasterize([(polygon, 1)], out_shape=data.shape, fill=0, transform=transform, dtype=np.uint8)
    valid_values = data[mask == 1]
    if valid_values.size == 0:
        return 0.0
    return float(valid_values.mean() / 255.0)


def postprocess_polygon_v2(polygon, mask_src, polygon_simplify_m: float):
    """v2-style 후처리.

    - 선형(elongated) 객체: skeleton 추출 → 추정 폭으로 buffer 재구성.
      buffer가 원본 폴리곤의 ``MIN_SOURCE_COVER`` 미만만 커버할 경우 (큰 객체 누락
      방지를 위한 fallback) smooth-only 폴리곤으로 회귀.
    - 비선형 객체: smooth + simplify 만 적용.
    """
    if polygon is None or polygon.is_empty:
        return None

    width_m, length_m = minimum_rotated_dimensions(polygon)
    elongation = float(length_m / max(width_m, 1e-6))
    area_m2 = float(polygon.area)
    is_linear = (
        elongation >= LINEAR_ELONGATION
        and length_m >= LINEAR_LENGTH_MIN_M
        and width_m <= LINEAR_WIDTH_MAX_M
    )

    if is_linear:
        mask, _, local_transform = component_mask_from_geometry(polygon, mask_src)
        lines = skeleton_segments(mask, local_transform)
        if lines:
            if len(lines) > 1:
                lines = [line for line in lines if line.length >= BRANCH_MIN_LENGTH_M] \
                    or [max(lines, key=lambda x: x.length)]
            merged = lines[0] if len(lines) == 1 else linemerge(unary_union(lines))
            simplified = [line.simplify(CENTERLINE_SIMPLIFY_M, preserve_topology=False)
                          for line in iter_lines(merged)]
            simplified = [line for line in simplified if line.length > 0]
            total_length = sum(line.length for line in simplified)
            if total_length > 0:
                est_width = float(np.clip(area_m2 / total_length,
                                          BUFFER_MIN_WIDTH_M, BUFFER_MAX_WIDTH_M))
                disp_width = float(np.clip(est_width * BUFFER_WIDTH_SCALE,
                                           BUFFER_MIN_WIDTH_M, BUFFER_MAX_WIDTH_M))
                buffer_radius = disp_width * 0.5 + BUFFER_PADDING_M
                buffered = unary_union([
                    line.buffer(buffer_radius, cap_style=1, join_style=1)
                    for line in simplified
                ])
                buffered = smooth_polygon(buffered, POLYGON_SMOOTH_M * 0.5, polygon_simplify_m)
                if not buffered.is_empty:
                    coverage = float(buffered.intersection(polygon).area / max(area_m2, 1e-9))
                    if coverage >= MIN_SOURCE_COVER:
                        return buffered

    # 비선형이거나 coverage 부족 → smooth-only fallback
    return smooth_polygon(polygon, POLYGON_SMOOTH_M, polygon_simplify_m)


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
    mask_path: Path, conf_path: Path, valid_path: Path,
    output_json_path: Path, img_name: str,
    min_area_m2: float, simplify_tolerance: float,
):
    """클래스 마스크를 폴리곤으로 변환 → v2-style 후처리 → 기존 GeoJSON 포맷으로 저장."""
    raw_features = []
    n_total_shapes = 0
    n_class_match = 0
    n_after_pre_area = 0
    n_after_postprocess = 0
    n_final = 0
    with rasterio.open(str(mask_path)) as mask_src, \
         rasterio.open(str(conf_path)) as conf_src, \
         rasterio.open(str(valid_path)) as valid_src:

        crs = mask_src.crs
        crs_str = f"EPSG:{crs.to_epsg()}" if crs else None

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
            if polygon is None or polygon.is_empty:
                continue
            if polygon.area < min_area_m2:
                continue
            n_after_pre_area += 1

            # confidence는 후처리 전 raw 폴리곤 기준 (postprocess가 도형을 변형하므로)
            conf_val = polygon_mean_conf(conf_src, polygon)

            transformed = postprocess_polygon_v2(polygon, mask_src, simplify_tolerance)
            if transformed is None:
                continue
            n_after_postprocess += 1

            for sub in iter_polygons(transformed):
                if sub.area < min_area_m2:
                    continue
                centroid = sub.centroid
                raw_features.append({
                    "polygon": sub,
                    "class_id": class_id,
                    "area": float(sub.area),
                    "conf": conf_val,
                    "cx": centroid.x,
                    "cy": centroid.y,
                })
                n_final += 1
    print(
        f"[vectorize] {img_name}: shapes={n_total_shapes}, "
        f"class_match={n_class_match}, after_pre_area>=({min_area_m2})={n_after_pre_area}, "
        f"after_postprocess={n_after_postprocess}, final={n_final}",
        flush=True,
    )

    # 기존 코드와 동일한 정렬: y desc, x desc
    raw_features.sort(key=lambda f: (f["cy"], f["cx"]), reverse=True)

    geojson_features = []
    for idx, f in enumerate(raw_features, start=1):
        geojson_features.append({
            "type": "Feature",
            "properties": {
                "CLS_ID": f["class_id"],
                "CLS_NAME": CLASS_NAME_MAP[f["class_id"]],
                "CONF": round(f["conf"] * 100.0, 2),
                "AREA": round(f["area"], 2),
                "ID": idx,
            },
            "geometry": mapping(f["polygon"]),
        })

    geojson_data = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": f"{crs_str}"}},
        "map_number": img_name,
        "features": geojson_features,
    }
    with open(output_json_path, "w") as f:
        json.dump(geojson_data, f, indent=4, ensure_ascii=False)

    return f"Vectorization done: {len(geojson_features)} features"


def main():
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
        raise RuntimeError("=> Please match before/after images")

    output_root = Path(args.output_path)
    output_root.mkdir(parents=True, exist_ok=True)

    init_status(args.output_path, args.status_file, total_step=len(t1_files))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model_for_inference(Path(args.model_path), device)
    overlap_px = overlap_to_pixels(args.overlap_ratio, args.patch_size)

    init_results_dir = output_root / "results"
    init_results_dir.mkdir(parents=True, exist_ok=True)

    final_name = None
    try:
        for img_filename in t1_files:
            base = img_filename.rsplit(".", 1)[0]
            final_name = base
            t1_path = init_t1_path / img_filename
            t2_path = init_t2_path / img_filename
            result_dir = init_results_dir / final_name
            result_dir.mkdir(parents=True, exist_ok=True)
            output_json_path = result_dir / f"{final_name}.json"

            update_status(args.output_path, args.status_file,
                          Status="in progress", final_name=final_name)

            # (1) 패치 (윈도우 좌표 계산)
            (_, clip_window, specs), elapsed_patch = step_patch(
                t1_path, t2_path, args.patch_size, overlap_px)
            update_status(args.output_path, args.status_file,
                          Process=20, CurrentTask="patch done",
                          _elapsed={"패치 생성": f"{round(elapsed_patch, 2)}s"})

            # (2) 추론
            (mask_path, conf_path, valid_path), elapsed_inf = step_inference(
                model, cfg, t1_path, t2_path, clip_window, specs,
                result_dir, final_name,
                args.batch_size, args.confidence_threshold, device)
            update_status(args.output_path, args.status_file,
                          Process=70, CurrentTask="inference done",
                          _elapsed={"추론": f"{round(elapsed_inf, 2)}s"})

            # (3) 재구성 (no-op, 추론 단계에서 직접 mosaic 작성)
            _, elapsed_rec = step_reconstruct()
            update_status(args.output_path, args.status_file,
                          Process=90, CurrentTask="reconstruct done",
                          _elapsed={"재구성": f"{round(elapsed_rec, 2)}s"})

            # (4) 벡터화
            _, elapsed_vec = step_vectorize(
                mask_path, conf_path, valid_path, output_json_path,
                final_name, args.min_area_m2, args.simplify_tolerance)
            update_status(args.output_path, args.status_file,
                          Process=100, Status="done",
                          CurrentTask="process completed",
                          _elapsed={"벡터화": f"{round(elapsed_vec, 2)}s"})

            # current_step 증가, 결과 JSON을 output 루트로 복사
            with open(os.path.join(args.output_path, args.status_file), "r") as f:
                st = json.load(f)
            st["Current_step"] += 1
            with open(os.path.join(args.output_path, args.status_file), "w") as f:
                json.dump(st, f, indent=4, ensure_ascii=False)

            subprocess.run(["cp", str(output_json_path), str(output_root)], check=True)
            print(f"{final_name}.tif process done")

        # 중간 산출물 정리
        subprocess.run(["rm", "-rf", str(init_results_dir)], check=False)
        print("=== All tasks completed ===")

    except Exception as ex:
        msg = f"Error during processing for {final_name}: {ex}"
        print(msg)
        log_error(args.output_path, msg)
        update_status(args.output_path, args.status_file, Status="failed")
        raise


if __name__ == "__main__":
    main()
