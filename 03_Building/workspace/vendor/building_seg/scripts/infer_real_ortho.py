from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import shutil
import sys
from typing import Sequence

import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.windows import Window
from shapely.geometry import mapping, shape
import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
COMMON_SRC = PROJECT_DIR.parents[0] / "common" / "src"
if str(COMMON_SRC) not in sys.path:
    sys.path.insert(0, str(COMMON_SRC))

from urban_cd.models import build_dino_upernet_segmentor


DEFAULT_CONFIG = (
    PROJECT_DIR / "configs" / "phase2_target_domain_dinov3_vitl16_lvd_lora_last4_upernet.yaml"
)
DEFAULT_CHECKPOINT = (
    PROJECT_DIR
    / "outputs"
    / "building_seg_phase2_target_domain_dinov3_vitl16_lvd_lora_last4_upernet"
    / "bseg_phase2_target_domain_main_v1-20260413-013537"
    / "best.pth"
)
DEFAULT_INPUT = Path("/home/user/data/ihlee/HiSup/anyang/input/anyang_2024.tif")
DEFAULT_OUTPUT_ROOT = Path("/home/user/data/ihlee/HiSup/anyang/output/urban_cd_v1_building_seg")


@dataclass(frozen=True)
class RasterInfo:
    image_path: str
    width: int
    height: int
    count: int
    crs: str | None
    res_x: float
    res_y: float
    pixel_area: float


@dataclass(frozen=True)
class TileSpec:
    tile_id: int
    row: int
    col: int
    read_x: int
    read_y: int
    read_width: int
    read_height: int
    write_x: int
    write_y: int
    write_width: int
    write_height: int
    crop_left: int
    crop_right: int
    crop_top: int
    crop_bottom: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run building segmentation on a large orthophoto raster")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Input orthophoto TIFF path")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="building_seg YAML config")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="best.pth checkpoint")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Root directory for run outputs")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run directory name")
    parser.add_argument("--patch-size", type=int, default=512, help="Tile size for model inference")
    parser.add_argument("--overlap", type=int, default=128, help="Overlap in pixels between adjacent tiles")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for binary building mask")
    parser.add_argument("--clip-window", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"), default=None)
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device, for example cuda:0 or cpu")
    parser.add_argument("--num-preview-tiles", type=int, default=8, help="Number of tile QC panels to save")
    parser.add_argument("--preview-max-size", type=int, default=2400, help="Longest side for preview PNGs")
    parser.add_argument("--min-area-m2", type=float, default=10.0, help="Minimum polygon area in square meters")
    parser.add_argument("--simplify-tolerance", type=float, default=0.15, help="Polygon simplify tolerance in CRS units")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it already exists")
    parser.add_argument(
        "--save-prob-float32",
        action="store_true",
        help="Additionally save the raw sigmoid probabilities as a float32 GeoTIFF "
             "(mosaic/building_prob_f32.tif). The uint8 raster is kept for compactness.",
    )
    return parser.parse_args()


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def validate_raster(image_path: Path) -> RasterInfo:
    with rasterio.open(image_path, IGNORE_COG_LAYOUT_BREAK="YES") as src:
        if src.count < 3:
            raise ValueError("Input raster must contain at least 3 bands")
        return RasterInfo(
            image_path=str(image_path),
            width=src.width,
            height=src.height,
            count=src.count,
            crs=src.crs.to_string() if src.crs else None,
            res_x=float(src.res[0]),
            res_y=float(src.res[1]),
            pixel_area=float(abs(src.transform.a * src.transform.e)),
        )


def build_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def resolve_clip_window(raster_info: RasterInfo, clip_window: Sequence[int] | None) -> tuple[int, int, int, int]:
    if clip_window is None:
        return 0, 0, raster_info.width, raster_info.height
    clip_x, clip_y, clip_w, clip_h = [int(v) for v in clip_window]
    if clip_w <= 0 or clip_h <= 0:
        raise ValueError("clip width and height must be positive")
    if clip_x < 0 or clip_y < 0:
        raise ValueError("clip origin must be non-negative")
    if clip_x + clip_w > raster_info.width or clip_y + clip_h > raster_info.height:
        raise ValueError("clip window exceeds raster bounds")
    return clip_x, clip_y, clip_w, clip_h


def build_tile_specs(clip_w: int, clip_h: int, patch_size: int, overlap: int) -> list[TileSpec]:
    if overlap < 0 or overlap >= patch_size:
        raise ValueError("overlap must be in [0, patch_size)")
    stride = patch_size - overlap
    xs = build_positions(clip_w, patch_size, stride)
    ys = build_positions(clip_h, patch_size, stride)
    left_half = overlap // 2
    right_half = overlap - left_half
    specs: list[TileSpec] = []
    tile_id = 0
    for row, y in enumerate(ys):
        read_h = min(patch_size, clip_h) if clip_h <= patch_size else patch_size
        for col, x in enumerate(xs):
            read_w = min(patch_size, clip_w) if clip_w <= patch_size else patch_size
            crop_left = 0 if col == 0 else left_half
            crop_right = 0 if col == len(xs) - 1 else right_half
            crop_top = 0 if row == 0 else left_half
            crop_bottom = 0 if row == len(ys) - 1 else right_half
            write_x = x + crop_left
            write_y = y + crop_top
            write_w = read_w - crop_left - crop_right
            write_h = read_h - crop_top - crop_bottom
            specs.append(
                TileSpec(
                    tile_id=tile_id,
                    row=row,
                    col=col,
                    read_x=x,
                    read_y=y,
                    read_width=read_w,
                    read_height=read_h,
                    write_x=write_x,
                    write_y=write_y,
                    write_width=write_w,
                    write_height=write_h,
                    crop_left=crop_left,
                    crop_right=crop_right,
                    crop_top=crop_top,
                    crop_bottom=crop_bottom,
                )
            )
            tile_id += 1
    return specs


def pad_tensor_to_multiple(tensor: torch.Tensor, multiple: int = 16) -> tuple[torch.Tensor, tuple[int, int]]:
    _, _, height, width = tensor.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, (0, 0)
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, (pad_h, pad_w)


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dino_path = resolve_path(cfg["model"]["backbone"]["checkpoint_path"], PROJECT_DIR)
    cfg["model"]["backbone"]["checkpoint_path"] = str(dino_path)
    model = build_dino_upernet_segmentor(cfg["model"])
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()
    return model, cfg


def prepare_run_dir(output_root: Path, run_name: str, overwrite: bool) -> dict[str, Path]:
    run_dir = output_root / run_name
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "run_dir": run_dir,
        "meta": run_dir / "meta",
        "mosaic": run_dir / "mosaic",
        "vector": run_dir / "vector",
        "qc": run_dir / "qc",
        "tile_qc": run_dir / "qc" / "tiles",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def make_default_run_name(image_path: Path, clip_window: Sequence[int] | None) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if clip_window is None:
        return f"{stamp}__{image_path.stem}"
    clip_x, clip_y, clip_w, clip_h = [int(v) for v in clip_window]
    return f"{stamp}__{image_path.stem}__clip_x{clip_x}_y{clip_y}_w{clip_w}_h{clip_h}"


def write_tile_index(csv_path: Path, specs: Sequence[TileSpec], clip_x: int, clip_y: int) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "tile_id",
                "row",
                "col",
                "global_read_x",
                "global_read_y",
                "read_width",
                "read_height",
                "local_write_x",
                "local_write_y",
                "write_width",
                "write_height",
            ]
        )
        for spec in specs:
            writer.writerow(
                [
                    spec.tile_id,
                    spec.row,
                    spec.col,
                    clip_x + spec.read_x,
                    clip_y + spec.read_y,
                    spec.read_width,
                    spec.read_height,
                    spec.write_x,
                    spec.write_y,
                    spec.write_width,
                    spec.write_height,
                ]
            )


def read_tile(
    src: rasterio.DatasetReader,
    clip_x: int,
    clip_y: int,
    spec: TileSpec,
) -> tuple[np.ndarray, np.ndarray]:
    read_window = Window(clip_x + spec.read_x, clip_y + spec.read_y, spec.read_width, spec.read_height)
    image = np.moveaxis(src.read(indexes=(1, 2, 3), window=read_window), 0, -1).astype(np.uint8)
    if src.count >= 4:
        valid = src.read(4, window=read_window) > 0
    else:
        valid = np.ones((spec.read_height, spec.read_width), dtype=bool)
    return image, valid


def preprocess_tiles(
    tiles: Sequence[np.ndarray],
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int]]:
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    batch = np.stack([(tile.astype(np.float32) / 255.0 - mean_arr) / std_arr for tile in tiles], axis=0)
    tensor = torch.from_numpy(np.moveaxis(batch, -1, 1)).to(device=device, dtype=torch.float32)
    tensor, pad = pad_tensor_to_multiple(tensor)
    return tensor, pad


def trim_tile(array: np.ndarray, spec: TileSpec) -> np.ndarray:
    return array[
        spec.crop_top : spec.crop_top + spec.write_height,
        spec.crop_left : spec.crop_left + spec.write_width,
    ]


def save_tile_preview(output_path: Path, image: np.ndarray, prob_u8: np.ndarray, mask: np.ndarray) -> None:
    prob_rgb = np.stack([prob_u8, np.zeros_like(prob_u8), np.zeros_like(prob_u8)], axis=-1)
    mask_rgb = np.stack([mask * 255, np.zeros_like(mask), np.zeros_like(mask)], axis=-1).astype(np.uint8)
    overlay = image.copy().astype(np.float32)
    building_mask = mask > 0
    overlay[building_mask] = 0.65 * overlay[building_mask] + 0.35 * np.array([255, 32, 32], dtype=np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    canvas = Image.new("RGB", (image.shape[1] * 2, image.shape[0] * 2))
    canvas.paste(Image.fromarray(image), (0, 0))
    canvas.paste(Image.fromarray(prob_rgb), (image.shape[1], 0))
    canvas.paste(Image.fromarray(overlay), (0, image.shape[0]))
    canvas.paste(Image.fromarray(mask_rgb), (image.shape[1], image.shape[0]))
    canvas.save(output_path)


def run_inference(
    model: torch.nn.Module,
    cfg: dict,
    image_path: Path,
    paths: dict[str, Path],
    clip_x: int,
    clip_y: int,
    clip_w: int,
    clip_h: int,
    specs: Sequence[TileSpec],
    batch_size: int,
    threshold: float,
    num_preview_tiles: int,
    device: torch.device,
    save_prob_float32: bool = False,
) -> dict[str, object]:
    prob_path = paths["mosaic"] / "building_prob_u8.tif"
    prob_f32_path = paths["mosaic"] / "building_prob_f32.tif"
    mask_path = paths["mosaic"] / "building_mask.tif"
    valid_path = paths["mosaic"] / "valid_mask.tif"

    with rasterio.open(image_path, IGNORE_COG_LAYOUT_BREAK="YES") as src:
        output_transform = src.window_transform(Window(clip_x, clip_y, clip_w, clip_h))
        output_profile = src.profile.copy()
    block_size = max(16, min(512, clip_w, clip_h))
    block_size = block_size - (block_size % 16)
    block_size = max(16, block_size)
    output_profile.update(
        driver="GTiff",
        height=clip_h,
        width=clip_w,
        count=1,
        dtype="uint8",
        compress="lzw",
        tiled=True,
        blockxsize=block_size,
        blockysize=block_size,
        transform=output_transform,
    )

    mean = cfg["data"]["aug"]["normalize_mean"]
    std = cfg["data"]["aug"]["normalize_std"]
    tile_preview_count = 0
    autocast_enabled = device.type == "cuda"
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    f32_profile = output_profile.copy()
    f32_profile.update(dtype="float32")

    from contextlib import ExitStack
    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(image_path, IGNORE_COG_LAYOUT_BREAK="YES"))
        prob_dst = stack.enter_context(rasterio.open(prob_path, "w", BIGTIFF="YES", **output_profile))
        prob_f32_dst = stack.enter_context(rasterio.open(prob_f32_path, "w", BIGTIFF="YES", **f32_profile)) if save_prob_float32 else None
        mask_dst = stack.enter_context(rasterio.open(mask_path, "w", BIGTIFF="YES", **output_profile))
        valid_dst = stack.enter_context(rasterio.open(valid_path, "w", BIGTIFF="YES", **output_profile))
        pbar = tqdm(total=len(specs), desc="Infer orthophoto tiles")
        for start in range(0, len(specs), batch_size):
            batch_specs = list(specs[start : start + batch_size])
            tiles: list[np.ndarray] = []
            valid_tiles: list[np.ndarray] = []
            for spec in batch_specs:
                tile, valid_tile = read_tile(src, clip_x, clip_y, spec)
                tiles.append(tile)
                valid_tiles.append(valid_tile)

            tensor, pad = preprocess_tiles(tiles, mean, std, device)
            with torch.inference_mode():
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                    logits = model(tensor)
                    probs = torch.sigmoid(logits).cpu().numpy()

            pad_h, pad_w = pad
            if pad_h or pad_w:
                probs = probs[:, :, : probs.shape[2] - pad_h, : probs.shape[3] - pad_w]

            for idx, spec in enumerate(batch_specs):
                prob = probs[idx, 0]
                prob_u8 = np.clip(np.rint(prob * 255.0), 0, 255).astype(np.uint8)
                mask = (prob >= threshold).astype(np.uint8)
                valid = valid_tiles[idx].astype(np.uint8)

                prob_trim = trim_tile(prob_u8, spec)
                mask_trim = trim_tile(mask, spec)
                valid_trim = trim_tile(valid, spec)
                mask_trim = np.where(valid_trim > 0, mask_trim, 0).astype(np.uint8)
                prob_trim = np.where(valid_trim > 0, prob_trim, 0).astype(np.uint8)

                write_window = Window(spec.write_x, spec.write_y, spec.write_width, spec.write_height)
                prob_dst.write(prob_trim, 1, window=write_window)
                mask_dst.write(mask_trim, 1, window=write_window)
                valid_dst.write(valid_trim, 1, window=write_window)

                if prob_f32_dst is not None:
                    prob_f32_trim = trim_tile(prob.astype(np.float32), spec)
                    prob_f32_trim = np.where(valid_trim > 0, prob_f32_trim, 0.0).astype(np.float32)
                    prob_f32_dst.write(prob_f32_trim, 1, window=write_window)

                if tile_preview_count < num_preview_tiles:
                    preview_path = paths["tile_qc"] / f"tile_{spec.tile_id:05d}_r{spec.row:03d}_c{spec.col:03d}.png"
                    save_tile_preview(preview_path, tiles[idx], prob_u8, mask.astype(np.uint8))
                    tile_preview_count += 1

            pbar.update(len(batch_specs))
        pbar.close()

    return {
        "prob_path": str(prob_path),
        "prob_f32_path": str(prob_f32_path) if save_prob_float32 else None,
        "mask_path": str(mask_path),
        "valid_path": str(valid_path),
        "tile_count": len(specs),
    }


def polygon_mean_probability(prob_src: rasterio.DatasetReader, polygon) -> float:
    window = rasterio.windows.from_bounds(*polygon.bounds, transform=prob_src.transform)
    window = window.round_offsets().round_lengths()
    window = Window(
        max(0, int(window.col_off)),
        max(0, int(window.row_off)),
        min(prob_src.width - max(0, int(window.col_off)), max(1, int(window.width))),
        min(prob_src.height - max(0, int(window.row_off)), max(1, int(window.height))),
    )
    data = prob_src.read(1, window=window)
    window_transform = prob_src.window_transform(window)
    mask = rasterize([(polygon, 1)], out_shape=data.shape, fill=0, transform=window_transform, dtype=np.uint8)
    valid_values = data[mask == 1]
    if valid_values.size == 0:
        return 0.0
    return float(valid_values.mean() / 255.0)


def vectorize_outputs(
    mask_path: Path,
    prob_path: Path,
    valid_path: Path,
    vector_dir: Path,
    min_area_m2: float,
    simplify_tolerance: float,
) -> dict[str, object]:
    geojson_path = vector_dir / "building.geojson"
    gpkg_path = vector_dir / "building.gpkg"
    csv_path = vector_dir / "building_features.csv"

    features_out: list[dict] = []
    mask_crs = None
    with rasterio.open(mask_path) as mask_src, rasterio.open(prob_path) as prob_src, rasterio.open(valid_path) as valid_src:
        mask_crs = mask_src.crs
        iterator = shapes(
            rasterio.band(mask_src, 1),
            mask=rasterio.band(valid_src, 1),
            transform=mask_src.transform,
            connectivity=8,
        )
        for geom, value in tqdm(iterator, desc="Vectorize mask"):
            if int(value) != 1:
                continue
            polygon = shape(geom)
            if polygon.is_empty or not polygon.is_valid:
                continue
            if simplify_tolerance > 0:
                polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)
            area_m2 = float(polygon.area)
            if area_m2 < min_area_m2:
                continue
            conf = polygon_mean_probability(prob_src, polygon)
            feature_id = len(features_out) + 1
            features_out.append(
                {
                    "type": "Feature",
                    "properties": {
                        "id": feature_id,
                        "label": "building",
                        "area_m2": round(area_m2, 3),
                        "prob_mean": round(conf, 4),
                    },
                    "geometry": mapping(polygon),
                }
            )

    feature_collection = {"type": "FeatureCollection", "features": features_out}
    with geojson_path.open("w", encoding="utf-8") as handle:
        json.dump(feature_collection, handle, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label", "area_m2", "prob_mean"])
        for feature in features_out:
            props = feature["properties"]
            writer.writerow([props["id"], props["label"], props["area_m2"], props["prob_mean"]])

    gpkg_written = False
    try:
        import geopandas as gpd

        gdf = gpd.GeoDataFrame.from_features(features_out, crs=mask_crs)
        gdf.to_file(gpkg_path, driver="GPKG")
        gpkg_written = True
    except Exception:
        gpkg_written = False

    total_area = round(sum(feature["properties"]["area_m2"] for feature in features_out), 3)
    return {
        "geojson_path": str(geojson_path),
        "csv_path": str(csv_path),
        "gpkg_path": str(gpkg_path) if gpkg_written else None,
        "feature_count": len(features_out),
        "total_building_area_m2": total_area,
    }


def save_preview(
    image_path: Path,
    mask_path: Path,
    qc_dir: Path,
    clip_x: int,
    clip_y: int,
    clip_w: int,
    clip_h: int,
    max_size: int,
) -> dict[str, str]:
    longest = max(clip_w, clip_h)
    scale = min(1.0, max_size / float(longest))
    out_w = max(1, int(round(clip_w * scale)))
    out_h = max(1, int(round(clip_h * scale)))
    read_window = Window(clip_x, clip_y, clip_w, clip_h)

    with rasterio.open(image_path, IGNORE_COG_LAYOUT_BREAK="YES") as src:
        image = np.moveaxis(
            src.read(indexes=(1, 2, 3), window=read_window, out_shape=(3, out_h, out_w), resampling=Resampling.bilinear),
            0,
            -1,
        )
    with rasterio.open(mask_path) as mask_src:
        mask = mask_src.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest)

    overlay = image.copy().astype(np.float32)
    building_mask = mask > 0
    overlay[building_mask] = 0.65 * overlay[building_mask] + 0.35 * np.array([255, 32, 32], dtype=np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    mask_rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    mask_rgb[building_mask] = np.array([255, 255, 255], dtype=np.uint8)

    image_preview = qc_dir / "preview_image.png"
    overlay_preview = qc_dir / "preview_overlay.png"
    mask_preview = qc_dir / "preview_mask.png"
    Image.fromarray(image).save(image_preview)
    Image.fromarray(overlay).save(overlay_preview)
    Image.fromarray(mask_rgb).save(mask_preview)
    return {
        "preview_image": str(image_preview),
        "preview_overlay": str(overlay_preview),
        "preview_mask": str(mask_preview),
    }


def write_run_readme(
    output_path: Path,
    run_name: str,
    raster_info: RasterInfo,
    config_path: Path,
    checkpoint_path: Path,
    clip_window: tuple[int, int, int, int],
    patch_size: int,
    overlap: int,
    batch_size: int,
    threshold: float,
    min_area_m2: float,
    inference_meta: dict[str, object],
    vector_meta: dict[str, object],
) -> None:
    clip_x, clip_y, clip_w, clip_h = clip_window
    lines = [
        f"# {run_name}",
        "",
        "## Summary",
        "",
        f"- Image: `{raster_info.image_path}`",
        f"- Config: `{config_path}`",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Raster size: `{raster_info.width} x {raster_info.height}`",
        f"- Clip window: `x={clip_x}, y={clip_y}, width={clip_w}, height={clip_h}`",
        f"- CRS: `{raster_info.crs}`",
        f"- Resolution: `{raster_info.res_x:.3f} x {abs(raster_info.res_y):.3f} m`",
        f"- Patch size: `{patch_size}`",
        f"- Overlap: `{overlap}`",
        f"- Batch size: `{batch_size}`",
        f"- Threshold: `{threshold}`",
        f"- Minimum polygon area: `{min_area_m2} m^2`",
        "",
        "## Outputs",
        "",
        f"- Tiles processed: `{inference_meta['tile_count']}`",
        "- Binary mask: `mosaic/building_mask.tif`",
        "- Probability raster: `mosaic/building_prob_u8.tif`",
        "- Valid mask: `mosaic/valid_mask.tif`",
        "- GeoJSON: `vector/building.geojson`",
        "- Feature CSV: `vector/building_features.csv`",
        f"- Feature count: `{vector_meta['feature_count']}`",
        f"- Total building area: `{vector_meta['total_building_area_m2']} m^2`",
        "",
        "## Quicklook",
        "",
        "- `qc/preview_overlay.png` overlays the final building mask on the orthophoto.",
        "- `qc/tiles/` stores tile-level QC panels for seam and threshold spot-checking.",
        "",
    ]
    if vector_meta.get("gpkg_path"):
        lines.insert(lines.index("## Quicklook") - 1, "- GPKG: `vector/building.gpkg`")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_run_config(
    output_path: Path,
    args: argparse.Namespace,
    raster_info: RasterInfo,
    clip_window: tuple[int, int, int, int],
    inference_meta: dict[str, object],
    vector_meta: dict[str, object],
    preview_meta: dict[str, str],
) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "raster_info": asdict(raster_info),
        "clip_window": {
            "x": clip_window[0],
            "y": clip_window[1],
            "width": clip_window[2],
            "height": clip_window[3],
        },
        "inference": inference_meta,
        "vector": vector_meta,
        "preview": preview_meta,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    image_path = resolve_path(args.input, PROJECT_DIR)
    config_path = resolve_path(args.config, PROJECT_DIR)
    checkpoint_path = resolve_path(args.checkpoint, PROJECT_DIR)
    output_root = resolve_path(args.output_root, PROJECT_DIR)

    raster_info = validate_raster(image_path)
    clip_window = resolve_clip_window(raster_info, args.clip_window)
    run_name = args.run_name or make_default_run_name(image_path, args.clip_window)
    paths = prepare_run_dir(output_root, run_name, args.overwrite)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, cfg = load_model(config_path, checkpoint_path, device)

    clip_x, clip_y, clip_w, clip_h = clip_window
    specs = build_tile_specs(clip_w, clip_h, args.patch_size, args.overlap)
    write_tile_index(paths["meta"] / "tile_index.csv", specs, clip_x, clip_y)

    inference_meta = run_inference(
        model=model,
        cfg=cfg,
        image_path=image_path,
        paths=paths,
        clip_x=clip_x,
        clip_y=clip_y,
        clip_w=clip_w,
        clip_h=clip_h,
        specs=specs,
        batch_size=args.batch_size,
        threshold=args.threshold,
        num_preview_tiles=args.num_preview_tiles,
        device=device,
        save_prob_float32=args.save_prob_float32,
    )
    preview_meta = save_preview(
        image_path=image_path,
        mask_path=Path(inference_meta["mask_path"]),
        qc_dir=paths["qc"],
        clip_x=clip_x,
        clip_y=clip_y,
        clip_w=clip_w,
        clip_h=clip_h,
        max_size=args.preview_max_size,
    )
    vector_meta = vectorize_outputs(
        mask_path=Path(inference_meta["mask_path"]),
        prob_path=Path(inference_meta["prob_path"]),
        valid_path=Path(inference_meta["valid_path"]),
        vector_dir=paths["vector"],
        min_area_m2=args.min_area_m2,
        simplify_tolerance=args.simplify_tolerance,
    )

    write_run_readme(
        output_path=paths["run_dir"] / "README.md",
        run_name=run_name,
        raster_info=raster_info,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        clip_window=clip_window,
        patch_size=args.patch_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        threshold=args.threshold,
        min_area_m2=args.min_area_m2,
        inference_meta=inference_meta,
        vector_meta=vector_meta,
    )
    write_run_config(
        output_path=paths["meta"] / "run_config.json",
        args=args,
        raster_info=raster_info,
        clip_window=clip_window,
        inference_meta=inference_meta,
        vector_meta=vector_meta,
        preview_meta=preview_meta,
    )
    print(json.dumps({"run_dir": str(paths["run_dir"]), **inference_meta, **vector_meta}, indent=2))


if __name__ == "__main__":
    main()
