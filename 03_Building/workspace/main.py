#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.windows import Window, from_bounds
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box, shape
from shapely.ops import unary_union


APP_DIR = Path(__file__).resolve().parent
VENDOR_DIR = APP_DIR / "vendor"
MODEL_DIR = APP_DIR / "model"

DXF_CONVERTER = VENDOR_DIR / "dxf_to_shp.py"
CHANGE_DETECTION_DIR = VENDOR_DIR / "change_detection"
BUILDING_SEG_DIR = VENDOR_DIR / "building_seg"

DEFAULT_SEG_CONFIG = (
    BUILDING_SEG_DIR
    / "configs"
    / "phase2_target_domain_dinov3_vitl16_lvd_lora_last4_upernet.yaml"
)
DEFAULT_SEG_CHECKPOINT = MODEL_DIR / "best.pth"

EPSG = 5186
DXF_BUILDING_CATEGORIES = ("기성건물", "수정도화")
RAW_TO_ERROR_NAME = {
    "신축": "초과 오류",
    "소멸": "누락 오류",
    "갱신": "묘사 오류",
}
ERROR_TO_ID = {
    "초과 오류": 1,
    "누락 오류": 2,
    "묘사 오류": 3,
}
DXF_LAYER_BY_ERROR = {
    "초과 오류": "EXCESS_ERROR",
    "누락 오류": "MISSING_ERROR",
    "묘사 오류": "DESCRIPTION_ERROR",
}
DXF_COLOR_BY_ERROR = {
    "초과 오류": 1,
    "누락 오류": 3,
    "묘사 오류": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Building change/error detection from one TIF and one or more DXF files"
    )
    parser.add_argument("--dataset_path", "--input", dest="dataset_path", type=str, default="/workspace/input")
    parser.add_argument("--tif_path", type=str, default=None)
    parser.add_argument(
        "--dxf_input",
        type=str,
        default=None,
        help="DXF file, DXF directory, comma-separated DXF list, or txt file with one DXF per line",
    )
    parser.add_argument("--output_path", "--output", dest="output_path", type=str, default="/workspace/output")
    parser.add_argument("--model_path", "--model", dest="model_path", type=str, default="/workspace/model")
    parser.add_argument("--epsg", type=int, default=EPSG)
    parser.add_argument("--dxf_workers", type=int, default=0)
    parser.add_argument("--layer_prefix", type=str, default="B")
    parser.add_argument("--dxf_xy_tol", type=float, default=0.01)
    parser.add_argument("--dxf_z_tol", type=float, default=1.0)

    parser.add_argument("--seg_config", type=str, default=None)
    parser.add_argument("--seg_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seg_threshold", type=float, default=0.5)
    parser.add_argument("--min_area_m2", type=float, default=10.0)
    parser.add_argument(
        "--result_min_area_m2",
        type=float,
        default=50.0,
        help="Minimum final error polygon area in square meters. Use 0 to disable.",
    )
    parser.add_argument("--simplify_tolerance", type=float, default=0.15)
    parser.add_argument("--num_preview_tiles", type=int, default=0)

    parser.add_argument("--cut_threshold", type=float, default=0.05)
    parser.add_argument("--cd_threshold", type=float, default=0.7)
    parser.add_argument(
        "--sheet_workers",
        "--parallel_sheets",
        dest="sheet_workers",
        type=int,
        default=3,
        help="Number of DXF sheets to process concurrently after DXF conversion.",
    )
    parser.add_argument(
        "--footprint_max_pixels",
        type=int,
        default=25_000_000,
        help="Maximum pixels to read when vectorizing the TIF valid footprint",
    )
    parser.add_argument(
        "--processing_area_mode",
        choices=("bbox", "convex_hull", "union"),
        default="bbox",
        help=(
            "Geometry used to decide the TIF segmentation area for each DXF sheet. "
            "bbox uses the full extent of DXF building polygons."
        ),
    )
    parser.add_argument(
        "--processing_bbox_buffer_m",
        type=float,
        default=0.0,
        help="Optional buffer in meters around the per-sheet processing area.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--also_geojson", action="store_true")
    parser.add_argument(
        "--keep_intermediate",
        action="store_true",
        help="Keep intermediate and report directories after successful completion.",
    )
    parser.add_argument(
        "--keep_status",
        action="store_true",
        help="Keep status.json after successful completion.",
    )
    args = parser.parse_args()
    model_root = Path(args.model_path)
    if args.seg_config is None:
        args.seg_config = str(model_root / "config.yaml")
        if not Path(args.seg_config).exists():
            args.seg_config = str(DEFAULT_SEG_CONFIG)
    if args.seg_checkpoint is None:
        args.seg_checkpoint = str(model_root / "best.pth")
    return args


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_python_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def normalize_path(path_text: str, base_dir: Optional[Path] = None) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    # Keep symlink paths intact. Several model/data paths in this project are
    # symlinks, and resolving them can jump outside the mounted Docker volume.
    return Path(os.path.abspath(os.fspath(path)))


def resolve_tif_path(args: argparse.Namespace) -> Path:
    if args.tif_path:
        tif_path = normalize_path(args.tif_path)
        if not tif_path.exists():
            raise FileNotFoundError(f"TIF not found: {tif_path}")
        return tif_path

    dataset = normalize_path(args.dataset_path)
    candidates: List[Path] = []
    for subdir in ("tif", "TIF", "T2", "."):
        root = dataset / subdir if subdir != "." else dataset
        if root.exists():
            candidates.extend(sorted(root.glob("*.tif")))
            candidates.extend(sorted(root.glob("*.tiff")))
    if not candidates:
        raise FileNotFoundError(
            "No TIF found. Pass --tif_path or place a .tif under dataset_path/tif."
        )
    return candidates[0].resolve()


def _read_dxf_list_file(list_path: Path) -> List[Path]:
    base = list_path.parent
    paths = []
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        paths.append(normalize_path(text, base))
    return paths


def resolve_dxf_inputs(args: argparse.Namespace) -> List[Path]:
    if args.dxf_input:
        text = args.dxf_input.strip()
        if "," in text:
            paths = [normalize_path(item.strip()) for item in text.split(",") if item.strip()]
        else:
            input_path = normalize_path(text)
            if input_path.is_dir():
                paths = sorted(input_path.glob("*.dxf"))
            elif input_path.suffix.lower() == ".txt":
                paths = _read_dxf_list_file(input_path)
            else:
                paths = [input_path]
    else:
        dataset = normalize_path(args.dataset_path)
        paths = []
        for subdir in ("dxf", "DXF", "T1"):
            root = dataset / subdir
            if root.exists():
                paths.extend(sorted(root.glob("*.dxf")))
        if not paths:
            paths.extend(sorted(dataset.glob("*.dxf")))

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"DXF input not found: {missing[:5]}")
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise FileNotFoundError(
            "No DXF found. Pass --dxf_input or place .dxf files under dataset_path/dxf."
        )
    return unique


def write_status(output_dir: Path, payload: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "status.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def convert_one_dxf(task: Tuple[str, str, int, str, float, float]) -> Dict[str, object]:
    dxf_path_text, out_dir_text, epsg, layer_prefix, xy_tol, z_tol = task
    dxf_path = Path(dxf_path_text)
    out_dir = Path(out_dir_text)
    try:
        converter = load_python_module(DXF_CONVERTER, "dxf_to_shp_worker")
        summary = converter.convert(
            str(dxf_path),
            str(out_dir),
            epsg=epsg,
            layer_prefix=layer_prefix,
            xy_tol=xy_tol,
            z_tol=z_tol,
            to_polygon=True,
        )
        return {
            "ok": True,
            "dxf_path": str(dxf_path),
            "sheet_id": dxf_path.stem,
            "out_dir": str(out_dir),
            "summary": summary,
        }
    except Exception as exc:
        return {
            "ok": False,
            "dxf_path": str(dxf_path),
            "sheet_id": dxf_path.stem,
            "out_dir": str(out_dir),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def run_dxf_conversion(
    dxf_paths: Sequence[Path],
    output_root: Path,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    raw_root = output_root / "intermediate" / "dxf_raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    workers = args.dxf_workers
    if workers <= 0:
        workers = max(1, min(len(dxf_paths), os.cpu_count() or 1, 4))
    tasks = [
        (
            str(path),
            str(raw_root / path.stem),
            int(args.epsg),
            args.layer_prefix,
            float(args.dxf_xy_tol),
            float(args.dxf_z_tol),
        )
        for path in dxf_paths
    ]
    log(f"DXF conversion start: {len(tasks)} file(s), workers={workers}")
    if workers == 1:
        results = [convert_one_dxf(task) for task in tasks]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_task = {executor.submit(convert_one_dxf, task): task for task in tasks}
            for future in as_completed(future_to_task):
                results.append(future.result())
    results.sort(key=lambda item: str(item["sheet_id"]))
    failures = [item for item in results if not item.get("ok")]
    if failures:
        detail_path = output_root / "dxf_conversion_errors.json"
        detail_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
        raise RuntimeError(f"DXF conversion failed for {len(failures)} file(s): {detail_path}")
    return results


def _category_from_shp_name(path: Path) -> Optional[str]:
    for category in DXF_BUILDING_CATEGORIES:
        if category in path.name:
            return category
    return None


def fix_polygon_gdf(gdf: gpd.GeoDataFrame, epsg: int) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gpd.GeoDataFrame(gdf, geometry="geometry", crs=f"EPSG:{epsg}")
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    if gdf.empty:
        return gpd.GeoDataFrame(gdf, geometry="geometry", crs=f"EPSG:{epsg}")
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=epsg)
    elif gdf.crs.to_epsg() != epsg:
        gdf = gdf.to_crs(epsg=epsg)
    try:
        gdf.geometry = gdf.geometry.make_valid()
    except Exception:
        gdf.geometry = gdf.geometry.buffer(0)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if not gdf.empty:
        gdf = gdf.explode(index_parts=False, ignore_index=True)
    return gdf


def load_converted_dxf_polygons(
    conversion: Dict[str, object],
    epsg: int,
) -> gpd.GeoDataFrame:
    out_dir = Path(str(conversion["out_dir"]))
    dxf_path = Path(str(conversion["dxf_path"]))
    frames = []
    for shp_path in sorted(out_dir.glob("*_polygon.shp")):
        category = _category_from_shp_name(shp_path)
        if category is None:
            continue
        gdf = gpd.read_file(shp_path)
        gdf = fix_polygon_gdf(gdf, epsg)
        if gdf.empty:
            continue
        gdf = gdf[["geometry"]].copy()
        gdf["DXF_CAT"] = category
        gdf["SHEET_ID"] = dxf_path.stem
        gdf["SRC_DXF"] = dxf_path.name
        frames.append(gdf)
    if not frames:
        return gpd.GeoDataFrame(
            {"DXF_CAT": [], "SHEET_ID": [], "SRC_DXF": []},
            geometry=[],
            crs=f"EPSG:{epsg}",
        )
    merged = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=f"EPSG:{epsg}")


def raster_valid_footprint(
    tif_path: Path,
    epsg: int,
    max_pixels: int,
) -> Polygon:
    with rasterio.open(tif_path, IGNORE_COG_LAYOUT_BREAK="YES") as src:
        if src.crs is None or src.crs.to_epsg() != epsg:
            raise ValueError(f"TIF CRS must be EPSG:{epsg}; got {src.crs}")
        raster_box = box(*src.bounds)
        if src.count < 4 and src.nodata is None:
            return raster_box

        total_pixels = src.width * src.height
        scale = max(1, int(math.ceil(math.sqrt(total_pixels / float(max_pixels)))))
        out_width = max(1, int(math.ceil(src.width / float(scale))))
        out_height = max(1, int(math.ceil(src.height / float(scale))))
        mask = src.dataset_mask(out_shape=(out_height, out_width), resampling=Resampling.nearest)
        if mask.size == 0 or int(mask.max()) == 0:
            return GeometryCollection()
        if int(mask.min()) > 0:
            return raster_box

        transform = src.transform * rasterio.Affine.scale(
            src.width / float(out_width),
            src.height / float(out_height),
        )
        geoms = [
            shape(geom)
            for geom, value in shapes(mask, mask=mask > 0, transform=transform)
            if int(value) > 0
        ]
        if not geoms:
            return GeometryCollection()
        footprint = unary_union(geoms).intersection(raster_box)
        if footprint.is_empty:
            return GeometryCollection()
        return footprint


def geometry_to_clip_window(tif_path: Path, geom) -> Tuple[int, int, int, int]:
    with rasterio.open(tif_path, IGNORE_COG_LAYOUT_BREAK="YES") as src:
        minx, miny, maxx, maxy = geom.bounds
        win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
        col0 = max(0, int(math.floor(win.col_off)))
        row0 = max(0, int(math.floor(win.row_off)))
        col1 = min(src.width, int(math.ceil(win.col_off + win.width)))
        row1 = min(src.height, int(math.ceil(win.row_off + win.height)))
        width = max(0, col1 - col0)
        height = max(0, row1 - row0)
        if width <= 0 or height <= 0:
            raise ValueError(f"Empty clip window for geometry bounds: {geom.bounds}")
        return col0, row0, width, height


def clip_gdf_to_geom(gdf: gpd.GeoDataFrame, geom, epsg: int) -> gpd.GeoDataFrame:
    if gdf.empty or geom.is_empty:
        return gpd.GeoDataFrame(gdf.iloc[0:0].copy(), geometry="geometry", crs=f"EPSG:{epsg}")
    area = gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs=f"EPSG:{epsg}")
    clipped = gpd.clip(gdf, area, keep_geom_type=True)
    return fix_polygon_gdf(clipped, epsg)


def build_processing_area(
    dxf_gdf: gpd.GeoDataFrame,
    tif_footprint,
    args: argparse.Namespace,
):
    if dxf_gdf.empty:
        return GeometryCollection()
    if args.processing_area_mode == "union":
        area_geom = unary_union(list(dxf_gdf.geometry))
    elif args.processing_area_mode == "convex_hull":
        area_geom = unary_union(list(dxf_gdf.geometry)).convex_hull
    else:
        minx, miny, maxx, maxy = dxf_gdf.total_bounds
        area_geom = box(float(minx), float(miny), float(maxx), float(maxy))
    if float(args.processing_bbox_buffer_m) != 0.0:
        area_geom = area_geom.buffer(float(args.processing_bbox_buffer_m))
    if area_geom.is_empty:
        return GeometryCollection()
    return area_geom.intersection(tif_footprint)


def load_building_seg_module():
    return load_python_module(BUILDING_SEG_DIR / "scripts" / "infer_real_ortho.py", "bseg_infer")


def load_building_seg_model(seg_module, args: argparse.Namespace):
    import torch

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    config_path = normalize_path(args.seg_config)
    checkpoint_path = normalize_path(args.seg_checkpoint)
    if not config_path.exists():
        raise FileNotFoundError(f"Segmentation config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {checkpoint_path}")
    device_text = args.device
    device = torch.device(device_text if torch.cuda.is_available() or device_text == "cpu" else "cpu")
    model, cfg = seg_module.load_model(config_path, checkpoint_path, device)
    return model, cfg, device, config_path, checkpoint_path


def run_segmentation_for_area(
    seg_module,
    model,
    cfg: dict,
    device,
    tif_path: Path,
    sheet_id: str,
    processing_area,
    output_root: Path,
    args: argparse.Namespace,
) -> gpd.GeoDataFrame:
    clip_window = geometry_to_clip_window(tif_path, processing_area)
    clip_x, clip_y, clip_w, clip_h = clip_window
    seg_root = output_root / "intermediate" / "building_seg"
    paths = seg_module.prepare_run_dir(seg_root, sheet_id, overwrite=args.overwrite)
    specs = seg_module.build_tile_specs(clip_w, clip_h, args.patch_size, args.overlap)
    seg_module.write_tile_index(paths["meta"] / "tile_index.csv", specs, clip_x, clip_y)
    log(
        f"{sheet_id}: TIF segmentation clip x={clip_x}, y={clip_y}, "
        f"w={clip_w}, h={clip_h}, tiles={len(specs)}"
    )
    inference_meta = seg_module.run_inference(
        model=model,
        cfg=cfg,
        image_path=tif_path,
        paths=paths,
        clip_x=clip_x,
        clip_y=clip_y,
        clip_w=clip_w,
        clip_h=clip_h,
        specs=specs,
        batch_size=args.batch_size,
        threshold=args.seg_threshold,
        num_preview_tiles=args.num_preview_tiles,
        device=device,
        save_prob_float32=False,
    )
    vector_meta = seg_module.vectorize_outputs(
        mask_path=Path(inference_meta["mask_path"]),
        prob_path=Path(inference_meta["prob_path"]),
        valid_path=Path(inference_meta["valid_path"]),
        vector_dir=paths["vector"],
        min_area_m2=args.min_area_m2,
        simplify_tolerance=args.simplify_tolerance,
    )
    vector_path = Path(vector_meta.get("gpkg_path") or vector_meta["geojson_path"])
    if not vector_path.exists():
        raise FileNotFoundError(f"Segmentation vector output not found: {vector_path}")
    tif_gdf = gpd.read_file(vector_path)
    tif_gdf = tif_gdf.set_crs(epsg=args.epsg, allow_override=True)
    tif_gdf = fix_polygon_gdf(tif_gdf, args.epsg)
    tif_gdf = clip_gdf_to_geom(tif_gdf, processing_area, args.epsg)
    tif_gdf = tif_gdf[["geometry"]].copy()
    tif_gdf["SRC"] = "TIF"
    tif_gdf["SHEET_ID"] = sheet_id
    return tif_gdf


def import_change_detection_modules():
    if str(CHANGE_DETECTION_DIR) not in sys.path:
        sys.path.insert(0, str(CHANGE_DETECTION_DIR))
    from src.core.polygon_matching import polygon_matching_algorithm
    from src.core.polygon_matching import polygon_matching_utils
    from src.utils import analysis_utils

    return polygon_matching_algorithm, polygon_matching_utils, analysis_utils


def write_match_input(gdf: gpd.GeoDataFrame, out_dir: Path, name: str, epsg: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    shp_path = out_dir / f"{name}.shp"
    slim = gdf[["geometry"]].copy()
    slim["SRC_ID"] = range(1, len(slim) + 1)
    slim = gpd.GeoDataFrame(slim, geometry="geometry", crs=f"EPSG:{epsg}")
    if slim.empty:
        import pyogrio

        pyogrio.write_dataframe(
            slim,
            shp_path,
            driver="ESRI Shapefile",
            geometry_type="Polygon",
            encoding="UTF-8",
        )
    else:
        slim.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")
    return shp_path


def build_error_result(
    tif_cd: gpd.GeoDataFrame,
    dxf_cd: gpd.GeoDataFrame,
    sheet_id: str,
    dxf_name: str,
    epsg: int,
) -> gpd.GeoDataFrame:
    frames = []
    tif_errors = tif_cd[tif_cd["cd_class"].isin(["소멸", "갱신"])].copy()
    if not tif_errors.empty:
        tif_errors["SIDE"] = "TIF"
        frames.append(tif_errors)
    dxf_errors = dxf_cd[dxf_cd["cd_class"].isin(["신축"])].copy()
    if not dxf_errors.empty:
        dxf_errors["SIDE"] = "DXF"
        frames.append(dxf_errors)

    columns = ["CLS_ID", "CLS_NAME", "AREA", "ID", "SIDE", "SHEET_ID", "SRC_DXF", "REL_CD", "CD_RAW"]
    if not frames:
        return gpd.GeoDataFrame({col: [] for col in columns}, geometry=[], crs=f"EPSG:{epsg}")

    result = pd.concat(frames, ignore_index=True)
    result = gpd.GeoDataFrame(result, geometry="geometry", crs=f"EPSG:{epsg}")
    relation_col = "Relation" if "Relation" in result.columns else "rel_cd"
    result["CD_RAW"] = result["cd_class"]
    result["CLS_NAME"] = result["CD_RAW"].map(RAW_TO_ERROR_NAME)
    result = result[result["CLS_NAME"].notna()].copy()
    result["CLS_ID"] = result["CLS_NAME"].map(ERROR_TO_ID).astype(int)
    result["AREA"] = result.geometry.area.round(2)
    result["SHEET_ID"] = sheet_id
    result["SRC_DXF"] = dxf_name
    result["REL_CD"] = result[relation_col].astype(str)
    result["ID"] = [
        f"{sheet_id}_{side}_{idx:06d}"
        for idx, side in enumerate(result["SIDE"].astype(str), start=1)
    ]
    result = result[columns + ["geometry"]].copy()
    return gpd.GeoDataFrame(result, geometry="geometry", crs=f"EPSG:{epsg}")


def filter_result_by_area(
    result: gpd.GeoDataFrame,
    min_area_m2: float,
    epsg: int,
) -> gpd.GeoDataFrame:
    if result.empty:
        return gpd.GeoDataFrame(result, geometry="geometry", crs=f"EPSG:{epsg}")
    filtered = result.copy()
    filtered["AREA"] = filtered.geometry.area.round(2)
    if min_area_m2 > 0:
        filtered = filtered[filtered["AREA"] >= float(min_area_m2)].copy()
    return gpd.GeoDataFrame(filtered, geometry="geometry", crs=f"EPSG:{epsg}")


def write_empty_shapefile(shp_path: Path, epsg: int) -> None:
    import pyogrio

    shp_path.parent.mkdir(parents=True, exist_ok=True)
    empty = gpd.GeoDataFrame(
        {
            "CLS_ID": pd.Series(dtype="int64"),
            "CLS_NAME": pd.Series(dtype="object"),
            "AREA": pd.Series(dtype="float64"),
            "ID": pd.Series(dtype="object"),
            "SIDE": pd.Series(dtype="object"),
            "SHEET_ID": pd.Series(dtype="object"),
            "SRC_DXF": pd.Series(dtype="object"),
            "REL_CD": pd.Series(dtype="object"),
            "CD_RAW": pd.Series(dtype="object"),
        },
        geometry=[],
        crs=f"EPSG:{epsg}",
    )
    pyogrio.write_dataframe(
        empty,
        shp_path,
        driver="ESRI Shapefile",
        geometry_type="Polygon",
        encoding="UTF-8",
    )


def iter_polygon_parts(geom) -> Iterable[Polygon]:
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        for part in geom.geoms:
            if not part.is_empty:
                yield part


def export_result_dxf(gdf: gpd.GeoDataFrame, dxf_path: Path) -> None:
    import ezdxf

    dxf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 6
    for error_name, layer_name in DXF_LAYER_BY_ERROR.items():
        if layer_name not in doc.layers:
            doc.layers.add(layer_name, color=DXF_COLOR_BY_ERROR.get(error_name, 7))
    msp = doc.modelspace()
    for _, row in gdf.iterrows():
        layer = DXF_LAYER_BY_ERROR.get(row["CLS_NAME"], "CHANGE_ERROR")
        for poly in iter_polygon_parts(row.geometry):
            exterior = [(float(x), float(y)) for x, y in poly.exterior.coords]
            if len(exterior) >= 4:
                msp.add_lwpolyline(exterior, format="xy", close=True, dxfattribs={"layer": layer})
            for ring in poly.interiors:
                coords = [(float(x), float(y)) for x, y in ring.coords]
                if len(coords) >= 4:
                    msp.add_lwpolyline(coords, format="xy", close=True, dxfattribs={"layer": layer})
    doc.saveas(str(dxf_path))


def export_result_files(
    result: gpd.GeoDataFrame,
    output_root: Path,
    sheet_id: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    result_dir = output_root / sheet_id
    result_dir.mkdir(parents=True, exist_ok=True)
    shp_path = result_dir / f"{sheet_id}_errors.shp"
    dxf_path = result_dir / f"{sheet_id}_errors.dxf"
    csv_path = result_dir / f"{sheet_id}_summary.csv"

    if result.empty:
        write_empty_shapefile(shp_path, args.epsg)
    else:
        result.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")
    export_result_dxf(result, dxf_path)

    summary = (
        result.groupby("CLS_NAME")
        .agg(count=("geometry", "count"), area=("AREA", "sum"))
        .reset_index()
        if not result.empty
        else pd.DataFrame(columns=["CLS_NAME", "count", "area"])
    )
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    if args.also_geojson:
        result.to_file(result_dir / f"{sheet_id}_errors.geojson", driver="GeoJSON")
    return {
        "sheet_id": sheet_id,
        "feature_count": int(len(result)),
        "shp": str(shp_path),
        "dxf": str(dxf_path),
        "summary_csv": str(csv_path),
        "by_class": summary.to_dict(orient="records"),
    }


def run_change_detection_for_sheet(
    tif_gdf: gpd.GeoDataFrame,
    dxf_gdf: gpd.GeoDataFrame,
    sheet_id: str,
    dxf_name: str,
    output_root: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    if tif_gdf.empty and dxf_gdf.empty:
        empty = gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{args.epsg}")
        return export_result_files(empty, output_root, sheet_id, args)

    input_root = output_root / "intermediate" / "change_inputs" / sheet_id
    tif_path = write_match_input(tif_gdf, input_root / "tif", "tif_building", args.epsg)
    dxf_path = write_match_input(dxf_gdf, input_root / "dxf", "dxf_building", args.epsg)

    algorithm, utils, analysis_utils = import_change_detection_modules()
    _, tif_cd, dxf_cd = algorithm.algorithm_pipeline(
        str(tif_path),
        str(dxf_path),
        str(output_root / "intermediate" / "change_graph" / sheet_id),
        args.cut_threshold,
    )
    tif_cd = utils.assign_cd_class(tif_cd, args.cd_threshold, "cd")
    tif_cd = utils.assign_class_10(tif_cd, "cd")
    dxf_cd = utils.assign_cd_class(dxf_cd, args.cd_threshold, "cd")
    dxf_cd = utils.assign_class_10(dxf_cd, "cd")

    report_dir = output_root / "reports" / sheet_id
    report_dir.mkdir(parents=True, exist_ok=True)
    analysis_utils.analysis_pipeline(tif_cd, dxf_cd).to_csv(
        report_dir / "class_report.csv",
        index=False,
        encoding="utf-8-sig",
    )

    result = build_error_result(tif_cd, dxf_cd, sheet_id, dxf_name, args.epsg)
    raw_feature_count = int(len(result))
    result = filter_result_by_area(result, args.result_min_area_m2, args.epsg)
    result_meta = export_result_files(result, output_root, sheet_id, args)
    result_meta["raw_feature_count"] = raw_feature_count
    result_meta["result_min_area_m2"] = float(args.result_min_area_m2)
    result_meta["filtered_feature_count"] = raw_feature_count - int(result_meta["feature_count"])
    return result_meta


def prepare_output_dir(output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and overwrite:
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)


def cleanup_intermediate_outputs(output_root: Path) -> Dict[str, object]:
    removed = []
    for name in ("intermediate", "reports", "visualization"):
        path = output_root / name
        if path.exists():
            shutil.rmtree(path)
            removed.append(name)
    for path in (output_root / "dxf_conversion_errors.json",):
        if path.exists():
            path.unlink()
            removed.append(path.name)
    return {"removed": removed}


def process_sheet(
    conversion: Dict[str, object],
    index: int,
    total: int,
    tif_path: Path,
    tif_footprint,
    output_root: Path,
    args: argparse.Namespace,
    seg_context: Optional[Tuple[object, object, dict, object]] = None,
) -> Dict[str, object]:
    sheet_start = time.time()
    sheet_id = str(conversion["sheet_id"])
    dxf_name = Path(str(conversion["dxf_path"])).name
    log(f"{sheet_id}: start ({index}/{total})")

    dxf_gdf = load_converted_dxf_polygons(conversion, args.epsg)
    if dxf_gdf.empty:
        log(f"{sheet_id}: no DXF building polygons in selected categories")
        empty = gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{args.epsg}")
        result_meta = export_result_files(empty, output_root, sheet_id, args)
        result_meta["tif_polygon_count"] = 0
        result_meta["dxf_polygon_count"] = 0
        result_meta["raw_feature_count"] = 0
        result_meta["result_min_area_m2"] = float(args.result_min_area_m2)
        result_meta["filtered_feature_count"] = 0
        result_meta["elapsed"] = f"{time.time() - sheet_start:.2f}s"
        return result_meta

    processing_area = build_processing_area(dxf_gdf, tif_footprint, args)
    if processing_area.is_empty:
        log(f"{sheet_id}: no intersection between DXF polygons and TIF footprint")
        empty = gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{args.epsg}")
        result_meta = export_result_files(empty, output_root, sheet_id, args)
        result_meta["tif_polygon_count"] = 0
        result_meta["dxf_polygon_count"] = int(len(dxf_gdf))
        result_meta["raw_feature_count"] = 0
        result_meta["result_min_area_m2"] = float(args.result_min_area_m2)
        result_meta["filtered_feature_count"] = 0
        result_meta["elapsed"] = f"{time.time() - sheet_start:.2f}s"
        return result_meta

    processing_dir = output_root / "intermediate" / "processing_area" / sheet_id
    processing_dir.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({"sheet_id": [sheet_id]}, geometry=[processing_area], crs=f"EPSG:{args.epsg}").to_file(
        processing_dir / "processing_area.shp",
        driver="ESRI Shapefile",
        encoding="utf-8",
    )

    dxf_clipped = clip_gdf_to_geom(dxf_gdf, processing_area, args.epsg)
    if seg_context is None:
        seg_module = load_building_seg_module()
        model, cfg, device, _, _ = load_building_seg_model(seg_module, args)
    else:
        seg_module, model, cfg, device = seg_context

    tif_gdf = run_segmentation_for_area(
        seg_module=seg_module,
        model=model,
        cfg=cfg,
        device=device,
        tif_path=tif_path,
        sheet_id=sheet_id,
        processing_area=processing_area,
        output_root=output_root,
        args=args,
    )
    result_meta = run_change_detection_for_sheet(
        tif_gdf=tif_gdf,
        dxf_gdf=dxf_clipped,
        sheet_id=sheet_id,
        dxf_name=dxf_name,
        output_root=output_root,
        args=args,
    )
    result_meta["tif_polygon_count"] = int(len(tif_gdf))
    result_meta["dxf_polygon_count"] = int(len(dxf_clipped))
    result_meta["processing_area_mode"] = args.processing_area_mode
    result_meta["processing_area_bounds"] = tuple(round(float(v), 3) for v in processing_area.bounds)
    result_meta["elapsed"] = f"{time.time() - sheet_start:.2f}s"
    log(
        f"{sheet_id}: done, errors={result_meta['feature_count']} "
        f"(filtered={result_meta.get('filtered_feature_count', 0)})"
    )
    return result_meta


def process_sheet_worker(payload: Tuple[Dict[str, object], int, int, str, object, str, argparse.Namespace]) -> Dict[str, object]:
    conversion, index, total, tif_path_text, tif_footprint, output_root_text, args = payload
    return process_sheet(
        conversion=conversion,
        index=index,
        total=total,
        tif_path=Path(tif_path_text),
        tif_footprint=tif_footprint,
        output_root=Path(output_root_text),
        args=args,
        seg_context=None,
    )


def main() -> None:
    args = parse_args()
    output_root = normalize_path(args.output_path)
    prepare_output_dir(output_root, args.overwrite)
    status = {
        "Status": "in progress",
        "CurrentTask": "init",
        "ElapsedTime": {},
        "Results": [],
    }
    write_status(output_root, status)
    t0 = time.time()

    tif_path = resolve_tif_path(args)
    dxf_paths = resolve_dxf_inputs(args)
    log(f"TIF: {tif_path}")
    log(f"DXF count: {len(dxf_paths)}")

    status["CurrentTask"] = "dxf_to_polygon"
    write_status(output_root, status)
    dxf_start = time.time()
    conversions = run_dxf_conversion(dxf_paths, output_root, args)
    status["ElapsedTime"]["dxf_to_polygon"] = f"{time.time() - dxf_start:.2f}s"
    write_status(output_root, status)

    status["CurrentTask"] = "load_tif_footprint"
    write_status(output_root, status)
    footprint_start = time.time()
    tif_footprint = raster_valid_footprint(tif_path, args.epsg, args.footprint_max_pixels)
    if tif_footprint.is_empty:
        raise RuntimeError("TIF valid footprint is empty")
    status["ElapsedTime"]["load_tif_footprint"] = f"{time.time() - footprint_start:.2f}s"
    write_status(output_root, status)

    config_path = normalize_path(args.seg_config)
    checkpoint_path = normalize_path(args.seg_checkpoint)
    status["SegmentationConfig"] = str(config_path)
    status["SegmentationCheckpoint"] = str(checkpoint_path)
    status["ResultMinAreaM2"] = float(args.result_min_area_m2)
    status["ProcessingAreaMode"] = args.processing_area_mode
    status["ProcessingAreaBufferM"] = float(args.processing_bbox_buffer_m)
    sheet_workers = max(1, min(int(args.sheet_workers), len(conversions)))
    status["SheetWorkers"] = sheet_workers
    write_status(output_root, status)

    per_sheet_meta = []
    if sheet_workers == 1:
        status["CurrentTask"] = "load_segmentation_model"
        write_status(output_root, status)
        seg_module = load_building_seg_module()
        model, cfg, device, _, _ = load_building_seg_model(seg_module, args)
        seg_context = (seg_module, model, cfg, device)
        for index, conversion in enumerate(conversions, start=1):
            status["CurrentTask"] = f"process_sheet:{conversion['sheet_id']}"
            status["CurrentStep"] = index
            status["TotalStep"] = len(conversions)
            write_status(output_root, status)
            result_meta = process_sheet(
                conversion=conversion,
                index=index,
                total=len(conversions),
                tif_path=tif_path,
                tif_footprint=tif_footprint,
                output_root=output_root,
                args=args,
                seg_context=seg_context,
            )
            per_sheet_meta.append(result_meta)
            status["Results"] = sorted(per_sheet_meta, key=lambda item: str(item["sheet_id"]))
            write_status(output_root, status)
    else:
        log(f"Sheet processing start: {len(conversions)} sheet(s), workers={sheet_workers}")
        status["CurrentTask"] = f"process_sheets_parallel:{sheet_workers}"
        status["CurrentStep"] = 0
        status["TotalStep"] = len(conversions)
        write_status(output_root, status)
        tasks = [
            (conversion, index, len(conversions), str(tif_path), tif_footprint, str(output_root), args)
            for index, conversion in enumerate(conversions, start=1)
        ]
        with ProcessPoolExecutor(max_workers=sheet_workers) as executor:
            future_to_sheet = {
                executor.submit(process_sheet_worker, task): str(task[0]["sheet_id"])
                for task in tasks
            }
            for future in as_completed(future_to_sheet):
                sheet_id = future_to_sheet[future]
                try:
                    result_meta = future.result()
                except Exception as exc:
                    status["Status"] = "failed"
                    status["CurrentTask"] = f"failed:{sheet_id}"
                    status["Error"] = str(exc)
                    write_status(output_root, status)
                    raise
                per_sheet_meta.append(result_meta)
                status["CurrentStep"] = len(per_sheet_meta)
                status["Results"] = sorted(per_sheet_meta, key=lambda item: str(item["sheet_id"]))
                write_status(output_root, status)

    status["Status"] = "done"
    status["CurrentTask"] = "done"
    status["ElapsedTime"]["total"] = f"{time.time() - t0:.2f}s"
    status["Results"] = sorted(per_sheet_meta, key=lambda item: str(item["sheet_id"]))
    if args.keep_intermediate:
        status["Cleanup"] = {"removed": []}
    else:
        status["Cleanup"] = cleanup_intermediate_outputs(output_root)
    write_status(output_root, status)
    if not args.keep_status:
        status_path = output_root / "status.json"
        if status_path.exists():
            status_path.unlink()
    log("All tasks completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[failed] {exc}", file=sys.stderr)
        traceback.print_exc()
        raise
