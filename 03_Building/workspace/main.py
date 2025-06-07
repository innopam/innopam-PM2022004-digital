import argparse
import os
import rasterio
import numpy as np
import rasterio.features
import json
import cv2
import geopandas as gpd
import math
import warnings
import onnxruntime as ort
import networkx as nx
import pandas as pd
import time

from more_itertools import chunked
from multiprocessing import Pool
from shapely.ops import unary_union
from rasterio.windows import Window
from shapely.geometry import shape
from rasterio.features import shapes
from rasterio.warp import transform
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")


def make_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", type=str, default="workspace/input", help="input folder containing T1/, T2/")
    parser.add_argument("-m", "--model", type=str, default="workspace/model", help="Model folder containing .onnx")
    parser.add_argument("-o", "--output", type=str, default="workspace/output", help="Output folder")

    parser.add_argument("-c", "--conf-threshold", type=float, default=None)
    parser.add_argument("-r", "--resolution", type=float, default=None)
    parser.add_argument("--classes", type=str, default=None)
    parser.add_argument("-t", "--max-threads", type=int, default=None)
    parser.add_argument("-b", "--batch-size", type=int, default=8)

    parser.add_argument("--cut-threshold", type=float, default=0.05)
    parser.add_argument("--cd-threshold", type=float, default=0.7)

    return parser


def resolve_paths(args):
    t1_path = os.path.join(args.input, "T1")
    t1_vec = [f for f in os.listdir(t1_path) if f.endswith(('.shp', '.geojson'))]
    if not t1_vec:
        raise FileNotFoundError("No vector file found in input/T1/")
    args.prev_gdf = os.path.join(t1_path, t1_vec[0])

    t2_path = os.path.join(args.input, "T2")
    t2_tifs = [f for f in os.listdir(t2_path) if f.endswith(".tif")]
    if not t2_tifs:
        raise FileNotFoundError("No .tif file found in input/T2/")
    args.geotiff = os.path.join(t2_path, t2_tifs[0])

    model_files = [f for f in os.listdir(args.model) if f.endswith(".onnx")]
    if not model_files:
        raise FileNotFoundError("No .onnx model found in model folder")
    args.model = os.path.join(args.model, model_files[0])

    return args


class ProgressBar:
    def __init__(self):
        self.pbar = tqdm(total=100, desc="Start")
        self.current = 0
        self.closed = False

    def update(self, text, perc=0):
        self.current += perc
        self.pbar.n = int(self.current)
        self.pbar.set_description(text)
        self.pbar.refresh()

        if self.current >= 100 and not self.closed:
            self.pbar.set_description("End")
            self.pbar.close()
            self.closed = True

    @staticmethod
    def write(text):
        tqdm.write(text)


class StatusManager:
    def __init__(self, dir_path="."):
        self.filepath = os.path.join(dir_path, "status.json")
        self.status = {
            "CurrentTask": "init",
            "ElapsedTime": {},
            "Error": None
        }

    def set_task(self, task_name):
        self.status["CurrentTask"] = task_name
        self.save()

    def record_time(self, label, seconds):
        self.status["ElapsedTime"][label] = f"{round(seconds, 2)}s"
        self.save()

    def log_error(self, msg):
        self.status["CurrentTask"] = "error"
        self.status["Error"] = msg
        self.save()

    def save(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True) if os.path.dirname(self.filepath) else None
        with open(self.filepath, "w") as f:
            json.dump(self.status, f, ensure_ascii=False, indent=4)

    def task(self, name):
        return _StatusTaskContext(self, name)


class _StatusTaskContext:
    def __init__(self, manager, name):
        self.manager = manager
        self.name = name

    def __enter__(self):
        self.manager.set_task(self.name)
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        self.manager.record_time(self.name, elapsed)
        if exc_type:
            self.manager.log_error(str(exc_val))


def import_shapefile(file_path, crs=5186):
    if os.path.isdir(file_path):
        vec_files = [f for f in os.listdir(file_path) if f.endswith(('.shp', '.geojson'))]
        if not vec_files:
            raise FileNotFoundError(f"No .shp or .geojson file found in directory: {file_path}")
        file_path = os.path.join(file_path, vec_files[0])  # 첫 번째 vector 파일 사용

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in [".shp", ".geojson"]:
        raise ValueError("Only .shp or .geojson")

    gdf = gpd.read_file(str(file_path))
    if gdf.crs != f"epsg:{crs}":
        gdf = gdf.to_crs(epsg=crs)

    return gdf


"""
1. TIF -> 건물 추론 모델 -> GDF
"""


# 1) 모델 및 설정 불러오기
def get_model_file(path):
    # 모델 경로 반환
    if os.path.isfile(path):
        return os.path.abspath(path)
    else:
        raise FileNotFoundError(f"Model file not found: {path}")


def create_session(model_file, max_threads=None):
    # ONNX 모델 로드 및 config 생성
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    if max_threads is not None:
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = max_threads

    providers = [
        'CUDAExecutionProvider',
        'CPUExecutionProvider'
    ]

    session = ort.InferenceSession(model_file, sess_options=options, providers=providers)
    inputs = session.get_inputs()
    if len(inputs) > 1:
        raise Exception("ONNX model: unsupported number of inputs")

    meta = session.get_modelmeta().custom_metadata_map

    config = {
        'det_type': json.loads(meta.get('det_type', '"YOLO_v5_or_v7_default"')),
        'det_conf': float(meta.get('det_conf', 0.3)),
        'det_iou_thresh': float(meta.get('det_iou_thresh', 0.1)),
        'classes': ['background', 'building'],
        'seg_thresh': float(meta.get('seg_thresh', 0.5)),
        'seg_small_segment': int(meta.get('seg_small_segment', 11)),
        'resolution': float(20),
        'class_names': json.loads(meta.get('class_names', '{}')),
        'model_type': json.loads(meta.get('model_type', '"Detector"')),
        'tiles_overlap': float(meta.get('tiles_overlap', 15)),  # percentage
        'tiles_size': inputs[0].shape[-1],
        'input_shape': inputs[0].shape,
        'input_name': inputs[0].name,
    }
    return session, config


def override_config(config, conf_threshold=None, resolution=None, classes=None):
    # 사용자 입력이 있다면 우선 순위
    if conf_threshold is not None:
        config['det_conf'] = conf_threshold
    if resolution is not None:
        config['resolution'] = resolution
    if classes is not None:
        cn_map = cls_names_map(config['class_names'])
        config['classes'] = [cn_map[cls_name] for cls_name in cn_map if cls_name in classes]
    return config


def cls_names_map(class_names):
    # {"0": "tree"} --> {"tree": 0}
    d = {}
    for i in class_names:
        d[class_names[i]] = int(i)
    return d


# 2) Import TIF, edit config
def load_raster(geotiff_path: str):
    raster = rasterio.open(geotiff_path, 'r')
    return raster


def get_input_resolution(raster) -> float:
    # tif의 transform 기반 해상도 계산
    input_res = round(max(abs(raster.transform[0]), abs(raster.transform[4])), 4) * 100
    if input_res <= 0:
        input_res = estimate_raster_resolution(raster)
    return input_res


def estimate_raster_resolution(raster):
    # transform 정보가 없을 경우 해상도 추정
    if raster.crs is None:
        return 10  # Wild guess cm/px

    bounds = raster.bounds
    width = raster.width
    height = raster.height
    crs = raster.crs
    res_x = (bounds.right - bounds.left) / width
    res_y = (bounds.top - bounds.bottom) / height

    if crs.is_geographic:
        center_lat = (bounds.top + bounds.bottom) / 2
        earth_radius = 6378137.0
        meters_lon = math.pi / 180 * earth_radius * math.cos(math.radians(center_lat))
        meters_lat = math.pi / 180 * earth_radius
        res_x *= meters_lon
        res_y *= meters_lat

    return round(max(abs(res_x), abs(res_y)), 4) * 100  # cm/px


# 3) 타일링-처리 준비
def compute_tiling_params(raster, config, input_res):
    # 타일 스케일 비율, 오버랩, 타일 리스트 생성
    model_res = config['resolution']
    scale_factor = max(1, int(model_res // input_res)) if input_res < model_res else 1
    height, width = raster.shape
    tiles_overlap = config['tiles_overlap'] / 100.0
    windows = generate_for_size(width, height, config['tiles_size'] * scale_factor, tiles_overlap, clip=False)
    return height, width, scale_factor, tiles_overlap, windows


def generate_for_size(width, height, max_window_size, overlap_percent, clip=True):
    # Window 생성
    window_size_x = max_window_size
    window_size_y = max_window_size

    # If the input data is smaller than the specified window size,
    # clip the window size to the input size on both dimensions
    if clip:
        window_size_x = min(window_size_x, width)
        window_size_y = min(window_size_y, height)

    # Compute the window overlap and step size
    window_overlap_x = int(math.floor(window_size_x * overlap_percent))
    window_overlap_y = int(math.floor(window_size_y * overlap_percent))
    step_size_x = window_size_x - window_overlap_x
    step_size_y = window_size_y - window_overlap_y

    # Determine how many windows we will need in order to cover the input data
    last_x = width - window_size_x
    last_y = height - window_size_y
    x_offsets = list(range(0, last_x + 1, step_size_x))
    y_offsets = list(range(0, last_y + 1, step_size_y))

    # Unless the input data dimensions are exact multiples of the step size,
    # we will need one additional row and column of windows to get 100% coverage
    if len(x_offsets) == 0 or x_offsets[-1] != last_x:
        x_offsets.append(last_x)
    if len(y_offsets) == 0 or y_offsets[-1] != last_y:
        y_offsets.append(last_y)

    # Generate the list of windows
    windows = []
    for x_offset in x_offsets:
        for y_offset in y_offsets:
            windows.append(Window(
                x_offset,
                y_offset,
                window_size_x,
                window_size_y,
            ))

    return windows


def determine_indexes(raster):
    # alpha 제거
    indexes = raster.indexes
    if len(indexes) > 1 and raster.colorinterp[-1] == rasterio.enums.ColorInterp.alpha:
        indexes = indexes[:-1]
    return indexes


# 4) 추론, 타일 마스크 생성 및 병합
def read_tile(args):
    # 타일 생성
    raster_path, window, indexes, config = args
    with rasterio.open(raster_path) as src:
        img = src.read(
            indexes=indexes,
            window=window,
            boundless=True,
            fill_value=0,
            out_shape=(len(indexes), config['tiles_size'], config['tiles_size']),
            resampling=rasterio.enums.Resampling.bilinear
        )
    return img


def preprocess_batch(image_list):
    # 배치 만들기
    stacked = np.stack(image_list, axis=0)  # (N, C, H, W) 또는 (N, H, W, C)
    return preprocess(stacked)


def execute_batch_segmentation(images_batch, session, config):
    images_batch = preprocess_batch(images_batch)
    outs = session.run(None, {config['input_name']: images_batch})
    final_out = outs[0][:, 0, :, :]
    return [final_out[i] for i in range(final_out.shape[0])]


def process_tiles(raster_path, windows, indexes, session, config, progress: ProgressBar, total_perc, batch_size):
    n = len(windows)
    read_perc = total_perc * 0.15
    infer_perc = total_perc * 0.75
    merge_perc = total_perc * 0.1
    per_tile_merge_perc = merge_perc / n

    # 타일 병렬 생성
    progress.update("Reading tiles", read_perc)
    args_list = [(raster_path, w, indexes, config) for w in windows]
    with Pool() as pool:
        tile_images = pool.map(read_tile, args_list)
    progress.write("Completed tile reading")

    # 추론
    tile_masks = []
    total_batches = len(tile_images) // batch_size + int(len(tile_images) % batch_size != 0)
    per_batch_perc = infer_perc / total_batches

    for i, batch in enumerate(chunked(tile_images, batch_size)):
        progress.update(f"Inference batch {i+1}/{total_batches}", perc=per_batch_perc)
        batch_masks = execute_batch_segmentation(batch, session, config)  # List[(H, W)]
        tile_masks.extend(batch_masks)
    progress.write(f"Completed inference")

    with rasterio.open(raster_path) as src:
        height, width = src.shape
        input_res = round(max(abs(src.transform[0]), abs(src.transform[4])), 4) * 100
    model_res = config['resolution']
    scale_factor = max(1, int(model_res // input_res)) if input_res < model_res else 1
    tiles_overlap = config['tiles_overlap'] / 100.0
    mask = np.zeros((height // scale_factor, width // scale_factor), dtype=np.uint8)

    for idx, w in enumerate(windows):
        progress.update(f"Merging tile {idx+1}/{n}", perc=per_tile_merge_perc)
        merge_mask(tile_masks[idx], mask, w, width, height, tiles_overlap, scale_factor)
    return mask


def preprocess(model_input):
    # 채널 정렬, 정규화
    s = model_input.shape
    if not len(s) in [3, 4]:
        raise Exception(f"Expected input with 3 or 4 dimensions, got: {s}")
    is_batched = len(s) == 4

    # expected: [batch],channel,height,width but could be: [batch],height,width,channel
    if s[-1] in [3, 4] and s[1] > s[-1]:
        if is_batched:
            model_input = np.transpose(model_input, (0, 3, 1, 2))
        else:
            model_input = np.transpose(model_input, (2, 0, 1))

    # add batch dimension (1, c, h, w)
    if not is_batched:
        model_input = np.expand_dims(model_input, axis=0)

    # drop alpha channel
    if model_input.shape[1] == 4:
        model_input = model_input[:, 0:3, :, :]

    if model_input.shape[1] != 3:
        raise Exception(f"Expected input channels to be 3, but got: {model_input.shape[1]}")

    # normalize
    if model_input.dtype == np.uint8:
        return (model_input / 255.0).astype(np.float32)

    if model_input.dtype.kind == 'f':
        min_value = float(model_input.min())
        value_range = float(model_input.max()) - min_value
    else:
        data_range = np.iinfo(model_input.dtype)
        min_value = 0
        value_range = float(data_range.max) - float(data_range.min)

    model_input = model_input.astype(np.float32)
    model_input -= min_value
    model_input /= value_range
    model_input[model_input > 1] = 1
    model_input[model_input < 0] = 0

    return model_input


# 5) 마스크 후처리
def merge_mask(tile_mask, mask, window, width, height, tiles_overlap=0, scale_factor=1.0):
    # 오버랩 계산하여 마스크 병합
    w = window
    row_off = int(w.row_off // scale_factor)  # int(np.round(w.row_off / scale_factor))
    col_off = int(w.col_off // scale_factor)  # int(np.round(w.col_off / scale_factor))
    tile_w, tile_h = tile_mask.shape

    pad_x = int(tiles_overlap * tile_w) // 2
    pad_y = int(tiles_overlap * tile_h) // 2

    pad_l = 0
    pad_r = 0
    pad_t = 0
    pad_b = 0

    if w.col_off > 0:
        pad_l = pad_x
    if w.col_off + w.width < width:
        pad_r = pad_x
    if w.row_off > 0:
        pad_t = pad_y
    if w.row_off + w.height < height:
        pad_b = pad_y

    row_off += pad_t
    col_off += pad_l
    tile_w -= pad_l + pad_r
    tile_h -= pad_t + pad_b

    tile_mask = tile_mask[np.newaxis, :, :]  # shape: (1, 512, 512)
    tile_mask = tile_mask[:, pad_t:pad_t + tile_h, pad_l:pad_l + tile_w]
    tr, sr = rect_intersect((col_off, row_off, tile_w, tile_h), (0, 0, mask.shape[1], mask.shape[0]))
    if tr is not None and sr is not None:
        mask[sr[1]:sr[1] + sr[3], sr[0]:sr[0] + sr[2]] = tile_mask[:, tr[1]:tr[1] + tr[3], tr[0]:tr[0] + tr[2]]
        # mask[sr[1]:sr[1]+sr[3], sr[0]:sr[0]+sr[2]] *= (idx + 1)


def rect_intersect(rect1, rect2):
    """
    Given two rectangles, compute the intersection rectangle and return
    its coordinates in the coordinate system of both rectangles.

    Each rectangle is represented as (x, y, width, height).

    Returns:
    - (r1_x, r1_y, iw, ih): Intersection in rect1's local coordinates
    - (r2_x, r2_y, iw, ih): Intersection in rect2's local coordinates
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    ix = max(x1, x2)  # Left boundary
    iy = max(y1, y2)  # Top boundary
    ix2 = min(x1 + w1, x2 + w2)  # Right boundary
    iy2 = min(y1 + h1, y2 + h2)  # Bottom boundary

    # Compute intersection
    iw = max(0, ix2 - ix)
    ih = max(0, iy2 - iy)

    # If no intersection
    if iw == 0 or ih == 0:
        return None, None

    # Compute local coordinates
    r1_x = ix - x1
    r1_y = iy - y1
    r2_x = ix - x2
    r2_y = iy - y2

    return (r1_x, r1_y, iw, ih), (r2_x, r2_y, iw, ih)


try:
    from scipy.ndimage import median_filter
except ImportError:
    def median_filter(arr, size=5):
        assert size % 2 == 1, "Kernel size must be an odd number."
        if arr.shape[0] <= size or arr.shape[1] <= size:
            return arr

        pad_size = size // 2
        padded = np.pad(arr, pad_size, mode='edge')
        shape = (arr.shape[0], arr.shape[1], size, size)
        strides = padded.strides + padded.strides
        view = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        return np.median(view, axis=(2, 3)).astype(arr.dtype)


def filter_small_segments(mask, config):
    # Better matches the logic from Deepness
    # where the parameter refers to the dilation/erode size
    # (sieve counts the number of pixels)
    ss = (config['seg_small_segment'] * 2) ** 2
    if ss > 0:
        # Remove small polygons
        rasterio.features.sieve(mask, ss, out=mask)
    return mask


def morphology_to_mask(mask, open_k=21, close_k=3, iterations=2):
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=iterations)
    morphed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel, iterations=iterations)
    return morphed


# 6) GDF 변환 및 후처리
def mask_to_gdf(raster, mask, config, scale_factor=1.0):
    affine = raster.transform * rasterio.Affine.scale(scale_factor, scale_factor)

    values = np.unique(mask)
    values = values[values != 0]

    target_value = values[0]

    geometries = []
    for geom, val in shapes(source=mask, mask=(mask == target_value), transform=affine):
        s = shape(geom)
        if s.is_valid and not s.is_empty:
            xs, ys = zip(*s.exterior.coords[:])
            x_new, y_new = transform(raster.crs, "EPSG:5186", xs, ys)
            projected_geom = shape({
                "type": "Polygon",
                "coordinates": [list(zip(x_new, y_new))]
            })
            geometries.append(projected_geom)

    return gpd.GeoDataFrame({'class': ['building'] * len(geometries), 'geometry': geometries}, crs="EPSG:5186")


def simplify_polygon(polygon, tolerance=1.0, preserve_topology=True):
    return polygon.simplify(tolerance, preserve_topology)


"""
2. Inference GDF, prev GDF -> 변화탐지 -> Inference result, prev result 
"""


# 1) 매칭 전처리
def indexing(poly1, poly2):
    # 고유 인덱스, 면적, spatial join
    poly1['poly1_idx'] = range(1, len(poly1) + 1)
    poly1 = poly1.reset_index(drop=True)

    poly2['poly2_idx'] = range(1, len(poly2) + 1)
    poly2 = poly2.reset_index(drop=True)

    poly1_area = poly1.geometry.area
    poly2_area = poly2.geometry.area

    poly1 = poly1.drop(columns=['area'], errors='ignore')
    idx_loc1 = poly1.columns.get_loc('poly1_idx')
    poly1.insert(loc=idx_loc1, column='area', value=poly1_area)

    poly2 = poly2.drop(columns=['area'], errors='ignore')
    idx_loc2 = poly2.columns.get_loc('poly2_idx')
    poly2.insert(loc=idx_loc2, column='area', value=poly2_area)

    outer_joined = outer_join(poly1, poly2, poly1_prefix="poly1", poly2_prefix="poly2")
    return poly1, poly2, outer_joined


def outer_join(poly1, poly2, poly1_prefix="poly1", poly2_prefix="poly2"):
    left_join = gpd.sjoin(poly1, poly2, how='left', predicate='intersects')
    right_join = gpd.sjoin(poly2, poly1, how='left', predicate='intersects')
    left_join.columns = [
        col.replace('_left', '_poly1').replace('_right', '_poly2')
        for col in left_join.columns
    ]

    right_join.columns = [
        col.replace('_left', '_poly2').replace('_right', '_poly1')
        for col in right_join.columns
    ]

    joined = pd.merge(left_join, right_join, how='outer', on=list(set(left_join.columns) & set(right_join.columns)))
    subset_cols = [f"{poly1_prefix}_idx", f"{poly2_prefix}_idx"]
    joined = joined.drop_duplicates(subset=subset_cols)
    joined = joined.reset_index(drop=True)
    return joined


# 2) 그래프 구성 및 정제
def build_graph(joined_df):
    # 공간적으로 매칭된 poly1, poly2 쌍으로부터 노드-링크 구조의 관계 그래프 생성
    nodes = set()
    links = []

    for _, row in joined_df.dropna(subset=["poly1_idx", "poly2_idx"]).iterrows():
        p1 = f"p1_{int(row['poly1_idx'])}"
        p2 = f"p2_{int(row['poly2_idx'])}"
        links.append({"source": p1, "target": p2})
        nodes.update([p1, p2])

    if "poly1_idx" in joined_df.columns:
        for p1 in joined_df["poly1_idx"].dropna().unique():
            nodes.add(f"p1_{int(p1)}")
    if "poly2_idx" in joined_df.columns:
        for p2 in joined_df["poly2_idx"].dropna().unique():
            nodes.add(f"p2_{int(p2)}")

    node_list = [{"id": n} for n in sorted(nodes)]

    return {"nodes": node_list, "links": links}


def add_energy_to_links(poly1, poly2, graph_dict):
    # 각 링크에 대해 IoU 계산
    poly1 = poly1.set_index("poly1_idx")
    poly2 = poly2.set_index("poly2_idx")

    for link in graph_dict["links"]:
        p1_idx = int(link["source"].replace("p1_", ""))
        p2_idx = int(link["target"].replace("p2_", ""))

        if p1_idx not in poly1.index or p2_idx not in poly2.index:
            raise ValueError(f"Missing poly1_idx {p1_idx} or poly2_idx {p2_idx} in geometry.")

        geom1 = poly1.loc[p1_idx, "geometry"]
        geom2 = poly2.loc[p2_idx, "geometry"]

        if geom1.is_empty or geom2.is_empty:
            raise ValueError(f"Empty geometry at poly1_idx {p1_idx} or poly2_idx {p2_idx}.")

        intersection = geom1.intersection(geom2)
        union = geom1.union(geom2)

        if union.area == 0 or intersection.area == 0:
            raise ValueError(f"No valid overlap between {p1_idx} and {p2_idx}.")

        link["energy"] = intersection.area / union.area

    return graph_dict


def split_graph_by_energy(poly1, poly2, graph_dict, threshold):
    # energy(IoU)가 낮은 링크 끊고 component 및 link 재구성
    poly1 = poly1.set_index("poly1_idx")
    poly2 = poly2.set_index("poly2_idx")

    G = nx.Graph()
    original_links = graph_dict["links"]
    original_nodes = graph_dict["nodes"]

    cut_links = []
    kept_links = []

    suppression = 0.7

    for link in original_links:
        energy = link.get("energy", 0)

        if energy >= threshold:
            G.add_edge(link["source"], link["target"], energy=energy)
            kept_links.append(link)
        else:
            p1_idx = int(link["source"].replace("p1_", ""))
            p2_idx = int(link["target"].replace("p2_", ""))
            geom1 = poly1.loc[p1_idx, "geometry"]
            geom2 = poly2.loc[p2_idx, "geometry"]
            area1 = poly1.loc[p1_idx, "area"]

            intersection = geom1.intersection(geom2)
            ol1 = intersection.area / area1 if area1 > 0 else 0

            if ol1 < suppression:
                cut_links.append(link)
            else:
                G.add_edge(link["source"], link["target"], energy=energy)
                kept_links.append(link)

    for node in original_nodes:
        G.add_node(node["id"])

    components_dict = {}
    new_node_list = []
    new_link_list = []

    for comp_idx, comp in enumerate(nx.connected_components(G)):
        poly1_set = sorted(int(n[3:]) for n in comp if n.startswith("p1_"))
        poly2_set = sorted(int(n[3:]) for n in comp if n.startswith("p2_"))

        components_dict[comp_idx] = {
            "poly1_set": poly1_set,
            "poly2_set": poly2_set
        }

        for n in comp:
            new_node_list.append({"id": n, "comp_idx": comp_idx})

        for u, v in G.subgraph(comp).edges:
            source, target = (u, v) if u.startswith("p1_") else (v, u)
            new_link_list.append({
                "source": source,
                "target": target,
                "comp_idx": comp_idx,
                "energy": G[source][target]["energy"]
            })

    summary = {
        "after_components": len(components_dict),
        "num_cut_links": len(cut_links)
    }
    return components_dict, {"nodes": new_node_list, "links": new_link_list}, cut_links, summary


# 3) 그래프 기반 정량 지표 계산
def mark_cut_links(poly1, poly2, cut_links):
    # link 제거 정보 기록
    cut_poly1_idxs = set(
        int(link["source"].replace("p1_", "")) for link in cut_links if link["source"].startswith("p1_"))
    cut_poly2_idxs = set(
        int(link["target"].replace("p2_", "")) for link in cut_links if link["target"].startswith("p2_"))

    poly1 = poly1.copy()
    poly2 = poly2.copy()

    poly1["cut_link"] = poly1["poly1_idx"].isin(cut_poly1_idxs)
    poly2["cut_link"] = poly2["poly2_idx"].isin(cut_poly2_idxs)

    return poly1, poly2


def attach_metrics_from_components(components_dict, poly1, poly2):
    # Relation, IoU, Overlap1,2 유형별 기록
    poly1 = poly1.copy()
    poly2 = poly2.copy()

    # 초기화
    metric_cols = [
        "comp_idx", "Relation",
        "iou_1n", "ol_pl1_1n", "ol_pl2_1n",
        "iou_n1", "ol_pl1_n1", "ol_pl2_n1",
        "iou_11", "ol_pl1_11", "ol_pl2_11",
        "iou_nn", "ol_pl1_nn", "ol_pl2_nn"
    ]
    for col in metric_cols[2:]:
        poly1[col] = np.nan
        poly2[col] = np.nan
    poly1["comp_idx"] = np.nan
    poly2["comp_idx"] = np.nan
    poly1["Relation"] = np.nan
    poly2["Relation"] = np.nan

    def calc_metrics(g1, g2):
        if g1 is None or g2 is None:
            return (np.nan, np.nan, np.nan)
        inter = g1.intersection(g2).area
        if inter == 0:
            return (0, 0, 0)
        return (
            inter / g1.union(g2).area,
            inter / g1.area if g1.area > 0 else 0,
            inter / g2.area if g2.area > 0 else 0
        )

    for comp_idx, comp in components_dict.items():
        p1_set = comp["poly1_set"]
        p2_set = comp["poly2_set"]

        rel = (
            "1:0" if len(p2_set) == 0 else
            "0:1" if len(p1_set) == 0 else
            "1:1" if len(p1_set) == 1 and len(p2_set) == 1 else
            "1:N" if len(p1_set) == 1 else
            "N:1" if len(p2_set) == 1 else "N:N"
        )

        poly1["Relation"] = poly1["Relation"].astype("object")
        poly2["Relation"] = poly2["Relation"].astype("object")

        poly1.loc[poly1['poly1_idx'].isin(p1_set), ["comp_idx", "Relation"]] = [comp_idx, rel]
        poly2.loc[poly2['poly2_idx'].isin(p2_set), ["comp_idx", "Relation"]] = [comp_idx, rel]

        if rel == "1:N":
            g1 = poly1.loc[poly1['poly1_idx'] == p1_set[0], "geometry"].values[0]
            g2_union = unary_union(poly2.loc[poly2['poly2_idx'].isin(p2_set), "geometry"])

            # n1: poly1 vs union(poly2)
            iou_n1, ol1_n1, ol2_n1 = calc_metrics(g1, g2_union)
            poly1.loc[poly1['poly1_idx'] == p1_set[0], ["iou_n1", "ol_pl1_n1", "ol_pl2_n1"]] = [iou_n1, ol1_n1, ol2_n1]
            poly2.loc[poly2['poly2_idx'].isin(p2_set), ["iou_n1", "ol_pl1_n1", "ol_pl2_n1"]] = [iou_n1, ol1_n1, ol2_n1]

            # nn: poly1 vs each poly2
            for p2 in p2_set:
                g2 = poly2.loc[poly2['poly2_idx'] == p2, "geometry"].values[0]
                iou, ol1, ol2 = calc_metrics(g1, g2)
                poly2.loc[poly2['poly2_idx'] == p2, ["iou_nn", "ol_pl1_nn", "ol_pl2_nn"]] = [iou, ol1, ol2]
            poly1.loc[poly1['poly1_idx'] == p1_set[0], ["iou_nn", "ol_pl1_nn", "ol_pl2_nn"]] = [iou, ol1, ol2]

        elif rel == "N:1":
            g2 = poly2.loc[poly2['poly2_idx'] == p2_set[0], "geometry"].values[0]
            g1_union = unary_union(poly1.loc[poly1['poly1_idx'].isin(p1_set), "geometry"])

            # 1n: union(poly1) vs poly2
            iou_1n, ol1_1n, ol2_1n = calc_metrics(g1_union, g2)
            poly1.loc[poly1['poly1_idx'].isin(p1_set), ["iou_1n", "ol_pl1_1n", "ol_pl2_1n"]] = [iou_1n, ol1_1n, ol2_1n]
            poly2.loc[poly2['poly2_idx'] == p2_set[0], ["iou_1n", "ol_pl1_1n", "ol_pl2_1n"]] = [iou_1n, ol1_1n, ol2_1n]

            # nn: each poly1 vs poly2
            for p1 in p1_set:
                g1 = poly1.loc[poly1['poly1_idx'] == p1, "geometry"].values[0]
                iou, ol1, ol2 = calc_metrics(g1, g2)
                poly1.loc[poly1['poly1_idx'] == p1, ["iou_nn", "ol_pl1_nn", "ol_pl2_nn"]] = [iou, ol1, ol2]
            poly2.loc[poly2['poly2_idx'] == p2_set[0], ["iou_nn", "ol_pl1_nn", "ol_pl2_nn"]] = [iou, ol1, ol2]

        elif rel == "N:N":
            g1_union = unary_union(poly1.loc[poly1['poly1_idx'].isin(p1_set), "geometry"])
            g2_union = unary_union(poly2.loc[poly2['poly2_idx'].isin(p2_set), "geometry"])

            # 11: union vs union
            iou_11, ol1_11, ol2_11 = calc_metrics(g1_union, g2_union)
            poly1.loc[poly1['poly1_idx'].isin(p1_set), ["iou_11", "ol_pl1_11", "ol_pl2_11"]] = [iou_11, ol1_11, ol2_11]
            poly2.loc[poly2['poly2_idx'].isin(p2_set), ["iou_11", "ol_pl1_11", "ol_pl2_11"]] = [iou_11, ol1_11, ol2_11]

            # 1n: union(poly1) vs each poly2
            for p2 in p2_set:
                g2 = poly2.loc[poly2['poly2_idx'] == p2, "geometry"].values[0]
                iou, ol1, ol2 = calc_metrics(g1_union, g2)
                poly2.loc[poly2['poly2_idx'] == p2, ["iou_1n", "ol_pl1_1n", "ol_pl2_1n"]] = [iou, ol1, ol2]
            poly1.loc[poly1['poly1_idx'].isin(p1_set), ["iou_1n", "ol_pl1_1n", "ol_pl2_1n"]] = [iou, ol1, ol2]

            # n1: each poly1 vs union(poly2)
            for p1 in p1_set:
                g1 = poly1.loc[poly1['poly1_idx'] == p1, "geometry"].values[0]
                iou, ol1, ol2 = calc_metrics(g1, g2_union)
                poly1.loc[poly1['poly1_idx'] == p1, ["iou_n1", "ol_pl1_n1", "ol_pl2_n1"]] = [iou, ol1, ol2]
            poly2.loc[poly2['poly2_idx'].isin(p2_set), ["iou_n1", "ol_pl1_n1", "ol_pl2_n1"]] = [iou, ol1, ol2]

        elif rel == "1:1":
            p1 = p1_set[0]
            p2 = p2_set[0]
            g1 = poly1.loc[poly1['poly1_idx'] == p1, "geometry"].values[0]
            g2 = poly2.loc[poly2['poly2_idx'] == p2, "geometry"].values[0]
            iou, ol1, ol2 = calc_metrics(g1, g2)
            poly1.loc[poly1['poly1_idx'] == p1, ["iou_nn", "ol_pl1_nn", "ol_pl2_nn"]] = [iou, ol1, ol2]
            poly2.loc[poly2['poly2_idx'] == p2, ["iou_nn", "ol_pl1_nn", "ol_pl2_nn"]] = [iou, ol1, ol2]

    return poly1, poly2


def add_component_sets_to_polys(poly1, poly2, components_dict):
    # 각 폴리곤에 본인이 속한 component, 관계가 있는 다른 폴리곤 기록
    poly1 = poly1.copy()
    poly2 = poly2.copy()

    # 새로운 열 초기화
    poly1['poly1_set'] = None
    poly1['poly2_set'] = None
    poly2['poly1_set'] = None
    poly2['poly2_set'] = None

    # poly1에 붙이기
    for idx, row in poly1.iterrows():
        p1_id = row['poly1_idx']
        for comp in components_dict.values():
            if p1_id in comp['poly1_set']:
                poly1.at[idx, 'poly1_set'] = comp['poly1_set']
                poly1.at[idx, 'poly2_set'] = comp['poly2_set']
                break  # 하나의 component만 매칭되므로 break

    # poly2에 붙이기
    for idx, row in poly2.iterrows():
        p2_id = row['poly2_idx']
        for comp in components_dict.values():
            if p2_id in comp['poly2_set']:
                poly2.at[idx, 'poly1_set'] = comp['poly1_set']
                poly2.at[idx, 'poly2_set'] = comp['poly2_set']
                break

    # 열 순서 정리: comp_idx 다음에 poly1_set, poly2_set 붙이기
    def insert_after(df, after_col, insert_cols):
        cols = df.columns.tolist()
        idx = cols.index(after_col) + 1
        for col in insert_cols:
            if col in cols:
                cols.remove(col)
        for i, col in enumerate(insert_cols):
            cols.insert(idx + i, col)
        return df[cols]

    poly1 = insert_after(poly1, 'comp_idx', ['poly1_set', 'poly2_set'])
    poly2 = insert_after(poly2, 'comp_idx', ['poly1_set', 'poly2_set'])

    return poly1, poly2


# 4) 변화 유형 분류
def assign_class(poly, threshold):
    poly = assign_cd_class(poly, threshold, "cd")
    poly = assign_class_10(poly, "cd")

    return poly


def assign_cd_class(poly, threshold, prefix="cd"):
    # IoU 및 Relation 기반 신축, 소멸, 변화없음, 갱신 판별
    class_col_name = f"{prefix}_class"  # 자동으로 열 이름 생성

    cd_class = np.full(len(poly), np.nan, dtype=object)

    cd_class[np.where(poly['Relation'] == '0:1')[0]] = '신축'
    cd_class[np.where(poly['Relation'] == '1:0')[0]] = '소멸'

    cd_class[np.where((poly['Relation'] == '1:1') & (poly['iou_nn'] > threshold))[0]] = '변화없음'
    cd_class[np.where((poly['Relation'] == '1:1') & (poly['iou_nn'] <= threshold))[0]] = '갱신'

    cd_class[np.where((poly['Relation'] == '1:N') & (poly['iou_n1'] > threshold))[0]] = '변화없음'
    cd_class[np.where((poly['Relation'] == '1:N') & (poly['iou_n1'] <= threshold))[0]] = '갱신'

    cd_class[np.where((poly['Relation'] == 'N:1') & (poly['iou_1n'] > threshold))[0]] = '변화없음'
    cd_class[np.where((poly['Relation'] == 'N:1') & (poly['iou_1n'] <= threshold))[0]] = '갱신'

    cd_class[np.where((poly['Relation'] == 'N:N') & (poly['iou_11'] > threshold))[0]] = '변화없음'
    cd_class[np.where((poly['Relation'] == 'N:N') & (poly['iou_11'] <= threshold))[0]] = '갱신'

    if class_col_name in poly.columns:
        poly = poly.drop(columns=[class_col_name])

    relation_loc = poly.columns.get_loc('Relation')
    poly.insert(loc=relation_loc + 1, column=class_col_name, value=cd_class)

    return poly


def assign_class_10(poly, prefix="cd"):
    # 10가지 클래스 생성
    class_col = f"{prefix}_class"
    class10_col = "class_10"

    def get_class(row):
        rel = row.get("Relation")
        cls = row.get(class_col)

        if rel == "1:0" and cls == "소멸":
            return "소멸"
        elif rel == "0:1" and cls == "신축":
            return "신축"
        elif rel == "1:1" and cls == "갱신":
            return "1:1 갱신"
        elif rel == "1:1" and cls == "변화없음":
            return "1:1 변화없음"
        elif rel == "1:N" and cls == "변화없음":
            return "1:N 변화없음"
        elif rel == "1:N" and cls == "갱신":
            return "1:N 갱신"
        elif rel == "N:1" and cls == "변화없음":
            return "N:1 변화없음"
        elif rel == "N:1" and cls == "갱신":
            return "N:1 갱신"
        elif rel == "N:N" and cls == "변화없음":
            return "N:N 변화없음"
        elif rel == "N:N" and cls == "갱신":
            return "N:N 갱신"
        else:
            return None  # 기타 처리 안 된 경우

    poly = poly.copy()
    poly[class10_col] = poly.apply(get_class, axis=1)

    relation_loc = poly.columns.get_loc("Relation")
    reordered = poly.pop(class10_col)
    poly.insert(loc=relation_loc + 1, column=class10_col, value=reordered)

    return poly


def cd_pipeline(dmap, seg, cd_threshold):
    dmap = assign_class(dmap, cd_threshold)
    seg = assign_class(seg, cd_threshold)
    dmap = dmap.rename(columns={"Relation": "rel_cd"})
    seg = seg.rename(columns={"Relation": "rel_cd"})
    result = pd.concat([dmap, seg[seg['cd_class'] == '신축']],
                       ignore_index=True)
    cd_class_map = {'변화없음': 0, '신축': 1, '소멸': 2, '갱신': 3}

    result = result.copy()
    result['CLS_NAME'] = result['cd_class']
    result['CLS_ID'] = result['CLS_NAME'].map(cd_class_map)
    result['AREA'] = result['geometry'].area.round(2)

    # 필요한 컬럼만 남기기
    result['ID'] = result.apply(
        lambda row: f"p_{int(row['poly1_idx'])}" if not pd.isna(row.get('poly1_idx')) else f"c_{int(row['poly2_idx'])}",
        axis=1
    )

    # 필요한 컬럼만 정리
    result = result[['CLS_ID', 'CLS_NAME', 'AREA', 'ID', 'geometry']]
    return result


def main():
    args = make_args().parse_args()
    progress = ProgressBar()
    status = StatusManager(args.output)
    with status.task("Loading input files"):
        args = resolve_paths(args)

    # 건물 추론 시작
    with status.task("Loading ONNX Model"):
        progress.write("Start Building Segmentation")
        # 모델 및 설정 불러오기
        progress.update("Loading ONNX Model", perc=5)
        session, config = create_session(get_model_file(args.model), max_threads=args.max_threads)
        config = override_config(
            config,
            conf_threshold=args.conf_threshold,
            resolution=args.resolution,
            classes=args.classes
        )
        progress.write("Completed Load Model")
    # 영상 데이터, 해상도 설정
    with status.task("Loading GeoTIFF"):
        progress.update("Loading GeoTIFF", perc=5)
        raster = load_raster(args.geotiff)
        progress.write("Completed Load GeoTIFF")

    with raster:
        input_res = get_input_resolution(raster)
        height, width, scale_factor, tiles_overlap, windows = compute_tiling_params(raster, config, input_res)
        indexes = determine_indexes(raster)

        with status.task("Processing Tiles"):
            mask = process_tiles(args.geotiff, windows, indexes, session, config, progress=progress, total_perc=55, batch_size=args.batch_size)
            progress.write("Completed Process tiles")

    with status.task("PostProcessing"):
        progress.update("PostProcessing", perc=5)
        # 마스크 후처리
        mask = median_filter(mask, 5)
        mask = filter_small_segments(mask, config)
        mask = morphology_to_mask(mask, open_k=21, close_k=3, iterations=1)
        mask = morphology_to_mask(mask, open_k=7, close_k=3, iterations=1)

        # GDF 변환 및 후처리
        inf_gdf = mask_to_gdf(raster, mask, config)
        inf_gdf = simplify_polygon(inf_gdf, tolerance=0.4, preserve_topology=True)
        inf_gdf = gpd.GeoDataFrame(geometry=inf_gdf)

        progress.write("Completed PostProcessing")
    # 변화 탐지 시작
    with status.task("Generating Graph"):
        progress.write("Start Change Detection")
        progress.update("Generating Graph", perc=5)

        # 매칭 전처리
        prev_gdf = import_shapefile(args.prev_gdf)
        prev_gdf, inf_gdf, joined = indexing(prev_gdf, inf_gdf)

        # 그래프 생성
        graph = build_graph(joined)
        graph = add_energy_to_links(prev_gdf, inf_gdf, graph)
        component, graph, cut_link, summary = split_graph_by_energy(prev_gdf, inf_gdf, graph, args.cut_threshold)
        progress.write("Completed Generating Graph")

    with status.task("Cal metrics"):
        # metrics 계산
        progress.update("Cal metrics", perc=20)
        prev_gdf, inf_gdf = mark_cut_links(prev_gdf, inf_gdf, cut_link)
        prev_gdf, inf_gdf = attach_metrics_from_components(component, prev_gdf, inf_gdf)

        progress.write("Completed Cal metrics")
    with status.task("Classification"):
        progress.update("Finalizing", perc=6)
        # 변화 유형 분류
        result = cd_pipeline(prev_gdf, inf_gdf, args.cd_threshold)
        os.makedirs(args.output, exist_ok=True)
        result.to_file(os.path.join(args.output, "result.json"), driver="GeoJSON", encoding="euc-kr")
    with status.task("Done"):
        progress.write("Done")


if __name__ == "__main__":
    main()
