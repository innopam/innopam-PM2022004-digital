import os
import ray
import json
import time
import imageio.v2 as imageio
import rasterio
import argparse
import numpy as np

from tqdm import tqdm
from rasterio.windows import Window
from rasterio.features import shapes, rasterize
from skimage.measure import label
from shapely.ops import unary_union
from shapely.geometry import Polygon, shape, mapping

from MambaCD.changedetection.apis import Inferencer2

# Ray 초기화 (필요하다면 수정)
ray.shutdown()
ray.init(num_cpus=os.cpu_count(), num_gpus=0)

def make_args():
    parser = argparse.ArgumentParser(description="Multiclass Change Detection system")
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ",
                        default=None, nargs='+')
    parser.add_argument('--model_path', type=str, default='/workspace/model/')
    parser.add_argument('--dataset_path', type=str, default='/workspace/sample_data/')
    parser.add_argument('--output_path', type=str, default='/workspace/out/')
    parser.add_argument('--multi_mode', type=bool, default=True)
    parser.add_argument('--data_name_list', type=list)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--overlap_ratio", type=str, default='25')
    parser.add_argument("--status_file", type=str, default="status.json")

    return parser

def log_error(log_file_path, message):
    with open(log_file_path,"a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

def update_status(status, status_file):
    with open(status_file,"w") as f:
        json.dump(status,f,indent=4)

def measure_time(func):
    def wrapper(*args, **kwargs):
        st= time.perf_counter()
        res= func(*args, **kwargs)
        et= time.perf_counter()
        elapsed= et-st
        return res, elapsed
    return wrapper

def calculate_patch_distribution(image_size, patch_size, save_dir, opt):
    y_size, x_size = image_size
    patch_height = patch_width = patch_size

    if opt == 'min':
        # 각 축의 패치 수 계산
        y_patches = y_size // patch_height + 1
        x_patches = x_size // patch_width + 1
    
        # 각 축의 나머지 계산
        y_remainder = patch_size * y_patches - y_size
        x_remainder = patch_size * x_patches - x_size
    
        # 중복 영역만 고려하여 overlap 비율 계산 (y_patches - 1, x_patches - 1)
        y_overlap_distribution = [0] * (y_patches - 1)
        x_overlap_distribution = [0] * (x_patches - 1)
    
        if y_remainder > 0:
            # 나머지를 적절히 나누기 위한 패치 수 결정
            y_chunks = int(np.floor(y_remainder / (y_patches - 1)))
            for i in range(y_patches - 1):
                if i < y_remainder % (y_patches - 1):
                    y_overlap_distribution[i] = y_chunks + 1
                else:
                    y_overlap_distribution[i] = y_chunks
    
        if x_remainder > 0:
            x_chunks = int(np.floor(x_remainder / (x_patches - 1)))
            for i in range(x_patches - 1):
                if i < x_remainder % (x_patches - 1):
                    x_overlap_distribution[i] = x_chunks + 1
                else:
                    x_overlap_distribution[i] = x_chunks
    else:
        if 0 <= float(opt) <= 100:
            overlap_ratio = float(opt) / 100.

            effective_patch_size_y = int(np.ceil(patch_height * (1 - overlap_ratio)))
            effective_patch_size_x = int(np.ceil(patch_width * (1 - overlap_ratio)))
    
            y_patches = int(np.ceil(y_size / effective_patch_size_y))
            x_patches = int(np.ceil(x_size / effective_patch_size_x))
    
            y_overlap_distribution = [(patch_height-effective_patch_size_y)] * (y_patches-1)
            x_overlap_distribution = [(patch_width-effective_patch_size_x)] * (x_patches-1)
        else:
            raise(ValueError("=> Please enter a valid overlap ratio (between 0 and 100)."))

    # 총 패치 수 계산
    total_patches = y_patches * x_patches

    # overlap 비율 계산
    y_overlap_ratio = np.mean([y / patch_height for y in y_overlap_distribution]) * 100
    x_overlap_ratio = np.mean([x / patch_width for x in x_overlap_distribution]) * 100

    # 각 overlap 분포의 개수 세기
    y_distribution_count = {}
    for value in y_overlap_distribution:
        if value in y_distribution_count:
            y_distribution_count[value] += 1
        else:
            y_distribution_count[value] = 1

    x_distribution_count = {}
    for value in x_overlap_distribution:
        if value in x_distribution_count:
            x_distribution_count[value] += 1
        else:
            x_distribution_count[value] = 1

    # patch_distribution.txt 파일에 저장
    with open(os.path.join(save_dir, "patch_distribution.txt"), "a") as f:
        f.write(f"Image size: {image_size}\n")
        f.write(f"Patch size: {patch_size}\n")
        f.write(f"Patches along Y-axis: {y_patches}\n")
        f.write(f"Patches along X-axis: {x_patches}\n")
        f.write(f"Total patches: {total_patches}\n")
        f.write(f"Y-axis overlap distribution: {y_distribution_count}\n")
        f.write(f"X-axis overlap distribution: {x_distribution_count}\n")
        f.write(f"Y-axis overlap ratio: {y_overlap_ratio:.2f}%\n")
        f.write(f"X-axis overlap ratio: {x_overlap_ratio:.2f}%\n")

    return y_patches, x_patches, y_overlap_distribution, x_overlap_distribution

@ray.remote
def patch_maker(y, x, t1_path, t2_path, save_dir, img_name, width, height, patch_size, x_overlap_distribution, y_overlap_distribution):
    with rasterio.open(t1_path, IGNORE_COG_LAYOUT_BREAK="YES") as src1, \
    rasterio.open(t2_path, IGNORE_COG_LAYOUT_BREAK="YES") as src2:
        start_x = max(0, x * patch_size - sum(x_overlap_distribution[:x]))
        start_y = max(0, y * patch_size - sum(y_overlap_distribution[:y]))
        end_x = min(start_x + patch_size, width)
        end_y = min(start_y + patch_size, height)

        # 디스크에 기록 (해당 window 영역)
        window = Window(start_x, start_y, end_x - start_x, end_y - start_y)

        patch_t1 = np.transpose(src1.read(window=window), (1, 2, 0))[:,:,:3].astype(np.uint8)
        patch_t2 = np.transpose(src2.read(window=window), (1, 2, 0))[:,:,:3].astype(np.uint8)
        # patch_t2 = adjust_img(patch_t2, patch_t1)

        imageio.imwrite(os.path.join(save_dir, "T1", f"{img_name}_{y}_{x}.png"), patch_t1)
        imageio.imwrite(os.path.join(save_dir, "T2", f"{img_name}_{y}_{x}.png"), patch_t2)

    with open(os.path.join(save_dir, "data_list.txt"), "a") as train_file:
        train_file.write(f"{img_name}_{y}_{x}.png\n")

def make_patch(t1_path, t2_path, save_dir, patch_size, img_name, overlap_ratio):
    """
    대규모 이미지를 청크(window) 단위로 나누어 패치를 생성.
    """
    with rasterio.open(t1_path, IGNORE_COG_LAYOUT_BREAK="YES") as src1, \
        rasterio.open(t2_path, "r+", IGNORE_COG_LAYOUT_BREAK="YES") as src2:
        if (src1.width != src2.width) or (src1.height != src2.height):
            raise ValueError("Shape mismatch")

        width, height = src1.width, src1.height
        metadata = src2.meta.copy()
        metadata.update({"driver": "GTiff", "dtype": "uint8",
                         "height": height, 'width': width,
                         'blockxsize': patch_size, 'blockysize': patch_size,
                         'tiled': False})
        if 'crs' in metadata:
            metadata['crs'] = str(metadata['crs'].to_wkt()) 
        metadata = {k: v for k, v in metadata.items() if k != 'count'}

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir,"T1"), exist_ok=True)
        os.makedirs(os.path.join(save_dir,"T2"), exist_ok=True)

        with open(os.path.join(save_dir, "patch_distribution.txt"), "w") as f:
            f.write(f"--------{img_name} patch distribution--------\n")
        
        y_patches, x_patches, y_overlap_distribution, x_overlap_distribution = calculate_patch_distribution((src1.height, src1.width), patch_size, save_dir, overlap_ratio)

        with open(os.path.join(save_dir, "data_list.txt"), "w") as f:
            pass

    tasks=[]
    for y in range(y_patches):
        for x in range(x_patches):
            tasks.append(patch_maker.remote(
                y=y, 
                x=x, 
                t1_path=t1_path, 
                t2_path=t2_path, 
                save_dir=save_dir, 
                img_name=img_name, 
                width=width, 
                height=height, 
                patch_size=patch_size, 
                x_overlap_distribution=x_overlap_distribution, 
                y_overlap_distribution=y_overlap_distribution
                ))
    pbar = tqdm(total=len(tasks), desc=f"Creating patches for {img_name}...")
    while tasks:
        done_tasks, tasks = ray.wait(tasks, num_returns=1)
        for _ in done_tasks:
            pbar.update(1)  # 진행률 업데이트
    pbar.close()  # 작업 완료 후 tqdm 닫기

    with open(os.path.join(save_dir, f"{img_name}_metadata.json"), 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

@measure_time
def inference(args, mode, img_name):
    # 모델 로직 건드리지 않음
    inf= Inferencer2(args, mode, img_name)
    inf.infer()
    return "추론 완료"

@ray.remote
def reconstruct_patch(y, x, color_map, img_name, out_dir, image_size, patch_size, x_overlap_list, y_overlap_list):
    patch_path = os.path.join(out_dir, "pred", f"{img_name}_{y}_{x}.png")
    conf_path = os.path.join(out_dir, "confidence", f"{img_name}_{y}_{x}_confidence.png")

    # 원본 TIFF 이미지의 크기 설정
    y_size, x_size = image_size
    
    # 존재하지 않는 패치 파일이 있을 경우 건너뜀
    if not os.path.exists(patch_path) or not os.path.exists(conf_path):
        return None

    # 패치 읽기
    patch_rgb = imageio.imread(patch_path)
    patch_conf = imageio.imread(conf_path)

    # RGB 패치를 클래스 ID로 변환
    patch_class = np.zeros((patch_rgb.shape[0], patch_rgb.shape[1]), dtype=np.uint8)
    for color, mapped_value in color_map.items():
        mask = np.all(patch_rgb == color, axis=-1)
        patch_class[mask] = mapped_value

    # 시작 좌표 계산
    start_x = max(0, x * patch_size - sum(x_overlap_list[:x]))
    start_y = max(0, y * patch_size - sum(y_overlap_list[:y]))
    end_x = min(start_x + patch_rgb.shape[1], x_size)
    end_y = min(start_y + patch_rgb.shape[0], y_size)

    # 디스크에 기록 (해당 window 영역)
    window = Window(start_x, start_y, end_x - start_x, end_y - start_y)

    return (patch_class[:end_y - start_y, :end_x - start_x],
            patch_conf[:end_y - start_y, :end_x - start_x],
            window)

def reconstruct_tiff_from_patches(save_dir, out_dir, mode, img_name):
    # (a) patch_distribution.txt 읽기
    patch_distribution_path = os.path.join(save_dir, "patch_distribution.txt")
    with open(patch_distribution_path, "r") as f:
        lines = f.readlines()
        image_size = tuple(map(int, lines[1].split(": ")[1].strip("()\n").split(", ")))
        patch_size = int(lines[2].split(": ")[1].strip())
        y_patches  = int(lines[3].split(": ")[1].strip())
        x_patches  = int(lines[4].split(": ")[1].strip())
        y_overlap_distribution = eval(lines[6].split("n: ")[1].strip())
        x_overlap_distribution = eval(lines[7].split("n: ")[1].strip())

        # 오버랩 리스트
        y_overlap_list = [k for k, v in y_overlap_distribution.items() for _ in range(v)]
        x_overlap_list = [k for k, v in x_overlap_distribution.items() for _ in range(v)]

    # (b) 메타데이터
    with open(os.path.join(save_dir, f"{img_name}_metadata.json"), 'r') as metadata_file:
        metadata = json.load(metadata_file)
        metadata.update({
            'tiled': True,
            'interleave': 'band'
        })

    # (c) color_map 설정
    if mode:
        color_map = {
            (0, 0, 0): 0, (255, 0, 0): 1, (0, 255, 0): 2,
            (0, 0, 255): 3, (255, 255, 0): 4, (255, 0, 255): 5,
            (0, 255, 255): 6, (255, 255, 255): 7
        }
    else:
        color_map = {(0, 0, 0): 0, (255, 255, 255): 1}

    # (d) 최종 결과 TIFF (없으면 생성)
    output_img_path  = os.path.join(out_dir, f"{img_name}.tif")
    output_conf_path = os.path.join(out_dir, f"{img_name}_conf.tif")
    if not os.path.exists(output_img_path):
        with rasterio.open(output_img_path, 'w', BIGTIFF='YES', count=1, **metadata) as dst_img, \
             rasterio.open(output_conf_path, 'w', BIGTIFF='YES', count=1, **metadata) as dst_conf:
            pass

    # (e) Ray로 패치 병렬 처리 -> (patch_class, patch_conf, window) 가져오기
    tasks_list = []
    for y in range(y_patches):
        for x in range(x_patches):
            ref = reconstruct_patch.remote(
                y=y,
                x=x,
                color_map=color_map,
                img_name=img_name, 
                out_dir=out_dir, 
                image_size=image_size,
                patch_size=patch_size, 
                x_overlap_list=x_overlap_list, 
                y_overlap_list=y_overlap_list
            )
            tasks_list.append((y, x, ref))

    # (f) 결과를 차례로 받아오면서(또는 전부 모아서) TIFF에 write
    pbar = tqdm(total=len(tasks_list), desc=f"Reconstructing {img_name}...")
    with rasterio.open(output_img_path, 'r+') as dst_img, \
         rasterio.open(output_conf_path, 'r+') as dst_conf:

        # Ray.wait()로 하나씩 가져오면서 쓰기(또는 ray.get(tasks)로 한꺼번에 받아도 됨)
        remaining_refs = [t[2] for t in tasks_list]
        while remaining_refs:
            done_refs, remaining_refs = ray.wait(remaining_refs, num_returns=1)
            done_ref = done_refs[0]
            # 어느 (y, x)에 해당하는지 찾아야 하므로 인덱스를 매칭
            done_task = None
            for (yy, xx, rr) in tasks_list:
                if rr == done_ref:
                    done_task = (yy, xx, rr)
                    break

            if done_task is not None:
                patch_result = ray.get(done_task[2])  # (patch_class, patch_conf, window) or None
                if patch_result is not None:
                    patch_class, patch_conf, window = patch_result
                    dst_img.write(patch_class, 1, window=window)
                    dst_conf.write(patch_conf, 1, window=window)

            pbar.update(1)
    pbar.close()

def get_overlapping_windows(height, width, patch_size, overlap=0.25):
    """오버랩된 윈도우 좌표 생성"""
    step_size = int(patch_size * (1 - overlap))
    windows = []
    
    for y in range(0, height, step_size):
        if y + patch_size > height:
            y = max(0, height - patch_size)
        for x in range(0, width, step_size):
            if x + patch_size > width:
                x = max(0, width - patch_size)
            windows.append(rasterio.windows.Window(x, y, 
                                                 min(patch_size, width - x),
                                                 min(patch_size, height - y)))
    return windows

@ray.remote
def polygonize_window(label_array, label_id, pixel_area, min_area, block_array, block_transform):
    obj_mask = label_array == label_id
    total_area = float(np.sum(obj_mask))

    if not np.any(obj_mask) or total_area * pixel_area < min_area:
        return

    unique_classes = np.unique(block_array[obj_mask])

    features = []
    for cls in unique_classes:
        if cls == 0:  # 배경 클래스 제외
            continue

        cls_mask = ((block_array == cls) & obj_mask).astype(np.uint8)

        polygons = []
        for geom, val in shapes(cls_mask, transform=block_transform):
            # val이 1이면 해당 픽셀이 cls_mask 내 'True(1)'인 영역
            if val == 1:
                poly = shape(geom)
                # 유효 여부와 최소 면적 조건 확인
                if poly.is_valid and poly.area > min_area:
                    polygons.append(poly)

        if polygons:
            outer_polygon = max(polygons, key=lambda p: p.area)
            features.append({
                "type": "Feature",
                "properties": {
                    "CLS_ID": int(cls),
                    "AREA": outer_polygon.area
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [list(outer_polygon.exterior.coords)]
                }
            })
    return features

def post_process_features(all_features, conf_path, transform, color_map, tolerance=0.5):
    class_features = {}
    for feature in all_features:
        cls_id = feature["properties"]["CLS_ID"]
        polygon = shape(feature["geometry"])
        if cls_id not in class_features:
            class_features[cls_id] = []
        class_features[cls_id].append(polygon)
    
    # 같은 클래스의 객체 병합
    merged_features = []
    for cls_id, polygons in class_features.items():
        union_polygon = unary_union(polygons)
        if union_polygon.geom_type == 'Polygon':
            polygons = [union_polygon]  # 리스트로 감싸기
        else:
            polygons = union_polygon.geoms
        for poly in polygons:
            if poly.is_valid:
                minx = abs(int((poly.bounds[0] - transform[2]) / transform[0]))
                miny = abs(int((poly.bounds[3] - transform[5]) / transform[4]))
                maxx = abs(int((poly.bounds[2] - transform[2]) / transform[0]))
                maxy = abs(int((poly.bounds[1] - transform[5]) / transform[4]))

                pixel_coords = [(
                    abs(abs(int((y - transform[2]) / transform[0])) - minx),
                    abs(abs(int((x - transform[5]) / transform[4])) - miny)) 
                    for y, x in poly.exterior.coords]
                
                # width와 height 계산
                width = max(1, abs(maxx - minx))
                height = max(1, abs(maxy - miny))
                
                window = Window(minx, miny, width, height)
                with rasterio.open(conf_path) as conf_src:
                    conf_raster = conf_src.read(1, window=window)

                    pixel_polygon = Polygon(pixel_coords)
                    rasterized = rasterize(
                        [(pixel_polygon, 1)],
                        out_shape=conf_raster.shape,
                        fill=0,
                        dtype=np.uint8)
    
                    # 마스크를 사용해 해당 영역의 conf_array 값 추출
                    masked_conf = conf_raster * rasterized  # 래스터화된 부분만 추출
                    valid_conf = masked_conf[masked_conf > 0]
                    if valid_conf.size == 0:
                        conf_mean = np.nan
                    else:
                        conf_mean = valid_conf.mean()

                merged_features.append({
                    "type": "Feature",
                    "properties": {
                        "CLS_ID": cls_id,
                        "CLS_NAME": color_map[cls_id],
                        "CONF": round(conf_mean, 2),
                        "AREA": round(poly.area, 2)
                    },
                    "geometry": mapping(poly)
                })
    
    # 유효한 features만 남김
    processed_features = [f for f in merged_features if f is not None]

    # 중심점 기준으로 정렬
    features_with_centroids = []
    for feature in processed_features:
        poly = shape(feature["geometry"])
        centroid = poly.centroid
        features_with_centroids.append((centroid.x, centroid.y, feature))
    
    # y좌표 오름차순, x좌표 오름차순으로 정렬
    sorted_features = sorted(features_with_centroids, 
                           key=lambda x: (x[1], x[0]), reverse=True)
    
    # 정렬된 순서대로 ID 부여
    for idx, (_, _, feature) in enumerate(sorted_features, start=1):
        poly = shape(feature["geometry"]).simplify(tolerance=tolerance)
        feature["geometry"] = mapping(poly)
        feature["properties"]["CLS_NAME"] = color_map[int(feature["properties"]["CLS_ID"])]
        feature["properties"]["ID"] = idx
    
    return [feature for _, _, feature in sorted_features]

def polygonize(out_dir, mode, img_name, patch_size, overlap=0.25, min_area=50):
    """메인 벡터화 함수"""
    input_tif_path = os.path.join(out_dir, f"{img_name}.tif")
    conf_path = os.path.join(out_dir, f"{img_name}_conf.tif")
    output_json_path = os.path.join(out_dir, f"{img_name}.json")

    if mode:
        color_map = {1: "신축", 2: "소멸", 3: "갱신", 4: "색상변화", 
        5: "더미 1", 6: "더미 2", 7: "더미 3"}
    else:
        color_map = {1: "변화"}

    with rasterio.open(input_tif_path) as src:
        transform = src.transform
        pixel_area = abs(transform[0] * transform[4])
        crs = src.crs
        if crs:
            crs = f"EPSG:{crs.to_epsg()}"
        geojson_data = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": f"{crs}"}},
        "map_number": img_name,
        "features": []
        }
        windows = get_overlapping_windows(src.height, src.width, patch_size, overlap)
        all_features = []
        # 블록 단위로 데이터 읽기 및 처리
        pbar = tqdm(total=len(windows), desc=f"Vectorizing {img_name}...")
        for window in windows:
            # 블록 데이터 읽기
            block_array = src.read(1, window=window)
            block_array_ref = ray.put(block_array)
            block_transform = src.window_transform(window)
            # 레이블링 및 객체 마스크 생성
            label_array = label(block_array > 0, connectivity=2)
            label_array_ref = ray.put(label_array)
            tasks = []
            for label_id in range(1, label_array.max() + 1):
                tasks.append(polygonize_window.remote(
                    label_array=label_array_ref,
                    label_id=label_id, 
                    pixel_area=pixel_area, 
                    min_area=min_area, 
                    block_array=block_array_ref, 
                    block_transform=block_transform, 
                ))
            while tasks:
                done_tasks, tasks = ray.wait(tasks, num_returns=1)
                for result in done_tasks:
                    feature = ray.get(result)
                    if feature:
                        all_features.extend(feature)
            pbar.update(1)  # 진행률 업데이트
        pbar.close()  # 작업 완료 후 tqdm 닫기
        # 중첩된 폴리곤 병합
        processed_features = post_process_features(all_features, conf_path, transform, color_map)
        geojson_data["features"] = processed_features
    # 결과 저장
    with open(output_json_path, "w") as f:
        json.dump(geojson_data, f, indent=4)

@measure_time
def reconstruct(infer_dataset_path, result_saved_path, mode, img_name):
    reconstruct_tiff_from_patches(infer_dataset_path, result_saved_path, mode, img_name)
    return "재구성 완료"

@measure_time
def vectorize(result_saved_path, mode, img_name, patch_size):
    polygonize(result_saved_path, mode, img_name, patch_size)
    return "벡터화 완료"

#######################################################
# 단계별 함수 (step_patch / step_inference / step_reconstruct / step_vectorize)
#######################################################
@measure_time
def step_patch(args, t1_file, t2_file, final_name):
    status_file= os.path.join(args.output_path, args.status_file)

    with open(status_file,"r") as f:
        st= json.load(f)

    patch_dir= os.path.join(args.output_path,"patches", final_name)
    os.makedirs(patch_dir, exist_ok=True)
    
    t1_path= os.path.join(args.dataset_path,"T1", t1_file)
    t2_path= os.path.join(args.dataset_path,"T2", t2_file)

    make_patch(t1_path, t2_path, patch_dir,
               args.patch_size, final_name, args.overlap_ratio)

    # step 완료 -> status 업데이트
    st["Process"]= 20
    st["ElapsedTime"]["패치 생성"]= "done"
    st["final_name"]= final_name
    with open(status_file,"w") as f:
        json.dump(st,f,indent=4)

    return f"Patch creation done: {final_name}"

@measure_time
def step_inference(args):
    """
    2) 추론
    - 이미 patch_dir/data_list.txt 존재
    - args.model_weights => 실제 model
    """
    status_file= os.path.join(args.output_path, args.status_file)
    with open(status_file,"r") as f:
        st= json.load(f)

    final_name= st.get("final_name")
    if not final_name:
        raise ValueError("No final_name in status. Did you run step_patch?")
    
    patch_dir= os.path.join(args.output_path,"patches", final_name)

    data_list_path= os.path.join(patch_dir,"data_list.txt")
    if not os.path.exists(data_list_path):
        raise FileNotFoundError("[step_inference] data_list.txt not found")

    with open(data_list_path,"r") as ff:
        dlist= [ln.strip() for ln in ff if ln.strip()]
    args.data_name_list= dlist
    
    # 실제 추론
    _, _ = inference(args, args.multi_mode, final_name)

    # status
    st["Process"]= 70
    st["ElapsedTime"]["추론"]= "done"
    with open(status_file,"w") as f:
        json.dump(st,f,indent=4)

    return "Inference done"

@measure_time
def step_reconstruct(args):
    """
    3) 재구성
    """
    status_file= os.path.join(args.output_path, args.status_file)
    with open(status_file,"r") as f:
        st= json.load(f)

    final_name= st.get("final_name")
    if not final_name:
        raise ValueError("no final_name in status")
    
    patch_dir= os.path.join(args.output_path,"patches", final_name)
    result_dir= os.path.join(args.output_path,"results",final_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    # reconstruct
    _, _ = reconstruct(patch_dir, result_dir, args.multi_mode, final_name)

    st["Process"]= 90
    st["ElapsedTime"]["재구성"]= "done"
    with open(status_file,"w") as f:
        json.dump(st,f,indent=4)

    return "Reconstruction done"

@measure_time
def step_vectorize(args):
    """
    4) 벡터화
    """
    status_file= os.path.join(args.output_path, args.status_file)
    with open(status_file,"r") as f:
        st= json.load(f)

    final_name= st.get("final_name")
    if not final_name:
        raise ValueError("no final_name in status")

    result_dir= os.path.join(args.output_path,"results",final_name)

    # vectorize
    _, _ = vectorize(result_dir, args.multi_mode, final_name, args.patch_size)

    st["Process"]= 100
    st["ElapsedTime"]["벡터화"]= "done"
    with open(status_file,"w") as f:
        json.dump(st,f, indent=4)

    return "Vectorization done"

def main():
    # 1) 인자 생성
    args = make_args().parse_args()
    
    args.model_weights = [os.path.join(args.model_path, pth) for pth in os.listdir(args.model_path) if pth.endswith('.pth')][0]

    # make patch
    init_infer_path = os.path.join(args.output_path,'patches')
    init_out_path = os.path.join(args.output_path, 'results')
    
    init_t1_path = os.path.join(args.dataset_path, 'T1/')
    init_t2_path = os.path.join(args.dataset_path, 'T2/')

    t1_list = [file for file in os.listdir(init_t1_path) if file.endswith(".tif")]
    t2_list = [file for file in os.listdir(init_t2_path) if file.endswith(".tif")]

    if len(t1_list) != len(t2_list):
        raise(RuntimeError("=> Please match before/after images"))

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    status_file = os.path.join(args.output_path, args.status_file)
    total_step, current_step, processing = len(t1_list), 0, 0
    # 초기화(혹은 이미 존재하면 갱신)
    with open(status_file,"w") as f:
        json.dump({
            "Total_step": total_step,
            "Current_step": current_step,
            "Process": processing,
            "Status": "pending",
            "ElapsedTime":{},
        }, f, indent=4)

    try:
        for t1, t2 in zip(t1_list, t2_list):
            t1_file = t1
            t2_file = t2
            # (2) final_name 생성
            base_t1 = t1_file.replace(".tif","")
            base_t2 = t2_file.replace(".tif","")
            if base_t1 == base_t2:
                final_name = base_t1
            else:
                final_name = f"{base_t1}_{base_t2}"

            # (3) patch 디렉토리, result 디렉토리 경로 등 잡아주기
            infer_dataset_path = os.path.join(init_infer_path, final_name)
            result_saved_path = os.path.join(init_out_path, final_name)

            with open(status_file,"r") as ff:
                st= json.load(ff)
            st["Status"] = 'in progress'
            st["final_name"]= final_name
            with open(status_file,"w") as ff:
                json.dump(st,ff,indent=4)

            # (4) 패치 생성
            if not os.path.exists(infer_dataset_path):
                _, time_spent = step_patch(args, t1_file, t2_file, final_name)
                with open(status_file,"r") as ff:
                    st= json.load(ff)
                st["ElapsedTime"]["패치 생성"]= f"{round(time_spent,2)}s"
                st["CurrentTask"]="patch done"
                with open(status_file,"w") as ff:
                    json.dump(st,ff,indent=4)

            if not os.path.exists(result_saved_path):
                _, time_spent= step_inference(args)
                with open(status_file,"r") as ff:
                    st= json.load(ff)
                st["ElapsedTime"]["추론"]= f"{round(time_spent,2)}s"
                st["CurrentTask"]="inference done"
                with open(status_file,"w") as ff:
                    json.dump(st,ff,indent=4)

            if not os.path.exists(os.path.join(result_saved_path, final_name+'.tif')):
                _, time_spent= step_reconstruct(args)
                with open(status_file,"r") as ff:
                    st= json.load(ff)
                st["ElapsedTime"]["재구성"]= f"{round(time_spent,2)}s"
                st["CurrentTask"]="reconstruct done"
                with open(status_file,"w") as ff:
                    json.dump(st,ff,indent=4)

            if not os.path.exists(os.path.join(result_saved_path, final_name+'.json')):
                _, time_spent= step_vectorize(args)
                with open(status_file,"r") as ff:
                    st= json.load(ff)
                st["ElapsedTime"]["벡터화"]= f"{round(time_spent,2)}s"
                st["CurrentTask"]="vectorize done"

                # done
                st["Status"]="done"
                st["CurrentTask"]="process completed"
                with open(status_file,"w") as ff:
                    json.dump(st,ff,indent=4)

            st["Current_step"] += 1
            with open(status_file,"w") as ff:
                json.dump(st,ff,indent=4)
            print(f"{final_name}.tif process done")

        print("=== All tasks completed ===")
    
    except Exception as ex:
        error_message = f"Error during processing for {final_name}: {str(ex)}"
        print(error_message)
        log_error(os.path.join(args.output_path, "error_log.txt"), error_message)
        with open(status_file,"r") as ff:
            st= json.load(ff)
        st["Status"] = 'failed'
        with open(status_file,"w") as ff:
            json.dump(st,ff,indent=4)

if __name__ == "__main__":
    main()
