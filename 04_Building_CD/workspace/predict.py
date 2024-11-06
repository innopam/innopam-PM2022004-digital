import os
import json
import imageio.v2 as imageio
import rasterio
import argparse
import subprocess
import numpy as np
import geopandas as gpd

from tqdm import tqdm
from rasterio.mask import mask
from osgeo import gdal, ogr, osr

from MambaCD.changedetection.apis import Inferencer

def make_args():
    parser = argparse.ArgumentParser(description="Multiclass Change Detection system")
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')
    parser.add_argument('--model_path', type=str, default='/workspace/model/')
    parser.add_argument('--dataset_path', type=str, default='/workspace/sample_data/')
    parser.add_argument('--output_path', type=str, default='/workspace/out/')
    parser.add_argument('--data_name_list', type=list)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--overlap_ratio", type=str, default='min')

    return parser

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

    return y_patches, x_patches, total_patches, y_overlap_distribution, x_overlap_distribution, opt

def make_patches(t1_path, t2_path, save_dir, patch_size, img_name, overlap_ratio):
    with rasterio.open(t1_path) as src:
        t1_image = np.transpose(src.read(), (1, 2, 0))[:,:,:3]
        metadata = src.meta.copy()
    with rasterio.open(t2_path) as src:
        t2_image = np.transpose(src.read(), (1, 2, 0))[:,:,:3]
    
    size_list = [t1_image.shape[:2],t2_image.shape[:2]]
    image_size = (min(t[0] for t in size_list), min(t[1] for t in size_list))

    metadata.update({"driver": "GTiff", "dtype": "uint8", "height": image_size[0], 'width': image_size[1]})
    metadata['crs'] = str(metadata['crs'].to_wkt()) if 'crs' in metadata else None
    metadata = {k: v for k, v in metadata.items() if k != 'count'}
    
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir,"T1"))
        os.makedirs(os.path.join(save_dir,"T2"))

    with open(os.path.join(save_dir, "patch_distribution.txt"), "a") as f:
        f.write(f"--------{img_name} patch distribution--------\n")

    y_patches, x_patches, total_patches, y_overlap_distribution, x_overlap_distribution, opt = calculate_patch_distribution(image_size, patch_size, save_dir, overlap_ratio)

    patch_height = patch_width = patch_size

    with tqdm(total=total_patches, desc=f"Creating patches for {img_name}") as pbar:
        with open(os.path.join(save_dir, "data_list.txt"), "a") as train_file:
            for y in range(y_patches):
                for x in range(x_patches):
                    if x == 0 and y == 0:
                        patch_t1 = t1_image[0:patch_height, 0:patch_width]
                        patch_t2 = t2_image[0:patch_height, 0:patch_width]
                    else:
                        start_x = x * patch_width - sum(x_overlap_distribution[:x])
                        start_y = y * patch_height - sum(y_overlap_distribution[:y])
                        if x == 0:
                            patch_t1 = t1_image[start_y:start_y + patch_height, 0:patch_width]
                            patch_t2 = t2_image[start_y:start_y + patch_height, 0:patch_width]
                        elif y == 0:
                            patch_t1 = t1_image[0:patch_height, start_x:start_x + patch_width]
                            patch_t2 = t2_image[0:patch_height, start_x:start_x + patch_width]
                        elif isinstance(opt, (int, float)):
                            patch_t1 = np.zeros((patch_height, patch_width))
                            patch_t2 = np.zeros((patch_height, patch_width))
        
                            start_x = x * patch_width - sum(x_overlap_distribution[:x])
                            start_y = y * patch_height - sum(y_overlap_distribution[:y])
                            end_y = image_size[0] - 1
                            end_x = image_size[1] - 1
                        
                            if x == x_patches-1 and y == y_patches-1:
                                patch_t1[:end_y - start_y + 1, :end_x - start_x + 1] = t1_image[start_y:end_y, start_x:end_x]
                                patch_t2[:end_y - start_y + 1, :end_x - start_x + 1] = t2_image[start_y:end_y, start_x:end_x]
                            elif x == x_patches-1:
                                patch_t1[:patch_height, :end_x - start_x + 1] = t1_image[start_y:start_y + patch_height, start_x:end_x]
                                patch_t2[:patch_height, :end_x - start_x + 1] = t2_image[start_y:start_y + patch_height, start_x:end_x]
                            elif y == y_patches-1:
                                patch_t1[:end_y - start_y + 1, :patch_width] = t1_image[start_y:end_y, start_x:start_x + patch_width]
                                patch_t2[:end_y - start_y + 1, :patch_width] = t2_image[start_y:end_y, start_x:start_x + patch_width]
                            else:
                                patch_t1 = t1_image[start_y:start_y + patch_height, start_x:start_x + patch_width]
                                patch_t2 = t2_image[start_y:start_y + patch_height, start_x:start_x + patch_width]
                        else:
                            patch_t1 = t1_image[start_y:start_y + patch_height, start_x:start_x + patch_width]
                            patch_t2 = t2_image[start_y:start_y + patch_height, start_x:start_x + patch_width]
        
                    imageio.imwrite(os.path.join(save_dir, "T1", f"{img_name}_{y}_{x}.png"), patch_t1)
                    imageio.imwrite(os.path.join(save_dir, "T2", f"{img_name}_{y}_{x}.png"), patch_t2)
    
                    train_file.write(f"{img_name}_{y}_{x}.png\n")
                    pbar.update(1)  # 프로그레스 바 업데이트

    with open(os.path.join(save_dir, f"{img_name}_metadata.json"), 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=4)
                
def polygonize(out_dir, img_name):
    input_tif_path = os.path.join(out_dir, f"{img_name}.tif")
    conf_path = os.path.join(out_dir, f"{img_name}_conf.tif")
    output_gpkg_path = os.path.join(out_dir, os.path.basename(input_tif_path).replace(".tif",".gpkg"))
    output_json_path = os.path.join(out_dir, os.path.basename(input_tif_path).replace(".tif",".json"))
    tmp = output_gpkg_path.replace(".gpkg", "_tmp.gpkg")

    subprocess.run(
        "gdal_polygonize.py -f GPKG %s %s merged CLS_ID" % (input_tif_path, tmp),
        shell=True,
    )

    # subprocess.run(
    #     "ogr2ogr -f GPKG -where CLS_ID!=0 %s %s" % (output_gpkg_path, tmp),
    #     shell=True,
    # )
    
    gdf = gpd.read_file(tmp)
    conf_score(conf_path, gdf)
    gdf[gdf['CLS_ID'] != 0].to_file(output_gpkg_path, layer="merged")  # 수정된 GeoDataFrame을 GeoPackage에 저장

    subprocess.run(
        "ogr2ogr -f GeoJSON -where CLS_ID!=0 %s %s" % (output_json_path, output_gpkg_path),
        shell=True,
    )
    os.remove(tmp)

    # GeoJSON 파일 포맷팅
    with open(output_json_path, 'r') as f:
        data = json.load(f)
    
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)  # indent=4로 JSON 포맷팅

def conf_score(conf_tif, gdf):
    src = rasterio.open(conf_tif)  # 신뢰도 TIFF 파일 열기
    with tqdm(total=len(gdf), desc=f"Adding Confidence score") as pbar:
        for i, value in gdf.iterrows():  # 각 폴리곤에 대해 반복
            out_image, _ = mask(src, [value.geometry])  # 폴리곤 영역의 마스크를 생성
            conf = out_image.sum() / np.count_nonzero(out_image) if np.count_nonzero(out_image) > 0 else 0  # 신뢰도 평균 계산
            gdf.loc[i, "CONF"] = round(conf, 2)  # 신뢰도 값을 GeoDataFrame에 추가
            pbar.update(1)  # 프로그레스 바 업데이트

def reconstruct_tiff_from_patches(save_dir, out_dir, img_name):
    # patch_distribution.txt에서 image_size와 patch_size, 패치 분포 및 겹침 정보 읽기
    patch_distribution_path = os.path.join(save_dir, "patch_distribution.txt")
    with open(patch_distribution_path, "r") as f:
        lines = f.readlines()
        image_size = tuple(map(int, lines[1].split(": ")[1].strip("()\n").split(", ")))
        patch_size = int(lines[2].split(": ")[1].strip())
        y_patches = int(lines[3].split(": ")[1].strip())
        x_patches = int(lines[4].split(": ")[1].strip())
        # Y-axis overlap distribution을 딕셔너리로 변환
        y_overlap_distribution = eval(lines[6].split("n: ")[1].strip())
        # X-axis overlap distribution을 딕셔너리로 변환
        x_overlap_distribution = eval(lines[7].split("n: ")[1].strip())

        # 리스트로 변환
        y_overlap_list = [k for k, v in y_overlap_distribution.items() for _ in range(v)]
        x_overlap_list = [k for k, v in x_overlap_distribution.items() for _ in range(v)]

    # 원본 TIFF 이미지의 크기 설정
    y_size, x_size = image_size
    patch_height = patch_width = patch_size  # 단일 패치 크기

    reconstructed_img = np.zeros((y_size, x_size), dtype=np.uint8)
    reconstructed_conf = np.zeros((y_size, x_size), dtype=np.uint8)
    reconstructed_img_rgb = np.zeros((y_size, x_size, 3), dtype=np.uint8)

    img_dir = os.path.join(out_dir, "pred")
    conf_dir = os.path.join(out_dir, "confidence")

    # 색상 맵 정의
    color_map = {(0, 0, 0): 0, (255, 0, 0): 1, (0, 255, 0): 2, (0, 0, 255): 3, (255, 255, 0): 4}

    # 각 타일을 읽어와서 원래 위치에 삽입
    for y in range(y_patches):
        for x in range(x_patches):
            patch_path = os.path.join(img_dir, f"{img_name}_{y}_{x}.png")
            conf_path = os.path.join(conf_dir, f"{img_name}_{y}_{x}_confidence.png")
            
            # 존재하지 않는 패치 파일이 있을 경우 건너뜀
            if not os.path.exists(patch_path):
                continue
            if not os.path.exists(conf_path):
                continue

            # 패치 읽기
            patch_rgb = imageio.imread(patch_path)
            patch = np.zeros((patch_rgb.shape[0], patch_rgb.shape[1]), dtype=np.int32)
            for color, mapped_value in color_map.items():
                mask = np.all(patch_rgb == color, axis=-1)
                patch[mask] = mapped_value
            patch_conf = imageio.imread(conf_path)

            if x == 0:
                start_x = 0
            else:
                start_x = x * patch_width - sum(x_overlap_list[:x])

            if y == 0:
                start_y = 0
            else:
                start_y = y * patch_height - sum(y_overlap_list[:y])

            # 타일을 재구성 이미지에 삽입
            end_y = start_y + patch_height
            end_x = start_x + patch_width
            reconstructed_img_rgb[start_y:end_y, start_x:end_x] = patch_rgb
            reconstructed_img[start_y:end_y, start_x:end_x] = patch
            reconstructed_conf[start_y:end_y, start_x:end_x] = patch_conf

    with open(os.path.join(save_dir, f"{img_name}_metadata.json"), 'r') as metadata_file:
        metadata = json.load(metadata_file)
    
    with rasterio.open(os.path.join(out_dir, f"{img_name}.tif"), 'w', count = 1, **metadata) as dst1:
        dst1.write(reconstructed_img[np.newaxis,:,:])
    reconstructed_img_rgb = reconstructed_img_rgb.transpose(2, 0, 1)
    with rasterio.open(os.path.join(out_dir, f"{img_name}_rgb.tif"), 'w', count = 3, **metadata) as dst2:
        dst2.write(reconstructed_img_rgb)
    with rasterio.open(os.path.join(out_dir, f"{img_name}_conf.tif"), 'w', count = 1, **metadata) as dst3:
        dst3.write(reconstructed_conf[np.newaxis,:,:])

def main():
    args = make_args().parse_args()

    # make patch
    init_infer_path = os.path.join(args.output_path,'patches')
    init_out_path = os.path.join(args.output_path, 'results')
    
    init_t1_path = os.path.join(args.dataset_path, 'T1/')
    init_t2_path = os.path.join(args.dataset_path, 'T2/')

    t1_list = [file for file in os.listdir(init_t1_path) if file.endswith(".tif")]
    t2_list = [file for file in os.listdir(init_t2_path) if file.endswith(".tif")]

    if t1_list != t2_list:
        raise(RuntimeError("=> Please match before/after images"))

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    status_path = args.output_path
    total_step, current_step, processing = len(t1_list), 0, 0

    status = {
    "Total_step": total_step,
    "Current_step": current_step,
    "Process": processing,
    "Status": "pending"
    }

    with open(os.path.join(status_path, "status.json"), "w") as s:
        json.dump(status, s, indent=4)
        
    # try:
    for t in t1_list:
        infer_dataset_path = os.path.join(init_infer_path, t.replace('.tif', ''))
        result_saved_path = os.path.join(init_out_path, t.replace('.tif', ''))

        t1_path = os.path.join(init_t1_path, t)
        t2_path = os.path.join(init_t2_path, t)

        # Image name
        n_idx1 = t1_path.rfind('/') + 1
        n_idx2 = t1_path.find('.', n_idx1)
        img_name = t1_path[n_idx1:n_idx2]

        if not os.path.exists(infer_dataset_path):
            print(f"Creating patches for {img_name}")
            make_patches(t1_path, t2_path, infer_dataset_path, args.patch_size, img_name, args.overlap_ratio)
            print(f"Finished creating patches for {img_name}")
            
            processing = 25
            status.update({"Process": processing, "Status": "running"})
            with open(os.path.join(status_path, "status.json"), "w") as s:
                json.dump(status, s, indent=4)
                
        if not os.path.exists(result_saved_path):
            with open(os.path.join(infer_dataset_path, "data_list.txt"), "r") as f:
                data_name_list = [data_name.strip() for data_name in f]
            args.data_name_list = data_name_list

            print(f"Performing inference on {img_name}")
            infer = Inferencer(args, img_name)
            infer.infer()
            print(f"Inference completed for {img_name}")
            
            processing = 50
            status.update({"Process": processing, "Status": "running"})
            with open(os.path.join(status_path, "status.json"), "w") as s:
                json.dump(status, s, indent=4)
                
        if not os.path.exists(os.path.join(result_saved_path, img_name+'.tif')):
            print(f"Reconstructing {img_name}")
            reconstruct_tiff_from_patches(infer_dataset_path, result_saved_path, img_name)
            
            processing = 75
            status.update({"Process": processing, "Status": "running"})
            with open(os.path.join(status_path, "status.json"), "w") as s:
                json.dump(status, s, indent=4)
                
        print(f"Vectorizing {img_name}")
        polygonize(result_saved_path, img_name)
        print("All processes completed.")
        
        processing = 100
        current_step += 1
        status.update({"Current_step": current_step, "Process": processing, "Status": "done"})
        with open(os.path.join(status_path, "status.json"), "w") as s:
            json.dump(status, s, indent=4)
                
    # except Exception as ex:
    #     print(ex)
    #     status.update({"Current_step": current_step, "Process": processing, "Status": "failed"})
    #     with open(os.path.join(status_path, "status.json"), "w") as s:
    #         json.dump(status, s, indent=4)

if __name__ == "__main__":
    main()
