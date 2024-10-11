import os
from glob import glob
import shutil
import subprocess
import argparse
import json
from tqdm import tqdm
from time import time

import numpy as np
import rasterio as rio
from rasterio.mask import mask
import geopandas as gpd
import cv2
import torch

from opencd.apis import OpenCDInferencer

# 1. argparser 만들기
# 2. input 이미지 불러오기
# 3. input 이미지 가공하기(tiling)

# 4. model 불러오기
# 5. model로 추론하기
# 6. 추론결과(array)에 좌표 입히기
# 7. 추론결과 merge하기

# 8. 후처리 실시(모폴로지 연산)
# 9. polygon 화 후 저장


def make_args():
    """
    img_1(.tif): Before 이미지(도엽별 항공영상)가 담긴 폴더 경로
    img_2(.tif): After 이미지(도엽별 항공영상)가 담긴 폴더 경로
    output_path(.gpkg): 최종 결과물 경로
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_1", type=str, default="/workspace/sample_data/A")
    parser.add_argument("--img_2", type=str, default="/workspace/sample_data/B")
    parser.add_argument("--output_path", type=str, default="/workspace/out/out.gpkg")

    parser.add_argument(
        "--config", type=str, default="/workspace/model/ban_vit-l14-georsclip.py"
    )
    parser.add_argument(
        "--pth",
        type=str,
        default="/workspace/model/ban_vit-l14-georsclip_iter_8000.pth",
    )
    parser.add_argument("--px", type=int, default=512)

    return parser


def load_model(model_path, weight_path):
    model = OpenCDInferencer(
        model=model_path,
        weights=weight_path,
        # If not gpu, will operate with cpu
        # device="cuda:0",
        classes=["unchanged", "changed"],
        palette=[[0, 0, 0], [255, 255, 255]],
    )
    return model


def retile(input, output, px):
    subprocess.run(
        "gdal_retile.py -ps %s %s -of GTiff -ot Byte -targetDir %s %s"
        % (str(px), str(px), output, input),
        shell=True,
    )


def run_cd(model, image_set):
    """
    image_set: [[img1_before, img1_after], [img2_before, img2_after], ..]
    """
    # predict_array = model(image_set)["predictions"]
    predict_array = model(image_set, return_datasamples=True)

    return predict_array


def save_array_as_geotiff(array, reference_tif_path, output_tif_path):
    with rio.open(reference_tif_path) as src:
        meta = src.meta.copy()
        meta.update(
            {
                "driver": "GTiff",
                "dtype": "uint8",
                "count": 1,
            }
        )
    with rio.open(output_tif_path, "w", **meta) as dst:
        dst.write(array, 1)


def tif_merge(input_folder_path, dst_path):
    subprocess.run(
        "gdal_merge.py -o %s %s" % (dst_path, os.path.join(input_folder_path, "*.tif")),
        shell=True,
    )
    subprocess.run("gdal_edit.py -a_srs EPSG:5186 %s" % (dst_path), shell=True)
    return dst_path


def post_processing(input_tif_path, output_tif_path):
    # kernel 값, erode 횟수, dialate 횟수는 경험적으로 조정 필요
    # kernel = np.ones((5, 5), np.uint8)
    # array = cv2.erode(array, kernel, iterations=10)
    # array = cv2.dilate(array, kernel, iterations=20)
    # array = cv2.erode(array, kernel, iterations=10)
    # kernel: (3,3)으로
    # erode, dilate: 여러차례씩 시행(한번씩 시행 금지), 횟수는 10회 이하로
    array = rio.open(input_tif_path).read(1)
    print(array.shape)
    kernel = np.ones((3, 3), np.uint8)
    array = cv2.erode(array, kernel, iterations=5)
    array = cv2.dilate(array, kernel, iterations=10)
    array = cv2.erode(array, kernel, iterations=5)
    save_array_as_geotiff(
        array, reference_tif_path=input_tif_path, output_tif_path=output_tif_path
    )
    return output_tif_path


def polygonize(input_tif_path, output_gpkg_path):
    tmp = output_gpkg_path.replace(".gpkg", "_tmp.gpkg")
    subprocess.run(
        "gdal_polygonize.py -f GPKG %s %s merged CLS_ID" % (input_tif_path, tmp),
        shell=True,
    )

    subprocess.run(
        "ogr2ogr -f GPKG -where CLS_ID!=0 %s %s" % (output_gpkg_path, tmp),
        shell=True,
    )
    os.remove(tmp)


def ogr_merge(input_gpkg_folder_path, output_gpkg_path):
    tmp = output_gpkg_path.replace(".gpkg", "_tmp.gpkg")
    subprocess.run(
        "ogrmerge.py -f GPKG -single -o %s %s"
        % (tmp, os.path.join(input_gpkg_folder_path, "*.gpkg")),
        shell=True,
    )

    subprocess.run(
        "ogr2ogr %s %s -nln merged -nlt MULTIPOLYGON -dialect sqlite -sql "
        '"SELECT ST_Union(geom), CLS_ID FROM merged GROUP BY CLS_ID" '
        "-f GPKG -explodecollections -a_srs EPSG:5186" % (output_gpkg_path, tmp),
        # "-f GPKG -explodecollections" % (output_gpkg_path, tmp),
        shell=True,
    )
    os.remove(tmp)


def conf_score(conf_tif, gpkg):
    gdf = gpd.read_file(gpkg)
    src = rio.open(conf_tif)

    for i, value in gdf.iterrows():
        out_image, _ = mask(src, [value.geometry])
        conf = out_image.sum() / np.count_nonzero(out_image)
        gdf.loc[i, "CONF"] = round(conf, 2)

    gdf.to_file(gpkg, layer="merged")


if __name__ == "__main__":
    start = time()
    args = make_args().parse_args()
    config = args.config
    pth = args.pth
    folder_1 = args.img_1
    folder_2 = args.img_2
    px = args.px
    root_path = os.path.dirname(args.output_path)
    out_name = os.path.basename(args.output_path)

    # 결과물, 중간산출물 폴더 생성
    output = os.path.join(root_path, "output")
    output_conf = os.path.join(root_path, "output_conf")
    post_processed = os.path.join(root_path, "post_processed")
    output_gpkg = os.path.join(root_path, "output_gpkg")
    post_processed_gpkg = os.path.join(root_path, "post_processed_gpkg")
    tile = os.path.join(root_path, "tile")
    new_folders = [
        output,
        output_conf,
        post_processed,
        output_gpkg,
        post_processed_gpkg,
        tile,
    ]

    for folder in new_folders:
        os.makedirs(folder, exist_ok=True)

    # logging status: running
    log = {}
    log_path = os.path.join(root_path, out_name.split(".")[0] + ".json")
    log["status"] = "running"
    with open(log_path, "w") as logfile:
        json.dump(log, logfile)

    # 모델 생성
    model = load_model(config, pth)

    # 항공영상 도엽별 Retile
    before_li = sorted(os.listdir(folder_1))
    after_li = sorted(os.listdir(folder_2))

    # Before, After 의 도엽별 이미지의 이름이 동일해야함
    if before_li != after_li:
        log["status"] = "failed"
        with open(log_path, "w") as logfile:
            json.dump(log, logfile)
        raise Exception("Image sets of Before and After should be same.")

    else:
        os.makedirs(os.path.join(tile, "A"), exist_ok=True)
        os.makedirs(os.path.join(tile, "B"), exist_ok=True)
        for folder in [folder_1, folder_2]:
            for file in before_li:
                if folder == folder_1:
                    retile(os.path.join(folder, file), os.path.join(tile, "A"), px)
                else:
                    retile(os.path.join(folder, file), os.path.join(tile, "B"), px)

        # Tile 이미지 셋 리스트 생성
        dataset_li = []
        file_li = sorted(os.listdir(os.path.join(tile, "A")))
        for file in file_li:
            cd_set = [
                os.path.join(os.path.join(tile, "A"), file),
                os.path.join(os.path.join(tile, "B"), file),
            ]
            dataset_li.append(cd_set)

        # 이미지 셋트별로 변화탐지 추론실시(추론 -> 좌표넣기 -> 후처리)
        for dataset in tqdm(dataset_li):

            # pred, conf 계산
            array = run_cd(model, [dataset])
            pred_array = np.array(array.pred_sem_seg.data.cpu())[0]
            conf_array = np.array(torch.sigmoid(array.seg_logits.data.cpu()[1])) * 100

            output_tif_path = os.path.join(output, os.path.basename(dataset[0]))
            output_tif_conf_path = os.path.join(
                output_conf, os.path.basename(dataset[0])
            )

            # 좌표 입히기
            save_array_as_geotiff(pred_array, dataset[0], output_tif_path)
            save_array_as_geotiff(conf_array, dataset[0], output_tif_conf_path)

            # 후처리
            post_processed_tif_path = os.path.join(
                post_processed, os.path.basename(dataset[0])
            )
            post_processing(output_tif_path, post_processed_tif_path)

            # Vectorize
            output_gpkg_path = os.path.join(
                output_gpkg, os.path.basename(dataset[0]).replace(".tif", ".gpkg")
            )
            post_processed_gpkg_path = os.path.join(
                post_processed_gpkg,
                os.path.basename(dataset[0]).replace(".tif", ".gpkg"),
            )
            polygonize(output_tif_path, output_gpkg_path)
            polygonize(post_processed_tif_path, post_processed_gpkg_path)

        # 전체 추론결과 합치기
        tif_merge(output_conf, os.path.join(root_path, "conf.tif"))
        ogr_merge(output_gpkg, os.path.join(root_path, "merged.gpkg"))
        # ogr_merge(post_processed_gpkg, os.path.join(root_path, "post_processed_merged.gpkg"))
        ogr_merge(post_processed_gpkg, os.path.join(root_path, out_name))

        # Confidence Score 구하기
        conf_score(
            os.path.join(root_path, "conf.tif"), os.path.join(root_path, out_name)
        )

        # logging status: done
        log["status"] = "done"
        with open(log_path, "w") as logfile:
            json.dump(log, logfile)

        # Remove tmp folders
        for folder in new_folders:
            shutil.rmtree(folder)
        os.remove(os.path.join(root_path, "merged.gpkg"))
        os.remove(os.path.join(root_path, "conf.tif"))

        print(f"Total time elapsed: {time()-start}")
