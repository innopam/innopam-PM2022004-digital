import argparse
from src.utils import io
from src.utils import analysis_utils
from src.core.polygon_matching import polygon_matching_utils, polygon_matching_algorithm
from src.common.path_loader import load_building_paths


def assign_class(poly, threshold):
    poly = polygon_matching_utils.assign_cd_class(poly, threshold, "cd")
    poly = polygon_matching_utils.assign_class_10(poly, "cd")

    return poly


def cd_pipeline(dmap_path, seg_path, dmap_output_path, seg_output_path, anl_output_path, cut_threshold, cd_threshold):
    _, dmap, seg = (polygon_matching_algorithm.algorithm_pipeline
                          (dmap_path, seg_path, anl_output_path, cut_threshold))
    dmap = assign_class(dmap, cd_threshold)
    seg = assign_class(seg, cd_threshold)
    dmap = polygon_matching_utils.bd_result_attach(dmap, seg)
    dmap = dmap.rename(columns={"Relation": "rel_cd"})
    seg = seg.rename(columns={"Relation": "rel_cd"})
    report = analysis_utils.analysis_pipeline(dmap, seg)
    io.export_file(dmap, dmap_output_path, 'dmap')
    io.export_file(seg, seg_output_path, 'seg')
    io.export_file(report, anl_output_path, 'analysis_result')


def main():
    parser = argparse.ArgumentParser(description="건물 변화 탐지 프로세스")
    parser.add_argument("--region", type=str, required=True, help="지역 이름 (예: gangseo)")
    parser.add_argument("--year", type=str, default=2022, help="기준 연도")
    parser.add_argument("--previous_year", type=str, default=2020, help="이전 연도")
    parser.add_argument("--cut_threshold", type=float, default=0.05, help="그래프 컷 임계값")
    parser.add_argument("--cd_threshold", type=float, default=0.7, help="변화 판별 임계값")

    args = parser.parse_args()

    paths = load_building_paths(args.region, args.year, args.previous_year)

    cd_pipeline(
        dmap_path=paths["GT_of_building_change_detection_prev"],
        seg_path=paths["evaluation_of_building_detection_predict"],
        dmap_output_path=paths["building_change_detection_result_prev"],
        seg_output_path=paths["building_change_detection_result_cur"],
        anl_output_path=paths["building_change_detection_result_anl"],
        cut_threshold=args.cut_threshold,
        cd_threshold=args.cd_threshold
    )


if __name__ == "__main__":
    main()
    print("end")
