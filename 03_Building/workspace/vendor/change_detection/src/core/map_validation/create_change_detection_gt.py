import argparse
from src.utils import io, analysis_utils
from src.core.polygon_matching import polygon_matching_utils, polygon_matching_algorithm
from src.common.path_loader import load_building_paths


def assign_class(poly, threshold):
    poly = polygon_matching_utils.assign_cd_class(poly, threshold, "gt")
    poly = polygon_matching_utils.assign_class_10(poly, "gt")

    return poly


def cd_pipeline(dmap1_path, dmap2_path, prev_output_path, cur_output_path, anl_output_path, cut_threshold, cd_threshold):
    _, dmap1, dmap2 = (polygon_matching_algorithm.algorithm_pipeline
                             (dmap1_path, dmap2_path, anl_output_path, cut_threshold))
    dmap1 = assign_class(dmap1, cd_threshold)
    dmap2 = assign_class(dmap2, cd_threshold)
    report = analysis_utils.analysis_pipeline(dmap1, dmap2)
    cols_to_drop = [
        'iou_nn', 'ol_pl1_nn', 'ol_pl2_nn',
        'iou_1n', 'ol_pl1_1n', 'ol_pl2_1n',
        'iou_n1', 'ol_pl1_n1', 'ol_pl2_n1',
        'iou_11', 'ol_pl1_11', 'ol_pl2_11',
        'comp_idx', 'poly1_set', 'poly2_set', 'cut_link', 'Relation'
    ]
    dmap1 = dmap1.drop(columns=[col for col in cols_to_drop if col in dmap1.columns])
    dmap2 = dmap2.drop(columns=[col for col in cols_to_drop if col in dmap2.columns])
    io.export_file(dmap1, prev_output_path, 'prev_dmap_add_error')
    io.export_file(dmap2, cur_output_path, 'cur_dmap_add_error')
    io.export_file(report, anl_output_path, 'analysis_result')


def main():
    parser = argparse.ArgumentParser(description="건물 변화 탐지 Ground Truth 생성")
    parser.add_argument("--region", type=str, required=True, help="지역 이름 (예: gangseo)")
    parser.add_argument("--year", type=str, default=2022, help="기준 연도")
    parser.add_argument("--previous_year", type=str, default=2020, help="이전 연도")
    parser.add_argument("--cut_threshold", type=float, default=0.05, help="그래프 컷 임계값")
    parser.add_argument("--cd_threshold", type=float, default=0.95, help="변화 판별 임계값")

    args = parser.parse_args()

    paths = load_building_paths(args.region, args.year, args.previous_year)

    cd_pipeline(
        dmap1_path=paths["previous_building_digital_map"],
        dmap2_path=paths["GT_of_building_detection"],
        prev_output_path=paths["GT_of_building_change_detection_prev"],
        cur_output_path=paths["GT_of_building_change_detection_cur"],
        anl_output_path=paths["GT_of_building_change_detection_anl"],
        cut_threshold=args.cut_threshold,
        cd_threshold=args.cd_threshold
    )


if __name__ == "__main__":
    main()
    print("end")
