import argparse
from src.utils import io
from src.utils import analysis_utils
from src.core.polygon_matching import polygon_matching_utils, polygon_matching_algorithm
from src.utils import evaluation_utils
from src.common.path_loader import load_building_paths


def assign_class(dmap, seg, bd_threshold):
    dmap = polygon_matching_utils.assign_bd_class_gt(dmap, bd_threshold)
    seg = polygon_matching_utils.assign_bd_class_seg(seg, bd_threshold)
    dmap = polygon_matching_utils.assign_class_bd_10(dmap, prefix="bd")
    seg = polygon_matching_utils.assign_class_bd_10(seg, prefix="bd")
    return dmap, seg


def evaluate_bd_pipeline(dmap_path, seg_path, dmap_output_path, seg_output_path, anl_output_path, eval_output_path, cut_threshold, bd_threshold):
    _, dmap, seg = (polygon_matching_algorithm.algorithm_pipeline
                             (dmap_path, seg_path, anl_output_path, cut_threshold))
    dmap, seg = assign_class(dmap, seg, bd_threshold)
    cols_to_drop = [
        'iou_nn', 'ol_pl1_nn', 'ol_pl2_nn',
        'iou_1n', 'ol_pl1_1n', 'ol_pl2_1n',
        'iou_n1', 'ol_pl1_n1', 'ol_pl2_n1',
        'iou_11', 'ol_pl1_11', 'ol_pl2_11',
        'comp_idx', 'poly1_set', 'poly2_set', 'cut_link'
    ]

    result = evaluation_utils.evaluate_bd(dmap, seg, bd_threshold)
    anl_result = analysis_utils.report_bd(dmap, seg)
    dmap = dmap.drop(columns=[col for col in cols_to_drop if col in dmap.columns])
    seg = seg.drop(columns=[col for col in cols_to_drop if col in seg.columns])
    dmap = dmap.rename(columns={"Relation": "rel_bd"})
    seg = seg.rename(columns={"Relation": "rel_bd"})
    io.export_file(dmap, dmap_output_path, 'gt')
    io.export_file(seg, seg_output_path, 'predict')
    io.export_file(anl_result, anl_output_path, 'bd_anl_result')
    io.export_file(result, eval_output_path, 'bd_evaluate_result')


def main():
    parser = argparse.ArgumentParser(description="건물 변화 탐지 프로세스")
    parser.add_argument("--region", type=str, required=True, help="지역 이름 (예: gangseo)")
    parser.add_argument("--year", type=str, default=2022, help="기준 연도")
    parser.add_argument("--previous_year", type=str, default=2020, help="이전 연도")
    parser.add_argument("--cut_threshold", type=float, default=0.05, help="그래프 컷 임계값")
    parser.add_argument("--bd_threshold", type=float, default=0.6, help="탐지 판별 임계값")

    args = parser.parse_args()

    paths = load_building_paths(args.region, args.year, args.previous_year)

    evaluate_bd_pipeline(
        dmap_path=paths["GT_of_building_detection"],
        seg_path=paths["building_inference"],
        dmap_output_path=paths["evaluation_of_building_detection_gt"],
        seg_output_path=paths["evaluation_of_building_detection_predict"],
        anl_output_path=paths["evaluation_of_building_detection_anl"],
        eval_output_path=paths["evaluation_of_building_detection"],
        cut_threshold=args.cut_threshold,
        bd_threshold=args.bd_threshold
    )


if __name__ == "__main__":
    main()
    print("end")
