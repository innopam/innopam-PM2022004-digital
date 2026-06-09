import argparse
import pandas as pd
from src.utils import io
from src.utils import evaluation_utils
from src.core.polygon_matching import polygon_matching_utils
from src.common.path_loader import load_building_paths


def decide_confusion_matrix(gt_prev, gt_cur, cd_prev, cd_cur):
    # 소멸, 갱신, 변화 없음 판단
    removed_updated_unchanged = evaluation_utils.compare_gt_cd_remove_updated_unchanged(gt_prev, cd_prev)
    new_temp = evaluation_utils.filter_new(gt_cur, cd_cur)
    new = evaluation_utils.compare_gt_cd_new(new_temp)
    confusion_matrix = pd.concat([removed_updated_unchanged, new], ignore_index=True)
    confusion_matrix = confusion_matrix.sort_values(by='gt_idx', ascending=True).reset_index(drop=True)
    cd_prev, cd_cur = polygon_matching_utils.confusion_matrix_to_cd(cd_prev, cd_cur, confusion_matrix)
    return confusion_matrix, cd_prev, cd_cur


def evaluate_cd(confusion_matrix):
    confusion_matrix = confusion_matrix.copy()

    # NaN 제거 (gt_class 또는 cd_class 모두 비어있는 행 제거)
    confusion_matrix = confusion_matrix.dropna(subset=["gt_class", "cd_class"], how="all")

    # 고유 클래스 목록
    all_classes = sorted(set(confusion_matrix["gt_class"].dropna().unique())
                         | set(confusion_matrix["cd_class"].dropna().unique()))

    results = []

    for cls in all_classes:
        # GT 기준
        gt_group = confusion_matrix[confusion_matrix["gt_class"] == cls]
        gt_tp = (gt_group["gt_status"] == "TP").sum()
        gt_fn = (gt_group["gt_status"] == "FN").sum()
        gt_total = gt_tp + gt_fn
        recall = round(gt_tp / gt_total, 3) if gt_total > 0 else 0.0

        # CD 기준
        cd_group = confusion_matrix[confusion_matrix["cd_class"] == cls]
        cd_tp = (cd_group["pred_stat"] == "TP").sum()
        cd_fp = (cd_group["pred_stat"] == "FP").sum()
        cd_total = cd_tp + cd_fp
        precision = round(cd_tp / cd_total, 3) if cd_total > 0 else 0.0

        results.append({
            "클래스": cls,
            "GT 수": gt_total,
            "Pred 수": cd_total,
            "TP": gt_tp,
            "FN": gt_fn,
            "FP": cd_fp,
            "재현율": recall,
            "정밀도": precision
        })

    df = pd.DataFrame(results)
    return df


def cd_evaluate_pipeline(gt_prev_path, gt_cur_path, cd_prev_path, cd_cur_path, output_path):
    gt_prev = io.import_shapefile(gt_prev_path, crs=5186)
    gt_cur = io.import_shapefile(gt_cur_path, crs=5186)
    cd_prev = io.import_shapefile(cd_prev_path, crs=5186)
    cd_cur = io.import_shapefile(cd_cur_path, crs=5186)
    confusion_matrix, cd_prev, cd_cur = decide_confusion_matrix(gt_prev, gt_cur, cd_prev, cd_cur)
    cd_prev = polygon_matching_utils.reorder_columns_after_cut_link(cd_prev)
    cd_cur = polygon_matching_utils.reorder_columns_after_cut_link(cd_cur)
    cd_evaluate_report = evaluate_cd(confusion_matrix)
    io.export_file(cd_prev, cd_prev_path, 'dmap')
    io.export_file(cd_cur, cd_cur_path, 'seg')
    io.export_file(confusion_matrix, output_path, 'cd_evaluate_result')
    io.export_file(cd_evaluate_report, output_path, 'cd_evaluate_result')


def main():
    parser = argparse.ArgumentParser(description="건물 변화 탐지 성능 평가")
    parser.add_argument("--region", type=str, required=True, help="지역 이름 (예: gangseo)")
    parser.add_argument("--year", type=str, default=2022, help="기준 연도")
    parser.add_argument("--previous_year", type=str, default=2020, help="이전 연도")

    args = parser.parse_args()

    paths = load_building_paths(args.region, args.year, args.previous_year)

    cd_evaluate_pipeline(
        gt_prev_path=paths["GT_of_building_change_detection_prev"],
        gt_cur_path=paths["GT_of_building_change_detection_cur"],
        cd_prev_path=paths["building_change_detection_result_prev"],
        cd_cur_path=paths["building_change_detection_result_cur"],
        output_path=paths["evaluation_of_building_change_detection"]
    )


if __name__ == "__main__":
    main()
    print("end")
