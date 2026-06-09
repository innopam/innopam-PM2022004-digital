import pandas as pd
import geopandas as gpd
from src.core.polygon_matching import polygon_matching_utils


def compare_gt_cd_remove_updated_unchanged(gt_prev, cd_prev):
    # 필요한 열만 추출
    gt_prev_sel = gt_prev[["poly1_idx", "gt_class", "geometry"]]
    cd_prev_sel = cd_prev[["poly1_idx", "cd_class"]]

    # poly1_idx 기준 join
    merged = pd.merge(gt_prev_sel, cd_prev_sel, on="poly1_idx", how="outer")

    merged = merged.rename(columns={"poly1_idx": "gt_idx"})
    merged['cd_idx'] = merged['gt_idx']

    # 매트릭스 열 초기화
    merged["gt_status"] = ""
    merged["pred_stat"] = ""

    # 조건에 따라 매트릭스 할당
    for idx, row in merged.iterrows():
        gt = row["gt_class"]
        cd = row["cd_class"]

        if pd.isna(gt) or pd.isna(cd):
            continue
        elif gt == cd:
            merged.at[idx, "gt_status"] = "TP"
            merged.at[idx, "pred_stat"] = "TP"
        else:
            merged.at[idx, "gt_status"] = "FN"
            merged.at[idx, "pred_stat"] = "FP"

    # GeoDataFrame으로 변환
    merged_gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=gt_prev.crs)

    return merged_gdf


def filter_new(gt_cur, cd_cur):
    # 신축 클래스 필터링
    gt_filtered = gt_cur[gt_cur['gt_class'] == '신축'].copy()
    cd_filtered = cd_cur[cd_cur['cd_class'] == '신축'].copy()

    # poly2_idx 열 이름 변경
    if 'poly2_idx' in gt_filtered.columns:
        gt_filtered = gt_filtered.rename(columns={'poly2_idx': 'gt_idx'})
    if 'poly2_idx' in cd_filtered.columns:
        cd_filtered = cd_filtered.rename(columns={'poly2_idx': 'cd_idx'})

    # 외부 조인 수행
    joined = polygon_matching_utils.outer_join(gt_filtered, cd_filtered, poly1_prefix="gt", poly2_prefix="cd")
    joined = joined[["gt_idx", "cd_idx"]]

    # 조건에 따른 geometry 땡겨오기
    geometries = []
    for _, row in joined.iterrows():
        gt_idx = row["gt_idx"]
        cd_idx = row["cd_idx"]

        if pd.isna(gt_idx) and not pd.isna(cd_idx):
            geom = cd_filtered.loc[cd_filtered["cd_idx"] == cd_idx, "geometry"]
        elif not pd.isna(gt_idx):
            geom = gt_filtered.loc[gt_filtered["gt_idx"] == gt_idx, "geometry"]
        else:
            geom = pd.Series([None])

        geometries.append(geom.values[0] if not geom.empty else None)

    joined["geometry"] = geometries
    joined_gdf = gpd.GeoDataFrame(joined, geometry="geometry", crs=gt_cur.crs)

    return joined_gdf


def compare_gt_cd_new(joined):
    joined = joined.copy()

    # 초기화
    joined['gt_class'] = pd.Series(dtype='object')
    joined['cd_class'] = pd.Series(dtype='object')
    joined['gt_status'] = pd.Series(dtype='object')
    joined['pred_stat'] = pd.Series(dtype='object')

    # gt_class 설정: gt_idx가 존재하면 '신축'
    joined.loc[joined['gt_idx'].notna(), 'gt_class'] = '신축'

    # cd_class 설정: cd_idx가 존재하면 '신축'
    joined.loc[joined['cd_idx'].notna(), 'cd_class'] = '신축'

    # 1. gt_idx가 NaN → pred_stat = 'FP'
    joined.loc[joined['gt_idx'].isna() & joined['cd_idx'].notna(), 'pred_stat'] = 'FP'

    # 2. cd_idx가 NaN → gt_status = 'FN'
    joined.loc[joined['cd_idx'].isna() & joined['gt_idx'].notna(), 'gt_status'] = 'FN'

    # 3. 둘 다 존재 → TP
    cond_tp = joined['gt_idx'].notna() & joined['cd_idx'].notna()
    joined.loc[cond_tp, ['gt_status', 'pred_stat']] = 'TP'

    return joined


def evaluate_bd(dmap, seg, bd_threshold):
    # dmap 기반 계산 (Recall 기준)
    dmap_tp = (dmap['bd_status'] == 'TP').sum()
    dmap_fn = (dmap['bd_status'] == 'FN').sum()
    gt_total = dmap_tp + dmap_fn
    recall = dmap_tp / gt_total if gt_total > 0 else 0

    # seg 기반 계산 (Precision 기준)
    seg_tp = (seg['bd_status'] == 'TP').sum()
    seg_fp = (seg['bd_status'] == 'FP').sum()
    pred_total = seg_tp + seg_fp
    precision = seg_tp / pred_total if pred_total > 0 else 0

    # F1-score 계산
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    # 결과 DataFrame
    result = pd.DataFrame([{
        "GT 수": gt_total,
        "Pred 수": pred_total,
        "TP": dmap_tp,
        "FN": dmap_fn,
        "FP": seg_fp,
        "재현율": round(recall, 3),
        "정밀도": round(precision, 3),
        "F1-score": round(f1_score, 3),
        "Threshold": bd_threshold
    }])

    return result