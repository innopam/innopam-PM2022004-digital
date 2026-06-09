import pandas as pd


def report_class_10(poly1, poly2):
    # 1. 필요한 열만 추출하고 concat
    df = pd.concat([
        poly1[["class_10", "geometry"]],
        poly2[["class_10", "geometry"]]
    ], ignore_index=True)

    # 2. 면적 계산
    df["area"] = df["geometry"].area

    # 3. 그룹화
    grouped = df.groupby("class_10").agg(
        count=("geometry", "count"),
        area=("area", "sum")
    ).reset_index()

    # 4. 비율 계산
    total_count = grouped["count"].sum()
    total_area = grouped["area"].sum()

    grouped["count_percent"] = (grouped["count"] / total_count * 100).round(2)
    grouped["area_percent"] = (grouped["area"] / total_area * 100).round(2)

    # 5. 열 순서 정리
    final_df = grouped[["class_10", "count", "count_percent", "area", "area_percent"]]

    return final_df


def report_bd(poly1, poly2):
    # 1. 필요한 열만 추출하고 concat
    df = pd.concat([
        poly1[["bd_class", "geometry"]],
        poly2[["bd_class", "geometry"]]
    ], ignore_index=True)

    # 2. 면적 계산
    df["area"] = df["geometry"].area

    # 3. 그룹화
    grouped = df.groupby("bd_class").agg(
        count=("geometry", "count"),
        area=("area", "sum")
    ).reset_index()

    # 4. 비율 계산
    total_count = grouped["count"].sum()
    total_area = grouped["area"].sum()

    grouped["count_percent"] = (grouped["count"] / total_count * 100).round(2)
    grouped["area_percent"] = (grouped["area"] / total_area * 100).round(2)

    # 5. 열 순서 정리
    final_df = grouped[["bd_class", "count", "count_percent", "area", "area_percent"]]

    return final_df


def analysis_pipeline(poly1, poly2):
    report = report_class_10(poly1, poly2)
    return report
