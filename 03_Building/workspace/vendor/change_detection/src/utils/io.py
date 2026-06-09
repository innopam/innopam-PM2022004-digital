import pandas as pd
import geopandas as gpd
import os
import rasterio


def export_file(df, output_path, file_name):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if isinstance(df, gpd.GeoDataFrame):
        full_path = os.path.join(output_path, f"{file_name}.shp")
        df.to_file(full_path, driver='ESRI Shapefile', encoding='euc-kr')

    elif isinstance(df, pd.DataFrame):
        full_path = os.path.join(output_path, f"{file_name}.csv")
        df.to_csv(full_path, index=False, encoding='utf-8-sig')

    else:
        raise TypeError("error")


def import_shapefile(file_path, crs=5186):
    # 디렉토리일 경우, 안에서 .shp 파일 찾기
    if os.path.isdir(file_path):
        shp_files = [f for f in os.listdir(file_path) if f.endswith('.shp')]
        if not shp_files:
            raise FileNotFoundError(f"No .shp file found in directory: {file_path}")
        file_path = os.path.join(file_path, shp_files[0])  # 첫 번째 shp 파일로 설정

    gdf = gpd.read_file(file_path)
    if gdf.crs != f"epsg:{crs}":
        gdf = gdf.to_crs(epsg=crs)
    return gdf


def import_tif(tif_path):
    with rasterio.open(tif_path) as src:
        data = src.read()         # shape: (bands, height, width)
        crs = src.crs
        meta = src.meta

    return data, crs, meta
