# 03_Building: TIF + DXF 건물 오류 탐지

1개 정사영상 TIF와 1개 이상의 DXF 도엽을 입력받아 도엽 단위 건물 오류를 산출합니다.

## 입력 구조

```text
workspace/
├── input/
│   ├── T1/        # DXF 파일들
│   └── T2/        # TIF 파일 1개
├── model/
│   ├── best.pth
│   └── dinov3-vitl16-pretrain-lvd1689m/
├── output/
└── main.py
```

배포 폴더에는 파일 배치 확인을 위한 placeholder가 들어 있습니다. 실제 실행 전에는 아래 이름을 유지한 채 운영 데이터/모델로 교체하세요.

- `workspace/input/T1/*.dxf`
- `workspace/input/T2/2025_Asan_index_9_12cm_cog.tif`
- `workspace/model/best.pth`
- `workspace/model/dinov3-vitl16-pretrain-lvd1689m/`

기본 규칙:

- `T1`: DXF 기반 건물 폴리곤
- `T2`: TIF 기반 건물 segmentation 결과
- 좌표계: `EPSG:5186`
- DXF 건물 카테고리: `기성건물 + 수정도화`

## 출력

도엽별로 다음 파일이 생성됩니다.

```text
workspace/output/
└── <sheet_id>/
    ├── <sheet_id>_errors.shp
    ├── <sheet_id>_errors.dxf
    └── <sheet_id>_summary.csv
```

오류 클래스:

- TIF에만 있음: `누락 오류`
- DXF에만 있음: `초과 오류`
- 양쪽에 있으나 형상이 다름: `묘사 오류`
- `변화없음`은 최종 오류 폴리곤으로 출력하지 않습니다.
- 최종 결과는 기본적으로 `AREA >= 50m²` 오류 폴리곤만 남깁니다.

처리 중에는 `status.json`, `intermediate/`, `reports/`가 임시 생성됩니다. 성공 종료 후 기본적으로 삭제하며, 디버깅을 위해 남기려면 `--keep_status`, `--keep_intermediate`를 사용합니다.

TIF는 전체 파일을 통째로 추론하지 않습니다. DXF 건물 영역과 TIF valid footprint의 교차 영역을 먼저 만들고, 해당 처리영역의 raster window만 원본 TIF에서 읽어 segmentation을 수행합니다.

## 실행

```bash
cd /home/hero/change_detection/innopam-PM2022004-digital/03_Building

export dataset_path=/workspace/input
export output_path=/workspace/output
export model_path=/workspace/model

docker-compose up --build
```

기본 실행은 도엽을 3개씩 병렬 처리합니다. GPU 메모리가 부족하면 `sheet_workers=1` 또는 `sheet_workers=2`로 낮추세요.

```bash
sheet_workers=1 docker-compose up
```

## 직접 실행 인자

```bash
python /workspace/main.py \
  --input /workspace/input \
  --output /workspace/output \
  --model /workspace/model \
  --result_min_area_m2 50 \
  --sheet_workers 3 \
  --overwrite
```

주요 옵션:

- `--tif_path`: TIF 직접 지정
- `--dxf_input`: DXF 파일, DXF 폴더, 쉼표 구분 리스트, 또는 txt 리스트
- `--dxf_workers`: DXF 병렬 변환 worker 수
- `--patch_size`: segmentation tile 크기
- `--overlap`: segmentation tile overlap
- `--batch_size`: GPU inference batch size
- `--seg_threshold`: 건물 mask threshold
- `--min_area_m2`: TIF segmentation polygon 최소 면적
- `--result_min_area_m2`: 최종 오류 폴리곤 최소 면적. 기본값 `50`
- `--cut_threshold`: polygon matching graph cut threshold
- `--cd_threshold`: 형상 변화 판정 IoU threshold
- `--sheet_workers`: 도엽 단위 병렬 처리 수. 기본값 `3`
- `--keep_intermediate`: 성공 후 중간 산출물과 report 보존
- `--keep_status`: 성공 후 `status.json` 보존
- `--overwrite`: 기존 출력 폴더를 지우고 재실행

## 로컬 테스트 기준

배포 폴더인 `03_Building`은 compact 실행 포맷 기준입니다. 로컬 검증은 다음 경로에서 진행합니다.

```text
/home/hero/change_detection/urban_cd_v1/building/tif_dxf_cd
```

테스트 데이터:

- TIF: `2025_Asan_index_9_12cm_cog.tif`
- DXF: `/home/hero/change_detection/urban_cd_v1/building/data/dxf/*.dxf`
- model: `building_seg` Phase2 `best.pth`

symlink target이 보이도록 `docker-compose.yml`은 `/home/hero/change_detection`와 `/home/user/data`를 함께 mount합니다.
