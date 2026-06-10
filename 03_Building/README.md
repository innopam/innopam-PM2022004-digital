# 03_Building: TIF + DXF 건물 오류 탐지

1개 정사영상 TIF와 1개 이상의 DXF 도엽을 입력받아 도엽 단위 건물 오류를 산출합니다.

## 처리 기준

- 좌표계는 `EPSG:5186` 고정입니다.
- DXF 입력은 `T1`, TIF 입력은 `T2` 기준으로 처리합니다.
- TIF 기반 건물 폴리곤이 변화탐지 `T1`, DXF 기반 건물 폴리곤이 변화탐지 `T2`로 들어갑니다.
- TIF에만 있는 건물은 `누락 오류`로 출력합니다.
- DXF에만 있는 건물은 `초과 오류`로 출력합니다.
- 양쪽에 있으나 형상이 다른 건물은 `묘사 오류`로 출력합니다.
- `변화없음`은 최종 결과로 출력하지 않습니다.
- 최종 오류 폴리곤은 기본적으로 `AREA >= 50m²`만 남깁니다.

## 배포 폴더 구조

```text
workspace/
├── input/
│   ├── T1/
│   │   ├── 36701075.dxf
│   │   ├── 36701076.dxf
│   │   └── 36701077.dxf
│   └── T2/
│       └── 2025_Asan_index_9_12cm_cog.tif
├── model/
│   ├── best.pth
│   └── dinov3-vitl16-pretrain-lvd1689m/
│       ├── .gitattributes
│       ├── LICENSE.md
│       ├── README.md
│       ├── config.json
│       ├── preprocessor_config.json
│       └── model.safetensors
├── output/
└── main.py
```

현재 repository에는 배치 확인용 placeholder가 포함되어 있습니다. 운영 전에 입력 데이터와 무거운 모델 가중치 파일만 실제 파일로 교체합니다. DINOv3의 경량 설정 파일(`config.json`, `preprocessor_config.json` 등)은 실제 파일을 그대로 포함했습니다.

## 실제 데이터 교체 항목

### 1. DXF 입력

교체 위치:

```text
workspace/input/T1/*.dxf
```

규칙:

- 처리할 모든 도엽 DXF를 `workspace/input/T1/` 아래에 넣습니다.
- 파일명 stem이 도엽 ID로 사용됩니다. 예: `36701075.dxf` -> 결과 폴더 `output/36701075/`
- 폴더 입력 방식에서는 `T1` 아래의 모든 `*.dxf`가 처리됩니다.
- 일부 파일만 처리하려면 실행 시 `--dxf_input`으로 DXF 파일, DXF 폴더, 쉼표 구분 리스트, 또는 txt 리스트를 지정합니다.

### 2. TIF 입력

교체 위치:

```text
workspace/input/T2/2025_Asan_index_9_12cm_cog.tif
```

규칙:

- 기본 실행은 `workspace/input/T2/` 아래의 첫 번째 `.tif` 또는 `.tiff`를 사용합니다.
- 운영 TIF 파일명이 달라도 동작하지만, 배포 혼선을 줄이려면 placeholder와 같은 이름으로 교체하는 것을 권장합니다.
- 다른 경로 또는 파일명을 직접 지정하려면 실행 시 `--tif_path`를 사용합니다.

### 3. 건물 segmentation checkpoint

교체 위치:

```text
workspace/model/best.pth
```

필수 내용:

- `building_seg` Phase2 최종 checkpoint 파일입니다.
- 현재 로컬 검증 기준 checkpoint는 Phase2 결과 디렉터리의 `best.pth`입니다.
- placeholder 텍스트 파일을 실제 `.pth` 바이너리 checkpoint로 교체해야 합니다.

### 4. DINOv3 backbone weight

교체 위치:

```text
workspace/model/dinov3-vitl16-pretrain-lvd1689m/model.safetensors
```

현재 포함된 경량 파일:

```text
dinov3-vitl16-pretrain-lvd1689m/
├── .gitattributes
├── LICENSE.md
├── README.md
├── config.json
└── preprocessor_config.json
```

중요:

- 위 경량 파일들은 원본 실제 파일이므로 그대로 commit합니다.
- `model.safetensors`만 무거운 실제 가중치 파일이어서 placeholder로 둡니다.
- 운영 전에는 `workspace/model/dinov3-vitl16-pretrain-lvd1689m/model.safetensors`를 실제 DINOv3 가중치 파일로 교체해야 합니다.
- 폴더 이름만 만들거나 `model.safetensors`를 빠뜨리면 `transformers`가 backbone을 로드할 수 없습니다.

## 실행

기본 실행:

```bash
cd /home/hero/change_detection/innopam-PM2022004-digital/03_Building
docker-compose up --build
```

기본값:

- 입력: `/workspace/input`
- 출력: `/workspace/output`
- 모델: `/workspace/model`
- 도엽 병렬 처리 수: `3`
- 최종 면적 필터: `50m²`
- 실행 시 기존 출력 삭제: `--overwrite`

GPU 메모리가 부족하면 도엽 병렬 처리 수를 낮춥니다.

```bash
sheet_workers=1 docker-compose up --build
```

추가 인자는 `extra_args`로 전달할 수 있습니다.

```bash
extra_args="--dxf_input /workspace/input/T1/36701075.dxf" docker-compose up --build
```

## 직접 실행

컨테이너 내부 또는 동일 환경에서 직접 실행할 때:

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

- `--tif_path`: 사용할 TIF 파일 직접 지정
- `--dxf_input`: DXF 파일, DXF 폴더, 쉼표 구분 리스트, 또는 txt 리스트 지정
- `--dxf_workers`: DXF 변환 병렬 worker 수
- `--sheet_workers`: 도엽 단위 병렬 처리 수. 기본값 `3`
- `--result_min_area_m2`: 최종 오류 폴리곤 최소 면적. 기본값 `50`
- `--min_area_m2`: TIF segmentation 폴리곤 최소 면적
- `--seg_threshold`: 건물 mask threshold
- `--cut_threshold`: polygon matching graph cut threshold
- `--cd_threshold`: 형상 변화 판정 IoU threshold
- `--processing_area_mode`: 도엽별 TIF 추론 처리영역 산정 방식. 기본값 `bbox`
- `--processing_bbox_buffer_m`: 처리영역 외곽 buffer. 기본값 `0`
- `--keep_intermediate`: 성공 후 중간 산출물과 report 보존
- `--keep_status`: 성공 후 `status.json` 보존
- `--also_geojson`: SHP/DXF/CSV 외 GeoJSON도 추가 출력
- `--overwrite`: 기존 출력 폴더 삭제 후 재실행

## 출력

최종 결과는 도엽 단위 폴더로 생성됩니다.

```text
workspace/output/
└── <sheet_id>/
    ├── <sheet_id>_errors.shp
    ├── <sheet_id>_errors.shx
    ├── <sheet_id>_errors.dbf
    ├── <sheet_id>_errors.prj
    ├── <sheet_id>_errors.cpg
    ├── <sheet_id>_errors.dxf
    └── <sheet_id>_summary.csv
```

repository에는 최근 로컬 검증 결과 예시로 `36701075`, `36701076`, `36701077` 3개 도엽 output이 포함되어 있습니다. 실제 실행 시 `--overwrite`가 적용되면 기존 예시 output은 삭제되고 새 결과로 대체됩니다.

처리 중에는 `status.json`, `intermediate/`, `reports/`가 임시 생성됩니다. 성공 종료 후 기본적으로 삭제됩니다.

## 성능 구조

전체 TIF를 통째로 추론하지 않습니다.

1. DXF를 건물 폴리곤으로 변환합니다.
2. DXF 건물 폴리곤 전체 BBox와 TIF valid footprint의 교차 영역을 처리 영역으로 산출합니다.
3. 원본 TIF에서 해당 처리 영역의 raster window만 읽어 segmentation을 수행합니다.
4. 도엽별로 TIF 기반 폴리곤과 DXF 기반 폴리곤을 비교합니다.

처리영역 기본값은 `bbox`입니다. `union`은 DXF 건물 폴리곤 내부만 추론/clip하므로 DXF에 없는 TIF 건물을 누락시킬 수 있어 기본 운영에는 사용하지 않습니다. 비교 진단이 필요할 때만 `--processing_area_mode union` 또는 `--processing_area_mode convex_hull`을 사용합니다.

도엽별 병렬 처리는 `--sheet_workers`로 조정합니다. 기본값은 `3`입니다.

## 배포 전 확인

실행 전 다음 항목을 확인하세요.

```bash
find workspace/input/T1 -name '*.dxf'
find workspace/input/T2 -name '*.tif' -o -name '*.tiff'
test -f workspace/model/best.pth
test -f workspace/model/dinov3-vitl16-pretrain-lvd1689m/config.json
test -f workspace/model/dinov3-vitl16-pretrain-lvd1689m/preprocessor_config.json
test -f workspace/model/dinov3-vitl16-pretrain-lvd1689m/model.safetensors
```

위 명령은 파일 존재 여부만 확인합니다. 현재 commit되는 `best.pth`, `model.safetensors`, 입력 TIF/DXF는 placeholder이므로 실제 추론 전에는 반드시 실제 파일로 교체해야 합니다. `config.json`, `preprocessor_config.json`, `LICENSE.md`, `README.md`, `.gitattributes`는 이미 실제 파일입니다.

## Git 주의사항

현재 repository에는 다음만 placeholder로 commit합니다.

- `workspace/input/T1/*.dxf`
- `workspace/input/T2/*.tif`
- `workspace/model/best.pth`
- `workspace/model/dinov3-vitl16-pretrain-lvd1689m/model.safetensors`

실제 TIF, DXF, `.pth`, `.safetensors`로 교체한 뒤에는 대용량 운영 데이터가 `git status`에 잡힐 수 있으므로 실제 데이터 교체분은 commit하지 마세요. DINOv3의 경량 설정 파일들은 실제 파일 그대로 commit합니다.

`workspace/output`은 기본적으로 ignore되지만, 예시 산출물인 `36701075`, `36701076`, `36701077` 폴더만 commit 가능하도록 예외 처리되어 있습니다.

## 로컬 테스트 기준

배포 폴더인 `03_Building`은 compact 실행 포맷입니다. 기능 검증과 샘플 실행은 다음 경로에서 진행합니다.

```text
/home/hero/change_detection/urban_cd_v1/building/tif_dxf_cd
```

최근 검증 조건:

- TIF: `/home/hero/change_detection/urban_cd_v1/building/data/tif/2025_Asan_index_9_12cm_cog.tif`
- DXF: `36701075.dxf`, `36701076.dxf`, `36701077.dxf`
- 출력: `/home/hero/change_detection/urban_cd_v1/building/tif_dxf_cd/workspace/output/<sheet_id>/`
