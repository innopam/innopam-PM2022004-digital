# 사용방법

```bash
cd ./innopam-PM2022004-digital/04_Building_CD

# 가중치 파일 배치 (별도 전달받은 실제 파일로 교체)
# 레포에는 같은 이름의 빈 더미 파일이 들어 있으니 덮어쓰기만 하면 됨.
#   - workspace/model/best.pth
#   - workspace/model/dinov3-vitl16-pretrain-lvd1689m/model.safetensors
# config.json, preprocessor_config.json 은 레포에 포함됨.

# (선택) 실제 가중치 교체 후 git이 변경사항으로 인식하지 않게 처리
git update-index --skip-worktree workspace/model/best.pth
git update-index --skip-worktree workspace/model/dinov3-vitl16-pretrain-lvd1689m/model.safetensors

# Docker image 생성
docker build -t building_cd:latest .

# input, output, model 경로 지정
export dataset_path=/workspace/input output_path=/workspace/output model_path=/workspace/model

# 추론 실행
docker-compose up
```

# Data

- input: T1, T2 시점의 이미지 쌍
    - `dataset_path` 경로에는 `T1`, `T2` 폴더가 존재해야 함
    - `T1`, `T2` 내부의 `.tif` 파일 이름과 크기가 동일해야 함
- output
    - 추론 경과: `status.json`
    - 최종 벡터 결과: `<이미지_이름>.json`
    - 내부 검증용 래스터: `results/<이미지_이름>/<이미지_이름>.tif`, `_conf.tif`, `_prob_1.tif`, `_prob_2.tif`, `_prob_3.tif`

# 폴더 구조

```bash
.
├── docker-compose.yml
├── dockerfile
├── README.md
└── workspace
    ├── input
    │   ├── T1
    │   │   └── sample.tif
    │   └── T2
    │       └── sample.tif
    ├── model
    │   ├── best.pth                                  # 더미 (실제 가중치로 교체)
    │   └── dinov3-vitl16-pretrain-lvd1689m
    │       ├── config.json
    │       ├── preprocessor_config.json
    │       └── model.safetensors                     # 더미 (실제 backbone으로 교체)
    ├── output
    │   ├── sample.json
    │   ├── status.json
    │   └── results
    │       └── sample
    │           ├── sample.tif
    │           ├── sample_conf.tif
    │           ├── sample_valid.tif
    │           ├── sample_prob_1.tif
    │           ├── sample_prob_2.tif
    │           ├── sample_prob_3.tif
    │           └── sample.json
    └── predict.py
```

# 모델

- `best.pth`: Building Phase 3 semantic change detector
    - DINOv3 ViT-L/16 + UPerNet
    - 최신 기준 실험: `phase3_update14_phase2init_lvd_lora_last4_rankaux_v1`
    - 지정 로그 기준 best checkpoint: epoch 17, `val_macro_change_f1=0.7167784635997279`
- `dinov3-vitl16-pretrain-lvd1689m/`: DINOv3 backbone (HuggingFace 형식)

# 추론 방식

- 기존 Mamba/Ray/PNG 패치 방식 대신 rasterio window 기반 직접 추론을 사용
- 기본 window size: `1024 x 1024`
- 기본 overlap: `25%` (`256px`)
- 중간 PNG 패치를 만들지 않으므로 디스크 I/O와 재구성 비용을 줄임

# predict.py 인자

- `--dataset_path`: input 데이터 경로 (default=`/workspace/input/`)
- `--output_path`: output 경로 (default=`/workspace/output/`)
- `--model_path`: 모델 경로 (default=`/workspace/model/`)
- `--patch_size`: 추론 window 크기 (default=`1024`)
- `--overlap_ratio`: window 중복도, `min` 또는 0~100 퍼센트 문자열 (default=`25`)
- `--batch_size`: GPU 추론 배치 크기 (default=`4`)
- `--confidence_threshold`: softmax confidence 하한 (default=`0.7`)
- `--min_component_pixels`: 클래스별 connected component 최소 픽셀 수 (default=`200`)
- `--min_area_m2`: GeoJSON 폴리곤 최소 면적 m² (default=`20`)
- `--simplify_tolerance`: 폴리곤 단순화 tolerance, m 단위 (default=`0.2`)

# 출력 포맷

- 클래스 3종
    - `CLS_ID=1`, `CLS_NAME="신축"`
    - `CLS_ID=2`, `CLS_NAME="소멸"`
    - `CLS_ID=3`, `CLS_NAME="갱신"`
- 출력 GeoJSON Feature properties
    - `CLS_ID`, `CLS_NAME`: 클래스 정보
    - `CONF`: 폴리곤 내 해당 클래스 평균 확률 (0~100)
    - `AREA`: 폴리곤 면적 (CRS 단위 제곱)
    - `ID`: 정렬된 일련번호
