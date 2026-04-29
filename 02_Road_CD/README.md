# 사용방법
```
cd ./innopam-PM2022004-digital/02_Road_CD

# 가중치 파일 배치 (별도 전달받은 실제 파일로 교체)
# 레포에는 같은 이름의 빈 더미 파일이 들어 있으니 덮어쓰기만 하면 됨.
#   - workspace/model/best.pth                                     (학습 가중치, ~1.3GB)
#   - workspace/model/dinov3-vitl16-pretrain-lvd1689m/model.safetensors  (DINOv3 backbone, ~1.2GB)
# config.json, preprocessor_config.json 은 레포에 정상 포함되어 있어 별도 전달 불필요.

# (선택) 실제 가중치 교체 후 git이 변경사항으로 인식하지 않게 처리
git update-index --skip-worktree workspace/model/best.pth
git update-index --skip-worktree workspace/model/dinov3-vitl16-pretrain-lvd1689m/model.safetensors

# 샘플 이미지 다운로드: T1, T2
curl -u $USER:$PASSWORD -o ./workspace/input/T1/367011758.tif https://nexus.innopam.kr/repository/models-repo/digital_land/02_Road_CD/input/T1/367011758.tif
curl -u $USER:$PASSWORD -o ./workspace/input/T2/367011758.tif https://nexus.innopam.kr/repository/models-repo/digital_land/02_Road_CD/input/T2/367011758.tif

# Docker image 생성
docker build --tag road_cd:1.0 .

# input, output 경로 지정
export dataset_path=/workspace/input output_path=/workspace/output

# 추론 실행
docker-compose up
```

# Data
- input: T1, T2 시점의 이미지(쌍)
    - dataset_path의 경로에는 T1, T2 폴더가 존재해야 함
    - T1, T2 내부의 모든 파일에 대해 추론 실시
    - T1, T2 폴더의 파일 이름, 크기가 동일해야 추론이 가능함
- output
    - 추론 경과 (status.json)
    - 추론 결과물 (이미지_이름.json)

# 폴더 구조
```
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
    │       └── model.safetensors                     # 더미 (실제 가중치로 교체)
    ├── output
    │   ├── sample.json
    │   └── status.json
    └── predict.py
```

# 상세 설명
1. workspace
    - host(./innopam-PM2022004-digital/02_Road_CD/workspace) <-> container(/workspace) volume bind

2. input: 추론 이미지 폴더
    - T1: Before 이미지
    - T2: After 이미지

3. output: 추론 결과물 폴더

4. model: 도로 변화탐지 모델 가중치 폴더
    - `best.pth`: 학습된 가중치 (urban_cd_v1 / DINOv3 ViT-L/16 + UPerNet, Phase 3 class-2)
    - `dinov3-vitl16-pretrain-lvd1689m/`: DINOv3 backbone (HuggingFace 형식)
        - `config.json`, `preprocessor_config.json` 은 레포에 포함됨
        - `model.safetensors` 만 별도 전달 후 교체

5. predict.py: 추론용 스크립트
    - `--dataset_path`: input 데이터 경로 (default=`/workspace/input/`)
    - `--output_path`: output 경로 (default=`/workspace/output/`)
    - `--model_path`: 모델 경로 (default=`/workspace/model/`)
    - `--patch_size`: 추론 패치 크기 (default=`1024`)
    - `--overlap_ratio`: 패치 영상 중복도, `min` 또는 0~100 퍼센트 문자열 (default=`25`)
    - `--batch_size`: GPU 추론 배치 크기 (default=`4`)
    - `--confidence_threshold`: softmax confidence 하한 (default=`0.45`)
    - `--min_area_m2`: 폴리곤 최소 면적 m² (default=`30`)
    - `--simplify_tolerance`: 폴리곤 단순화 tolerance, m 단위 (default=`0.8`)

# 후처리 (v2 centerline + buffer)
- 추론 결과 폴리곤은 [postprocess_road_display_v2.py](https://github.com/innopam/urban_cd_v1/blob/main/road_cd/scripts/postprocess_road_display_v2.py) 와 동일한 로직으로 후처리됨
- 동작
    - **선형 객체** (elongation ≥ 1.9, length ≥ 10m, width ≤ 32m): skeletonize → centerline 추출 → 추정 폭(5~14m, scale 1.18, padding 0.8m)으로 buffer 재구성
    - **비선형 객체**: smooth(1.0m) + simplify(`--simplify_tolerance`)
    - **Coverage fallback**: centerline buffer가 원본 폴리곤의 85% 미만을 커버할 경우, 큰/넓은 객체 누락 방지를 위해 smooth-only로 회귀

# 모델 / 출력 포맷
- 클래스 2종
    - `CLS_ID=1`, `CLS_NAME="신설/확장"`
    - `CLS_ID=2`, `CLS_NAME="철거/축소"`
- 출력 GeoJSON Feature properties
    - `CLS_ID`, `CLS_NAME`: 클래스 정보
    - `CONF`: 폴리곤 내 평균 confidence (0~100)
    - `AREA`: 후처리된 폴리곤 면적 (CRS 단위 제곱)
    - `ID`: 정렬된 일련번호
