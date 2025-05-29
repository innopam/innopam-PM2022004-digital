# 사용방법
```
cd ./innopam-PM2022004-digital/02_Road_CD

# 모델 다운로드
curl -u $USER:$PASSWORD -o ./workspace/model/model.pth https://nexus.innopam.kr/repository/models-repo/digital_land/02_Road_CD/model.pth

docker build --tag road_cd:1.0 .
export dataset_path=/workspace/input output_path=/workspace/output
docker-compose up
```

# Data
- input: T1, T2 시점의 이미지(쌍)
    - dataset_path의 경로에는 T1, T2 폴더가 존재해야
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
    │   ├── model.pth
    │   └── vssm_base_224_class2.yaml
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

4. model: 도로 변화탐지 모델로 구성
    - (.yml): 모델 파라미터 파일
    - (.pth): 가중치 파일

5. predict.py: 추론용 스크립트
    - --dataset_path: input 데이터 경로
    - --output_path: output 경로
    - --model_path: 모델 경로 (default=/workspace/model)
    - --patch_size: AI데이터 패치 크기 (default=512)
    - --overlap_ratio: 패치 영상 중복도 (default="min")