## 1. 사용방법
```bash
cd ./innopam-PM2022004-digital/02_Road_CD
docker build -t opencd_pytorch:latest .
export img_1=/workspace/sample_data/A img_2=/workspace/sample_data/B output_path=/workspace/out/out.gpkg && docker-compose up
```
- docker-compose up 전, input, output 경로(container 내부 경로)를 환경변수로 지정해야 함
    - img_1: input(Before 이미지) 폴더 경로
    - img_2: input(After 이미지) 폴더 경로
    - output_path: output(.gpkg) 저장 경로

## 2. 데이터 처리 과정
1. 데이터 Tiling(추가예정)
2. 변화탐지 추론(.tif)
3. 후처리(.tif)
4. 결과 merge(.tif)
5. polygonize(.gpkg)
6. confidence score 계산(.gpkg)

## 3. input, output
- input
    - Before, After 두 시기의 항공영상(폴더)
    - A, B 폴더의 파일은 동일한 이름의 파일끼리 한 쌍을 이룸
- output
    - out.gpkg, out.json
    - 좌표는 EPSG:5186

## 4. directory 구조
```bash
.
├── docker-compose.yml
├── dockerfile
├── READMD.md
└── workspace
    ├── model
    │   ├── ban_vit-l14-georsclip_iter_8000.pth
    │   └── ban_vit-l14-georsclip.py
    ├── predict.py
    ├── sample_data
    │   ├── A
    │   ├── B
    └── out
        ├── out.gpkg
        └── out.json
```
- workspace
    - model, sample_data, predict.py -> container 생성시 /workspace 경로에 bind 되어야함
        - model: config(.py), 가중치(.pth)로 구성
        - sample_data: A(Before 이미지), B(After 이미지)로 구성
        - predict.py: 추론 + 후처리 코드
            - 매개변수: default값 지정 되어있음
                - --config: config(.py) 파일 경로
                - --pth: 가중치(.pth) 파일 경로
                - --img_1: A(Before 이미지) 폴더 경로
                - --img_2: B(After 이미지) 폴더 경로
                - --output_path: 결과물 생성파일 경로(.gpkg), 경로 없을시 폴더 생성함

    - out: 결과물 생성 예시. container 내부의 /workspace가 아닌 다른 폴더로 지정 가능
- docker-compose.yml
    - gpu, cpu 모두 추론가능
    - cpu 추론시 deploy 부분 주석처리 필요
    - 30장 추론시 예상 소요시간: gpu(62초), cpu(336초)