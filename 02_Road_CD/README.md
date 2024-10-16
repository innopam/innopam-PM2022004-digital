## 1. 사용방법
```bash
cd ./innopam-PM2022004-digital/02_Road_CD
docker build -t opencd_pytorch:latest .
export before_img_dir_path=/workspace/sample_data/A after_img_dir_path=/workspace/sample_data/B output_file_path=/workspace/out/out.gpkg \
&& docker-compose up
```
- docker-compose up 전, input, output 경로(container 내부 경로)를 환경변수로 지정해야 함
    - before_img_dir_path: input(Before 이미지) 폴더 경로
    - after_img_dir_path: input(After 이미지) 폴더 경로
    - output_file_path: output(.gpkg) 저장 경로

## 2. 데이터 처리 과정
1. 데이터 Tiling
2. 변화탐지 추론(.tif)
3. 후처리(.tif)
4. 결과 merge(.tif)
5. polygonize(.gpkg)
6. confidence score 계산(.gpkg)

## 3. input, output
- input
    - Before, After 두 시기의 도엽단위의 항공영상(폴더)
    - A, B 폴더의 파일은 동일한 이름의 파일끼리 한 쌍을 이룸
- output
    - out.gpkg, out.json
    - 좌표는 EPSG:5186
### 예시
- input: Before 이미지(2020년도 377051713도엽)

<img width="566" alt="스크린샷 2024-10-11 오후 3 26 52" src="https://github.com/user-attachments/assets/110e42da-d42e-441d-a968-8c8857b2d58e"></br>
- input: After 이미지(2022년도 377051713도엽)

<img width="563" alt="스크린샷 2024-10-11 오후 3 27 07" src="https://github.com/user-attachments/assets/2ea5e7bf-fdb0-49ee-a848-212479c5fd40"></br>
- output: Before 이미지(2022년도 377051713도엽)에 추론결과 OVERLAP

<img width="563" alt="스크린샷 2024-10-11 오후 3 27 25" src="https://github.com/user-attachments/assets/9ac12269-27d6-47e7-b405-263399ffb570">

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
            - --before_img_dir_path: A(Before 이미지) 폴더 경로
            - --after_img_dir_path: B(After 이미지) 폴더 경로
            - --output_file_path: 결과물 생성파일 경로(.gpkg), 경로 없을시 폴더 생성함
            - --config: config(.py) 파일 경로
            - --pth: 가중치(.pth) 파일 경로
            - --px: 이미지 추론시 retile 사이즈(pixel)

    - out: 결과물 생성 예시. container 내부의 /workspace가 아닌 다른 폴더로 지정 가능
- docker-compose.yml
    - gpu, cpu 모두 추론가능
    - cpu 추론시 deploy 부분 주석처리 필요
    - 30장 추론시 예상 소요시간: gpu(62초), cpu(336초)
