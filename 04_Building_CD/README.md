
## 사용방법

```bash
cd ./innopam-PM202204-digital/04_Building_CD
docker build -t building_cd:latest .
# 디버그
export dataset_path=/workspace/input output_path=/workspace/out model_path=/workspace/model && docker compose run mambacd
# 실행
export dataset_path=/workspace/input output_path=/workspace/out model_path=/workspace/model && docker compose up
```

## 데이터 처리 과정
1. AI 데이터 제작 (.tif -> .png)
2. 변화탐지 추론
3. 출력물 병합 (.png -> .tif)
4. 벡터화 (.tif -> .geojson)
5. 후처리 (.geojson -> .geojson)

## Data
- Input
	- 전처리가 완료된 Before/After 영상
	- T1, T2 폴더에 같은 이름과 크기로 저장
- Output
	- [이미지 이름].tif, [이미지 이름]_rgb.tif, [이미지 이름]_conf.tif > 래스터 데이터
	- [이미지 이름].json > 벡터 데이터
	- status.json > 진행도
	
## 폴더 구조
```bash
├── docker-compose.yml
├── dockerfile
├── READMD.md
└── workspace
    ├── model
    │   ├── init_train_aihub.pth
	│   ├── vssm_base_224.yaml
    ├── predict.py
    ├── input
    │   ├── T1
    │   ├── T2
    └── output
        ├── status.json
        ├── patches
			├──377052100
				├──T1
				├──T2
				├──377052100_metadata.json
				├──data_list.txt
				├──patch_distribution.txt
		└──results
			├──377052100
				├──confidence
				├──pred
				├──377052100.json
				├──377052100.tif
				└──377052100_conf.tif
		
```
- workspace
	- model, sample_data, predict.py -> container 생성시 /workspace 경로에 bind 되어야함
		- model: config(.yaml), 가중치(.pth)로 구성
		- input: T1(Before 이미지), T2(After 이미지)로 구성
		- predict.py: 추론코드
			- --dataset_path: Before/After 영상 경로
			- --output_path: 결과물 저장 경로
			- --model_path: 모델/가중치 파일 경로
			- --multi_mode: 멀티 클래스 변화탐지 여부 (default=True)
			- --patch_size: AI데이터 패치 크기
			- --overlap_ratio: 패치 영상 중복도 (default='min')
		- output: container 내부의 /workspace가 아닌 다른 폴더로 지정 가능
