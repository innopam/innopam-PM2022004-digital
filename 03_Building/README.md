

### 사용 방법

```bash
cd ./innopam-PM202204-digital/03_Building
docker build -t building:latest .
# 디버그
export dataset_path=/workspace/input output_path=/workspace/output model_path=/workspace/model && docker compose run dt
# 실행
export dataset_path=/workspace/input output_path=/workspace/output model_path=/workspace/model && docker compose up
```
---

### 데이터 처리 과정

### 1. 건물 추론
1. 모델 및 설정 불러오기
2. TIF 이미지 불러오기 및 config 설정
3. 타일 분할 및 처리 준비
4. 추론 실행, 타일 마스크 생성 및 병합
5. 마스크 후처리
6. GeoDataFrame(GDF) 변환 및 후처리

### 2. 변화 탐지
1. GDF 매칭 및 전처리
2. 그래프 구성 및 객체 정제
3. 그래프 기반 정량 지표 계산
4. 변화 유형 분류

---

### 데이터

### Input
- 전처리된 **과거 수치지도** 및 **현재 영상**
- `input/T1`, `input/T2` 폴더에 각각 저장

### Output
- `output/cur_result.geojson`: 현재 시점 건물 추론 결과
- `output/prev_result.geojson`: 과거 시점 건물 추론 결과
- `output/status.json`: 전체 파이프라인 진행도 상태
---

### 폴더 구조
```
workspace/
├── input/
│   ├── T1/  # 과거 수치지도
│   └── T2/  # 현재 영상
├── model/
│   └── building_2005_deepness.onnx
├── output/
│   └── cur_result.goejson
│   └── prev_result.geojson
│   └── status.json
└── main.py
```
---

### 구성 파일
- `model`: ONNX 형식의 모델 포함
- `main.py`: 전체 파이프라인 실행 코드
- `requirements.txt`: Python 의존 패키지 목록

### 파라미터

```python
parser.add_argument("-i", "--input", type=str, default="workspace/input", help="input folder containing T1/, T2/")
parser.add_argument("-m", "--model", type=str, default="workspace/model", help="Model folder containing .onnx")
parser.add_argument("-o", "--output", type=str, default="workspace/output", help="Output folder")

parser.add_argument("-c", "--conf-threshold", type=float, default=None)
parser.add_argument("-r", "--resolution", type=float, default=None)
parser.add_argument("--classes", type=str, default=None)
parser.add_argument("-t", "--max-threads", type=int, default=None)

parser.add_argument("--cut-threshold", type=float, default=0.05)
parser.add_argument("--cd-threshold", type=float, default=0.7)
```
