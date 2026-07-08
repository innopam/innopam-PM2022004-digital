# Model Setup

이 저장소는 실행 코드와 경로 구조만 Git으로 관리하고, 실제 모델 가중치는 별도 저장소에서 받아 배치한다.

## 기본 원칙

- `workspace/model/best.pth`와 `workspace/model/dinov3-vitl16-pretrain-lvd1689m/model.safetensors`는 경로 보존용 0바이트 dummy 파일이다.
- dummy 파일은 실제 추론에 사용할 수 없다.
- 실제 `.pth`와 `model.safetensors`는 Nexus 같은 모델 저장소에서 받아 로컬에 배치한다.
- `config.json`, `preprocessor_config.json`은 Git에 포함되어 있으므로 별도로 받을 필요가 없다.
- 인증 정보는 문서나 Git에 쓰지 않고 실행 환경의 `NEXUS_USERNAME`, `NEXUS_PASSWORD` 등으로 관리한다.

## 필요한 파일

fresh clone 후 변화탐지 엔진을 실행하려면 아래 파일이 필요하다.

| 용도 | 로컬 배치 경로 | 현재 기준 크기 |
| --- | --- | ---: |
| Road checkpoint | `02_Road_CD/workspace/model/best_road.pth` | `1347389551` bytes |
| Building checkpoint | `04_Building_CD/workspace/model/best_building.pth` | `1400951863` bytes |
| Road DINOv3 backbone | `02_Road_CD/workspace/model/dinov3-vitl16-pretrain-lvd1689m/model.safetensors` | `1212559808` bytes |
| Building DINOv3 backbone | `04_Building_CD/workspace/model/dinov3-vitl16-pretrain-lvd1689m/model.safetensors` | `1212559808` bytes |

Road와 Building의 `model.safetensors`가 같은 DINOv3 backbone이면 Nexus에는 shared 파일 하나만 올리고, 실행 환경에서는 양쪽 경로에 같은 파일을 복사해 둔다.

## 권장 Nexus URI 기록

모델 파일을 Nexus에 올린 뒤, 이 절의 placeholder를 실제 URI로 갱신한다.

```text
ROAD_CHECKPOINT_URI=https://nexus.innopam.kr/repository/models-repo/digital_land/02_Road_CD/workspace/model/best_road.pth
BUILDING_CHECKPOINT_URI=https://nexus.innopam.kr/repository/models-repo/digital_land/04_Building_CD/workspace/model/best_building.pth
DINO_BACKBONE_URI=https://nexus.innopam.kr/repository/models-repo/digital_land/shared/dinov3-vitl16-pretrain-lvd1689m/model.safetensors
```

URI가 확정되면 checksum도 같이 기록한다.

```text
ROAD_CHECKPOINT_SHA256=<sha256>
BUILDING_CHECKPOINT_SHA256=<sha256>
DINO_BACKBONE_SHA256=<sha256>
```

checksum은 모델 파일이 있는 머신에서 계산한다.

```bash
sha256sum 02_Road_CD/workspace/model/best_road.pth
sha256sum 04_Building_CD/workspace/model/best_building.pth
sha256sum 02_Road_CD/workspace/model/dinov3-vitl16-pretrain-lvd1689m/model.safetensors
```

## 수동 다운로드 예시

아래 명령은 URI가 확정된 뒤 실행자가 직접 값을 채워 사용하는 예시다.

```bash
export NEXUS_USERNAME=<username>
export NEXUS_PASSWORD=<password>

curl -fL -u "$NEXUS_USERNAME:$NEXUS_PASSWORD" \
  -o 02_Road_CD/workspace/model/best_road.pth \
  "$ROAD_CHECKPOINT_URI"

curl -fL -u "$NEXUS_USERNAME:$NEXUS_PASSWORD" \
  -o 04_Building_CD/workspace/model/best_building.pth \
  "$BUILDING_CHECKPOINT_URI"

curl -fL -u "$NEXUS_USERNAME:$NEXUS_PASSWORD" \
  -o 02_Road_CD/workspace/model/dinov3-vitl16-pretrain-lvd1689m/model.safetensors \
  "$DINO_BACKBONE_URI"

cp \
  02_Road_CD/workspace/model/dinov3-vitl16-pretrain-lvd1689m/model.safetensors \
  04_Building_CD/workspace/model/dinov3-vitl16-pretrain-lvd1689m/model.safetensors
```

## 배치 확인

모델 파일을 받은 뒤 아래 명령으로 0바이트 dummy가 아닌 실제 파일인지 확인한다.

```bash
test -s 02_Road_CD/workspace/model/best_road.pth
test -s 04_Building_CD/workspace/model/best_building.pth
test -s 02_Road_CD/workspace/model/dinov3-vitl16-pretrain-lvd1689m/model.safetensors
test -s 04_Building_CD/workspace/model/dinov3-vitl16-pretrain-lvd1689m/model.safetensors
```

크기도 함께 확인한다.

```bash
find 02_Road_CD/workspace/model 04_Building_CD/workspace/model \
  -maxdepth 3 -type f -printf '%s %p\n' | sort -k2
```

## Git Changes 정리

실제 모델을 배치하면 dummy로 추적 중인 `model.safetensors`가 modified로 보일 수 있다. 실행 환경에서는 아래처럼 로컬 전용으로 숨긴다.

```bash
git update-index --skip-worktree \
  02_Road_CD/workspace/model/best.pth \
  02_Road_CD/workspace/model/dinov3-vitl16-pretrain-lvd1689m/model.safetensors \
  04_Building_CD/workspace/model/best.pth \
  04_Building_CD/workspace/model/dinov3-vitl16-pretrain-lvd1689m/model.safetensors
```

`best_road.pth`와 `best_building.pth`는 Git에 추가하지 않는다. 필요하면 로컬 전용 ignore 파일에 추가한다.

```bash
cat >> .git/info/exclude <<'EOF'
02_Road_CD/workspace/model/best_road.pth
04_Building_CD/workspace/model/best_building.pth
EOF
```

## 실행 전 체크

Road:

```bash
cd 02_Road_CD
test -s workspace/model/best_road.pth
test -s workspace/model/dinov3-vitl16-pretrain-lvd1689m/model.safetensors
docker build --tag road_cd:1.0 .
docker-compose up
```

Building:

```bash
cd 04_Building_CD
test -s workspace/model/best_building.pth
test -s workspace/model/dinov3-vitl16-pretrain-lvd1689m/model.safetensors
docker build -t building_cd:latest .
docker-compose up
```

## 이후 자동화 계획

자동화가 필요해지면 이 문서를 기준으로 `models.lock.json`과 `scripts/ensure_models.py`를 추가한다. 자동화 스크립트는 로컬 파일의 size/sha256을 확인하고, 누락되었거나 checksum이 다를 때만 Nexus에서 다운로드하도록 만든다.
