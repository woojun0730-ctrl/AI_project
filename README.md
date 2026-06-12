# 🚧 실제 블랙박스 환경을 위한 YOLOv11 기반 도로 위험물 탐지

> **학습 데이터와 실전 블랙박스 환경 간의 도메인 갭을 분석하고, 전처리 기법(ROI Crop, CLAHE)의 효과와 한계를 실증한 AI 프로젝트**

---

## 팀원

| 이름 | 학번 |
|------|------|
| 이찬민 | 23619019 |
| 김우준 | 22619005 |
| 박채원 | 24619019 |
| 한지은 | 24619040 |
---

## 프로젝트 개요

포트홀(Pothole)·균열(Crack)은 매년 교통사고를 급증시키고, 주행 중 운전자의 시야가 제한되는 환경에서는 치명적인 2차 사고를 유발한다.

공개 데이터셋은 대부분 **정지 상태의 고화질 이미지**로 구성되어 있다.
이 프로젝트에서는 실제 주행 중 촬영된 블랙박스 영상에 YOLOv11 모델을 적용했을 때 발생하는 **성능 저하 현상**을 정량적으로 분석하고, ROI Crop·CLAHE 등의 전처리 기법이 성능에 미치는 영향을 실험적으로 검증한다.

### 탐지 대상 클래스

| 클래스 ID | 이름 | 라벨 수 |
|-----------|------|---------|
| 0 | Pothole (포트홀) | 1,261개 |
| 1 | Crack (균열) | 2,519개 |
| 2 | Manhole (맨홀) | 957개 |

---

## 데이터셋

| 항목 | 내용 |
|------|------|
| 공식 명칭 | Road Damage Dataset: Potholes, Cracks and Manholes |
| 제공 기관 | Kaggle (글로벌 오픈소스 플랫폼) |
| 출처 URL |[Kaggle 링크](https://www.kaggle.com/datasets/lorenzoarcioni/road-damagedataset-potholes-cracks-and-manholes) |
| 수집 기간 | 2024 ~ 2025 (Version 4) |
| 이미지 수 | 총 2,009장 (Train 1,607장 / Val 402장) |
| 바운딩 박스 수 | 총 4,737개 |

> **학습 환경 vs. 실전 환경 차이**  
> - 학습 데이터: 고화질, 정면 카메라, 낮은 노이즈  
> - 실전 데이터 (직접 수집한 블랙박스): 저화질, 빛 반사, 주행 중 노이즈

---

## 파이프라인 구조

```
Dataset
  │
  ▼
Data Processing
  ├─ ROI Crop       (Crop.ipynb, Crop_train.ipynb)
  └─ CLAHE          (YOLO_Clahe.ipynb)
  │
  ▼
Model Training (YOLOv11)
  ├─ Baseline       (Baseline.ipynb)
  └─ Augmentation   (YOLO_Val.ipynb)
  │
  ▼
Test (실제_영상_테스트.ipynb)
```

### 전처리 기법 설명

**ROI Crop**
- 이미지 상단 45%, 하단 15%를 잘라내어 도로 영역에 집중
- YOLO 형식의 바운딩 박스 좌표를 크롭 후 이미지 기준으로 자동 재계산
- 한글 파일명 처리를 위한 `imread_korean` / `imwrite_korean` 함수 포함

**CLAHE (Contrast Limited Adaptive Histogram Equalization)**
- LAB 색공간에서 L 채널에만 적용 (clipLimit=3.0, tileGridSize=8×8)
- 명암 대비를 강조하여 저조도 환경에서의 탐지력 향상 목적

---

## 실험 결과

모든 실험은 Val 데이터셋 기준 mAP로 평가하였습니다.

| 실험 | 적용 기법 | mAP | 분석 |
|------|-----------|-----|------|
| **Baseline** | YOLO11s | **0.557** | 해외 데이터 기반 순수 탐지 성능 |
| Model Upgrade | YOLO11m | 0.557 | 모델 체급 업그레이드 효과 없음 |
| AUG | Augmentation | 0.543 | 인위적 증강이 오히려 노이즈를 증폭 |
| Processing | ROI Crop | 0.535 | 원근감·주변 맥락 정보 손실로 성능 저하 |
| Processing | CLAHE | 0.477 | 아스팔트 질감 노이즈 증폭 → 오탐지 급증 |

> **Test 지표를 사용하지 않은 이유**  
> 학습 데이터셋과 테스트 데이터셋 간의 YAML 클래스 순서 불일치 문제가 발생하였고, 강제 변환 시 데이터 왜곡 우려로 정량적 mAP를 산출하지 못하였다. 대신 실제 블랙박스 영상에서 바운딩 박스가 정확히 매핑되는지를 정성적으로 확인하는 방식으로 평가하였다.

---

## 파일 구성

```
📦 프로젝트 루트
├── Baseline.ipynb          # YOLO11s 기본 학습
├── YOLO_Val.ipynb          # 데이터 분할 및 Augmentation 실험
├── Crop.ipynb              # ROI Crop 전처리 (Train/Val 데이터)
├── Crop_train.ipynb        # Crop된 데이터로 YOLO 학습
├── YOLO_Clahe.ipynb        # CLAHE 전처리 적용 및 학습
├── 실제_영상_테스트.ipynb   # 블랙박스 영상 실전 테스트
└── AI프로젝트_최종_발표.pptx
```

---

## 실행 방법

모든 노트북은 **Google Colab** 환경 기준으로 작성되었다.
Google Drive 마운트 후 아래 순서대로 실행하면 된다.

### 1. 환경 설정

```bash
pip install ultralytics
```

### 2. 데이터 준비 — Val 중복 제거

`YOLO_Val.ipynb` 실행 → Train 폴더에서 Val과 겹치는 이미지 자동 삭제

```
dataset_yolo/
├── train/
│   ├── images/   # 1,607장
│   └── labels/
├── val/
│   ├── images/   # 402장
│   └── labels/
└── data.yaml
```

### 3. (옵션) ROI Crop 전처리

`Crop.ipynb` 실행 → `dataset_cropped/` 폴더 생성

```python
# 크롭 비율 조정 가능 (기본값: 상단 45%, 하단 15% 제거)
crop_dataset(img_dir, label_dir, out_img_dir, out_label_dir,
             top_ratio=0.45, bottom_ratio=0.15)
```

### 4. (옵션) CLAHE 전처리

`YOLO_Clahe.ipynb` 실행 → `dataset_clahe/` 폴더 생성

```python
# CLAHE 파라미터 조정 가능
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
```

### 5. 모델 학습

```python
from ultralytics import YOLO

model = YOLO('yolo11s.pt')
model.train(
    data='경로/data.yaml',
    epochs=100,
    patience=20,
    imgsz=640,
    batch=16,
    device=0,
)
```

### 6. 실전 영상 테스트

`실제_영상_테스트.ipynb` 실행

```python
model = YOLO('best.pt 경로')
results = model.predict(
    source='테스트 이미지 폴더',
    conf=0.05,
    imgsz=640,
    save=True,
)
```

---

## 결론 및 한계점

**결론**
- 해외 공개 데이터셋으로 학습된 YOLO11s를 국내 블랙박스 영상에 적용할 때 **도메인 갭으로 인한 명확한 성능 저하**가 발생함을 확인
- ROI Crop과 CLAHE는 의도와 달리 모델이 필요로 하는 **원근감·맥락 정보를 손실**시키거나 **미세 노이즈를 증폭**시키는 역효과를 냄

**한계점**
- YAML 클래스 구조 불일치로 테스트셋에 대한 **정량적 mAP 산출 불가**
- 바운딩 박스를 육안으로 확인하는 **정성적 평가에만 의존**, 객관적 통계 입증에 한계

---

## 향후 과제

1. **데이터 중심 접근** — 국내 도로·블랙박스 환경에 맞는 데이터셋을 직접 구축하고 Fine-tuning 진행
2. **YAML 오류 수정** — 클래스 구조 불일치 자동 해결 및 레이블 통합 전처리 스크립트 개발
3. **임베디드 최적화** — 차량용 블랙박스·내비게이션 등 저사양 환경을 위한 모델 경량화 연구
