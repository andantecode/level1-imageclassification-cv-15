# 🖍️ Sketch Image Classification

**프로젝트 주제**:<br>
인간의 상상력을 반영하는 추상적이고 단순한 형태의 스케치 이미지를 분류하는 모델

## 프로젝트 개요

이 프로젝트는 스케치 이미지 분류를 위해 ConvNextV2와 Eva02 모델을 활용하며, stacking 기법과 Stratified KFold를 적용하여 분류 성능을 극대화하는 것을 목표로 한다. 또한, 데이터 전처리 파이프라인에서는 기하학적 변환, Mixup 및 Cutmix 기법을 적용하여 데이터 증강을 수행한다.

![Image](https://github.com/user-attachments/assets/eb4552d5-96b6-4db0-9482-d8ca0d59262c)

## 프로젝트 구조

```bash
sketch/
├── datasets/               # 데이터셋 로딩 및 전처리 관련 코드
├── models/                 # 모델 정의 (ConvNextV2, Eva02, Stacking 및 Single 모델 등)
├── tools/                  # 학습 및 평가 스크립트 (train.py, inference.py)
├── utils/                  # 유틸리티 함수 및 도구 (데이터 변환, 콜백 등)
└── sketch_predictor.py     # 추론용 스크립트 (Streamlit 앱 포함)
```

## 데이터 전처리 파이프라인

**기하학적 변환(50%)**:<br>

- HorizontalFlip(p=0.5):
  50% 확률로 이미지를 좌우 반전

- VerticalFlip(p=0.5):
  50% 확률로 이미지를 상하 반전

- Rotate(limit=15, p=0.5):
  50% 확률로 최대 15도 범위 내에서 무작위 회전

- Affine(scale=(1, 1.5), shear=(-10, 10), p=0.5):
  50% 확률로 이미지 스케일을 1배에서 1.5배 사이로 조절하고, -10도에서 10도 범위의 시어(shear)를 적용

- ElasticTransform(alpha=10, sigma=50, p=0.5):
  50% 확률로 탄성 변형을 적용하여 이미지에 비선형 왜곡 효과 부여

- RandomBrightnessContrast(p=0.5):
  50% 확률로 이미지의 밝기와 대비를 무작위로 조절

- MotionBlur(blur_limit=(3, 7), p=0.5):
  50% 확률로 3~7 범위의 모션 블러 효과 적용

**Mixup(25%)**:<br>

- 배치 샘플 중 랜덤으로 2개씩 선택하여 Mixup 적용
  - lam: [0.3, 0.7]

**Cutmix(25%)**:<br>

- 배치 샘플 중 랜덤으로 2개씩 선택하여 Cutmix 적용
  - lam: [0.3, 0.7]

## 모델 구성

- ConvNextV2와 Eva02 모델을 활용
- 두 모델의 출력을 stacking하여 feature를 결합하는 방식 적용
- Stratified KFold 기법을 활용해 데이터를 여러 Fold로 분할하여 학습

## 사용 방법

1. 모델 학습

- Git 저장소 클론
  ```bash
  git clone https://github.com/andantecode/level1-imageclassification-cv-15.git
  ```
- 프로젝트 메인 디렉토리 이동
  ```bash
  cd sketch
  ```
- 학습 스크립트 실행
  ```bash
  python tools/train.py
  ```

2. 프로토타입 실행 (Streamlit)
   ```bash
   streamlit run sketch_predictor.py -- --checkpoint_path <체크포인트_경로> --train_dir <train dataset 경로> --train_info_file <train.csv_경로> --model_type stacking
   ```
