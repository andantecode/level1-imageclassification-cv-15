import sys, os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import random
import argparse

# 프로젝트 루트 경로 추가 (예: models, tools, utils 폴더가 있는 상위 디렉토리)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.inference import load_models  # refactored load_models (모델 타입 옵션 포함)
from utils.transform import AlbumentationsTransform
from datasets.dataset import CustomDataset  # 추후 필요한 경우
from classes import IMAGENET2012_CLASSES

def parse_args():
    parser = argparse.ArgumentParser(description="Streamlit Inference App Arguments")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/fold_1/test.ckpt", help="Checkpoint 파일 경로")
    parser.add_argument("--train_dir", type=str, default="../data/train", help="Train images 폴더 경로")
    parser.add_argument("--train_info_file", type=str, default="../data/train.csv", help="Train info CSV 파일 경로")
    parser.add_argument("--model_type", type=str, choices=["stacking", "single"], default="single", help="사용할 모델 타입")
    args, unknown = parser.parse_known_args()  # Streamlit 관련 인자 무시
    return args

args = parse_args()

# Streamlit 페이지 설정
st.set_page_config(page_title='Sketch Image Classification', layout='wide')
st.title(':crystal_ball: Sketch Image Classifier in 500 classes :crystal_ball:')
st.markdown('---')
st.info('분류하고 싶은 이미지를 업로드 해주세요!')

# 이미지 업로드 위젯
uploaded_file = st.file_uploader('이미지 업로드', type=['jpg', 'png', 'jpeg'], label_visibility="hidden")

# 모델 타입 선택 (stacking / single)
model_type = st.radio("모델 타입 선택", options=["stacking", "single"], index=1)

# st.cache_resource로 모델 로드 (checkpoint 폴더 경로와 모델 타입 옵션 전달)
@st.cache_resource
def load_model(file_path: str, model_type: str):
    return load_models(file_path=file_path, model_type=model_type)

# 캐시된 train_info 로드 함수
@st.cache_resource
def load_train_info(train_info_file: str):
    return pd.read_csv(train_info_file)

# 실제 inference 함수
def perform_inference(models, device, transform, image_file):
    # 업로드된 파일을 cv2 이미지로 변환
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 전처리 및 배치 차원 추가
    image = transform(image)
    image = image.unsqueeze(0)

    # 각 모델을 device로 옮기고 eval 모드 설정
    for model in models:
        model.to(device)
        model.eval()

    print(models)

    with torch.no_grad():
        image = image.to(device)
        all_logits = []
        for model in models:
            logits = model(image)
            all_logits.append(logits)
        # 여러 모델의 결과 평균 내기
        avg_logits = torch.mean(torch.stack(all_logits), dim=0)
        avg_probs = F.softmax(avg_logits, dim=1)
        preds = avg_probs.argmax(dim=1)
        predicted_probabilities = avg_probs[torch.arange(avg_probs.size(0)), preds]

    return preds, predicted_probabilities

def show_result(image_file, probability, info_df: pd.DataFrame, class_number: int):
    # 원본 이미지 로드
    original_image = Image.open(image_file)
    
    # 해당 클래스의 예시 이미지 선택 (랜덤 샘플)
    class_info_candidates = info_df[info_df['target'] == class_number]
    if len(class_info_candidates) == 0:
        st.error("해당 클래스에 해당하는 이미지 정보가 없습니다.")
        return
    random_index = random.randint(0, max(0, len(class_info_candidates) - 1))
    class_info = class_info_candidates.iloc[random_index]
    
    class_image_path = os.path.join(args.train_dir, class_info['image_path'])
    class_name = class_info['class_name']
    # IMAGENET2012_CLASSES 사전에서 클래스 이름 가져오기 (없으면 원래 값 사용)
    class_name = IMAGENET2012_CLASSES.get(class_name, class_name)
    try:
        class_image = Image.open(class_image_path)
    except Exception as e:
        st.error(f"클래스 이미지 로드 실패: {e}")
        return

    # 두 컬럼에 원본 이미지와 예시 이미지 나란히 표시
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image.resize((512, 512)), caption='업로드 한 이미지', use_container_width=True)
    with col2:
        st.image(class_image.resize((512, 512)), caption=f'클래스 {class_name}의 예시 이미지', use_container_width=True)

    # 신뢰도에 따라 정보 또는 에러 메시지 출력
    if probability.item() > 0.5:
        st.info(f'예측된 클래스는 {class_name}이고, 신뢰도는 {probability.item():.2%}입니다.')
    else:
        st.error(f'예측된 클래스는 {class_name}이고, 신뢰도는 {probability.item():.2%}입니다.')

# 전처리 transform 생성
transform = AlbumentationsTransform(is_train=False)
# 모델 로드 (checkpoint 폴더 경로와 선택한 모델 타입 전달)
models = load_model(args.checkpoint_path, args.model_type)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 학습 데이터 CSV 로드 (예시 이미지 선택을 위한 정보)
train_info = load_train_info(args.train_info_file)


# 버튼 클릭 시 추론 수행
if st.button("예측하기") and uploaded_file is not None:
    # inference 수행 후 결과 반환
    preds, probability = perform_inference(models, device, transform, uploaded_file)
    # 결과 출력
    show_result(uploaded_file, probability, train_info, preds.item())
