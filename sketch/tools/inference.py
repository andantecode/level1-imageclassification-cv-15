import os
import sys
import argparse
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import tqdm

# 프로젝트 루트 경로 추가 (tools/inference.py에서 상대경로 문제 해결)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 모델 임포트 (옵션에 따라 single, stacking 선택)
from models.single_model import SingleClassifier
from models.stacking_model import StackingClassifier
from datasets.dataset import CustomDataset
from utils.transform import AlbumentationsTransform


def load_models(root_path: str, model_type: str):
    models = []
    # 모델 타입에 따라 불러올 클래스를 선택
    ModelClass = StackingClassifier if model_type == "stacking" else SingleClassifier

    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file.endswith(".ckpt"):  # checkpoint 파일 확장자가 .ckpt라고 가정
                file_path = os.path.join(dirpath, file)
                print(f"Loading checkpoint: {file_path}")
                models.append(ModelClass.load_from_checkpoint(file_path))
    return models


def inference(models: List, device: torch.device, test_loader: DataLoader):
    # 모든 모델을 device에 올리고 eval 모드 전환
    for model in models:
        model.to(device)
        model.eval()

    predictions = []
    with torch.no_grad():
        for images in tqdm.tqdm(test_loader):
            images = images.to(device)

            all_logits = []
            for model in models:
                logits = model(images)
                all_logits.append(logits)

            # 여러 모델의 예측값 평균 내기
            avg_logits = torch.mean(torch.stack(all_logits), dim=0)
            avg_probs = F.softmax(avg_logits, dim=1)
            preds = avg_probs.argmax(dim=1)
            predictions.extend(preds.cpu().detach().numpy())

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Inference script for classification models")
    parser.add_argument("--root_path", type=str, default="./checkpoints", help="Root path for checkpoint files")
    parser.add_argument("--model_type", type=str, choices=["stacking", "single"], default="stacking", help="Select model type")
    parser.add_argument("--test_data_dir", type=str, default="../data/test", help="Test data directory")
    parser.add_argument("--test_info_file", type=str, default="../data/test.csv", help="CSV file containing test info")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for DataLoader")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 테스트 데이터셋 로드
    test_info = pd.read_csv(args.test_info_file)
    test_dataset = CustomDataset(
        root_dir=args.test_data_dir,
        info_df=test_info,
        transform=AlbumentationsTransform(is_train=False),
        is_inference=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )
    
    # 모델 로드 (옵션에 따라 single / stacking)
    models = load_models(root_path=args.root_path, model_type=args.model_type)
    if not models:
        print("No checkpoint files found!")
        return

    predictions = inference(models=models, device=device, test_loader=test_loader)
    
    # 결과 저장
    test_info["target"] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    os.makedirs("./result", exist_ok=True)
    result_path = "./result/eva02convnext2_kfold.csv"
    test_info.to_csv(result_path, index=False)
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
