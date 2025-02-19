import os
import argparse
import pandas as pd
from torch.utils.data import Subset, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold

# 모델 관련 import (파일 분리한 대로)
from models.single_model import SingleClassifier
from models.stacking_model import StackingClassifier
from datasets.dataset import CustomDataset
from utils.transform import AlbumentationsTransform

def load_data(train_data_dir, train_info_file):
    train_info = pd.read_csv(train_info_file)
    num_classes = len(train_info['target'].unique())
    
    dataset = CustomDataset(
        root_dir=train_data_dir,
        info_df=train_info,
        transform=AlbumentationsTransform(is_train=True)
    )
    return dataset, num_classes

def create_model(model_type, train_dataset, val_dataset, num_classes, lr, weight_decay, model_size):
    if model_type == "stacking":
        return StackingClassifier(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_classes=num_classes,
            lr=lr,
            weight_decay=weight_decay,
            model_size=model_size
        )
    else:  # "single"
        return SingleClassifier(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_classes=num_classes,
            lr=lr,
            weight_decay=weight_decay,
            model_size=model_size
        )

def train_model(model, fold, output_dir, max_steps, gradient_clip_val):
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=False,
        mode='min'
    )

    checkpoint_dir = os.path.join(output_dir, f"fold_{fold + 1}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=2,
        monitor="val_loss",
        mode="min",
        save_weights_only=True
    )

    trainer = Trainer(
        max_steps=max_steps,
        gradient_clip_val=gradient_clip_val,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='gpu'
    )
    trainer.fit(model)

def main():
    parser = argparse.ArgumentParser(description="Train model with optional KFold and model type selection.")
    parser.add_argument("--train_data_dir", type=str, default="../data/train", help="Path to training data directory")
    parser.add_argument("--train_info_file", type=str, default="../data/train.csv", help="Path to CSV file with train info")
    parser.add_argument("--model_type", type=str, choices=["single", "stacking"], default="single", help="Select model type: single or stacking")
    parser.add_argument("--model_size", type=str, choices=["tiny", "large"], default="tiny", help="Select model size: tiny or large")
    parser.add_argument("--kfold", action="store_true", help="Use StratifiedKFold for training")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of splits for KFold")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--gradient_clip_val", type=float, default=2.0, help="Gradient clipping value")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()

    dataset, num_classes = load_data(args.train_data_dir, args.train_info_file)

    # StratifiedKFold 방식으로 학습할 경우
    if args.kfold:
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        # CustomDataset에 targets 속성이 있다고 가정
        for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, dataset.targets)):
            print(f"Fold {fold + 1}")
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            model = create_model(args.model_type, train_subset, val_subset, num_classes, args.lr, args.weight_decay, args.model_size)
            train_model(model, fold, args.output_dir, args.max_steps, args.gradient_clip_val)
    else:
        # 단일 train/val 분할 (예: 80/20)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_subset, val_subset = random_split(dataset, [train_size, val_size])
        model = create_model(args.model_type, train_subset, val_subset, num_classes, args.lr, args.weight_decay, args.model_size)
        train_model(model, 0, args.output_dir, args.max_steps, args.gradient_clip_val)

if __name__ == "__main__":
    main()
