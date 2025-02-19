import pytorch_lightning as pl
from transformers import ConvNextV2ForImageClassification
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from .utils import mixup_cutmix_collate_fn, SoftTargetCrossEntropy


# 베이스 클래스: 공통된 초기화 및 데이터로더, 학습/검증 스텝 등
class BaseClassifier(pl.LightningModule):
    def __init__(self, 
                 train_dataset=None, 
                 val_dataset=None, 
                 num_classes=500, 
                 lr=3e-5, 
                 weight_decay=1e-2,
                 model_size: str = "large"   # "large" 혹은 "tiny"
                ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.model_size = model_size

        # model_size에 따라 convnext 모델 선택
        if self.model_size == "tiny":
            self.convnext = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-tiny-1k-224")
            self.convnext_output_dim = 768
        else:
            self.convnext = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-large-22k-224")
            self.convnext_output_dim = 1536

        # 기존 분류기 제거
        self.convnext.classifier = nn.Identity()

        # 공통 손실 함수
        self.loss_fn_crossentropy = nn.CrossEntropyLoss()
        self.loss_fn = SoftTargetCrossEntropy()

    def forward(self, pixel_values):
        # 서브클래스에서 구현
        raise NotImplementedError("서브클래스에서 forward 메소드를 구현해야 함")

    def training_step(self, batch, batch_idx):
        pixel_values, labels = batch
        logits = self.forward(pixel_values)
        loss = self.loss_fn(logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values, labels = batch
        logits = self.forward(pixel_values)
        loss = self.loss_fn_crossentropy(logits, labels)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        # 기본적으로 convnext와 classifier의 파라미터만 포함.
        optimizer_params = [
            {'params': self.convnext.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4},
            {'params': self.classifier.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay},
        ]
        optimizer = torch.optim.AdamW(optimizer_params, lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=8,
            collate_fn=mixup_cutmix_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=8,
            drop_last=False
        )