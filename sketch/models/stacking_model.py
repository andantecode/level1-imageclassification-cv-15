from models.base_model import BaseClassifier
import timm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

# stacking 옵션: convnext와 eva02를 모두 사용하여 feature를 결합하는 버전
class StackingClassifier(BaseClassifier):
    def __init__(self, 
                 train_dataset=None, 
                 val_dataset=None, 
                 num_classes=500, 
                 lr=3e-5, 
                 weight_decay=1e-2,
                 model_size: str = "tiny"
                ):
        super().__init__(train_dataset, val_dataset, num_classes, lr, weight_decay, model_size)

        # eva02 모델 추가
        if self.model_size == "tiny":
            self.eva02 = timm.create_model('eva02_tiny_patch14_224.mim_in22k', pretrained=True, num_classes=0)
            self.eva02_output_dim = 192
        else:
            self.eva02 = timm.create_model('eva02_large_patch14_clip_224.merged2b', pretrained=True, num_classes=0)
            self.eva02_output_dim = 1024

        # stacking 시엔 convnext와 eva02의 feature를 결합함.
        combined_dim = self.convnext_output_dim + self.eva02_output_dim
        self.classifier = nn.Linear(combined_dim, self.num_classes)

    def forward(self, pixel_values):
        # convnext로부터 feature 추출
        convnext_features = self.convnext(pixel_values).logits

        # eva02로부터 feature 추출 (eva02는 timm 모델이므로 forward_features와 forward_head를 사용)
        eva02_out = self.eva02.forward_features(pixel_values)
        eva02_features = self.eva02.forward_head(eva02_out, pre_logits=True)

        # 두 feature 결합 후 classifier 통과
        combined_features = torch.cat((convnext_features, eva02_features), dim=1)
        logits = self.classifier(combined_features)
        return logits

    def configure_optimizers(self):
        optimizer_params = [
            {'params': self.convnext.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4},
            {'params': self.eva02.parameters(), 'lr': 1e-5, 'weight_decay': 1e-2},
            {'params': self.classifier.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay},
        ]
        optimizer = torch.optim.AdamW(optimizer_params, lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
        return [optimizer], [scheduler]