from models.base_model import BaseClassifier
import torch.nn as nn

# 단일 모델 버전: convnext만 사용하여 분류하는 경우
class SingleClassifier(BaseClassifier):
    def __init__(self, 
                 train_dataset=None, 
                 val_dataset=None, 
                 num_classes=500, 
                 lr=3e-5, 
                 weight_decay=1e-2,
                 model_size: str = "large"
                ):
        super().__init__(train_dataset, val_dataset, num_classes, lr, weight_decay, model_size)
        # classifier는 convnext output dimension만 사용
        self.classifier = nn.Linear(self.convnext_output_dim, self.num_classes)

    def forward(self, pixel_values):
        convnext_features = self.convnext(pixel_values).logits
        logits = self.classifier(convnext_features)
        return logits