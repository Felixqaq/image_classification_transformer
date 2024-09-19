import torch
import torch.nn as nn
from torchvision import models

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=10):
        super(VisionTransformer, self).__init__()
        # 使用已預訓練的 ViT 模型
        self.vit = models.vit_b_16(pretrained=True)
        # 更改最後一層的全連接層來匹配輸出類別數量
        self.vit.heads = nn.Linear(self.vit.heads.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)
