from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class ResNetItemEmbedder(nn.Module):
    def __init__(self, embedding_dim: int = 512, backbone: str = "resnet50", pretrained: bool = True) -> None:
        super().__init__()
        if backbone == "resnet50":
            model = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT if pretrained else None)
            feat_dim = 2048
        elif backbone == "resnet101":
            model = tvm.resnet101(weights=tvm.ResNet101_Weights.DEFAULT if pretrained else None)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove classifier, keep global average pooling output
        modules = list(model.children())[:-1]  # drop fc
        self.backbone = nn.Sequential(*modules)
        self.proj = nn.Linear(feat_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        feats = self.backbone(x)  # (B, C, 1, 1)
        feats = feats.flatten(1)  # (B, C)
        emb = self.proj(feats)    # (B, D)
        # Apply L2 normalization as specified in requirements
        emb = F.normalize(emb, p=2, dim=1)
        return emb
