from __future__ import annotations

import timm
import torch.nn as nn


def build_vit_classifier(
    num_classes: int,
    backbone: str = "vit_base_patch16_224",
    pretrained: bool = True,
    drop_rate: float = 0.1,
) -> nn.Module:
    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model
