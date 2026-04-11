from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CurriculumConfig:
    num_classes: int
    stage1_epochs: int = 10
    stage2_epochs: int = 25
    total_epochs: int = 50
    label_smoothing_major: float = 0.02
    label_smoothing_minor: float = 0.08


class AsymmetricLabelSmoothingCE(nn.Module):
    """Cross-entropy with per-class label smoothing."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        class_weights: torch.Tensor,
        smoothing_per_class: torch.Tensor,
    ) -> torch.Tensor:
        # Build smoothed one-hot targets with class-specific smoothing.
        with torch.no_grad():
            n = target.shape[0]
            smoothed = torch.zeros((n, self.num_classes), device=logits.device, dtype=logits.dtype)
            eps = smoothing_per_class[target].unsqueeze(1)
            smoothed.fill_(eps / max(self.num_classes - 1, 1))
            smoothed.scatter_(1, target.unsqueeze(1), 1.0 - eps.squeeze(1))

        log_probs = F.log_softmax(logits, dim=1)
        sample_weight = class_weights[target]
        loss = -(smoothed * log_probs).sum(dim=1)
        loss = loss * sample_weight
        return loss.mean()


def build_stage_weights(
    epoch: int,
    class_counts: torch.Tensor,
    per_class_f1: torch.Tensor,
    cfg: CurriculumConfig,
) -> torch.Tensor:
    """
    Stage-1: uniform.
    Stage-2: inverse-frequency weighting.
    Stage-3: inverse-F1 weighting (hard classes get larger weights).
    """
    eps = 1e-6
    c = cfg.num_classes
    class_counts = class_counts.float()
    base = torch.ones(c, dtype=torch.float32, device=class_counts.device)

    if epoch < cfg.stage1_epochs:
        w = base
    elif epoch < cfg.stage2_epochs:
        inv_freq = class_counts.sum() / (c * (class_counts + eps))
        w = inv_freq
    else:
        inv_f1 = 1.0 / (per_class_f1 + eps)
        inv_f1 = inv_f1 / inv_f1.mean().clamp_min(eps)
        w = inv_f1

    w = w / w.mean().clamp_min(eps)
    return w


def build_smoothing_vector(
    class_counts: torch.Tensor,
    cfg: CurriculumConfig,
) -> torch.Tensor:
    """Assign stronger smoothing to minority classes."""
    counts = class_counts.float()
    median_count = torch.median(counts)
    smoothing = torch.full_like(counts, cfg.label_smoothing_major, dtype=torch.float32)
    smoothing[counts < median_count] = cfg.label_smoothing_minor
    return smoothing.clamp(0.0, 0.2)


def per_class_f1_from_predictions(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    scores = []
    for c in range(num_classes):
        tp = ((y_true == c) & (y_pred == c)).sum().float()
        fp = ((y_true != c) & (y_pred == c)).sum().float()
        fn = ((y_true == c) & (y_pred != c)).sum().float()
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        scores.append(f1)
    return torch.stack(scores, dim=0)


def state_dict_for_logging(
    class_weights: torch.Tensor,
    smoothing: torch.Tensor,
) -> Dict[str, list]:
    return {
        "class_weights": [float(x) for x in class_weights.detach().cpu().tolist()],
        "class_smoothing": [float(x) for x in smoothing.detach().cpu().tolist()],
    }
