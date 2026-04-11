from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import ImageFolderFlat, build_eval_transform, build_train_transform
from losses import (
    AsymmetricLabelSmoothingCE,
    CurriculumConfig,
    build_stage_weights,
    build_smoothing_vector,
    per_class_f1_from_predictions,
    state_dict_for_logging,
)
from model import build_vit_classifier


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(
    train_dir: str,
    val_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, Dict]:
    train_ds = ImageFolderFlat(train_dir, transform=build_train_transform(image_size))
    val_ds = ImageFolderFlat(val_dir, transform=build_eval_transform(image_size))
    if train_ds.meta.class_to_idx != val_ds.meta.class_to_idx:
        raise ValueError("Class folders differ between train and val.")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    meta = {
        "class_to_idx": train_ds.meta.class_to_idx,
        "idx_to_class": train_ds.meta.idx_to_class,
        "class_counts": train_ds.meta.class_counts.tolist(),
    }
    return train_loader, val_loader, meta


def compute_binary_metrics(y_true: np.ndarray, prob_pos: np.ndarray, threshold: float):
    y_pred = (prob_pos >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int):
    model.eval()
    all_probs = []
    all_preds = []
    all_y = []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        all_probs.append(probs.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_y.append(y.detach().cpu())

    probs = torch.cat(all_probs, dim=0).numpy()
    preds = torch.cat(all_preds, dim=0).numpy()
    y = torch.cat(all_y, dim=0).numpy()

    result = {
        "macro_f1": float(f1_score(y, preds, average="macro")),
        "per_class_f1": [],
    }
    for c in range(num_classes):
        f1c = f1_score((y == c).astype(int), (preds == c).astype(int), zero_division=0)
        result["per_class_f1"].append(float(f1c))

    if num_classes == 2:
        auc = roc_auc_score(y, probs[:, 1])
        fpr, tpr, th = roc_curve(y, probs[:, 1])
        youden = tpr - fpr
        best_i = int(np.argmax(youden))
        threshold = float(th[best_i])
        result["auroc"] = float(auc)
        result["threshold_youden"] = threshold
        result.update(compute_binary_metrics(y, probs[:, 1], threshold))
    else:
        auc = roc_auc_score(y, probs, average="macro", multi_class="ovr")
        result["auroc"] = float(auc)
        result["threshold_youden"] = None

    return result, y, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=str, required=True)
    ap.add_argument("--val-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--backbone", type=str, default="vit_base_patch16_224")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--stage1-epochs", type=int, default=10)
    ap.add_argument("--stage2-epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, meta = build_loaders(
        args.train_dir, args.val_dir, args.image_size, args.batch_size, args.num_workers
    )
    num_classes = len(meta["class_to_idx"])
    class_counts = torch.tensor(meta["class_counts"], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_vit_classifier(
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = AsymmetricLabelSmoothingCE(num_classes=num_classes)

    cfg = CurriculumConfig(
        num_classes=num_classes,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        total_epochs=args.epochs,
    )
    smoothing_vec = build_smoothing_vector(class_counts, cfg).to(device)
    per_class_f1 = torch.ones(num_classes, dtype=torch.float32, device=device)

    best_auroc = -1.0
    history = []

    for epoch in range(args.epochs):
        model.train()

        # Warmup + cosine.
        if epoch < args.warmup_epochs:
            lr = args.lr * (epoch + 1) / max(args.warmup_epochs, 1)
        else:
            progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
            cos = 0.5 * (1 + np.cos(np.pi * progress))
            lr = args.min_lr + (args.lr - args.min_lr) * cos
        for g in optimizer.param_groups:
            g["lr"] = lr

        class_weights = build_stage_weights(
            epoch=epoch,
            class_counts=class_counts.to(device),
            per_class_f1=per_class_f1,
            cfg=cfg,
        )

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        running_loss = 0.0
        for x, y, _ in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(
                logits=logits,
                target=y,
                class_weights=class_weights,
                smoothing_per_class=smoothing_vec,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics, y_true, y_pred = evaluate(model, val_loader, device, num_classes)
        per_class_f1 = per_class_f1_from_predictions(
            torch.tensor(y_true), torch.tensor(y_pred), num_classes
        ).to(device)

        epoch_log = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_loss,
            **val_metrics,
            **state_dict_for_logging(class_weights, smoothing_vec),
        }
        history.append(epoch_log)

        auroc = float(val_metrics["auroc"])
        if auroc > best_auroc:
            best_auroc = auroc
            ckpt = {
                "model_state_dict": model.state_dict(),
                "meta": meta,
                "args": vars(args),
                "best_val": val_metrics,
            }
            torch.save(ckpt, out_dir / "best.pt")

        print(
            f"Epoch {epoch + 1}: "
            f"loss={train_loss:.4f}, auroc={val_metrics['auroc']:.4f}, macro_f1={val_metrics['macro_f1']:.4f}"
        )

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Training complete. Best AUROC={best_auroc:.4f}")
    print(f"Saved: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
