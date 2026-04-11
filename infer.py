from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from data import build_eval_transform
from model import build_vit_classifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--image-size", type=int, default=224)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    meta = ckpt["meta"]
    num_classes = len(meta["class_to_idx"])
    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()} if isinstance(next(iter(meta["idx_to_class"].keys())), str) else meta["idx_to_class"]
    backbone = ckpt["args"]["backbone"]

    model = build_vit_classifier(num_classes=num_classes, backbone=backbone, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    img = Image.open(args.input).convert("RGB")
    x = build_eval_transform(args.image_size)(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0].cpu()

    pred_idx = int(torch.argmax(prob).item())
    print(f"input: {Path(args.input)}")
    print(f"pred: {idx_to_class[pred_idx]}")
    for i in range(num_classes):
        print(f"{idx_to_class[i]}: {float(prob[i]):.6f}")


if __name__ == "__main__":
    main()
