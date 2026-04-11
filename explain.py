from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from data import build_eval_transform
from model import build_vit_classifier


def reshape_transform(tensor, height=14, width=14):
    # ViT token map reshape for CAM.
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--output", type=str, default="cam.png")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--target-class", type=int, default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    meta = ckpt["meta"]
    num_classes = len(meta["class_to_idx"])
    backbone = ckpt["args"]["backbone"]

    model = build_vit_classifier(num_classes=num_classes, backbone=backbone, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    img = Image.open(args.input).convert("RGB")
    x = build_eval_transform(args.image_size)(img).unsqueeze(0).to(device)
    rgb = np.array(img.resize((args.image_size, args.image_size))).astype(np.float32) / 255.0

    with torch.no_grad():
        pred = torch.softmax(model(x), dim=1)[0]
        pred_idx = int(torch.argmax(pred).item())
    target_idx = pred_idx if args.target_class is None else args.target_class

    # Last transformer block norm layer is a common CAM target for ViT.
    target_layers = [model.blocks[-1].norm1]
    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
        use_cuda=torch.cuda.is_available(),
    )
    grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(target_idx)])[0]
    vis = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
    Image.fromarray(vis).save(args.output)

    print(f"pred_class={pred_idx}, target_class={target_idx}")
    print(f"saved_cam={Path(args.output)}")


if __name__ == "__main__":
    main()
