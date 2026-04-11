from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def estimate_pupil_center(gray: np.ndarray) -> tuple[int, int]:
    # Pupil is often the darkest region. Use low-intensity centroid as a robust proxy.
    thresh = np.percentile(gray, 8)
    mask = gray <= thresh
    if mask.sum() < 20:
        h, w = gray.shape
        return w // 2, h // 2
    ys, xs = np.nonzero(mask)
    return int(xs.mean()), int(ys.mean())


def crop_iris_square(img: Image.Image, scale: float = 0.72) -> Image.Image:
    arr = np.array(img.convert("L"))
    h, w = arr.shape
    cx, cy = estimate_pupil_center(arr)
    side = int(min(h, w) * scale)
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    if (x2 - x1) != side:
        x1 = max(0, x2 - side)
    if (y2 - y1) != side:
        y1 = max(0, y2 - side)
    return img.crop((x1, y1, x2, y2))


def process_dir(src: Path, dst: Path, scale: float):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for p in src.rglob("*"):
        if p.suffix.lower() not in exts:
            continue
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(p).convert("RGB")
        cropped = crop_iris_square(img, scale=scale)
        cropped.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True, help="Source image root.")
    ap.add_argument("--dst", type=str, required=True, help="Output root.")
    ap.add_argument("--scale", type=float, default=0.72, help="Crop side ratio of min(H, W).")
    args = ap.parse_args()

    process_dir(Path(args.src), Path(args.dst), args.scale)
    print("Done.")


if __name__ == "__main__":
    main()
