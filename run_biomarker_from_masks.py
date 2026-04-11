from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from kf_biomarker import KFMetricConfig, compute_kf_metrics


def _read_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _read_mask(path: str) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return (arr > 127).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True, help="CSV with image/mask paths.")
    ap.add_argument("--output", type=str, required=True, help="Output metrics CSV.")
    ap.add_argument("--default-px-per-mm", type=float, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    required = ["image_path", "kf_mask_path", "iris_mask_path", "pupil_mask_path"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column in manifest: {c}")

    cfg = KFMetricConfig()
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img = _read_rgb(str(row["image_path"]))
        kf_mask = _read_mask(str(row["kf_mask_path"]))
        iris_mask = _read_mask(str(row["iris_mask_path"]))
        pupil_mask = _read_mask(str(row["pupil_mask_path"]))
        px_per_mm = row["px_per_mm"] if "px_per_mm" in df.columns and pd.notna(row["px_per_mm"]) else args.default_px_per_mm

        metrics = compute_kf_metrics(
            image_rgb_uint8=img,
            kf_mask=kf_mask,
            iris_mask=iris_mask,
            pupil_mask=pupil_mask,
            px_per_mm=px_per_mm,
            cfg=cfg,
        ).to_dict()

        base = row.to_dict()
        base.update(metrics)
        rows.append(base)

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
