from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class KFMetricConfig:
    num_angles: int = 360
    num_radial: int = 128
    ring_search_start_ratio: float = 0.55  # search from mid-periphery to limbus
    min_angle_positive_px: int = 2
    # KF Index weights
    w_width: float = 0.35
    w_coverage: float = 0.40
    w_density: float = 0.20
    w_texture: float = 0.05


@dataclass
class KFMetrics:
    ring_width_px: float
    ring_width_mm: Optional[float]
    ring_width_ratio: float
    angular_coverage_ratio: float
    angular_coverage_deg: float
    pigment_density: float
    texture_entropy: float
    texture_variance: float
    kf_index: float
    grade: int

    def to_dict(self) -> Dict:
        return asdict(self)


def _centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        h, w = mask.shape
        return w / 2.0, h / 2.0
    return float(xs.mean()), float(ys.mean())


def _estimate_radius(mask: np.ndarray) -> float:
    area = float(mask.sum())
    if area <= 0:
        return 1.0
    return float(np.sqrt(area / np.pi))


def _sample_polar(
    arr: np.ndarray,
    center_x: float,
    center_y: float,
    r_min: float,
    r_max: float,
    num_angles: int,
    num_radial: int,
) -> np.ndarray:
    # Output shape: [num_angles, num_radial]
    thetas = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    rs = np.linspace(r_min, r_max, num_radial)
    rr, tt = np.meshgrid(rs, thetas)
    xs = center_x + rr * np.cos(tt)
    ys = center_y + rr * np.sin(tt)

    h, w = arr.shape[:2]
    xi = np.clip(np.rint(xs).astype(np.int32), 0, w - 1)
    yi = np.clip(np.rint(ys).astype(np.int32), 0, h - 1)
    return arr[yi, xi]


def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    # rgb float [0,1]
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _rgb_saturation(rgb: np.ndarray) -> np.ndarray:
    cmax = rgb.max(axis=-1)
    cmin = rgb.min(axis=-1)
    delta = cmax - cmin
    sat = np.where(cmax > 1e-6, delta / (cmax + 1e-6), 0.0)
    return sat


def _entropy_16bins(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=16, range=(0, 1), density=True)
    hist = hist + 1e-9
    ent = -np.sum(hist * np.log(hist))
    ent_max = np.log(16.0)
    return float(ent / ent_max)


def grade_from_coverage(coverage_deg: float) -> int:
    if coverage_deg < 5:
        return 0
    if coverage_deg < 90:
        return 1
    if coverage_deg < 180:
        return 2
    if coverage_deg < 330:
        return 3
    return 4


def compute_kf_metrics(
    image_rgb_uint8: np.ndarray,
    kf_mask: np.ndarray,
    iris_mask: np.ndarray,
    pupil_mask: np.ndarray,
    px_per_mm: Optional[float] = None,
    cfg: KFMetricConfig = KFMetricConfig(),
) -> KFMetrics:
    """
    Compute KF ring quantitative biomarkers from image + segmentation masks.
    mask arrays can be bool or {0,1}.
    """
    rgb = np.asarray(image_rgb_uint8, dtype=np.float32) / 255.0
    kf = np.asarray(kf_mask > 0, dtype=np.uint8)
    iris = np.asarray(iris_mask > 0, dtype=np.uint8)
    pupil = np.asarray(pupil_mask > 0, dtype=np.uint8)

    cx, cy = _centroid(pupil if pupil.sum() > 20 else iris)
    r_pupil = _estimate_radius(pupil)
    r_iris = _estimate_radius(iris)
    r_start = r_pupil + (r_iris - r_pupil) * cfg.ring_search_start_ratio
    r_end = r_iris

    polar_kf = _sample_polar(
        kf, cx, cy, r_start, r_end, cfg.num_angles, cfg.num_radial
    )  # [A, R] binary
    polar_kf = (polar_kf > 0).astype(np.uint8)

    positive_angles = polar_kf.sum(axis=1) >= cfg.min_angle_positive_px
    coverage_ratio = float(positive_angles.mean())
    coverage_deg = coverage_ratio * 360.0

    dr = (r_end - r_start) / max(cfg.num_radial - 1, 1)
    widths = []
    for a in range(cfg.num_angles):
        row = polar_kf[a]
        if row.sum() == 0:
            continue
        idx = np.where(row > 0)[0]
        width_px = (idx.max() - idx.min() + 1) * dr
        widths.append(width_px)
    ring_width_px = float(np.median(widths)) if widths else 0.0
    ring_width_ratio = float(ring_width_px / max(r_iris, 1e-6))
    ring_width_mm = None if px_per_mm is None else float(ring_width_px / max(px_per_mm, 1e-6))

    # Pigment and texture from ring region.
    ring_region = (kf > 0)
    if ring_region.sum() < 20:
        pigment_density = 0.0
        texture_entropy = 0.0
        texture_variance = 0.0
    else:
        gray = _rgb_to_gray(rgb)
        sat = _rgb_saturation(rgb)
        gray_ring = gray[ring_region]
        sat_ring = sat[ring_region]
        # Higher for darker + more saturated deposits.
        darkness = 1.0 - float(gray_ring.mean())
        chroma = float(sat_ring.mean())
        pigment_density = float(np.clip(0.7 * darkness + 0.3 * chroma, 0.0, 1.0))
        texture_entropy = _entropy_16bins(gray_ring)
        texture_variance = float(np.var(gray_ring))

    kf_index = (
        cfg.w_width * np.clip(ring_width_ratio / 0.15, 0.0, 1.0)
        + cfg.w_coverage * coverage_ratio
        + cfg.w_density * pigment_density
        + cfg.w_texture * np.clip(texture_entropy, 0.0, 1.0)
    )
    kf_index = float(np.clip(kf_index, 0.0, 1.0))
    grade = grade_from_coverage(coverage_deg)

    return KFMetrics(
        ring_width_px=ring_width_px,
        ring_width_mm=ring_width_mm,
        ring_width_ratio=ring_width_ratio,
        angular_coverage_ratio=coverage_ratio,
        angular_coverage_deg=coverage_deg,
        pigment_density=pigment_density,
        texture_entropy=texture_entropy,
        texture_variance=texture_variance,
        kf_index=kf_index,
        grade=grade,
    )
