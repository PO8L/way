from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr, linregress
from sklearn.metrics import roc_auc_score, roc_curve


def _safe_spearman(x: pd.Series, y: pd.Series) -> Dict:
    tmp = pd.concat([x, y], axis=1).dropna()
    if len(tmp) < 5:
        return {"n": int(len(tmp)), "rho": None, "p": None}
    rho, p = spearmanr(tmp.iloc[:, 0], tmp.iloc[:, 1])
    return {"n": int(len(tmp)), "rho": float(rho), "p": float(p)}


def biomarker_correlations(df: pd.DataFrame) -> Dict:
    out = {}
    if "kf_index" not in df.columns:
        raise ValueError("Column 'kf_index' is required.")

    targets = [
        "urine_copper_24h",
        "ceruloplasmin",
        "hepatic_copper",
    ]
    for t in targets:
        if t in df.columns:
            out[t] = _safe_spearman(df["kf_index"], df[t])
    return out


def phenotype_analysis(df: pd.DataFrame) -> Dict:
    if "phenotype" not in df.columns or "kf_index" not in df.columns:
        return {}
    tmp = df[["phenotype", "kf_index"]].dropna()
    groups = []
    group_data = {}
    for g, gdf in tmp.groupby("phenotype"):
        vals = gdf["kf_index"].values
        group_data[str(g)] = {
            "n": int(len(vals)),
            "median": float(np.median(vals)),
            "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
        }
        if len(vals) > 0:
            groups.append(vals)
    result = {"group_summary": group_data}
    if len(groups) >= 2:
        h, p = kruskal(*groups)
        result["kruskal"] = {"H": float(h), "p": float(p)}
    else:
        result["kruskal"] = None
    return result


def _patient_slope(gdf: pd.DataFrame, y_col: str) -> float | None:
    if y_col not in gdf.columns:
        return None
    sdf = gdf[["months_from_baseline", y_col]].dropna()
    if len(sdf) < 3:
        return None
    lr = linregress(sdf["months_from_baseline"].values, sdf[y_col].values)
    return float(lr.slope)


def longitudinal_response(df: pd.DataFrame) -> Dict:
    required = {"patient_id", "months_from_baseline", "kf_index"}
    if not required.issubset(set(df.columns)):
        return {}

    slopes = []
    for pid, gdf in df.groupby("patient_id"):
        kf_slope = _patient_slope(gdf, "kf_index")
        uc_slope = _patient_slope(gdf, "urine_copper_24h") if "urine_copper_24h" in df.columns else None
        cp_slope = _patient_slope(gdf, "ceruloplasmin") if "ceruloplasmin" in df.columns else None
        slopes.append(
            {
                "patient_id": pid,
                "kf_slope_per_month": kf_slope,
                "urine_copper_slope_per_month": uc_slope,
                "ceruloplasmin_slope_per_month": cp_slope,
            }
        )
    slope_df = pd.DataFrame(slopes)

    out = {"n_patients": int(slope_df["patient_id"].nunique())}
    if "kf_slope_per_month" in slope_df.columns:
        kf_vals = slope_df["kf_slope_per_month"].dropna().values
        if len(kf_vals) > 0:
            out["kf_slope_summary"] = {
                "n": int(len(kf_vals)),
                "median": float(np.median(kf_vals)),
                "iqr": float(np.percentile(kf_vals, 75) - np.percentile(kf_vals, 25)),
            }
    if {"kf_slope_per_month", "urine_copper_slope_per_month"}.issubset(set(slope_df.columns)):
        tmp = slope_df[["kf_slope_per_month", "urine_copper_slope_per_month"]].dropna()
        if len(tmp) >= 5:
            r, p = spearmanr(tmp.iloc[:, 0], tmp.iloc[:, 1])
            out["slope_correlation_kf_vs_urine_copper"] = {"n": int(len(tmp)), "rho": float(r), "p": float(p)}

    return out


def early_screening_threshold(df: pd.DataFrame, min_sensitivity: float = 0.9) -> Dict:
    required = {"label_wilson", "kf_index"}
    if not required.issubset(set(df.columns)):
        return {}
    tmp = df[["label_wilson", "kf_index"]].dropna()
    if len(tmp) < 20:
        return {}

    y = tmp["label_wilson"].astype(int).values
    s = tmp["kf_index"].astype(float).values
    auc = roc_auc_score(y, s)
    fpr, tpr, thr = roc_curve(y, s)
    valid = np.where(tpr >= min_sensitivity)[0]
    if len(valid) == 0:
        idx = int(np.argmax(tpr - fpr))
    else:
        # maximize specificity under sensitivity constraint
        idx = int(valid[np.argmin(fpr[valid])])
    threshold = float(thr[idx])
    sensitivity = float(tpr[idx])
    specificity = float(1.0 - fpr[idx])
    return {
        "n": int(len(tmp)),
        "auroc": float(auc),
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "min_sensitivity_constraint": float(min_sensitivity),
    }


def generate_report(df: pd.DataFrame, min_sensitivity: float = 0.9) -> Dict:
    return {
        "correlation_with_copper_markers": biomarker_correlations(df),
        "phenotype_association": phenotype_analysis(df),
        "longitudinal_response": longitudinal_response(df),
        "early_screening_threshold": early_screening_threshold(df, min_sensitivity=min_sensitivity),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Clinical CSV with kf_index and markers.")
    ap.add_argument("--output-json", type=str, required=True)
    ap.add_argument("--min-sensitivity", type=float, default=0.9)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    report = generate_report(df, min_sensitivity=args.min_sensitivity)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
