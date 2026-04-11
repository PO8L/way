from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_FEATURES = [
    "kf_index",
    "angular_coverage_deg",
    "ring_width_mm",
    "pigment_density",
    "rec_percent",
    "urine_copper_24h",
    "ceruloplasmin",
    "hepatic_copper",
    "phenotype",
]


def _pick_existing(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def build_pipeline(df: pd.DataFrame, feature_cols: List[str]) -> Pipeline:
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
    )

    return Pipeline([("preprocess", pre), ("model", model)])


def evaluate_cv(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    group_col: str = "patient_id",
    n_splits: int = 5,
):
    df = df.copy()
    df = df.dropna(subset=[label_col])
    y = df[label_col].astype(int).values

    if group_col in df.columns:
        groups = df[group_col].astype(str).values
    else:
        groups = np.arange(len(df)).astype(str)
        n_splits = min(n_splits, 3)

    pipe = build_pipeline(df, features)
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))

    oof_prob = np.zeros(len(df), dtype=np.float64)
    for tr, te in gkf.split(df, y, groups):
        xtr = df.iloc[tr][features]
        xte = df.iloc[te][features]
        ytr = y[tr]
        pipe.fit(xtr, ytr)
        oof_prob[te] = pipe.predict_proba(xte)[:, 1]

    auc = roc_auc_score(y, oof_prob)
    ap = average_precision_score(y, oof_prob)
    brier = brier_score_loss(y, oof_prob)
    fpr, tpr, thr = roc_curve(y, oof_prob)
    youden = tpr - fpr
    i = int(np.argmax(youden))
    metric = {
        "n": int(len(df)),
        "positive_rate": float(y.mean()),
        "auroc": float(auc),
        "auprc": float(ap),
        "brier": float(brier),
        "youden_threshold": float(thr[i]),
        "sensitivity_at_youden": float(tpr[i]),
        "specificity_at_youden": float(1.0 - fpr[i]),
    }

    # fixed high-sensitivity point (screening-friendly)
    valid = np.where(tpr >= 0.9)[0]
    if len(valid) > 0:
        j = int(valid[np.argmin(fpr[valid])])
        metric["threshold_at_sens_ge_0.9"] = float(thr[j])
        metric["sensitivity_at_fixed"] = float(tpr[j])
        metric["specificity_at_fixed"] = float(1.0 - fpr[j])

    return metric, oof_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="CSV with image biomarkers + labs.")
    ap.add_argument("--label-col", type=str, default="high_copper_state", help="Binary target.")
    ap.add_argument("--output-json", type=str, required=True)
    ap.add_argument("--output-pred-csv", type=str, default=None)
    ap.add_argument("--features", type=str, default=",".join(DEFAULT_FEATURES))
    ap.add_argument("--group-col", type=str, default="patient_id")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    feature_cols = [x.strip() for x in args.features.split(",") if x.strip()]
    feature_cols = _pick_existing(df, feature_cols)
    if len(feature_cols) == 0:
        raise ValueError("No valid feature columns found in input CSV.")

    metric, oof_prob = evaluate_cv(
        df=df,
        features=feature_cols,
        label_col=args.label_col,
        group_col=args.group_col,
        n_splits=5,
    )
    out = {
        "label_col": args.label_col,
        "group_col": args.group_col,
        "features_used": feature_cols,
        "cv_metrics": metric,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics: {out_path}")

    if args.output_pred_csv:
        pred_df = df.copy()
        pred_df["risk_score_oof"] = oof_prob
        pred_path = Path(args.output_pred_csv)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
        print(f"Saved predictions: {pred_path}")


if __name__ == "__main__":
    main()
