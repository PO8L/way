#!/usr/bin/env python
"""Full-flow standalone entrypoint for the best closed-book EEG emotion model.

Default model, chosen as the conservative closed-book submission candidate:
    uncertainty_medium_w20
    + 5% native_integrated_temperature
    + temperature calibration

Aggressive alternative:
    same blend
    + beta calibration

The default is intentionally conservative. Nested meta-CV does not prove that
the 5% native branch improves balanced accuracy; it is kept only as a tiny
regularizing/calibration blend because it marginally lowers OOF log loss.

The 75% reference spreadsheet is not loaded by this pipeline. Keep that file in
a separate diagnostic script after all model decisions are frozen.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm

    def progress(iterable, *, total: int | None = None, desc: str = ""):
        return tqdm(iterable, total=total, desc=desc, ncols=100)

except Exception:

    def progress(iterable, *, total: int | None = None, desc: str = ""):
        return iterable

from domain_shift_calibration_study import CALIBRATORS
from emotion_transfer_algorithms import RANDOM_SEED, metrics_from_predictions, topk_subject_predictions, write_json
from hybrid_seed_nb_w25_corrected_model import resolve_public_test_dir
from native_integrated_main_upgrade import public_predict as run_native_public
from native_integrated_main_upgrade import run_loso as run_native_loso
from video_source_generalization_study import resolve_competition_train_dir


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "best_closed_book_eeg_model"
DEFAULT_NATIVE_WEIGHT = 0.05
WEIGHT_GRID: Sequence[float] = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
CALIBRATOR_GRID: Sequence[str] = ("temperature", "beta")

VARIANTS = {
    "stable_temperature": {
        "final_calibrator": "temperature",
        "description": "safer closed-book default; same BA as medium baseline, best loss among fixed low-weight blends",
    },
    "aggressive_beta": {
        "final_calibrator": "beta",
        "description": "higher OOF BA, but more exposed to calibration/model-selection variance",
    },
}

DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / MODEL_NAME
DEFAULT_SOURCE_DIR = PROJECT_DIR / "public_video_sources"
DEFAULT_NATIVE_OUTPUT_DIR = PROJECT_DIR / "outputs" / "native_integrated_main_upgrade"
DEFAULT_DOMAIN_OUTPUT_DIR = PROJECT_DIR / "outputs" / "domain_shift_calibration_study"
DEFAULT_UNCERTAINTY_OUTPUT_DIR = PROJECT_DIR / "outputs" / "uncertainty_corrector_study"
DEFAULT_BASE_PUBLIC = PROJECT_DIR / "outputs" / "hybrid_seed_nb_w25_submission" / "public_test_predictions.with_prob.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def metric_block(y: np.ndarray, prob: np.ndarray, subjects: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    out = metrics_from_predictions(y, pred, prob)
    top4 = topk_subject_predictions(subjects, prob, k=4)
    out["top4_balanced_accuracy"] = float(metrics_from_predictions(y, top4, prob)["balanced_accuracy"])
    return out


def require_file(path: Path, purpose: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}")
    return path


def run_script(script: str, args: Sequence[str], cwd: Path) -> None:
    cmd = [sys.executable, str(PROJECT_DIR / script), *args]
    subprocess.run(cmd, cwd=str(cwd), check=True)


def frozen_native_args(args: argparse.Namespace) -> argparse.Namespace:
    # Intentionally frozen: these hyperparameters match the final closed-book submission.
    # Do not auto-sync with native_integrated_main_upgrade defaults.
    return argparse.Namespace(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        source_dir=args.source_dir,
        output_dir=args.native_output_dir,
        reference_75=None,
        seed=args.seed,
        native_k=240,
        native_components=32,
        shared_components=12,
        shared_c=1.0,
        robust_v2_components=64,
        robust_v2_c=1.0,
        population_components=8,
        population_c=0.1,
        robust_common_weight=0.75,
        robust_group_weight=0.25,
        corrector_c=0.2,
    )


def ensure_prerequisites(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.domain_output_dir.mkdir(parents=True, exist_ok=True)
    args.uncertainty_output_dir.mkdir(parents=True, exist_ok=True)
    args.native_output_dir.mkdir(parents=True, exist_ok=True)

    domain_oof = args.domain_output_dir / "oof_predictions.csv"
    if args.refresh_prerequisites or not domain_oof.exists():
        run_script(
            "domain_shift_calibration_study.py",
            [
                "--train-dir",
                str(args.train_dir),
                "--source-dir",
                str(args.source_dir),
                "--output-dir",
                str(args.domain_output_dir),
                "--no-reference-75",
                "--seed",
                str(args.seed),
            ],
            PROJECT_DIR,
        )

    medium_public = args.uncertainty_output_dir / "uncertainty_medium_w20_public_test.with_prob.csv"
    if args.refresh_prerequisites or not medium_public.exists():
        run_script(
            "uncertainty_corrector_study.py",
            [
                "--train-dir",
                str(args.train_dir),
                "--source-dir",
                str(args.source_dir),
                "--output-dir",
                str(args.uncertainty_output_dir),
                "--no-reference-75",
                "--seed",
                str(args.seed),
            ],
            PROJECT_DIR,
        )

    native_oof = args.native_output_dir / "native_integrated_oof_predictions.csv"
    native_public_raw = args.native_output_dir / "native_integrated_hybrid_public_test.with_prob.csv"
    native_public_temperature = args.native_output_dir / "native_integrated_hybrid_temperature_public_test.with_prob.csv"
    if args.refresh_prerequisites or not native_oof.exists() or not native_public_raw.exists() or not native_public_temperature.exists():
        n_args = frozen_native_args(args)
        loso = run_native_loso(n_args)
        public = run_native_public(n_args, "native_integrated_hybrid", "temperature")
        write_json(
            args.native_output_dir / "native_integrated_refresh_for_best_closed_book.json",
            {
                "selected_calibration": {
                    "prob_col": "native_integrated_hybrid",
                    "method": "temperature",
                    "stats": loso["calibration"]["native_integrated_hybrid"]["temperature"],
                },
                "public_candidate": public,
            },
        )


def build_oof_frame(args: argparse.Namespace) -> pd.DataFrame:
    domain_oof = require_file(args.domain_output_dir / "oof_predictions.csv", "domain/medium OOF predictions")
    native_oof = require_file(args.native_output_dir / "native_integrated_oof_predictions.csv", "native OOF predictions")
    domain = pd.read_csv(domain_oof)
    native = pd.read_csv(native_oof)
    required_domain = {"subject", "trial_id", "y", "medium_prob"}
    required_native = {"subject", "trial_id", "y", "native_integrated_hybrid"}
    missing_domain = required_domain - set(domain.columns)
    missing_native = required_native - set(native.columns)
    if missing_domain:
        raise RuntimeError(f"domain OOF missing columns: {sorted(missing_domain)}")
    if missing_native:
        raise RuntimeError(f"native OOF missing columns: {sorted(missing_native)}")
    frame = domain.merge(
        native[["subject", "trial_id", "y", "native_integrated_hybrid"]],
        on=["subject", "trial_id", "y"],
        how="inner",
    )
    if len(frame) != len(domain) or len(frame) != len(native):
        raise RuntimeError(f"OOF merge mismatch: domain={len(domain)}, native={len(native)}, merged={len(frame)}")
    if frame.duplicated(["subject", "trial_id"]).any():
        raise RuntimeError("Duplicate (subject, trial_id) keys detected after OOF merge -- check for duplicate rows in input CSVs.")
    return frame


def subject_crossfit(prob: np.ndarray, y: np.ndarray, subjects: np.ndarray, calibrator: str, desc: str) -> np.ndarray:
    out = np.zeros_like(prob, dtype=float)
    unique_subjects = sorted(set(subjects.tolist()))
    for sid in progress(unique_subjects, total=len(unique_subjects), desc=desc):
        val = subjects == sid
        train = ~val
        model = CALIBRATORS[calibrator]().fit(prob[train], y[train])
        out[val] = model.predict(prob[val])
    return out


def subject_folds(subjects: np.ndarray, n_folds: int, seed: int) -> Sequence[tuple[np.ndarray, np.ndarray]]:
    unique = np.asarray(sorted(set(map(str, subjects.tolist()))))
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    parts = np.array_split(shuffled, min(n_folds, len(shuffled)))
    folds = []
    for part in parts:
        val_subjects = set(map(str, part.tolist()))
        val = np.asarray([str(s) in val_subjects for s in subjects])
        folds.append((~val, val))
    return folds


def crossfit_by_subject_folds(prob: np.ndarray, y: np.ndarray, subjects: np.ndarray, calibrator: str, n_folds: int, seed: int) -> np.ndarray:
    out = np.zeros_like(prob, dtype=float)
    for train, val in subject_folds(subjects, n_folds=n_folds, seed=seed):
        model = CALIBRATORS[calibrator]().fit(prob[train], y[train])
        out[val] = model.predict(prob[val])
    return out


def validation_probabilities(args: argparse.Namespace, frame: pd.DataFrame, final_calibrator: str) -> pd.DataFrame:
    y = frame["y"].to_numpy(dtype=int)
    subjects = frame["subject"].to_numpy(dtype=str)
    medium = frame["medium_prob"].to_numpy(dtype=float)
    native = frame["native_integrated_hybrid"].to_numpy(dtype=float)
    native_temp = subject_crossfit(native, y, subjects, "temperature", "native temp OOF")
    blend_raw = (1.0 - args.native_weight) * medium + args.native_weight * native_temp
    final = subject_crossfit(blend_raw, y, subjects, final_calibrator, f"{final_calibrator} OOF")
    out = frame[["subject", "trial_id", "y", "medium_prob", "native_integrated_hybrid"]].copy()
    out["native_temperature_oof"] = native_temp
    out["blend_raw_prob"] = blend_raw
    out["prob"] = final
    out["Emotion_label"] = (final >= 0.5).astype(int)
    return out


def validate(args: argparse.Namespace) -> dict[str, object]:
    ensure_prerequisites(args)
    final_calibrator = VARIANTS[args.variant]["final_calibrator"]
    frame = build_oof_frame(args)
    y = frame["y"].to_numpy(dtype=int)
    subjects = frame["subject"].to_numpy(dtype=str)
    val = validation_probabilities(args, frame, final_calibrator)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    val.to_csv(args.output_dir / f"{MODEL_NAME}_{args.variant}_oof_predictions.csv", index=False, encoding="utf-8-sig")

    medium = val["medium_prob"].to_numpy(dtype=float)
    blend = val["blend_raw_prob"].to_numpy(dtype=float)
    final = val["prob"].to_numpy(dtype=float)
    return {
        "variant": args.variant,
        "description": VARIANTS[args.variant]["description"],
        "formula": f"{1.0 - args.native_weight:.3f} * uncertainty_medium_w20 + {args.native_weight:.3f} * native_integrated_temperature + {final_calibrator} calibration",
        "metrics": {
            "medium_prob_baseline": metric_block(y, medium, subjects),
            "blend_raw": metric_block(y, blend, subjects),
            args.variant: metric_block(y, final, subjects),
        },
    }


def meta_predict_outer(
    medium_train: np.ndarray,
    native_train: np.ndarray,
    y_train: np.ndarray,
    subjects_train: np.ndarray,
    medium_val: np.ndarray,
    native_val: np.ndarray,
    weight: float,
    final_calibrator: str,
    seed: int,
) -> np.ndarray:
    native_temp_train = crossfit_by_subject_folds(native_train, y_train, subjects_train, "temperature", n_folds=5, seed=seed)
    blend_train = (1.0 - weight) * medium_train + weight * native_temp_train
    native_model = CALIBRATORS["temperature"]().fit(native_train, y_train)
    native_temp_val = native_model.predict(native_val)
    blend_val = (1.0 - weight) * medium_val + weight * native_temp_val
    final_model = CALIBRATORS[final_calibrator]().fit(blend_train, y_train)
    return final_model.predict(blend_val)


def choose_meta_on_train(
    medium: np.ndarray,
    native: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    seed: int,
) -> dict[str, object]:
    rows = []
    for weight in WEIGHT_GRID:
        for calibrator in CALIBRATOR_GRID:
            pred = np.zeros_like(y, dtype=float)
            for fold_idx, (inner_train, inner_val) in enumerate(subject_folds(subjects, n_folds=5, seed=seed)):
                pred[inner_val] = meta_predict_outer(
                    medium[inner_train],
                    native[inner_train],
                    y[inner_train],
                    subjects[inner_train],
                    medium[inner_val],
                    native[inner_val],
                    weight,
                    calibrator,
                    seed + 100 + fold_idx,
                )
            stats = metric_block(y, pred, subjects)
            rows.append({"weight": float(weight), "calibrator": calibrator, "stats": stats})
    rows.sort(key=lambda row: (-row["stats"]["balanced_accuracy"], row["stats"]["log_loss"]))
    return rows[0]


def nested_weight_cv(args: argparse.Namespace) -> dict[str, object]:
    ensure_prerequisites(args)
    frame = build_oof_frame(args)
    y = frame["y"].to_numpy(dtype=int)
    subjects = frame["subject"].to_numpy(dtype=str)
    medium = frame["medium_prob"].to_numpy(dtype=float)
    native = frame["native_integrated_hybrid"].to_numpy(dtype=float)
    outputs = {
        "nested_selected": np.zeros_like(y, dtype=float),
        "fixed_weight0_temperature": np.zeros_like(y, dtype=float),
        "fixed_005_temperature": np.zeros_like(y, dtype=float),
        "fixed_005_beta": np.zeros_like(y, dtype=float),
    }
    fold_rows = []
    for fold_idx, (outer_train, outer_val) in enumerate(progress(subject_folds(subjects, n_folds=6, seed=args.seed), total=6, desc="6-fold nested weight"), start=1):
        choice = choose_meta_on_train(medium[outer_train], native[outer_train], y[outer_train], subjects[outer_train], args.seed + fold_idx)
        candidates = {
            "nested_selected": (float(choice["weight"]), str(choice["calibrator"])),
            "fixed_weight0_temperature": (0.0, "temperature"),
            "fixed_005_temperature": (0.05, "temperature"),
            "fixed_005_beta": (0.05, "beta"),
        }
        for name, (weight, calibrator) in candidates.items():
            outputs[name][outer_val] = meta_predict_outer(
                medium[outer_train],
                native[outer_train],
                y[outer_train],
                subjects[outer_train],
                medium[outer_val],
                native[outer_val],
                weight,
                calibrator,
                args.seed + 1000 + fold_idx,
            )
        fold_rows.append(
            {
                "fold": fold_idx,
                "val_subjects": sorted(set(subjects[outer_val].tolist())),
                "selected_weight": float(choice["weight"]),
                "selected_calibrator": str(choice["calibrator"]),
                "inner_stats": choice["stats"],
            }
        )
    out = {
        "baseline_medium": metric_block(y, medium, subjects),
        "folds": fold_rows,
        "selection_counts": {
            "weights": {str(k): int(v) for k, v in pd.Series([r["selected_weight"] for r in fold_rows]).value_counts().sort_index().items()},
            "calibrators": {str(k): int(v) for k, v in pd.Series([r["selected_calibrator"] for r in fold_rows]).value_counts().sort_index().items()},
        },
    }
    for name, prob in outputs.items():
        out[name] = metric_block(y, prob, subjects)
    pd.DataFrame(
        {
            "subject": subjects,
            "trial_id": frame["trial_id"].to_numpy(),
            "y": y,
            "medium_prob": medium,
            **outputs,
        }
    ).to_csv(args.output_dir / f"{MODEL_NAME}_nested_weight_oof_predictions.csv", index=False, encoding="utf-8-sig")
    return out


def fit_public_final_prob(
    args: argparse.Namespace,
    medium_public_prob: np.ndarray,
    native_public_raw_prob: np.ndarray,
    final_calibrator: str,
) -> np.ndarray:
    frame = build_oof_frame(args)
    y = frame["y"].to_numpy(dtype=int)
    subjects = frame["subject"].to_numpy(dtype=str)
    medium = frame["medium_prob"].to_numpy(dtype=float)
    native = frame["native_integrated_hybrid"].to_numpy(dtype=float)
    native_temp_oof = subject_crossfit(native, y, subjects, "temperature", "native temp train OOF")
    blend_train_oof = (1.0 - args.native_weight) * medium + args.native_weight * native_temp_oof

    public_fold_probs = []
    unique_subjects = sorted(set(subjects.tolist()))
    for sid in progress(unique_subjects, total=len(unique_subjects), desc="public LOSO calibrator ensemble"):
        train = subjects != sid
        native_model = CALIBRATORS["temperature"]().fit(native[train], y[train])
        native_public_temp = native_model.predict(native_public_raw_prob)
        blend_public = (1.0 - args.native_weight) * medium_public_prob + args.native_weight * native_public_temp
        final_model = CALIBRATORS[final_calibrator]().fit(blend_train_oof[train], y[train])
        public_fold_probs.append(final_model.predict(blend_public))
    return np.mean(np.vstack(public_fold_probs), axis=0)


def public_sort_key(user_id: str) -> int:
    # Natural numeric order: P_test1 < P_test2 < ... < P_test10
    return int(str(user_id).replace("P_test", ""))


def predict(args: argparse.Namespace) -> dict[str, object]:
    ensure_prerequisites(args)
    final_calibrator = VARIANTS[args.variant]["final_calibrator"]
    medium_path = require_file(args.uncertainty_output_dir / "uncertainty_medium_w20_public_test.with_prob.csv", "medium public predictions")
    native_path = require_file(args.native_output_dir / "native_integrated_hybrid_public_test.with_prob.csv", "native public predictions")
    base_path = require_file(args.base_public, "base public predictions")

    medium = pd.read_csv(medium_path).rename(columns={"Emotion_label": "medium_label", "prob": "medium_prob"})
    native = pd.read_csv(native_path)
    native_col = "prob"
    public = medium.merge(native[["user_id", "trial_id", native_col]], on=["user_id", "trial_id"], how="left")
    if public[native_col].isna().any():
        raise RuntimeError("Native public probabilities did not align with medium public predictions.")

    medium_public_prob = public["medium_prob"].to_numpy(dtype=float)
    native_public_raw_prob = public[native_col].to_numpy(dtype=float)
    blend_raw = (1.0 - args.native_weight) * medium_public_prob + args.native_weight * native_public_raw_prob
    final_prob = fit_public_final_prob(args, medium_public_prob, native_public_raw_prob, final_calibrator)

    out = public[["user_id", "trial_id"]].copy()
    out["Emotion_label"] = (final_prob >= 0.5).astype(int)
    out["prob"] = final_prob
    out["blend_raw_prob"] = blend_raw
    out["medium_prob"] = public["medium_prob"].to_numpy(dtype=float)
    out["native_raw_prob"] = native_public_raw_prob

    base = pd.read_csv(base_path).rename(columns={"Emotion_label": "base_label", "prob": "base_prob"})
    n_before = len(out)
    out = out.merge(base[["user_id", "trial_id", "base_label", "base_prob"]], on=["user_id", "trial_id"], how="left")
    if len(out) != n_before:
        raise RuntimeError(f"Base merge changed row count: {n_before} -> {len(out)}. Check for duplicate (user_id, trial_id) in base file.")
    if out[["base_label", "base_prob"]].isna().any(axis=None):
        missing = out[out["base_label"].isna()][["user_id", "trial_id"]].to_dict(orient="records")
        raise RuntimeError(f"Base merge produced NaN -- {len(missing)} rows unmatched: {missing[:5]}")
    out["changed_vs_base"] = out["Emotion_label"] != out["base_label"]

    # Sort by natural numeric user order, then trial_id, to match expected submission row order.
    out["_user_order"] = out["user_id"].map(public_sort_key)
    out = out.sort_values(["_user_order", "trial_id"]).drop(columns=["_user_order"]).reset_index(drop=True)

    xlsx = args.output_dir / f"{MODEL_NAME}_{args.variant}_public_test.xlsx"
    csv = args.output_dir / f"{MODEL_NAME}_{args.variant}_public_test.with_prob.csv"
    out[["user_id", "trial_id", "Emotion_label"]].to_excel(xlsx, index=False)
    out.to_csv(csv, index=False, encoding="utf-8-sig")
    changed = out[out["changed_vs_base"]]
    return {
        "xlsx": str(xlsx),
        "csv": str(csv),
        "changed_vs_base_rows": int(out["changed_vs_base"].sum()),
        "changed_rows": changed[["user_id", "trial_id", "base_label", "Emotion_label", "prob", "base_prob"]].to_dict(orient="records"),
        "calibration_contract": "public probabilities are averaged over LOSO calibrator ensemble",
    }


def write_readme(report: dict[str, object], path: Path) -> None:
    lines = [
        "# Best Closed-Book EEG Model",
        "",
        "Default recommendation: `stable_temperature`.",
        "",
        "Pipeline:",
        "1. Run/consume `uncertainty_medium_w20` as the main medium-probability model.",
        "2. Run/consume `native_integrated_hybrid` built from EEG-native dynamic connectivity, temporal dynamics, and compact native PCs.",
        "3. Temperature-calibrate the native integrated signal.",
        f"4. Blend probabilities: `{1.0 - report['native_weight']:.3f} * medium + {report['native_weight']:.3f} * native_temperature`.",
        "5. Apply final calibration: `temperature` for stable default, `beta` for aggressive alternative.",
        "6. Public predictions use an LOSO calibrator ensemble instead of one full-fit calibrator.",
        "",
        "This main pipeline never loads the 75 percent reference label file.",
        "The `top4BA` metric is diagnostic: it forces four positive trials per subject, matching the known 4 neutral/4 positive training design; it is not the submitted prediction rule.",
        "",
    ]
    if "validation" in report:
        lines.append("## Validation")
        for name, stats in report["validation"]["metrics"].items():
            lines.append(
                f"- `{name}`: BA={stats['balanced_accuracy']:.4f}, acc={stats['accuracy']:.4f}, "
                f"loss={stats['log_loss']:.4f}, top4BA={stats['top4_balanced_accuracy']:.4f}"
            )
        lines.append("")
    if "variant_comparison" in report:
        lines.append("## Variant Comparison")
        for variant, detail in report["variant_comparison"].items():
            # detail["metrics"] keys are set by validate() using args.variant — must match outer key.
            stats = detail["metrics"][variant]
            lines.append(f"- `{variant}`: BA={stats['balanced_accuracy']:.4f}, loss={stats['log_loss']:.4f}")
        lines.append("")
    if "nested_weight_cv" in report:
        nested = report["nested_weight_cv"]
        lines.append("## 6-Fold Nested Weight Check")
        lines.append("Outer split uses six subject folds. Inner split selects native weight and final calibrator using only training subjects.")
        for name in ("baseline_medium", "nested_selected", "fixed_weight0_temperature", "fixed_005_temperature", "fixed_005_beta"):
            stats = nested[name]
            lines.append(f"- `{name}`: BA={stats['balanced_accuracy']:.4f}, loss={stats['log_loss']:.4f}")
        lines.append(f"- selected weights: `{nested['selection_counts']['weights']}`")
        lines.append(f"- selected calibrators: `{nested['selection_counts']['calibrators']}`")
        lines.append("")
    if "public_prediction" in report:
        pred = report["public_prediction"]
        lines.append("## Public Prediction")
        lines.append(
            f"- file: `{Path(pred['xlsx']).name}`, changed_vs_base={pred['changed_vs_base_rows']}, "
            f"contract={pred['calibration_contract']}"
        )
        for row in pred["changed_rows"]:
            lines.append(
                f"  - {row['user_id']} trial {int(row['trial_id'])}: "
                f"{int(row['base_label'])} -> {int(row['Emotion_label'])}, prob={row['prob']:.4f}, base={row['base_prob']:.4f}"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Best full-flow closed-book EEG model.")
    parser.add_argument("--mode", choices=("validate", "predict", "validate_predict", "compare_variants", "nested_weight"), default="validate_predict")
    parser.add_argument("--variant", choices=tuple(VARIANTS.keys()), default="stable_temperature")
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--domain-output-dir", type=Path, default=DEFAULT_DOMAIN_OUTPUT_DIR)
    parser.add_argument("--uncertainty-output-dir", type=Path, default=DEFAULT_UNCERTAINTY_OUTPUT_DIR)
    parser.add_argument("--native-output-dir", type=Path, default=DEFAULT_NATIVE_OUTPUT_DIR)
    parser.add_argument("--base-public", type=Path, default=DEFAULT_BASE_PUBLIC)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--native-weight", type=float, default=DEFAULT_NATIVE_WEIGHT)
    parser.add_argument("--skip-nested-check", action="store_true")
    parser.add_argument("--refresh-prerequisites", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.train_dir is None and args.mode in {"validate", "validate_predict", "compare_variants", "nested_weight", "predict"}:
        args.train_dir = resolve_competition_train_dir()
    if args.test_dir is None and args.mode in {"predict", "validate_predict"}:
        args.test_dir = resolve_public_test_dir(None)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "model_name": MODEL_NAME,
        "variant": args.variant,
        "native_weight": args.native_weight,
        "final_calibrator": VARIANTS[args.variant]["final_calibrator"],
        "reference_75_used_for_training": False,
        "reference_75_loaded_for_diagnostics": False,
    }
    if args.mode in {"validate", "validate_predict"}:
        report["validation"] = validate(args)
    if args.mode == "validate_predict" and not args.skip_nested_check:
        report["nested_weight_cv"] = nested_weight_cv(args)
    if args.mode == "compare_variants":
        comparison = {}
        original_variant = args.variant
        for variant in VARIANTS:
            args.variant = variant
            comparison[variant] = validate(args)
        args.variant = original_variant
        report["variant_comparison"] = comparison
    if args.mode == "nested_weight":
        report["nested_weight_cv"] = nested_weight_cv(args)
    if args.mode in {"predict", "validate_predict"}:
        report["public_prediction"] = predict(args)

    report_path = args.output_dir / f"{MODEL_NAME}_{args.variant}_report.json"
    readme_path = args.output_dir / f"{MODEL_NAME}_{args.variant}_README.md"
    write_json(report_path, report)
    write_readme(report, readme_path)
    print(json.dumps({"report": str(report_path), "readme": str(readme_path), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
