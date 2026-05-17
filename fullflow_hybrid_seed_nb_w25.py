#!/usr/bin/env python
"""
Independent full-flow entrypoint for the first 72.5 public-score model:
hybrid_seed_nb_w25.

Model structure:
1. robust_v2 in-domain backbone
2. SEED-only public video EEG helper branch
3. normal-bias fusion with helper weight = 0.25

This file is submission-safe: it can train, validate, predict, or
train_and_predict from one script, and public output is sorted as
P_test1..P_test10, trial 1..8.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm

    def progress(iterable, *, total: int | None = None, desc: str = "", leave: bool = True):
        return tqdm(iterable, total=total, desc=desc, leave=leave, ncols=100)

except Exception:

    def progress(iterable, *, total: int | None = None, desc: str = "", leave: bool = True):
        return iterable

from advanced_emotion_fusion import (
    build_robust_v2_dataset,
    build_test_robust_v2_features,
    robust_v2_trial_probabilities,
    train_robust_v2_bundle,
)
from emotion_transfer_algorithms import (
    RANDOM_SEED,
    EnhancedWindowDataset,
    aggregate_trial_probabilities,
    build_subject_summary_from_windows,
    subject_holdout_splits,
    topk_subject_predictions,
    write_json,
)
from robust_v2_generalization_study import build_target_raw_window_dataset
from video_source_generalization_study import (
    SharedFeatureDataset,
    VIDEO_SOURCES,
    TARGET_SHARED_BANDS,
    build_target_shared_dataset,
    candidate_metrics,
    differential_entropy_features,
    fit_video_branch,
    hybrid_probabilities,
    load_video_source_dataset,
    population_depressed_map,
    predict_video_branch,
    resolve_competition_train_dir as _legacy_resolve_train_dir,
    subset_video_source_dataset,
    summarize_channel_band_matrix,
)


SEED_ONLY: Sequence[str] = ("SEED",)
MODEL_NAME = "fullflow_hybrid_seed_nb_w25"
ORIGINAL_MODEL_NAME = "hybrid_seed_nb_w25"
VIDEO_WEIGHT = 0.25
VIDEO_MODE = "normal_bias"
TRAIN_DIR_LABEL = "\u8bad\u7ec3\u96c6"
TEST_DIR_LABEL = "\u516c\u5f00\u6d4b\u8bd5\u96c6"


@dataclass
class HybridSeedNBW25Bundle:
    robust_bundle: object
    video_bundle: object
    config: Dict[str, object]


def resolve_competition_dir(explicit: Optional[Path], target_label: str) -> Path:
    if explicit is not None:
        explicit = explicit.resolve()
        if not explicit.exists():
            raise FileNotFoundError(f"Path does not exist: {explicit}")
        return explicit

    try:
        if target_label == TRAIN_DIR_LABEL:
            return _legacy_resolve_train_dir()
    except Exception:
        pass

    search_roots = [
        Path(r"D:\软件\赛题四数据集及说明文档"),
        Path(r"D:\软件"),
        Path(r"D:\软件"),
        Path("D:/"),
        Path(r"C:\Users\lyg\Documents"),
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob(target_label):
            if path.is_dir():
                mat_count = sum(1 for _ in path.rglob("*.mat"))
                if mat_count > 0:
                    return path
    raise FileNotFoundError(f"Could not auto-locate {target_label}. Pass it explicitly.")


def public_user_order(user_id: str) -> int:
    match = re.fullmatch(r"P_test(\d+)", str(user_id))
    if not match:
        raise ValueError(f"Unexpected public user_id: {user_id!r}")
    return int(match.group(1))


def sort_public_submission(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_user_order"] = out["user_id"].map(public_user_order)
    out = out.sort_values(["_user_order", "trial_id"]).drop(columns=["_user_order"])
    return out.reset_index(drop=True)


def build_test_shared_dataset(test_dir: Path) -> SharedFeatureDataset:
    rows: List[np.ndarray] = []
    subject_ids: List[str] = []
    trial_ids: List[str] = []
    feature_names: Optional[List[str]] = None

    from emotion_transfer_algorithms import load_mat_auto, natural_key, split_test_recording

    test_files = sorted(test_dir.glob("*.mat"), key=natural_key)
    for mat_path in progress(test_files, total=len(test_files), desc="shared test features"):
        obj = load_mat_auto(mat_path)
        key = "test_eeg_c" if "test_eeg_c" in obj else next(iter(obj.keys()))
        user_id = mat_path.stem
        for trial_idx, segment in enumerate(split_test_recording(obj[key]), start=1):
            de = differential_entropy_features(segment)
            feat, names = summarize_channel_band_matrix(de)
            rows.append(feat)
            subject_ids.append(user_id)
            trial_ids.append(f"{user_id}_trial_{trial_idx}")
            if feature_names is None:
                feature_names = names

    if not rows or feature_names is None:
        raise RuntimeError(f"No usable public-test windows found in {test_dir}")

    return SharedFeatureDataset(
        x=np.vstack(rows).astype(np.float64),
        y=np.zeros(len(rows), dtype=np.int64),
        subject_ids=np.asarray(subject_ids),
        group_labels=np.zeros(len(rows), dtype=np.int64),
        trial_ids=np.asarray(trial_ids),
        feature_names=feature_names,
    )


def train_full_bundle(args: argparse.Namespace) -> HybridSeedNBW25Bundle:
    robust_dataset = build_robust_v2_dataset(args.train_dir)
    raw_dataset = build_target_raw_window_dataset(args.train_dir)
    shared_dataset = build_target_shared_dataset(raw_dataset)
    source_dataset = subset_video_source_dataset(load_video_source_dataset(args.source_dir), SEED_ONLY)

    robust_bundle = train_robust_v2_bundle(robust_dataset, args, seed=args.seed)
    video_bundle = fit_video_branch(
        shared_dataset,
        source_dataset,
        components=args.shared_components,
        c_value=args.shared_c,
        seed=args.seed + 1000,
    )
    return HybridSeedNBW25Bundle(
        robust_bundle=robust_bundle,
        video_bundle=video_bundle,
        config={
            "model_name": MODEL_NAME,
            "video_source_domains": list(SEED_ONLY),
            "video_weight": VIDEO_WEIGHT,
            "video_mode": VIDEO_MODE,
            "seed": args.seed,
            "shared_components": args.shared_components,
            "shared_c": args.shared_c,
            "robust_v2_components": args.robust_v2_components,
            "robust_v2_c": args.robust_v2_c,
            "population_components": args.population_components,
            "population_c": args.population_c,
            "robust_common_weight": args.robust_common_weight,
            "robust_group_weight": args.robust_group_weight,
        },
    )


def save_bundle(bundle: HybridSeedNBW25Bundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)


def load_bundle(path: Path) -> HybridSeedNBW25Bundle:
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, HybridSeedNBW25Bundle):
        raise TypeError(f"Unexpected bundle type in {path}")
    return bundle


def validate_bundle(args: argparse.Namespace) -> Dict[str, object]:
    robust_dataset = build_robust_v2_dataset(args.train_dir)
    raw_dataset = build_target_raw_window_dataset(args.train_dir)
    shared_dataset = build_target_shared_dataset(raw_dataset)
    source_dataset = subset_video_source_dataset(load_video_source_dataset(args.source_dir), SEED_ONLY)

    if args.validation_scheme == "loso":
        unique_subjects = sorted(set(map(str, robust_dataset.subject_ids)))
        splits = []
        sid_array = np.asarray(list(map(str, robust_dataset.subject_ids)))
        for subject_id in unique_subjects:
            val_mask = sid_array == subject_id
            train_mask = ~val_mask
            splits.append((train_mask, val_mask))
    else:
        splits = subject_holdout_splits(
            robust_dataset.subject_ids,
            robust_dataset.group_labels,
            n_folds=min(args.folds, len(np.unique(robust_dataset.subject_ids))),
            seed=args.seed,
        )

    fold_rows: List[Dict[str, object]] = []
    for fold_idx, (train_mask, val_mask) in enumerate(progress(splits, total=len(splits), desc="hybrid validate"), start=1):
        robust_train = EnhancedWindowDataset(
            x=robust_dataset.x[train_mask].copy(),
            y_emotion=robust_dataset.y_emotion[train_mask].copy(),
            subject_ids=robust_dataset.subject_ids[train_mask].copy(),
            group_labels=robust_dataset.group_labels[train_mask].copy(),
            trial_ids=robust_dataset.trial_ids[train_mask].copy(),
            feature_names=list(robust_dataset.feature_names),
        )
        shared_train = SharedFeatureDataset(
            x=shared_dataset.x[train_mask].copy(),
            y=shared_dataset.y[train_mask].copy(),
            subject_ids=shared_dataset.subject_ids[train_mask].copy(),
            group_labels=shared_dataset.group_labels[train_mask].copy(),
            trial_ids=shared_dataset.trial_ids[train_mask].copy(),
            feature_names=list(shared_dataset.feature_names),
        )

        robust_bundle = train_robust_v2_bundle(robust_train, args, seed=args.seed + fold_idx)
        video_bundle = fit_video_branch(
            shared_train,
            source_dataset,
            components=args.shared_components,
            c_value=args.shared_c,
            seed=args.seed + 1000 + fold_idx,
        )

        dep_prob_map = population_depressed_map(robust_bundle, robust_dataset, val_mask)
        trial_y_r, trial_prob_r, trial_subjects_r, _aux = robust_v2_trial_probabilities(
            robust_bundle,
            robust_dataset,
            val_mask,
            args,
        )
        trial_y_v, trial_prob_v, trial_subjects_v = predict_video_branch(video_bundle, shared_dataset, val_mask)
        if not np.array_equal(trial_y_r, trial_y_v) or not np.array_equal(trial_subjects_r, trial_subjects_v):
            raise RuntimeError("Validation trial ordering mismatch between robust and video branches.")

        trial_prob_h = hybrid_probabilities(
            trial_prob_r,
            trial_prob_v,
            trial_subjects_r,
            dep_prob_map,
            weight=VIDEO_WEIGHT,
            mode=VIDEO_MODE,
        )
        metrics = candidate_metrics(trial_y_r, trial_prob_h, trial_subjects_r)
        fold_rows.append(
            {
                "fold": fold_idx,
                "val_subjects": sorted(set(map(str, robust_dataset.subject_ids[val_mask]))),
                "metrics": metrics,
            }
        )

    summary = {
        "mean_accuracy": float(np.mean([row["metrics"]["accuracy"] for row in fold_rows])),
        "mean_balanced_accuracy": float(np.mean([row["metrics"]["balanced_accuracy"] for row in fold_rows])),
        "balanced_accuracy_std": float(np.std([row["metrics"]["balanced_accuracy"] for row in fold_rows])),
        "mean_log_loss": float(np.mean([row["metrics"]["log_loss"] for row in fold_rows])),
        "log_loss_std": float(np.std([row["metrics"]["log_loss"] for row in fold_rows])),
        "mean_top4_balanced_accuracy": float(np.mean([row["metrics"]["top4_balanced_accuracy"] for row in fold_rows])),
    }
    return {
        "model_name": MODEL_NAME,
        "validation_scheme": args.validation_scheme,
        "config": {
            "folds": len(fold_rows),
            "seed": args.seed,
            "video_weight": VIDEO_WEIGHT,
            "video_mode": VIDEO_MODE,
        },
        "summary": summary,
        "folds": fold_rows,
    }


def robust_test_trial_probabilities(bundle: HybridSeedNBW25Bundle, user_ids: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    from emotion_transfer_algorithms import center_features_by_subject

    x_centered = center_features_by_subject(x_test, user_ids)
    common_prob = bundle.robust_bundle.all_model.predict_proba(x_centered)[:, 1]
    normal_prob = bundle.robust_bundle.normal_model.predict_proba(x_centered)[:, 1]
    patient_prob = bundle.robust_bundle.depressed_model.predict_proba(x_centered)[:, 1]

    ordered_trial_ids: List[str] = []
    counters: Dict[str, int] = {}
    for uid in user_ids:
        counters[str(uid)] = counters.get(str(uid), 0) + 1
        ordered_trial_ids.append(f"{uid}_trial_{counters[str(uid)]}")
    ordered_trial_ids_arr = np.asarray(ordered_trial_ids)

    subject_x, _dummy_y, subject_names = build_subject_summary_from_windows(
        x_test,
        user_ids,
        np.zeros(len(user_ids), dtype=np.int64),
        ordered_trial_ids_arr,
    )
    depressed_prob = bundle.robust_bundle.population_model.predict_proba(subject_x)[:, 1]
    dep_map = {str(sid): float(prob) for sid, prob in zip(subject_names, depressed_prob)}
    dep_weight = np.asarray([dep_map[str(uid)] for uid in user_ids], dtype=np.float64)
    routed_prob = (1.0 - dep_weight) * normal_prob + dep_weight * patient_prob
    robust_prob = (
        bundle.config["robust_common_weight"] * common_prob
        + bundle.config["robust_group_weight"] * routed_prob
    ) / (bundle.config["robust_common_weight"] + bundle.config["robust_group_weight"])

    trial_y, trial_prob, trial_subjects, _trial_names = aggregate_trial_probabilities(
        np.zeros(len(user_ids), dtype=np.int64),
        robust_prob,
        user_ids,
        ordered_trial_ids_arr,
    )
    return trial_prob, trial_subjects, dep_map


def predict_public_test(args: argparse.Namespace) -> pd.DataFrame:
    bundle = load_bundle(args.model_in)
    robust_x_test, robust_user_ids = build_test_robust_v2_features(args.test_dir)
    shared_test = build_test_shared_dataset(args.test_dir)

    robust_trial_prob, robust_trial_subjects, dep_map = robust_test_trial_probabilities(
        bundle,
        robust_user_ids,
        robust_x_test,
    )
    shared_mask = np.ones(shared_test.x.shape[0], dtype=bool)
    _trial_y_v, video_trial_prob, video_trial_subjects = predict_video_branch(
        bundle.video_bundle,
        shared_test,
        shared_mask,
    )
    if not np.array_equal(robust_trial_subjects, video_trial_subjects):
        raise RuntimeError("Public-test trial ordering mismatch between robust and video branches.")

    hybrid_trial_prob = hybrid_probabilities(
        robust_trial_prob,
        video_trial_prob,
        robust_trial_subjects,
        dep_map,
        weight=VIDEO_WEIGHT,
        mode=VIDEO_MODE,
    )
    if args.use_top4_postprocess:
        labels = topk_subject_predictions(robust_trial_subjects, hybrid_trial_prob, k=4)
    else:
        labels = (hybrid_trial_prob >= 0.5).astype(int)

    rows: List[Dict[str, object]] = []
    per_user_counts: Dict[str, int] = {}
    for user_id, prob, label in zip(robust_trial_subjects, hybrid_trial_prob, labels):
        per_user_counts[str(user_id)] = per_user_counts.get(str(user_id), 0) + 1
        rows.append(
            {
                "user_id": str(user_id),
                "trial_id": per_user_counts[str(user_id)],
                "Emotion_label": int(label),
                "prob": float(prob),
            }
        )
    df = sort_public_submission(pd.DataFrame(rows))
    args.prediction_out.parent.mkdir(parents=True, exist_ok=True)
    df[["user_id", "trial_id", "Emotion_label"]].to_excel(args.prediction_out, index=False)
    df.to_csv(args.prediction_out.with_suffix(".with_prob.csv"), index=False, encoding="utf-8-sig")
    return df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone hybrid_seed_nb_w25 model entrypoint.")
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent.parent / "tmp" / "video_eeg_public")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs" / MODEL_NAME)
    parser.add_argument("--model-out", type=Path, default=None)
    parser.add_argument("--model-in", type=Path, default=None)
    parser.add_argument("--prediction-out", type=Path, default=None)
    parser.add_argument("--validation-out", type=Path, default=None)
    parser.add_argument("--mode", choices=("train", "validate", "predict", "train_and_predict"), default="train")
    parser.add_argument("--validation-scheme", choices=("kfold", "loso"), default="kfold")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--shared-components", type=int, default=12)
    parser.add_argument("--shared-c", type=float, default=1.0)
    parser.add_argument("--robust-v2-components", type=int, default=64)
    parser.add_argument("--robust-v2-c", type=float, default=1.0)
    parser.add_argument("--population-components", type=int, default=8)
    parser.add_argument("--population-c", type=float, default=0.1)
    parser.add_argument("--robust-common-weight", type=float, default=0.75)
    parser.add_argument("--robust-group-weight", type=float, default=0.25)
    parser.add_argument("--use-top4-postprocess", action="store_true")
    return parser


def finalize_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_out = args.model_out or (args.output_dir / f"{ORIGINAL_MODEL_NAME}.pkl")
    args.model_in = args.model_in or args.model_out
    args.prediction_out = args.prediction_out or (args.output_dir / "public_test_predictions_ordered.xlsx")
    args.validation_out = args.validation_out or (args.output_dir / "validation_report.json")

    if args.mode in {"train", "validate", "train_and_predict"}:
        args.train_dir = resolve_competition_dir(args.train_dir, TRAIN_DIR_LABEL)
    if args.mode in {"predict", "train_and_predict"}:
        args.test_dir = resolve_competition_dir(args.test_dir, TEST_DIR_LABEL)
    return args


def main() -> None:
    parser = build_arg_parser()
    args = finalize_paths(parser.parse_args())

    if args.mode == "train":
        bundle = train_full_bundle(args)
        save_bundle(bundle, args.model_out)
        print(json.dumps({"model_name": MODEL_NAME, "model_out": str(args.model_out)}, ensure_ascii=False, indent=2))
        return

    if args.mode == "validate":
        report = validate_bundle(args)
        write_json(args.validation_out, report)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return

    if args.mode == "predict":
        df = predict_public_test(args)
        print(json.dumps({"prediction_rows": int(len(df)), "prediction_out": str(args.prediction_out)}, ensure_ascii=False, indent=2))
        return

    bundle = train_full_bundle(args)
    save_bundle(bundle, args.model_out)
    df = predict_public_test(args)
    print(
        json.dumps(
            {
                "model_name": MODEL_NAME,
                "model_out": str(args.model_out),
                "prediction_rows": int(len(df)),
                "prediction_out": str(args.prediction_out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
