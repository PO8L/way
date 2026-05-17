#!/usr/bin/env python
"""Standalone inference for hybrid_seed_nb_w25 + conservative residual corrector.

The corrector is trained from out-of-fold training predictions only. The
public-test 75% reference spreadsheet is not used for training, tuning, or
model selection.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from advanced_emotion_fusion import build_robust_v2_dataset, build_test_robust_v2_features, train_robust_v2_bundle
from emotion_transfer_algorithms import (
    RANDOM_SEED,
    EnhancedWindowDataset,
    aggregate_trial_probabilities,
    build_subject_summary_from_windows,
    center_features_by_subject,
    write_json,
)
from hybrid_seed_nb_w25_model import (
    HybridSeedNBW25Bundle,
    TRAIN_DIR_LABEL,
    build_test_shared_dataset,
    load_bundle,
    resolve_competition_dir,
    save_bundle,
    train_full_bundle,
)
from p_test3_style_corrector_study import (
    BranchFrame,
    SEED_ONLY,
    VIDEO_MODE,
    VIDEO_WEIGHT,
    build_corrector_features,
    feature_family,
    fit_corrector,
    loso_splits,
)
from robust_v2_generalization_study import build_target_raw_window_dataset
from video_source_generalization_study import (
    SharedFeatureDataset,
    build_target_shared_dataset,
    fit_video_branch,
    hybrid_probabilities,
    load_video_source_dataset,
    predict_video_branch,
    subset_video_source_dataset,
)

try:
    from tqdm.auto import tqdm

    def progress(iterable: Iterable, *, total: int | None = None, desc: str = "", leave: bool = True):
        return tqdm(iterable, total=total, desc=desc, leave=leave, ncols=100)

except Exception:

    def progress(iterable: Iterable, *, total: int | None = None, desc: str = "", leave: bool = True):
        return iterable


MODEL_NAME = "hybrid_seed_nb_w25_corrected_conflict_t18_w10"
CORRECTOR_WEIGHT = 0.10
CONFLICT_THRESHOLD = 0.18
MARGIN_THRESHOLD = 0.25
SELECTED_FAMILIES: Sequence[str] = (
    "enhanced_other",
    "fine_logrel",
    "band_ratio",
    "logrel_gamma",
    "logrel_beta",
    "theta_logcov",
    "alpha_logcov",
    "beta_logcov",
    "gamma_logcov",
)


def resolve_public_test_dir(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        explicit = explicit.resolve()
        if not explicit.exists():
            raise FileNotFoundError(f"Path does not exist: {explicit}")
        return explicit

    for root in (Path(r"D:\软件"), Path("D:/"), Path(r"C:\Users\lyg")):
        if not root.exists():
            continue
        for match in root.rglob("P_test3.mat"):
            folder = match.parent
            if len(list(folder.glob("P_test*.mat"))) >= 10:
                return folder
    raise FileNotFoundError("Could not auto-locate public test directory containing P_test*.mat files.")


@dataclass
class CorrectedHybridBundle:
    base_bundle: HybridSeedNBW25Bundle
    corrector: object
    corrector_feature_names: List[str]
    config: Dict[str, object]


def _aggregate_prob_with_names(
    y: np.ndarray,
    prob: np.ndarray,
    subject_ids: np.ndarray,
    trial_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return aggregate_trial_probabilities(y, prob, subject_ids, trial_ids)


def robust_trial_ids_for_rows(user_ids: Sequence[str]) -> np.ndarray:
    counters: Dict[str, int] = {}
    trial_ids: List[str] = []
    for uid_raw in user_ids:
        uid = str(uid_raw)
        counters[uid] = counters.get(uid, 0) + 1
        trial_ids.append(f"{uid}_trial_{counters[uid]}")
    return np.asarray(trial_ids)


def robust_frame_from_dataset(bundle, dataset: EnhancedWindowDataset, mask: np.ndarray, args: argparse.Namespace) -> Tuple[BranchFrame, np.ndarray]:
    x_mask = center_features_by_subject(dataset.x[mask], dataset.subject_ids[mask])
    y_mask = dataset.y_emotion[mask]
    sid_mask = dataset.subject_ids[mask]
    tid_mask = dataset.trial_ids[mask]

    common_w = bundle.all_model.predict_proba(x_mask)[:, 1]
    normal_w = bundle.normal_model.predict_proba(x_mask)[:, 1]
    patient_w = bundle.depressed_model.predict_proba(x_mask)[:, 1]

    subject_x, _subject_y, subject_names = build_subject_summary_from_windows(
        dataset.x[mask],
        dataset.subject_ids[mask],
        dataset.group_labels[mask],
        dataset.trial_ids[mask],
    )
    dep_subject = bundle.population_model.predict_proba(subject_x)[:, 1]
    dep_map = {str(sid): float(prob) for sid, prob in zip(subject_names, dep_subject)}
    dep_w = np.asarray([dep_map[str(sid)] for sid in sid_mask], dtype=np.float64)
    routed_w = (1.0 - dep_w) * normal_w + dep_w * patient_w
    robust_w = (
        args.robust_common_weight * common_w + args.robust_group_weight * routed_w
    ) / (args.robust_common_weight + args.robust_group_weight)

    trial_y, common_t, trial_subjects, trial_names = _aggregate_prob_with_names(y_mask, common_w, sid_mask, tid_mask)
    _trial_y, normal_t, _subjects, _names = _aggregate_prob_with_names(y_mask, normal_w, sid_mask, tid_mask)
    _trial_y, patient_t, _subjects, _names = _aggregate_prob_with_names(y_mask, patient_w, sid_mask, tid_mask)
    _trial_y, routed_t, _subjects, _names = _aggregate_prob_with_names(y_mask, routed_w, sid_mask, tid_mask)
    _trial_y, robust_t, _subjects, _names = _aggregate_prob_with_names(y_mask, robust_w, sid_mask, tid_mask)
    dep_t = np.asarray([dep_map[str(sid)] for sid in trial_subjects], dtype=np.float64)
    zeros = np.zeros_like(robust_t)
    return (
        BranchFrame(
            y=trial_y,
            subjects=trial_subjects,
            robust_prob=robust_t,
            video_prob=zeros.copy(),
            hybrid_prob=robust_t.copy(),
            common_prob=common_t,
            normal_prob=normal_t,
            patient_prob=patient_t,
            routed_prob=routed_t,
            dep_prob=dep_t,
        ),
        trial_names,
    )


def add_video(frame: BranchFrame, video_prob: np.ndarray) -> BranchFrame:
    dep_map = {str(sid): float(prob) for sid, prob in zip(frame.subjects, frame.dep_prob)}
    hybrid = hybrid_probabilities(frame.robust_prob, video_prob, frame.subjects, dep_map, VIDEO_WEIGHT, VIDEO_MODE)
    return BranchFrame(
        y=frame.y,
        subjects=frame.subjects,
        robust_prob=frame.robust_prob,
        video_prob=video_prob,
        hybrid_prob=hybrid,
        common_prob=frame.common_prob,
        normal_prob=frame.normal_prob,
        patient_prob=frame.patient_prob,
        routed_prob=frame.routed_prob,
        dep_prob=frame.dep_prob,
    )


def trial_feature_summaries_aligned(
    dataset: EnhancedWindowDataset,
    mask: np.ndarray,
    selected_families: Sequence[str],
    ordered_trial_names: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    x_centered = center_features_by_subject(dataset.x[mask], dataset.subject_ids[mask])
    trial_ids = np.asarray(list(map(str, dataset.trial_ids[mask])))
    families = np.asarray([feature_family(name) for name in dataset.feature_names])
    names: List[str] = []
    for fam in selected_families:
        names.extend([f"{fam}_mean", f"{fam}_std", f"{fam}_p90", f"{fam}_p10"])
    rows: List[np.ndarray] = []
    for trial_name in ordered_trial_names:
        row_mask = trial_ids == str(trial_name)
        block = x_centered[row_mask]
        parts: List[float] = []
        for fam in selected_families:
            fam_mask = families == fam
            vals = block[:, fam_mask].reshape(-1)
            if vals.size == 0:
                parts.extend([0.0, 0.0, 0.0, 0.0])
            else:
                parts.extend(
                    [
                        float(np.mean(vals)),
                        float(np.std(vals)),
                        float(np.quantile(vals, 0.90)),
                        float(np.quantile(vals, 0.10)),
                    ]
                )
        rows.append(np.asarray(parts, dtype=np.float64))
    return np.vstack(rows), names


def fit_oof_corrector(args: argparse.Namespace) -> Tuple[object, List[str], Dict[str, object]]:
    robust_dataset = build_robust_v2_dataset(args.train_dir)
    raw_dataset = build_target_raw_window_dataset(args.train_dir)
    shared_dataset = build_target_shared_dataset(raw_dataset)
    source_dataset = subset_video_source_dataset(load_video_source_dataset(args.source_dir), SEED_ONLY)

    meta_rows: List[np.ndarray] = []
    meta_y: List[np.ndarray] = []
    feature_names: Optional[List[str]] = None
    folds = loso_splits(robust_dataset.subject_ids)
    for fold_idx, (train_mask, val_mask) in enumerate(progress(folds, total=len(folds), desc="OOF corrector"), start=1):
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
        val_frame, val_trial_names = robust_frame_from_dataset(robust_bundle, robust_dataset, val_mask, args)
        trial_y_v, video_val, subject_v_val = predict_video_branch(video_bundle, shared_dataset, val_mask)
        if not np.array_equal(val_frame.y, trial_y_v) or not np.array_equal(val_frame.subjects, subject_v_val):
            raise RuntimeError("Validation robust/video ordering mismatch while fitting OOF corrector.")
        val_frame = add_video(val_frame, video_val)
        family_x, family_names = trial_feature_summaries_aligned(robust_dataset, val_mask, SELECTED_FAMILIES, val_trial_names)
        x_val, feature_names = build_corrector_features(val_frame, family_x, family_names)
        meta_rows.append(x_val)
        meta_y.append(val_frame.y)
    x_meta = np.vstack(meta_rows)
    y_meta = np.concatenate(meta_y)
    corrector = fit_corrector(x_meta, y_meta, args.seed + 9000, args.corrector_c)
    info = {
        "oof_rows": int(x_meta.shape[0]),
        "oof_positive_rate": float(np.mean(y_meta)),
        "oof_subjects": int(len(np.unique(robust_dataset.subject_ids))),
    }
    return corrector, list(feature_names or []), info


def public_base_frame(bundle: HybridSeedNBW25Bundle, test_dir: Path, args: argparse.Namespace) -> Tuple[BranchFrame, np.ndarray, EnhancedWindowDataset]:
    robust_x_test, robust_user_ids = build_test_robust_v2_features(test_dir)
    trial_ids = robust_trial_ids_for_rows(robust_user_ids)
    x_centered = center_features_by_subject(robust_x_test, robust_user_ids)
    common_w = bundle.robust_bundle.all_model.predict_proba(x_centered)[:, 1]
    normal_w = bundle.robust_bundle.normal_model.predict_proba(x_centered)[:, 1]
    patient_w = bundle.robust_bundle.depressed_model.predict_proba(x_centered)[:, 1]

    subject_x, _dummy_y, subject_names = build_subject_summary_from_windows(
        robust_x_test,
        robust_user_ids,
        np.zeros(len(robust_user_ids), dtype=np.int64),
        trial_ids,
    )
    dep_subject = bundle.robust_bundle.population_model.predict_proba(subject_x)[:, 1]
    dep_map = {str(sid): float(prob) for sid, prob in zip(subject_names, dep_subject)}
    dep_w = np.asarray([dep_map[str(uid)] for uid in robust_user_ids], dtype=np.float64)
    routed_w = (1.0 - dep_w) * normal_w + dep_w * patient_w
    robust_w = (
        bundle.config["robust_common_weight"] * common_w + bundle.config["robust_group_weight"] * routed_w
    ) / (bundle.config["robust_common_weight"] + bundle.config["robust_group_weight"])

    zeros_y = np.zeros(len(robust_user_ids), dtype=np.int64)
    trial_y, common_t, trial_subjects, trial_names = _aggregate_prob_with_names(zeros_y, common_w, robust_user_ids, trial_ids)
    _trial_y, normal_t, _subjects, _names = _aggregate_prob_with_names(zeros_y, normal_w, robust_user_ids, trial_ids)
    _trial_y, patient_t, _subjects, _names = _aggregate_prob_with_names(zeros_y, patient_w, robust_user_ids, trial_ids)
    _trial_y, routed_t, _subjects, _names = _aggregate_prob_with_names(zeros_y, routed_w, robust_user_ids, trial_ids)
    _trial_y, robust_t, _subjects, _names = _aggregate_prob_with_names(zeros_y, robust_w, robust_user_ids, trial_ids)
    dep_t = np.asarray([dep_map[str(sid)] for sid in trial_subjects], dtype=np.float64)

    public_dataset = EnhancedWindowDataset(
        x=robust_x_test,
        y_emotion=zeros_y,
        subject_ids=np.asarray(robust_user_ids),
        group_labels=np.zeros(len(robust_user_ids), dtype=np.int64),
        trial_ids=trial_ids,
        feature_names=list(bundle.robust_bundle.feature_names),
    )
    frame = BranchFrame(
        y=trial_y,
        subjects=trial_subjects,
        robust_prob=robust_t,
        video_prob=np.zeros_like(robust_t),
        hybrid_prob=robust_t.copy(),
        common_prob=common_t,
        normal_prob=normal_t,
        patient_prob=patient_t,
        routed_prob=routed_t,
        dep_prob=dep_t,
    )
    return frame, trial_names, public_dataset


def predict_corrected_public(bundle: CorrectedHybridBundle, test_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    base_frame, trial_names, public_dataset = public_base_frame(bundle.base_bundle, test_dir, args)
    shared_test = build_test_shared_dataset(test_dir)
    _trial_y_v, video_prob, video_subjects = predict_video_branch(
        bundle.base_bundle.video_bundle,
        shared_test,
        np.ones(shared_test.x.shape[0], dtype=bool),
    )
    if not np.array_equal(base_frame.subjects, video_subjects):
        raise RuntimeError("Public robust/video ordering mismatch.")
    base_frame = add_video(base_frame, video_prob)
    family_x, family_names = trial_feature_summaries_aligned(public_dataset, np.ones(public_dataset.x.shape[0], dtype=bool), SELECTED_FAMILIES, trial_names)
    x_public, feature_names = build_corrector_features(base_frame, family_x, family_names)
    corr_prob = bundle.corrector.predict_proba(x_public)[:, 1]
    conflict = np.abs(base_frame.video_prob - base_frame.robust_prob)
    margin = np.abs(base_frame.robust_prob - 0.5)
    triggered = (conflict >= CONFLICT_THRESHOLD) & (margin <= MARGIN_THRESHOLD)
    eff_w = np.where(triggered, CORRECTOR_WEIGHT, 0.0)
    corrected_prob = (1.0 - eff_w) * base_frame.hybrid_prob + eff_w * corr_prob
    labels = (corrected_prob >= 0.5).astype(int)

    rows: List[Dict[str, object]] = []
    per_user: Dict[str, int] = {}
    for uid, base_prob, cprob, corr, trig, label, robust_prob, video in zip(
        base_frame.subjects,
        base_frame.hybrid_prob,
        corrected_prob,
        corr_prob,
        triggered,
        labels,
        base_frame.robust_prob,
        base_frame.video_prob,
    ):
        uid_str = str(uid)
        per_user[uid_str] = per_user.get(uid_str, 0) + 1
        rows.append(
            {
                "user_id": uid_str,
                "trial_id": per_user[uid_str],
                "Emotion_label": int(label),
                "corrected_prob": float(cprob),
                "base_hybrid_prob": float(base_prob),
                "corrector_prob": float(corr),
                "triggered": bool(trig),
                "robust_prob": float(robust_prob),
                "video_prob": float(video),
            }
        )
    return pd.DataFrame(rows)


def save_corrected_bundle(bundle: CorrectedHybridBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)


def compare_p_test3(df: pd.DataFrame, reference_path: Optional[Path], first_submission_path: Optional[Path]) -> pd.DataFrame:
    out = df[df["user_id"] == "P_test3"].copy()
    if first_submission_path is not None and first_submission_path.exists():
        first = pd.read_excel(first_submission_path)
        first = first.rename(columns={first.columns[0]: "user_id", first.columns[1]: "trial_id", first.columns[2]: "first_submission_label"})
        out = out.merge(first[["user_id", "trial_id", "first_submission_label"]], on=["user_id", "trial_id"], how="left")
    if reference_path is not None and reference_path.exists():
        ref = pd.read_excel(reference_path)
        ref = ref.rename(columns={ref.columns[0]: "user_id", ref.columns[1]: "trial_id", ref.columns[2]: "reference_75_label"})
        out = out.merge(ref[["user_id", "trial_id", "reference_75_label"]], on=["user_id", "trial_id"], how="left")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=Path(r"C:\Users\lyg\Documents\Playground\eeg_competition_algorithms\public_video_sources"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"C:\Users\lyg\Documents\Playground\eeg_competition_algorithms\outputs\hybrid_seed_nb_w25_corrected"))
    parser.add_argument("--base-model-in", type=Path, default=Path(r"C:\Users\lyg\Documents\Playground\eeg_competition_algorithms\outputs\hybrid_seed_nb_w25_submission\hybrid_seed_nb_w25.pkl"))
    parser.add_argument("--model-out", type=Path, default=None)
    parser.add_argument("--prediction-out", type=Path, default=None)
    parser.add_argument("--reference-75", type=Path, default=Path(r"C:\Users\lyg\xwechat_files\wxid_qnea4ikicpvm22_75eb\msg\file\2026-05\public_test_submission_compact_rank_router_robust917.xlsx"))
    parser.add_argument("--first-submission", type=Path, default=Path(r"C:\Users\lyg\xwechat_files\wxid_qnea4ikicpvm22_75eb\msg\file\2026-05\最终提交结果_hybrid_seed_nb_w25.xlsx"))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--shared-components", type=int, default=12)
    parser.add_argument("--shared-c", type=float, default=1.0)
    parser.add_argument("--robust-v2-components", type=int, default=64)
    parser.add_argument("--robust-v2-c", type=float, default=1.0)
    parser.add_argument("--population-components", type=int, default=8)
    parser.add_argument("--population-c", type=float, default=0.1)
    parser.add_argument("--robust-common-weight", type=float, default=0.75)
    parser.add_argument("--robust-group-weight", type=float, default=0.25)
    parser.add_argument("--corrector-c", type=float, default=0.2)
    parser.add_argument("--train-base-if-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_out = args.model_out or (args.output_dir / f"{MODEL_NAME}.pkl")
    args.prediction_out = args.prediction_out or (args.output_dir / f"{MODEL_NAME}_public_test.xlsx")
    args.train_dir = resolve_competition_dir(args.train_dir, TRAIN_DIR_LABEL)
    args.test_dir = resolve_public_test_dir(args.test_dir)

    corrector, corrector_feature_names, corrector_info = fit_oof_corrector(args)
    if args.base_model_in.exists():
        base_bundle = load_bundle(args.base_model_in)
    elif args.train_base_if_missing:
        base_bundle = train_full_bundle(args)
        save_bundle(base_bundle, args.base_model_in)
    else:
        raise FileNotFoundError(f"Base model not found: {args.base_model_in}")

    corrected_bundle = CorrectedHybridBundle(
        base_bundle=base_bundle,
        corrector=corrector,
        corrector_feature_names=corrector_feature_names,
        config={
            "model_name": MODEL_NAME,
            "base_model": "hybrid_seed_nb_w25",
            "corrector_weight": CORRECTOR_WEIGHT,
            "conflict_threshold": CONFLICT_THRESHOLD,
            "margin_threshold": MARGIN_THRESHOLD,
            "selected_families": list(SELECTED_FAMILIES),
            "corrector_training": "LOSO out-of-fold meta-features only",
            "reference_75_used": False,
            **corrector_info,
        },
    )
    save_corrected_bundle(corrected_bundle, args.model_out)
    df = predict_corrected_public(corrected_bundle, args.test_dir, args)
    df[["user_id", "trial_id", "Emotion_label"]].to_excel(args.prediction_out, index=False)
    df.to_csv(args.prediction_out.with_suffix(".with_prob.csv"), index=False, encoding="utf-8-sig")
    p3 = compare_p_test3(df, args.reference_75, args.first_submission)
    p3.to_csv(args.output_dir / "P_test3_corrected_comparison.csv", index=False, encoding="utf-8-sig")
    report = {
        "model_out": str(args.model_out),
        "prediction_out": str(args.prediction_out),
        "rows": int(len(df)),
        "triggered_rows": int(df["triggered"].sum()),
        "p_test3": p3.to_dict(orient="records"),
        "config": corrected_bundle.config,
    }
    write_json(args.output_dir / "corrected_public_test_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
