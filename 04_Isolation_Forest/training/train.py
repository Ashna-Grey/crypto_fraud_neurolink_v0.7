#!/usr/bin/env python3
"""Train B6b Isolation Forest and an OOB-referenced bootstrap ensemble."""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.isolation_forest_utils import (  # noqa: E402
    FEATURES_B6B,
    FEATURES_23,
    MODEL_CONFIG,
    read_and_validate_training_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with all 20 B6b features; the original 23-feature source is accepted.")
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts"), help="Output directory for model and training artifacts.")
    parser.add_argument("--n-estimators", type=int, default=300, help="Trees per Isolation Forest model.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5, help="Folds for a legacy/raw-score OOF diagnostic reference.")
    parser.add_argument("--ensemble-size", type=int, default=25, help="Number of bootstrap Isolation Forest models for mu/sigma-squared estimates.")
    return parser.parse_args()


def _make_model(n_estimators: int, seed: int) -> IsolationForest:
    return IsolationForest(
        n_estimators=n_estimators,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )


def main():
    args = parse_args()
    if args.n_estimators < 50:
        raise ValueError("Use at least 50 trees; 300 is the established configuration.")
    if args.folds < 3:
        raise ValueError("At least 3 folds are required for the raw-score OOF diagnostic reference.")
    if args.ensemble_size < 2:
        raise ValueError("At least two bootstrap models are required to estimate sample variance.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_raw, preprocessing = read_and_validate_training_csv(args.input)
    X_raw = X_raw[FEATURES_B6B].reset_index(drop=True)
    n_rows = len(X_raw)
    if n_rows < max(args.folds * 2, 100):
        raise ValueError("Too few rows for the requested OOF and bootstrap procedure.")

    # --------------------------------------------------------
    # A. Five-fold OOF raw score reference (kept for diagnostics)
    # --------------------------------------------------------
    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=2026)
    oof_scores = np.full(n_rows, np.nan, dtype=float)
    fold_rows = []

    for fold_no, (train_idx, valid_idx) in enumerate(kfold.split(X_raw), start=1):
        X_train_raw = X_raw.iloc[train_idx]
        X_valid_raw = X_raw.iloc[valid_idx]
        fold_medians = X_train_raw.median()
        if fold_medians.isna().any():
            bad = fold_medians[fold_medians.isna()].index.tolist()
            raise ValueError("All-missing training-fold feature(s): " + ", ".join(bad))

        X_train = X_train_raw.fillna(fold_medians)
        X_valid = X_valid_raw.fillna(fold_medians)
        fold_model = _make_model(args.n_estimators, args.random_state + fold_no)
        fold_model.fit(X_train)
        oof_scores[valid_idx] = -fold_model.score_samples(X_valid)
        fold_rows.append({
            "fold": fold_no,
            "training_rows": int(len(train_idx)),
            "validation_rows": int(len(valid_idx)),
            "validation_score_mean": float(oof_scores[valid_idx].mean()),
            "validation_score_std": float(oof_scores[valid_idx].std()),
        })
        print(f"Completed raw-score OOF fold {fold_no}/{args.folds}")

    if not np.all(np.isfinite(oof_scores)):
        raise RuntimeError("Out-of-fold raw-score reference contains invalid values.")

    # --------------------------------------------------------
    # B. Retain a canonical single model for diagnostics/compatibility
    # --------------------------------------------------------
    global_medians = X_raw.median()
    if global_medians.isna().any():
        raise ValueError("At least one feature is entirely missing in the full training data.")
    X_imputed = X_raw.fillna(global_medians)

    final_model = _make_model(args.n_estimators, args.random_state)
    final_model.fit(X_imputed)
    in_sample_scores = -final_model.score_samples(X_imputed)

    # --------------------------------------------------------
    # C. Bootstrap ensemble; each member gets an OOB reference
    # --------------------------------------------------------
    # For each member, sample n rows with replacement. Rows not sampled form
    # the out-of-bag (OOB) reference set used to map its raw score to percentile.
    # Percentiles are on [0,1] and express relative abnormality, NOT fraud probability.
    rng = np.random.default_rng(args.random_state + 100_003)
    ensemble_members = []
    member_summary = []

    for member_idx in range(args.ensemble_size):
        member_seed = int(args.random_state + 10_007 + member_idx)
        member_rng = np.random.default_rng(member_seed)
        sampled_idx = member_rng.integers(0, n_rows, size=n_rows)
        inbag_mask = np.zeros(n_rows, dtype=bool)
        inbag_mask[np.unique(sampled_idx)] = True
        oob_idx = np.flatnonzero(~inbag_mask)

        if len(oob_idx) < max(30, int(0.05 * n_rows)):
            raise RuntimeError(
                f"Bootstrap member {member_idx} has too few OOB rows: {len(oob_idx)}"
            )

        # Fit the imputer statistics on the bootstrap sample only.
        X_inbag_raw = X_raw.iloc[sampled_idx]
        member_medians = X_inbag_raw.median()
        if member_medians.isna().any():
            bad = member_medians[member_medians.isna()].index.tolist()
            raise ValueError("Bootstrap member has all-missing feature(s): " + ", ".join(bad))

        X_inbag = X_inbag_raw.fillna(member_medians)
        X_oob = X_raw.iloc[oob_idx].fillna(member_medians)

        member_model = _make_model(args.n_estimators, member_seed)
        member_model.fit(X_inbag)
        oob_scores = -member_model.score_samples(X_oob)
        oob_sorted = np.sort(np.asarray(oob_scores, dtype=float))

        ensemble_members.append({
            "model": member_model,
            "medians": {c: float(member_medians[c]) for c in FEATURES_B6B},
            "score_reference_sorted": oob_sorted,
            "seed": member_seed,
            "oob_rows": int(len(oob_idx)),
        })

        member_summary.append({
            "member": member_idx + 1,
            "seed": member_seed,
            "inbag_draws": int(len(sampled_idx)),
            "unique_inbag_rows": int(inbag_mask.sum()),
            "oob_rows": int(len(oob_idx)),
            "oob_score_mean": float(np.mean(oob_scores)),
            "oob_score_std": float(np.std(oob_scores)),
        })
        print(f"Completed bootstrap member {member_idx + 1}/{args.ensemble_size} (OOB rows={len(oob_idx)})")

    # --------------------------------------------------------
    # D. Shared feature ranges/reference used for input checks/explanations
    # --------------------------------------------------------
    feature_reference_sorted = {
        c: np.sort(X_imputed[c].to_numpy(dtype=float))
        for c in FEATURES_B6B
    }
    feature_min = {c: float(X_raw[c].min()) for c in FEATURES_B6B}
    feature_max = {c: float(X_raw[c].max()) for c in FEATURES_B6B}
    oof_sorted = np.sort(oof_scores)

    metadata = {
        **MODEL_CONFIG,
        "n_estimators_per_member": int(args.n_estimators),
        "random_state": int(args.random_state),
        "rows_used_for_final_fit": int(n_rows),
        "features_in_source_schema": int(preprocessing["audit"]["feature_count_before_selection"]),
        "feature_columns": FEATURES_B6B,
        "out_of_fold_reference_folds": int(args.folds),
        "out_of_fold_reference_rows": int(len(oof_scores)),
        "bootstrap_ensemble_size": int(args.ensemble_size),
        "bootstrap_sampling": "n rows sampled with replacement per member; OOB rows score the member-specific reference distribution",
        "mu_definition": "if_mu = mean of member-specific smoothed empirical anomaly percentiles on [0,1]",
        "sigma_squared_definition": "if_sigma_squared = sample variance (ddof=1) of member-specific anomaly percentiles; measures bootstrap model disagreement on relative anomaly severity",
        "important_semantics": "if_mu is NOT a fraud probability; if_sigma_squared is NOT yet a calibrated predictive variance and must not be used as 1/sigma^2 fusion weight before validation",
        "training_dataset_name": Path(args.input).name,
        "missing_values_imputed_in_final_fit": int(preprocessing["audit"]["missing_values_before_imputation"]),
        "thresholds_from_oof_reference": {
            "p90": float(np.quantile(oof_scores, 0.90)),
            "p95": float(np.quantile(oof_scores, 0.95)),
            "p99": float(np.quantile(oof_scores, 0.99)),
        },
        "interpretation_warning": "Anomaly percentile and if_mu are relative abnormality measures, not fraud probabilities or confirmed-fraud labels.",
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }

    bundle = {
        "model": final_model,
        "ensemble_members": ensemble_members,
        "feature_columns": FEATURES_B6B,
        "medians": {c: float(global_medians[c]) for c in FEATURES_B6B},
        "score_reference_sorted": oof_sorted,
        "feature_reference_sorted": feature_reference_sorted,
        "feature_min": feature_min,
        "feature_max": feature_max,
        "metadata": metadata,
    }

    joblib.dump(bundle, output_dir / "isolation_forest_b6b.joblib", compress=3)

    schema = {
        "model_feature_count": len(FEATURES_B6B),
        "feature_columns_in_order": FEATURES_B6B,
        "excluded_source_columns": MODEL_CONFIG["excluded_features"],
        "accepted_source_feature_schema": FEATURES_23,
        "extra_columns": "Extra columns are ignored by the model and preserved as passthrough fields in predictions.",
    }
    write_json(output_dir / "feature_schema.json", schema)
    write_json(output_dir / "model_config.json", metadata)
    write_json(output_dir / "training_audit.json", preprocessing["audit"])
    write_json(output_dir / "oof_fold_summary.json", {"folds": fold_rows})
    write_json(output_dir / "bootstrap_ensemble_summary.json", {
        "ensemble_size": args.ensemble_size,
        "member_summaries": member_summary,
    })

    reference_frame = pd.DataFrame({
        "oof_anomaly_score": oof_scores,
        "oof_anomaly_percentile": [
            (np.searchsorted(oof_sorted, score, side="right") + 0.5) / (len(oof_sorted) + 1)
            for score in oof_scores
        ],
    })
    reference_frame.to_csv(output_dir / "oof_score_reference.csv", index=False)

    report = {
        "rows": int(n_rows),
        "features": int(len(FEATURES_B6B)),
        "bootstrap_ensemble_size": int(args.ensemble_size),
        "trees_per_member": int(args.n_estimators),
        "total_isolation_trees_in_bootstrap_ensemble": int(args.ensemble_size * args.n_estimators),
        "in_sample_anomaly_score_mean_single_model": float(in_sample_scores.mean()),
        "in_sample_anomaly_score_std_single_model": float(in_sample_scores.std()),
        "in_sample_score_quantiles_single_model": {
            f"p{p}": float(np.quantile(in_sample_scores, p / 100))
            for p in [50, 90, 95, 99]
        },
        "oof_score_mean_single_model_reference": float(oof_scores.mean()),
        "oof_score_std_single_model_reference": float(oof_scores.std()),
        "mu_semantics": "mean bootstrap-member OOB-referenced anomaly percentile; not fraud probability",
        "sigma_squared_semantics": "sample variance across member anomaly percentiles; bootstrap model disagreement",
        "fraud_metrics": None,
        "reason_fraud_metrics_unavailable": "Source data contains no verified fraud labels and no wallet identifiers.",
        "duplicate_audit": preprocessing["audit"],
        "in_sample_metrics_are_not_independent_evaluation": True,
    }
    write_json(output_dir / "training_report.json", report)

    print("\nFinal B6b model bundle trained successfully.")
    print(f"Rows: {n_rows:,}")
    print(f"Features: {len(FEATURES_B6B)}")
    print(f"Bootstrap ensemble size: {args.ensemble_size}")
    print(f"Trees per ensemble member: {args.n_estimators}")
    print(f"Model artifact: {output_dir / 'isolation_forest_b6b.joblib'}")
    print("if_mu = mean relative anomaly percentile [0,1]; NOT fraud probability.")
    print("if_sigma_squared = sample variance of member percentiles; model disagreement only.")
    print("Fraud-probability calibration and fusion weighting require a compatible labelled calibration dataset.")


if __name__ == "__main__":
    main()
