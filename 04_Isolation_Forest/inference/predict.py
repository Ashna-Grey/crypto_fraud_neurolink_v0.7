#!/usr/bin/env python3
"""Score B6b rows and return anomaly-severity mu plus bootstrap disagreement variance."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.isolation_forest_utils import (  # noqa: E402
    FEATURES_B6B,
    anomaly_band,
    create_feature_explanation,
    empirical_percentile,
    load_bundle,
)
from src.live_input import normalize_column_names, prepare_raw_live_input


def _smoothed_fraction_percentile(score: float, sorted_reference: np.ndarray) -> float:
    """Return smoothed empirical percentile as a fraction in (0,1)."""
    return empirical_percentile(score, sorted_reference) / 100.0


def score_dataframe(
    frame: pd.DataFrame,
    bundle: dict,
    top_explanations: int = 3,
    input_format: str = "preprocessed",
) -> pd.DataFrame:
    """Validate and score rows, returning IF mu and sigma^2.

    if_mu is the mean of bootstrap members' OOB-referenced anomaly percentiles.
    if_sigma_squared is the sample variance of those percentiles across members.
    Neither field is a calibrated fraud probability or a fusion-ready variance.
    """
    if top_explanations < 1:
        raise ValueError("top_explanations must be at least 1")
    if input_format not in {"preprocessed", "raw-live"}:
        raise ValueError("input_format must be either 'preprocessed' or 'raw-live'.")
    if input_format == "raw-live":
        frame, _ = prepare_raw_live_input(frame)
    else:
        frame = normalize_column_names(frame)
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Input contains duplicate column names: {duplicates}")

    features = bundle["feature_columns"]
    missing_features = [c for c in features if c not in frame.columns]
    if missing_features:
        raise ValueError("Input is missing required B6b features: " + ", ".join(missing_features))

    raw = frame[features].copy()
    numeric = pd.DataFrame(index=frame.index)
    for column in features:
        converted = pd.to_numeric(raw[column], errors="coerce")
        invalid_parse = raw[column].notna() & converted.isna()
        if invalid_parse.any():
            examples = raw.loc[invalid_parse, column].head(3).tolist()
            raise ValueError(f"Non-numeric values in feature {column!r}; examples: {examples}")
        numeric[column] = converted

    if len(frame) == 0:
        raise ValueError("Input CSV contains no data rows.")

    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError("Input contains infinite feature values.")

    missing_mask = numeric.isna()
    missing_count = missing_mask.sum(axis=1).to_numpy(dtype=int)
    unusable_mask = missing_count == len(features)

    # Data-quality ranges are computed from the source training dataset.
    range_flags = pd.DataFrame(False, index=frame.index, columns=features)
    for column in features:
        valid = ~missing_mask[column]
        range_flags.loc[valid, column] = (
            (numeric.loc[valid, column] < bundle["feature_min"][column])
            | (numeric.loc[valid, column] > bundle["feature_max"][column])
        )
    outside_range_count = range_flags.sum(axis=1).to_numpy(dtype=int)

    members = bundle.get("ensemble_members")
    if not members or len(members) < 2:
        raise ValueError(
            "This artifact has no bootstrap ensemble. Re-run training/train.py "
            "with the current package to generate if_mu and if_sigma_squared."
        )

    member_raw_scores = []
    member_percentiles = []

    # Every member has its own training-only median imputer and OOB score reference.
    for member in members:
        member_imputed = numeric.fillna(member["medians"])
        member_scores = -member["model"].score_samples(member_imputed[features])
        member_ref = np.asarray(member["score_reference_sorted"], dtype=float)
        member_u = np.asarray([
            _smoothed_fraction_percentile(float(s), member_ref)
            for s in member_scores
        ], dtype=float)
        member_raw_scores.append(member_scores)
        member_percentiles.append(member_u)

    raw_matrix = np.vstack(member_raw_scores)       # B x N
    percentile_matrix = np.vstack(member_percentiles)  # B x N

    if_mu = percentile_matrix.mean(axis=0)
    if_sigma_squared = percentile_matrix.var(axis=0, ddof=1)
    if_sigma = np.sqrt(np.maximum(if_sigma_squared, 0.0))
    anomaly_score_mean = raw_matrix.mean(axis=0)
    anomaly_score_variance = raw_matrix.var(axis=0, ddof=1)
    anomaly_percentile = if_mu * 100.0

    result = pd.DataFrame(index=np.arange(len(frame)))
    result["source_row_index"] = np.arange(len(frame), dtype=int)

    # Preserve external identifiers and other non-feature columns.
    for column in frame.columns:
        if column not in features and column != "source_row_index":
            result[column] = frame[column].reset_index(drop=True)

    result["if_mu"] = if_mu
    result["if_sigma_squared"] = if_sigma_squared
    result["if_sigma"] = if_sigma
    result["if_ensemble_size"] = len(members)
    result["anomaly_score"] = anomaly_score_mean
    result["anomaly_score_variance"] = anomaly_score_variance
    result["anomaly_percentile"] = anomaly_percentile
    result["anomaly_priority_band"] = [anomaly_band(v) for v in anomaly_percentile]
    result["if_mu_semantics"] = "mean_relative_anomaly_percentile_not_fraud_probability"
    result["if_sigma_squared_semantics"] = "bootstrap_model_disagreement_not_fusion_calibrated_variance"
    result["if_prediction"] = result["anomaly_priority_band"]
    result["missing_feature_count"] = missing_count
    result["missing_feature_ratio"] = missing_count / float(len(features))
    result["outside_training_minmax_count"] = outside_range_count
    result["outside_training_minmax_features"] = [
        ";".join([feature for feature in features if bool(range_flags.iloc[pos][feature])])
        for pos in range(len(frame))
    ]

    quality_flags = []
    for missing_num, outside_num in zip(missing_count, outside_range_count):
        flags = []
        if missing_num > 0:
            flags.append("MISSING_VALUES_IMPUTED")
        if outside_num > 0:
            flags.append("OUTSIDE_TRAINING_RANGE")
        if missing_num == len(features):
            flags.append("ALL_REQUIRED_FEATURES_MISSING")
        quality_flags.append(";".join(flags) if flags else "OK")
    result["data_quality_flag"] = quality_flags
    result["input_quality_status"] = [
        "UNUSABLE" if bool(unusable_mask[pos]) else ("WARN" if quality_flags[pos] != "OK" else "PASS")
        for pos in range(len(frame))
    ]

    # Explanations are marginal feature-tail summaries, not model attributions.
    global_imputed = numeric.fillna(bundle["medians"])
    result["marginal_feature_explanation"] = [
        create_feature_explanation(
            global_imputed.iloc[pos][features],
            bundle["feature_reference_sorted"],
            top_n=top_explanations,
        )
        for pos in range(len(global_imputed))
    ]

    # A row with no observed model features has no defensible anomaly assessment.
    # Do not present the all-median score as a usable prediction.
    if unusable_mask.any():
        no_score_columns = [
            "if_mu", "if_sigma_squared", "if_sigma", "anomaly_score",
            "anomaly_score_variance", "anomaly_percentile"
        ]
        result.loc[unusable_mask, no_score_columns] = np.nan
        result.loc[unusable_mask, "anomaly_priority_band"] = "NOT_SCORED_INPUT_UNUSABLE"
        result.loc[unusable_mask, "if_prediction"] = "NOT_SCORED_INPUT_UNUSABLE"
        result.loc[unusable_mask, "marginal_feature_explanation"] = "NOT_AVAILABLE_INPUT_UNUSABLE"
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV containing all 20 B6b features; 23-feature source CSVs are accepted too.")
    parser.add_argument("--model", default=str(ROOT / "artifacts" / "isolation_forest_b6b.joblib"))
    parser.add_argument("--output", required=True, help="Destination CSV path.")
    parser.add_argument("--top-explanations", type=int, default=3)
    parser.add_argument(
        "--input-format", choices=["preprocessed", "raw-live"], default="preprocessed",
        help="Use preprocessed for the model-space CSV; use raw-live to apply the verified signed-log1p transform to raw live feature exports.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    frame = pd.read_csv(input_path)
    bundle = load_bundle(args.model)
    result = score_dataframe(
        frame, bundle, top_explanations=args.top_explanations, input_format=args.input_format
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print("Prediction completed.")
    print(f"Rows scored: {len(result):,}")
    print(f"Output: {output_path.resolve()}")
    print(f"Input format: {args.input_format}")
    print(f"Bootstrap ensemble members: {len(bundle['ensemble_members'])}")
    print(f"Mean if_mu (relative anomaly percentile): {result['if_mu'].mean():.6f}")
    print(f"Mean if_sigma_squared (model disagreement): {result['if_sigma_squared'].mean():.8f}")
    print("Output meanings: if_mu is NOT fraud probability; if_sigma_squared is NOT yet fusion-calibrated.")
    print("Anomaly priority bands:")
    print(result["anomaly_priority_band"].value_counts(dropna=False).to_string())
    print("Input quality status:")
    print(result["input_quality_status"].value_counts(dropna=False).to_string())
    print("Data quality flags:")
    print(result["data_quality_flag"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
