#!/usr/bin/env python3
"""Audit live-input compatibility and feature-distribution diagnostics for B6b."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.isolation_forest_utils import FEATURES_B6B, load_bundle  # noqa: E402
from src.live_input import normalize_column_names, prepare_raw_live_input  # noqa: E402


def _numeric_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise ValueError("Input is missing required B6b features: " + ", ".join(missing))
    numeric = pd.DataFrame(index=frame.index)
    for feature in features:
        raw = frame[feature]
        converted = pd.to_numeric(raw, errors="coerce")
        bad = raw.notna() & converted.isna()
        if bad.any():
            raise ValueError(
                f"Non-numeric values in feature {feature!r}; examples: "
                f"{raw.loc[bad].head(3).tolist()}"
            )
        numeric[feature] = converted
    if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Input contains infinite feature values.")
    return numeric


def build_feature_audit(numeric: pd.DataFrame, bundle: dict, max_out_of_range_pct: float, median_shift_iqr: float) -> pd.DataFrame:
    rows = []
    row_count = max(len(numeric), 1)
    for feature in bundle["feature_columns"]:
        values = numeric[feature].dropna().to_numpy(dtype=float)
        train_ref = np.asarray(bundle["feature_reference_sorted"][feature], dtype=float)
        train_q = np.quantile(train_ref, [0.01, 0.25, 0.50, 0.75, 0.99])
        train_min = float(bundle["feature_min"][feature])
        train_max = float(bundle["feature_max"][feature])
        missing_count = int(numeric[feature].isna().sum())
        if values.size:
            live_q = np.quantile(values, [0.01, 0.50, 0.99])
            live_min = float(np.min(values))
            live_max = float(np.max(values))
            outside_count = int(((values < train_min) | (values > train_max)).sum())
            live_median = float(live_q[1])
            iqr = float(train_q[3] - train_q[1])
            scale = iqr if iqr > 1e-12 else max(abs(float(train_q[2])), 1.0)
            median_shift = abs(live_median - float(train_q[2])) / scale
            outside_pct = 100.0 * outside_count / len(values)
        else:
            live_q = [np.nan, np.nan, np.nan]
            live_min = live_max = live_median = median_shift = outside_pct = np.nan
            outside_count = 0
        warning = bool(
            (np.isfinite(outside_pct) and outside_pct > max_out_of_range_pct)
            or (np.isfinite(median_shift) and median_shift > median_shift_iqr)
        )
        rows.append({
            "feature": feature,
            "required": True,
            "live_missing_count": missing_count,
            "live_missing_pct": 100.0 * missing_count / row_count,
            "live_nonmissing_count": int(values.size),
            "training_min": train_min,
            "training_p01": float(train_q[0]),
            "training_q1": float(train_q[1]),
            "training_median": float(train_q[2]),
            "training_q3": float(train_q[3]),
            "training_p99": float(train_q[4]),
            "training_max": train_max,
            "live_min": live_min,
            "live_p01": float(live_q[0]),
            "live_median": live_median,
            "live_p99": float(live_q[2]),
            "live_max": live_max,
            "outside_training_minmax_count": outside_count,
            "outside_training_minmax_pct_of_observed": outside_pct,
            "absolute_median_shift_in_training_iqr": median_shift,
            "distribution_warning_heuristic": warning,
        })
    return pd.DataFrame(rows)


def compare_reference_rows(numeric: pd.DataFrame, reference_path: str | Path, bundle: dict) -> dict:
    """Compare rows only when caller confirms matching row order and identity."""
    reference = normalize_column_names(pd.read_csv(reference_path))
    missing = [feature for feature in bundle["feature_columns"] if feature not in reference.columns]
    if missing:
        raise ValueError("Reference dataset is missing required B6b features: " + ", ".join(missing))
    if len(reference) != len(numeric):
        raise ValueError(
            "--verify-row-alignment requires equal row counts; "
            f"input has {len(numeric)} rows and reference has {len(reference)} rows."
        )
    ref_num = _numeric_features(reference, bundle["feature_columns"])
    live_filled = numeric.fillna(bundle["medians"])
    ref_filled = ref_num.fillna(bundle["medians"])
    delta = np.abs(live_filled.to_numpy(dtype=float) - ref_filled.to_numpy(dtype=float))
    within_tolerance = np.isclose(
        live_filled.to_numpy(dtype=float), ref_filled.to_numpy(dtype=float),
        rtol=1e-7, atol=1e-7,
    )
    return {
        "reference_data": Path(reference_path).name,
        "row_alignment_assumed_by_user": True,
        "rows_compared": int(len(numeric)),
        "feature_cells_compared": int(delta.size),
        "feature_cells_matching_tolerance": int(within_tolerance.sum()),
        "feature_cell_match_pct": float(100.0 * within_tolerance.mean()) if delta.size else 100.0,
        "max_absolute_difference": float(delta.max()) if delta.size else 0.0,
        "exact_transform_reconstruction_within_tolerance": bool(within_tolerance.all()),
        "tolerance": {"rtol": 1e-7, "atol": 1e-7},
        "note": "Only valid when the input and reference are the same wallets in the same row order.",
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Live wallet CSV to audit.")
    parser.add_argument("--model", default=str(ROOT / "artifacts" / "isolation_forest_b6b.joblib"))
    parser.add_argument("--input-format", choices=["raw-live", "preprocessed"], default="raw-live")
    parser.add_argument("--output-dir", required=True, help="Folder for JSON and CSV audit reports.")
    parser.add_argument("--reference-data", help="Optional transformed reference CSV, e.g. the B6b preprocessed training CSV.")
    parser.add_argument(
        "--verify-row-alignment", action="store_true",
        help="Compare each row to --reference-data. Use only when both files contain the same rows in the same order.",
    )
    parser.add_argument("--max-out-of-range-pct", type=float, default=10.0)
    parser.add_argument("--median-shift-iqr", type=float, default=3.0)
    parser.add_argument("--max-missing-cell-pct", type=float, default=5.0)
    parser.add_argument("--prepared-output", help="Optional path to save transformed feature rows (including passthrough columns).")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_out_of_range_pct < 0 or args.median_shift_iqr < 0 or args.max_missing_cell_pct < 0:
        raise ValueError("Diagnostic thresholds must be non-negative.")
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    original = pd.read_csv(input_path)
    if original.empty:
        raise ValueError("Input CSV contains no data rows.")
    if args.input_format == "raw-live":
        prepared, preprocessing_audit = prepare_raw_live_input(original)
    else:
        prepared = normalize_column_names(original)
        preprocessing_audit = {
            "input_format": "preprocessed",
            "feature_value_transform": "none; input assumed already in the model's transformed B6b feature space",
            "missing_value_handling": "Preserve NaN for assessment; inference uses stored training medians",
        }

    bundle = load_bundle(args.model)
    features = list(bundle["feature_columns"])
    missing_features = [f for f in features if f not in prepared.columns]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if missing_features:
        summary = {
            "input_file": input_path.name,
            "input_rows": int(len(prepared)),
            "input_format": args.input_format,
            "schema_status": "FAIL",
            "missing_required_feature_columns": missing_features,
            "note": "Prediction must not run until the live feature builder supplies these B6b columns.",
        }
        (output_dir / "live_input_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        pd.DataFrame({"missing_required_feature": missing_features}).to_csv(
            output_dir / "live_feature_compatibility.csv", index=False
        )
        raise SystemExit(f"Schema compatibility FAILED; see {output_dir / 'live_input_audit.json'}")

    numeric = _numeric_features(prepared, features)
    feature_audit = build_feature_audit(numeric, bundle, args.max_out_of_range_pct, args.median_shift_iqr)
    feature_audit.to_csv(output_dir / "live_feature_compatibility.csv", index=False)

    missing_per_row = numeric.isna().sum(axis=1)
    nonmissing = numeric.notna()
    out_of_range_per_feature = []
    total_outside = 0
    for feature in features:
        observed = numeric[feature].dropna()
        outside = ((observed < bundle["feature_min"][feature]) | (observed > bundle["feature_max"][feature])).sum()
        total_outside += int(outside)
        out_of_range_per_feature.append(int(outside))

    reference_check = None
    if args.verify_row_alignment:
        if not args.reference_data:
            raise ValueError("--verify-row-alignment requires --reference-data.")
        reference_check = compare_reference_rows(numeric, args.reference_data, bundle)

    schema_ok = not missing_features
    drift_features = feature_audit.loc[feature_audit["distribution_warning_heuristic"], "feature"].tolist()
    missing_cell_pct = float(100.0 * numeric.isna().sum().sum() / max(numeric.size, 1))
    missingness_warning = missing_cell_pct > args.max_missing_cell_pct
    if not schema_ok:
        overall = "FAIL_SCHEMA"
    elif missingness_warning and drift_features:
        overall = "SCHEMA_OK_INPUT_QUALITY_AND_DISTRIBUTION_REVIEW"
    elif missingness_warning:
        overall = "SCHEMA_OK_INPUT_QUALITY_REVIEW"
    elif drift_features:
        overall = "SCHEMA_OK_DISTRIBUTION_REVIEW"
    else:
        overall = "SCHEMA_OK_NO_DEFAULT_WARNING"

    summary = {
        "input_file": input_path.name,
        "input_format": args.input_format,
        "input_rows": int(len(prepared)),
        "required_feature_count": len(features),
        "present_required_feature_count": len(features) - len(missing_features),
        "missing_required_feature_columns": missing_features,
        "schema_status": "PASS" if schema_ok else "FAIL",
        "input_rows_with_any_missing_feature": int((missing_per_row > 0).sum()),
        "input_rows_with_all_features_missing": int((missing_per_row == len(features)).sum()),
        "total_missing_feature_cells_before_imputation": int(numeric.isna().sum().sum()),
        "missing_feature_cell_pct": missing_cell_pct,
        "missingness_warning": bool(missingness_warning),
        "input_quality_warning_heuristics": {
            "feature_cell_missing_pct_greater_than": args.max_missing_cell_pct,
            "warning_is_heuristic_not_a_statistical_test": True,
        },
        "total_observed_values_outside_training_minmax": int(total_outside),
        "observed_feature_cell_count": int(nonmissing.sum().sum()),
        "observed_values_outside_training_minmax_pct": float(100.0 * total_outside / max(int(nonmissing.sum().sum()), 1)),
        "distribution_warning_features": drift_features,
        "distribution_warning_heuristics": {
            "observed_values_outside_training_minmax_pct_greater_than": args.max_out_of_range_pct,
            "absolute_live_median_shift_greater_than_training_iqr_multiple": args.median_shift_iqr,
            "warning_is_heuristic_not_a_statistical_test": True,
            "an_out_of_range_value_can_be_a_true_behavioural_anomaly_not_a_bad_input": True,
        },
        "preprocessing": preprocessing_audit,
        "reference_row_comparison": reference_check,
        "overall_assessment": overall,
        "outputs": {
            "feature_audit_csv": "live_feature_compatibility.csv",
            "summary_json": "live_input_audit.json",
        },
    }
    (output_dir / "live_input_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.prepared_output:
        prepared_output = Path(args.prepared_output)
        prepared_output.parent.mkdir(parents=True, exist_ok=True)
        prepared.to_csv(prepared_output, index=False)
        summary["outputs"]["prepared_input_csv"] = prepared_output.name
        (output_dir / "live_input_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("Live-input compatibility audit complete.")
    print(f"Rows audited: {len(prepared):,}")
    print(f"Schema: {summary['schema_status']}")
    print(f"Missing feature cells before imputation: {summary['total_missing_feature_cells_before_imputation']:,} ({summary['missing_feature_cell_pct']:.3f}%)")
    print(f"Missingness review threshold: {args.max_missing_cell_pct:.3f}%; triggered: {summary['missingness_warning']}")
    print(f"Observed values outside training min/max: {summary['total_observed_values_outside_training_minmax']:,} ({summary['observed_values_outside_training_minmax_pct']:.3f}%)")
    print(f"Distribution warning features: {len(drift_features)}")
    if reference_check is not None:
        print(f"Row-aligned transform reconstruction: {reference_check['feature_cell_match_pct']:.6f}% cells match")
        print(f"Maximum absolute difference: {reference_check['max_absolute_difference']:.3g}")
    print(f"Overall: {overall}")
    print(f"Reports: {output_dir.resolve()}")
    print("Note: distribution thresholds are engineering heuristics, not calibrated tests.")


if __name__ == "__main__":
    main()
