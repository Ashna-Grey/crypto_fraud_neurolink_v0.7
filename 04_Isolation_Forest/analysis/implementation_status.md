# B6b implementation status — Stage 1 μ / σ² integrated

## Implemented

- Frozen 20-feature B6b schema.
- Canonical 300-tree Isolation Forest for diagnostic compatibility.
- **25-member bootstrap Isolation Forest ensemble**, 300 trees per member.
- Each member uses a bootstrap sample and a member-specific out-of-bag score-reference distribution.
- `if_mu`: mean OOB-referenced member anomaly percentile on [0,1].
- `if_sigma_squared`: sample variance of member anomaly percentiles; bootstrap disagreement measure.
- `if_sigma`: square root of the variance.
- Mean raw anomaly score and raw-score variance for diagnostics.
- Training-median imputation; each bootstrap member has its own fitted medians.
- Input feature validation, raw-live signed-log1p preparation, data-quality flags, row-level explanations and wallet-ID passthrough.
- `if_prediction` as an explicit alias of the anomaly-priority band.
- `PASS`, `WARN`, and `UNUSABLE` input-quality statuses; fully missing feature rows have numeric anomaly outputs suppressed.
- Repeatable `inference/audit_live_input.py` with per-feature range/distribution diagnostics.
- Smoke test and nine unit tests.

## Verification for the updated bundle

- All 25 bootstrap members trained successfully on the supplied 10,000-row, 20-feature B6b representation.
- Smoke test passed.
- Nine unit tests passed.
- Raw-live inference scored all 10,000 rows successfully.
- The example full-dataset run reported mean `if_mu` 0.499399 and mean `if_sigma_squared` 0.00094405.
- In the raw-live run, priority bands were 9,030 `ROUTINE`, 495 `ELEVATED_ANOMALY`, 372 `HIGH_ANOMALY`, and 103 `EXTREME_ANOMALY`. Input-quality status was `PASS` for 6,054 rows and `WARN` for 3,946 rows due to missing feature values; no row had all 20 features missing.

These numbers are specific to the supplied source dataset and its model fit. Because the source rows contributed to model training, they are not independent test results. `PASS` means no required feature values were missing and no observed feature value exceeded the training min/max; it is not a general guarantee of input validity.

## Live-input compatibility audit

The bundled raw live export has all 20 required model features after surrounding column-name whitespace is removed. The verified signed-log1p transform plus stored global training median reconstruction matches the bundled 20-column preprocessed reference for all 200,000 feature cells within `rtol=1e-7`, `atol=1e-7` (maximum absolute difference `4.66e-10`). The raw export contains 16,666 missing feature cells (8.333%); one observed feature cell lies outside stored training min/max. No per-feature distribution warning heuristic triggered; however, raw feature-cell missingness was 8.333%, above the audit’s 5% dataset-level review threshold, so the overall status is `SCHEMA_OK_INPUT_QUALITY_REVIEW`. Details and per-feature metrics are in `analysis/live_input_compatibility.md` and `analysis/live_input_audit/`.

## Semantics and not-yet-implemented work

- `if_mu` is mean relative anomaly severity, **not a calibrated fraud probability**.
- `if_sigma_squared` quantifies bootstrap-member disagreement on that relative-anomaly scale, **not yet an error-calibrated predictive variance**.
- Do not use inverse variance weighting for fusion yet.
- To construct a fraud-probability mean comparable with `rf_mu_calibrated`, first establish a schema-compatible labelled calibration set, fit score-to-label calibration without test leakage, and evaluate it on a held-out test set.
- Conformal anomaly p-values would be a separate layer; they are not a substitute for either output.
