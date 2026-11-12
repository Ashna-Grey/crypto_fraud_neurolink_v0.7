# 04 — Isolation Forest (B6b)

This module provides an unsupervised anomaly-ranking component for the Ethereum wallet feature dataset. It remains separate from the supervised Random Forest and deterministic Rule Engine until the decision-fusion layer is implemented and validated.

## Current configuration

- Algorithm: `sklearn.ensemble.IsolationForest`
- Feature set: **B6b, 20 features**
- Trees per model: `300`
- Bootstrap ensemble size: `25` members by default
- `contamination="auto"`
- Base seed: `42`
- Scaling: none
- Raw member score: `-model.score_samples(X)`; larger means more anomalous
- Each ensemble member has its own out-of-bag (OOB) score reference

B6b was selected as a working configuration after experiments B1–B22. It is a stability-based engineering choice, not proof of superior fraud-detection accuracy.

## The current μ and σ² output contract (Stage 1)

The current data file does not contain verified fraud labels. Therefore the implemented pair describes **relative anomaly severity and bootstrap-model disagreement**, not fraud probability and not yet a fusion-calibrated predictive variance.

For member \(b\), define the raw anomaly score:

\[
s_b(x)=-\operatorname{score\_samples}_b(x)
\]

Each member is trained on a bootstrap sample and scores its out-of-bag rows to build a sorted score reference \(R_b\). Convert a new row's score to a smoothed empirical percentile:

\[
u_b(x)=\frac{\#\{z\in R_b:s_b(z)\leq s_b(x)\}+0.5}{|R_b|+1},\quad u_b(x)\in(0,1).
\]

The output pair is:

\[
\boxed{\mu_{IF}(x)=\frac{1}{B}\sum_{b=1}^{B}u_b(x)}
\]

\[
\boxed{\sigma^2_{IF}(x)=\frac{1}{B-1}\sum_{b=1}^{B}\left(u_b(x)-\mu_{IF}(x)\right)^2}
\]

The CSV fields are:

- `if_mu`: mean member-specific anomaly percentile on [0,1]; **not fraud probability**.
- `if_sigma_squared`: sample variance of member-specific anomaly percentiles; a measure of bootstrap-model disagreement on the relative-anomaly scale.
- `if_sigma`: square root of `if_sigma_squared`.
- `anomaly_score`: mean raw anomaly score across ensemble members.
- `anomaly_score_variance`: sample variance of raw scores across members (diagnostic only).
- `anomaly_percentile`: `100 * if_mu`.
- `anomaly_priority_band`: `ROUTINE`, `ELEVATED_ANOMALY`, `HIGH_ANOMALY`, or `EXTREME_ANOMALY`, based on `anomaly_percentile`.
- `missing_feature_count`: required input values replaced by member-specific training medians.
- `outside_training_minmax_count`: required feature values outside the observed training ranges.
- `data_quality_flag`: `OK`, `MISSING_VALUES_IMPUTED`, `OUTSIDE_TRAINING_RANGE`, `ALL_REQUIRED_FEATURES_MISSING`, or the applicable semicolon-separated combination.
- `marginal_feature_explanation`: reference-tail summary; not SHAP, causal evidence, or an exact tree-path explanation.
- Extra columns such as wallet identifiers are preserved as passthrough fields.

**Do not use** `1 / if_sigma_squared` as a fusion weight yet. First validate whether the variance predicts meaningful error/reliability differences. To make `if_mu` comparable in meaning to `rf_mu_calibrated`, a compatible labelled calibration dataset and held-out evaluation are required.

## Repository tree

```text
04_Isolation_Forest/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   ├── feature_schema.json
│   └── model_config.json
├── artifacts/
│   ├── isolation_forest_b6b.joblib
│   ├── feature_schema.json
│   ├── model_config.json
│   ├── oof_score_reference.csv
│   ├── oof_fold_summary.json
│   ├── bootstrap_ensemble_summary.json
│   ├── training_audit.json
│   ├── training_report.json
│   └── README.md
├── training/
│   └── train.py
├── inference/
│   ├── __init__.py
│   ├── predict.py
│   └── test_inference.py
├── src/
│   ├── __init__.py
│   └── isolation_forest_utils.py
├── tests/
│   ├── smoke_test.py
│   └── test_pipeline.py
└── analysis/
    ├── experimental_summary.md
    ├── implementation_status.md
    ├── mu_sigma_methodology.md
    └── Isolation_Forest_Experimental_Report_B1-B22.pdf
```

## Installation (Windows PowerShell)

Run from this folder:

```powershell
py -3.13 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The bundled `.joblib` was generated with Python 3.13.5, scikit-learn 1.8.0, NumPy 2.3.5, pandas 2.2.3 and joblib 1.5.3. Pickle/joblib artifacts are environment-sensitive; use the pinned dependencies. If the versions differ, retrain rather than assuming compatibility.

## Tests

```powershell
python .\tests\smoke_test.py
python -m unittest discover -s tests -v
```

Tests use synthetic rows based on stored feature medians; the original full dataset is not needed to execute the tests.

## Predict on a CSV

Input must contain all 20 B6b features. A 23-feature source CSV is accepted; the three excluded transaction-count features are ignored by the estimator. Extra columns are preserved where possible.

```powershell
python .\inference\predict.py `
  --input "C:\path\to\wallet_features.csv" `
  --output "C:\path\to\B6b_predictions.csv"
```

## Retrain

```powershell
python .\training\train.py `
  --input "C:\path\to\D2_IsolationForest_preprocessed_23_features.csv" `
  --output-dir ".\artifacts" `
  --ensemble-size 25 `
  --n-estimators 300 `
  --random-state 42
```

The training command constructs a 25-member bootstrap ensemble by default, computes an OOB score reference for each member, saves the model bundle and writes audit/summary artifacts. It also constructs a five-fold out-of-fold raw-score reference for diagnostics. Retraining overwrites artifacts; keep a backup if needed.

## Interpretation and limitations

The available 10,000-row dataset has 23 numeric features but no verified fraud labels or wallet identifiers. As a result:

- anomaly scores and `if_mu` are **not fraud probabilities**;
- `if_sigma_squared` measures bootstrap-model disagreement on anomaly-percentile values, and is **not yet a validated predictive variance suitable for inverse-variance fusion**;
- no precision, recall, F1, ROC-AUC, PR-AUC or fraud yield is claimed;
- row indexes cannot be mapped to Ethereum addresses without a valid identifier source;
- the 94 cross-configuration consensus candidates are exploratory anomaly candidates, not confirmed fraudulent wallets;
- possible caps in count features need confirmation from the original preprocessing/collection pipeline;
- the supplied fitted artifact was trained on the supplied dataset; predictions on those same rows are in-sample diagnostics, not independent evaluation.

See [`analysis/mu_sigma_methodology.md`](analysis/mu_sigma_methodology.md) for formulas and relevant research.

## Live-input compatibility and quality diagnostics

The live collector's raw export is **not directly in model feature space**. Use `--input-format raw-live` to trim surrounding whitespace from field names and apply the verified signed-log1p transform to the 14 fields listed in `config/feature_schema.json`. The remaining six B6b features are passed through unchanged. Do not select `raw-live` for a CSV that has already been transformed; doing so would transform those values twice.

### Audit before deployment

```powershell
python .\inference\audit_live_input.py `
  --input "C:\path\to\D2_live_wallet_dataset.csv" `
  --input-format raw-live `
  --reference-data "C:\path\to\D2_IsolationForest_preprocessed_23_features.csv" `
  --verify-row-alignment `
  --output-dir ".\analysis\live_input_audit"
```

Use `--verify-row-alignment` only when the input and reference files are known to contain the same wallets in exactly the same row order. It reconstructs the selected model-feature matrix after applying the stored global training medians to missing values. For unrelated future/live batches, omit both `--reference-data` and `--verify-row-alignment`; the audit will still compare the batch to model feature ranges and stored feature distributions.

The audit writes `live_input_audit.json` and `live_feature_compatibility.csv`. Per-feature diagnostics include missing percentage, training min/max and quantiles, live quantiles, out-of-training-range frequency, and median shift in training-IQR units. The default warning heuristics are more than 5% of required feature cells missing, more than 10% of observed values outside training min/max for a feature, or a live median shift above 3 training IQRs. These are engineering review heuristics—not hypothesis tests, calibrated drift detectors, or proofs that a wallet is malicious. Out-of-range values can be authentic behavioural extremes.

### Score a raw live export

```powershell
python .\inference\predict.py `
  --input "C:\path\to\D2_live_wallet_dataset.csv" `
  --input-format raw-live `
  --output "C:\path\to\B6b_live_predictions.csv"
```

### Input quality fields in prediction output

- `if_prediction`: an alias of `anomaly_priority_band`; values are `ROUTINE`, `ELEVATED_ANOMALY`, `HIGH_ANOMALY`, `EXTREME_ANOMALY`, or `NOT_SCORED_INPUT_UNUSABLE`.
- `input_quality_status`: `PASS`, `WARN`, or `UNUSABLE`.
- `missing_feature_count` / `missing_feature_ratio`: missing required feature values before imputation.
- `outside_training_minmax_count` / `outside_training_minmax_features`: observed features outside the model's training min/max.
- `data_quality_flag`: machine-readable reason(s), including `MISSING_VALUES_IMPUTED`, `OUTSIDE_TRAINING_RANGE`, and `ALL_REQUIRED_FEATURES_MISSING`.

A row with all 20 required model features missing is marked `UNUSABLE`; its numeric anomaly outputs are suppressed rather than reporting a meaningless score of an all-median vector. Rows with some missing values can still be scored after imputation, but their quality status remains `WARN`. A range warning is deliberately not a hard failure because a genuine behavioural anomaly can be outside the training range.

### Exact output meanings

| Output | Meaning | Not the same as |
|---|---|---|
| `if_mu` | Mean of the 25 members' OOB-referenced anomaly percentiles, in [0, 1]. | Fraud probability or calibrated confidence. |
| `if_sigma_squared` | Sample variance (ddof=1) of member anomaly percentiles. | Calibrated predictive-error variance or a validated fusion weight. |
| `if_sigma` | Square root of `if_sigma_squared`. | Standard error of a fraud-probability estimate. |
| `anomaly_score` | Mean raw anomaly score `-score_samples(X)` across members; higher means more anomalous. | A probability. |
| `anomaly_score_variance` | Sample variance of the members' raw anomaly scores; diagnostic scale differs from `if_sigma_squared`. | A probability-scale uncertainty estimate. |
| `anomaly_percentile` | `100 * if_mu`. | Percent chance of fraud. |
| `anomaly_priority_band` / `if_prediction` | Relative anomaly-severity band at the configured percentile cutoffs. | Confirmed fraudulent/non-fraudulent classification. |
| `marginal_feature_explanation` | The most extreme feature-wise marginal reference percentiles. | SHAP value, causal explanation, or exact tree-path attribution. |

No new fraud probability is introduced by this hardening pass. `if_mu` remains anomaly severity; `if_sigma_squared` remains within-IF ensemble disagreement. Do not use inverse-variance weighting until uncertainty is validated against an appropriate target.
