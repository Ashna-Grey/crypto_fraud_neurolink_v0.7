# Live-input compatibility audit — B6b Isolation Forest

## Scope

This audit checks whether the bundled raw live export can be converted into the exact 20-feature space expected by the fitted B6b model. It does not validate fraud-detection accuracy, because the Isolation Forest training dataset has no verified fraud labels.

## Verified preprocessing contract

The input `D2_live_wallet_dataset.csv` has 10,000 rows and includes all 20 required B6b features after trimming surrounding column-name whitespace. Fourteen features use the signed-log1p transformation

\[
T(x)=\operatorname{sign}(x)\log(1+|x|).
\]

Those features are: `total ether received`, `avg val received`, `Avg min between received tnx`, `max value received`, `ERC20 min val rec`, `min value received`, `total Ether sent`, `ERC20 avg val rec`, `ERC20 max val rec`, `Avg min between sent tnx`, `max val sent`, `avg val sent`, `ERC20 uniq sent token name`, and `current_eth_balance`. The other six model features are unchanged. Missing values are kept missing by the raw transformer; inference records the missingness and uses the fitted bundle's stored per-member medians.

## Results on the bundled raw/preprocessed pair

- Raw live input rows: **10,000**; model-required B6b features: **20/20 present** after column-name normalization.
- Transformed feature cells compared with the selected columns in `D2_IsolationForest_preprocessed_23_features.csv`: **200,000**.
- Feature cells matching within `rtol=1e-7`, `atol=1e-7`: **100.000000%**.
- Maximum absolute difference after applying the transformer and stored global training medians: **4.66 × 10⁻¹⁰**.
- Missing feature cells in the raw live file before inference imputation: **16,666 / 200,000 (8.333%)**.
- Observed values outside stored training min/max after transformation: **1**, or **0.001%** of observed feature cells.
- Default distribution-review heuristic triggered features: **0**.
- Dataset-level missingness review threshold: **5%** of model-feature cells; this input has **8.333%**, so the overall audit status is `SCHEMA_OK_INPUT_QUALITY_REVIEW` even though the schema matches and no default per-feature distribution warning triggered.

The row-aligned feature comparison is valid for this bundled pair because it represents the same 10,000-row source in the same order. Do not use row-aligned comparison for unrelated live batches.

## Interpretation

The raw live export can be transformed into the model's expected feature scale. Raw-live scoring should use `--input-format raw-live`; previously transformed input should use `--input-format preprocessed`. Using the wrong mode can double-transform values or feed raw values directly into the estimator.

The 8.333% raw feature-cell missingness remains operationally important: rows with missing values are imputed and flagged `WARN`; a row missing all 20 features is `UNUSABLE` and receives no numeric anomaly output. The one observed out-of-range value is a review signal, not automatically a bad input—it could be a real behavioural extreme.

The audit uses transparent engineering review heuristics: more than 5% of required feature cells missing, more than 10% of a feature’s observed values outside training min/max, or a live median shift greater than 3 training IQRs. These thresholds are not statistically calibrated and should be monitored on subsequent live batches. The audit verifies feature compatibility for this data pair, not compatibility with every future collector output or every future Ethereum population.

## Re-run

```powershell
python .\inference\audit_live_input.py `
  --input "C:\path\to\D2_live_wallet_dataset.csv" `
  --input-format raw-live `
  --reference-data "C:\path\to\D2_IsolationForest_preprocessed_23_features.csv" `
  --verify-row-alignment `
  --output-dir ".\analysis\live_input_audit"
```
