# Isolation Forest μ and σ² methodology (Stage 1)

## Purpose

This stage adds a mathematically explicit output pair to the B6b Isolation Forest detector. The current 10,000-row source file has no verified fraud labels, so the pair represents **anomaly severity and bootstrap-model disagreement**, not fraud probability and not yet fusion-calibrated predictive variance.

## Bootstrap ensemble

Train \(B=25\) Isolation Forest members. Each member uses a bootstrap sample of size \(n\), sampled with replacement, then scores its out-of-bag (OOB) rows to construct a member-specific reference distribution.

For member \(b\), define the raw anomaly score

\[
s_b(x)=-\operatorname{score\_samples}_b(x),
\]

where larger scores mean greater abnormality. Convert that score to a smoothed empirical percentile using member \(b\)'s sorted OOB scores \(R_b\):

\[
u_b(x)=\frac{\#\{z\in R_b:s_b(z)\leq s_b(x)\}+0.5}{|R_b|+1},\qquad u_b(x)\in(0,1).
\]

The two requested outputs are

\[
\boxed{\mu_{IF}(x)=\frac{1}{B}\sum_{b=1}^{B}u_b(x)}
\]

and

\[
\boxed{\sigma^2_{IF}(x)=\frac{1}{B-1}\sum_{b=1}^{B}\left(u_b(x)-\mu_{IF}(x)\right)^2}.
\]

`if_mu` is the mean relative anomaly percentile. `if_sigma_squared` is sample variance across bootstrap-member percentile estimates and represents model disagreement on this relative-anomaly scale.

## Interpretation cautions

- `if_mu` is not \(P(\mathrm{fraud}\mid x)\).
- `if_sigma_squared` is not yet a validated predictive-error variance.
- Do not use \(w_{IF}=1/\sigma^2_{IF}\) for fusion until the scale and its relationship to observed error are validated, and the fused engines estimate a comparable target.
- Anomaly scores and percentiles are relative to a reference distribution; they do not confirm fraud.

## Next stage: supervised calibration

To produce an IF fraud-likelihood estimate comparable in meaning to `rf_mu_calibrated`, a compatible labelled calibration dataset is required. Verify schema equivalence, preprocessing, and row/wallet identity first. Fit a score-to-label calibrator on dedicated calibration data and evaluate Brier score, log-loss, expected calibration error, discrimination, and uncertainty/error relationships on an untouched test set. Any conformal anomaly p-value is a separate output, not a replacement for \(\mu\) or \(\sigma^2\).

## Relevant research

1. Pelletier, B. (2024). *On the statistical properties of the isolation forest anomaly detection method*. Electronic Journal of Statistics, 18(2), 4322–4381. DOI: [10.1214/24-EJS2305](https://doi.org/10.1214/24-EJS2305). Studies the statistical behaviour of Isolation Forest scores and limitations such as masking.
2. Garg, I. & Majumder, S. (2025). *On the Integration of Cross-Conformal Prediction, Ensembles, and Sampling for Uncertainty Quantification in One-Class Anomaly Detection*. Proceedings of Machine Learning Research, 266, 687–705. [Paper](https://proceedings.mlr.press/v266/garg25a.html). Combines sample/ensemble-based uncertainty with conformal calibration and notes that ensemble variability alone does not give statistical false-positive guarantees.
3. Hennhöfer, O., Kirsch, M. & Preisach, C. (2026). *Conformal Anomaly Detection in Python: Moving Beyond Heuristic Thresholds with nonconform*. Proceedings of Machine Learning Research, 329, 613–632. [Paper](https://proceedings.mlr.press/v329/hennhofer26a.html). Discusses calibrating anomaly scores into conformal p-values under exchangeability assumptions.
