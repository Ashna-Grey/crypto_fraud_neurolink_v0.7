# v0.7: Empirically trained, calibrated evidence arbitration

## What replaced hand-picked rules?

In `FUSION_MODE=moe` (default), there are no hand-selected numerical voting weights or IF-versus-RF winner rules in the decision. The trained model estimates output probability from a context-gated mixture of three **label-supervised** estimates:

1. Original RF **raw** fraud probability.
2. Logistic IF evidence adapter, trained on D1 labels using original D2-trained IF percentile, IF applicability, variance, activity measurements.
3. Logistic rule evidence adapter, trained on D1 labels using eight original rule indicators, missing-rule flags, coverage, and activity.

The contextual gating features include original RF probability, IF anomaly percentile, IF applicability, wallet activity measurements, rule count/coverage and RF path contribution. A softmax gate is fitted by minimizing log loss, not manually fixed. The mixture is calibrated by sigmoid on a held-out labeled D1 calibration partition.

## Equations and attribution

For wallet evidence vector `z`, component estimates `p_RF(z)`, `p_IF_adapter(z)`, and `p_RULE_adapter(z)` are combined by a trained gate:

\[
w_k(z)=\frac{e^{g_k(z)}}{\sum_j e^{g_j(z)}},\qquad
p_{mix}(z)=\sum_k w_k(z)p_k(z),
\]

where `g_k(z)` are linear learned gating logits. Each adapter is a logistic classifier with learned coefficients. Let the held-out sigmoid map `p_mix` to:

\[
p_{cal}(z)=\sigma\left(a\,\operatorname{logit}(p_{mix}(z))+b\right).
\]

The learned `a,b` come from a D1 calibration partition. Original IF is **not** a fraud probability; its score becomes a *feature* of a supervised adapter only. Rule activation does not prove fraud and contributes only through the trained adapter and learned gating weights.

The final binary classification uses an **equal-cost 0.5 Bayes decision boundary** on `p_cal`. User-visible advisories do not override this classification. `confidence_set_90pct_nominal` is a separate conformal output fitted on a distinct D1 conformal partition and **has no guaranteed 90% coverage under distribution shift**.

## Interpretable comparison

`FUSION_MODE=gam` invokes the fitted logistic spline GAM with selected interactions, trained on out-of-fold feature records. All main and interaction log-odds terms add exactly. This is an original project application of standard additive logistic model mathematics, **not** a claim to have trained an InterpretML Explainable Boosting Machine. The optional actual EBM training script is `scripts/train_optional_ebm.py`; its output is not a deployed model.

## Data and safeguards

- The original D1 split metadata are used to assign roles. Exact feature-profile duplicates across split roles are screened out.
- Meta-training RF scores are generated through **fivefold out-of-fold RF training**, avoiding direct RF training-label reuse in the meta-features. Saved original RF is used for calibration and historical test partition predictions.
- IF model remains the D2-trained unlabeled original. Rule indicator extraction is deterministic with frozen thresholds where available.
- Separate calibration and conformal partitions are used. The historical test partition, due to earlier D1 exploratory experiments, does **not** constitute independent external validation.
- Rule and IF feature availability are included in the learned evidence vector; missing data are not silently presented as confirmed negative evidence.
- We do not have an independent labeled live Ethereum wallet dataset. The learned probability is calibrated for a D1 population under modeling assumptions, **not verified live fraud risk**.

## Reproducible findings

`fraud_reasoning/artifacts/historical_comparison_report.json`:

| Historical D1 metric (n=924, fraud=164) | Original RF calibrated | GAM | MoE |
|---|---:|---:|---:|
| ROC-AUC | 0.997733 | 0.996895 | 0.997713 |
| PR-AUC | 0.990848 | 0.987221 | 0.990848 |
| Brier (lower is better) | 0.014295 | 0.017840 | 0.013886 |

MoE gating overwhelmingly favored RF: **mean RF weight 0.9999871**. The paired bootstrap interval for MoE minus RF Brier includes zero. This means adding IF/rule experts **was not proven to improve predictive performance or to substantively contribute** in the current evaluation. Present this negative result transparently in research reporting.

## Key provenance

- Wolpert (1992): _Stacked generalization_, Neural Networks, DOI [10.1016/S0893-6080(05)80023-1](https://doi.org/10.1016/S0893-6080(05)80023-1).
- Jacobs, Jordan, Nowlan, Hinton (1991): _Adaptive Mixtures of Local Experts_, Neural Computation, DOI [10.1162/neco.1991.3.1.79](https://doi.org/10.1162/neco.1991.3.1.79).
- Vovk, Gammerman, Shafer (2005): _Algorithmic Learning in a Random World_, Springer DOI [10.1007/b106715](https://doi.org/10.1007/b106715).

These sources establish general methods, **not** the novelty, applicability, or empirical effectiveness of this particular Ethereum fraud pipeline.
