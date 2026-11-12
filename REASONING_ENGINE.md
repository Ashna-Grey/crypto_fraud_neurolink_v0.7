# Two different forms of reasoning in v0.7

## Deterministic factual context (retained, non-decisional)

`fraud_reasoning/logic.py` maps equivalent feature names, calculates an **average over an observed span** only when timestamps permit, and reports context such as high total volume with slow average activity. These are descriptive alerts, **not** conclusive fraud/legitimacy rules. Legacy policy thresholds in this module are **not** used to override the v0.7 learned binary prediction.

## Learned arbitration (decisional)

`fraud_reasoning/moe.py` uses three fraud-evidence experts: original RF raw estimate, a **supervised** logistic adapter of IF distributional evidence, and a **supervised** logistic adapter of rule conditions. Training constructs out-of-fold IF/rule adapter outputs for the gate and fits learned softmax routing weights via log-loss minimization. A post-hoc held-out sigmoid maps the routed result to a D1-calibrated fraud estimate.

`FUSION_MODE=gam` instead fits spline main effects and explicit RF×IF, RF×activity, and RF×rule interactions with regularized logistic coefficients. Its term values add exactly to calibrated log-odds.

The JSON exposes each original engine, source explanations, rule evidence coverage, contextual facts, model gate contributions, and the computed probability. A `review_recommended` advisory does not alter the primary binary decision. A nominal 90% conformal prediction set is provided separately.

## Interpreting a disagreement

If RF flags a wallet for transactions but the observed average activity rate is low, the system does not hardcode a choice to trust IF. It supplies the RF raw score, RF transaction-volume path attribution, IF relative anomaly, rule alerts, activity rate and missingness to a **trained** model. That model may learn to favor RF, IF-derived evidence, or rules on a supported distribution. On current D1, the learned MoE favors RF nearly exclusively. The evidence **does not** establish that IF is preferable for slow wallets.

Changing transaction rate without recalculating correlated wallet features is only a partial sensitivity test, not a causal counterfactual.
