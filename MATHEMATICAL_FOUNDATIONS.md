# Mathematical foundations: NeuroLink v0.7.0

All stated equations are either standard published components or explicit engineering implementations. **Their combination does not establish novelty.** Numerical performance claims require independent external evaluation.

## 1. Stacked generalization and out-of-fold evidence

Wolpert (1992) [R1] proposed second-stage learning from base learners' out-of-sample predictions. Here, for each D1 training record x, RF raw score and RF tree-path contributions are produced from an RF fit that excluded that record's fold. A fixed original D2-trained Isolation Forest provides anomaly measurements without D1 labels. The original rule predicates provide behavior flags without using the record's label.

RF feature groups use an exact additive path decomposition of raw tree probability, not a causal explanation or SHAP attribution. These feature groups, IF anomaly severity, token/transaction context and rule flags form evidence vector z(x).

The original IF bootstrap output is **not** a fraud probability. A supervised logistic adapter on IF-derived evidence is part of the proposed MoE and uses fraud labels to learn a D1-conditional probability.

## 2. Contextual Mixture of Experts (default)

Jacobs, Jordan, Nowlan and Hinton (1991) [R2] introduced adaptive mixtures with learned gates. Our implementation is an explicit, supervised adaptation:

\[
p_{RF}(x)=p_{RF,raw}(x),\quad
p_{IF}(x)=\sigma(\beta_{IF}^\top z_{IF}(x)),\quad
p_R(x)=\sigma(\beta_R^\top z_R(x)).
\]

The latter two are label-fitted fraud-evidence adapters, not the unsupervised IF score or rule score themselves. Their **out-of-fold** training predictions are used to fit the gate. Full adapters are refitted on meta-training rows for inference.

\[
w_k(x)=\frac{\exp(g_k(x))}{\sum_j\exp(g_j(x))},\quad
p_{mix}(x)=\sum_k w_k(x)p_k(x).
\]

In this implementation g_k is affine in standardized context features and the weights are fitted by minimizing Bernoulli log loss with L2 regularization. Each gate-feature logit contribution is reproducible exactly. A high gate weight does NOT mean an expert is correct with that probability.

## 3. Sigmoid probability calibration

Calibration is trained on a split separate from the MoE/GAM meta-training set [R3]. For the MoE, the held-out calibration labels fit:

\[
p_{cal}(x)=\sigma\left(a\,\mathrm{logit}\bigl(p_{mix}(x)\bigr)+b\right).
\]

Here a and b are fitted with logistic regression on held-out D1 calibration records, not manually chosen. A 0.5 decision threshold is the conventional equal-misclassification-cost Bayes classification boundary under calibrated probabilities and matching deployment class distributions, **not** a universal cost-optimized fraud threshold.

## 4. Additive GAM with pairwise spline interactions (comparison mode)

Following the general additive + pairwise idea of Lou et al. (2013) [R4], our implemented model uses regularized spline bases and three selected tensor-product spline interactions:

\[
\mathrm{logit}\,p_{GAM}(x)=\beta_0+\sum_j f_j(z_j(x))
+f_{RF,IF}(z_{RF},z_{IF})+f_{RF,rate}(z_{RF},z_{rate})
+f_{RF,rules}(z_{RF},z_{rules}).
\]

All spline coefficients are trained with penalized logistic loss. The resulting logit is sigmoid-calibrated on the separate D1 calibration partition. Exact additive contributions **after** the sigmoid-calibration logit transformation are returned in `learned_arbitration.terms` for `FUSION_MODE=gam`.

**This is not InterpretML's Explainable Boosting Machine**. A separate optional EBM experiment is supplied, but was not trained or executed in the offline package-building environment.

## 5. Split-conformal prediction sets

For the distinct original conformal partition of n=927 records, compute nonconformity s_i=1-p_cal(y_i|x_i). At nominal miscoverage alpha=0.1 use [R5]:

\[
q=\mathrm{Quantile}_{\lceil(n+1)(1-\alpha)\rceil/n}(\{s_i\}),\quad
\Gamma(x)=\{y:1-p_{cal}(y\mid x)\le q\}.
\]

The returned set supplements the automatic binary decision. Its marginal coverage guarantee requires an exchangeable, correctly separated calibration/test distribution. Dataset shift or previous D1 development can invalidate practical coverage expectations. The historical test is **not** an independent source population.

## 6. What is proposed rather than established?

Using original RF path attributions, D2 IF applicability indicators and rule provenance in an evidence vector with a contextual learned gate is our *implementation hypothesis*. No individual feature contribution proves fraud. The formulas above have published precedents, and empirical results here do not demonstrate a new superior method.

## 7. Historical experimental result

924 unique D1 evaluation records (164 fraud); see `fraud_reasoning/artifacts/historical_comparison_report.json`:

| Model | ROC-AUC | PR-AUC | Brier score |
|---|---:|---:|---:|
| Original calibrated RF | 0.997733 | 0.990848 | 0.014295 |
| Learned MoE | 0.997713 | 0.990848 | 0.013886 |
| Learned spline GAM | 0.996895 | 0.987221 | 0.017840 |

The paired 95% bootstrap interval for MoE Brier minus RF Brier is [-0.001388, +0.000397], which **includes zero**. We cannot claim a statistically supported improvement. MoE assigns almost all its weight to RF on this evaluation population. Performance on genuine live wallets remains unknown.

## Bibliography

See `RESEARCH_REFERENCES.md` for verified DOI/publisher links to [R1–R5].

---

# v0.7 calibrated learned alternatives (DEFAULT)

**All v0.6 heuristics above are legacy only** (`FUSION_MODE=legacy`). The v0.7 default is the supervised MoE described in [LEARNED_ARBITRATION_V07.md](LEARNED_ARBITRATION_V07.md). In `FUSION_MODE=gam`, a regularized spline logistic meta-model is used instead. These learned approaches should not be called the original belief-fusion formula, nor the subjective-logic consensus operator.

The MoE uses three *labeled fraud-evidence predictions*, each derived from one engine's evidence. The IF and rule predictions are supervised adapters and are **not** the original engine's anomaly score or raw rule score. The local gating logit coefficients are fitted to D1 labels using proper binary cross-entropy loss. The fitted gate `softmax(g_k(z))` and mixture `sum_k w_k(z)*p_k(z)` follow the standard mixture-of-experts formulation (Jacobs et al., 1991). The held-out sigmoid calibration `sigma(a*logit(p_mix)+b)` is an application of standard logistic probability calibration. See [LEARNED_ARBITRATION_V07.md](LEARNED_ARBITRATION_V07.md) for partition roles and the exact empirical comparison.

The GAM alternative models log-odds as `beta_0 + sum_j f_j(z_j) + sum_(i,j) f_ij(z_i,z_j)`, with basis functions implemented using `SplineTransformer` and learned regularized logistic coefficients. It is **not** an InterpretML EBM.

The **0.5 classification boundary** is an assumed equal-misclassification-cost decision rule, not a newly measured optimal cost-sensitive fraud threshold. The nominal conformal 90% set uses a separate calibration partition but has no distribution-free validity guarantee under the observed D1/D2 population shift.
