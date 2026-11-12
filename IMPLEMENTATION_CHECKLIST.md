# NeuroLink v0.7.0 implementation checklist

## Implemented and locally executed

- [x] Preserve original trained 20-feature RF and 20-feature IF bootstrap artifacts.
- [x] Reuse eight expert rule modules, report missing inputs/thresholds.
- [x] Exact original RF tree-path raw-probability attribution and group-level decomposition.
- [x] IF anomaly percentile separate from fraud probability.
- [x] Cross-fitted supervised RF inputs for learned meta-training (6,514 unique training records).
- [x] Separate supervised IF and rule fraud-evidence adapters with out-of-fold gate-training outputs.
- [x] Trained context-conditioned softmax Mixture-of-Experts routing model.
- [x] Trained regularized tensor-spline GAM comparison model.
- [x] Held-out logistic sigmoid probability calibration (923 records).
- [x] Independent conformal-quantile calculation (927 distinct original conformal records).
- [x] Historical, deduplicated 924-record D1 evaluation and original RF comparison.
- [x] Conditional paired bootstrap comparisons and historical subgroup/calibration summaries.
- [x] Portable JSON artifact inference for MoE and GAM, avoiding a new sklearn model persistence dependency.
- [x] FastAPI routes, strict validation, batch inference, docs, tests, Dockerfile, render.yaml.
- [x] Local end-to-end inference with original RF, IF, and rules.

## Implemented but NOT executed / validated

- [ ] Optional true InterpretML EBM training script (`scripts/train_optional_ebm.py`). Could not install interpret-core locally.
- [ ] Docker image build. Docker daemon unavailable in this environment.
- [ ] Render live v0.7.0 deployment and load/health checks.

## Blocked by missing source data / external validity

- [ ] Independent, newly labeled Ethereum wallet population with verified fraud provenance.
- [ ] Reliable timestamped transaction histories for burst and dormancy verification.
- [ ] Complete contract and ERC20 outgoing-address rule inputs/thresholds.
- [ ] Demonstrated benefit of IF/rules over RF alone, with statistically valid external comparison.
- [ ] Calibrated probability quality and subgroup robustness on new live-wallet distributions.

The project is **deployable as packaged and locally tested**, but is not a scientifically validated new fraud detector or forensic proof tool.
