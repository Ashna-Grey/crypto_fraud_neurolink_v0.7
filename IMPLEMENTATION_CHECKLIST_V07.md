# Implementation checklist, version 0.7

## Complete and locally verified

- [x] Preserve trusted trained RF, imputer, calibrator, and IF bootstrap ensemble.
- [x] Preserve eight original rule implementations, partial evidence and unavailability indicators.
- [x] Cross-fitting script for RF out-of-fold evidence.
- [x] Disjoint original D1 roles for meta-training, probability calibration, conformal calibration, and historical evaluation; deduplicate identical feature profiles across roles.
- [x] Learned logistic adapters for IF and rule evidence.
- [x] Learned context-dependent softmax MoE gate, fitted on labeled D1 training evidence.
- [x] Held-out sigmoid probability calibration.
- [x] Fitted regularized spline GAM benchmark with attributable main and pairwise effects.
- [x] Separate conformal calibration and nominal prediction sets.
- [x] RF original attribution and IF descriptive anomaly evidence remain in JSON.
- [x] API exposes trained fusion, component outputs, warnings, and model status.
- [x] Portable JSON of meta models included for inference.
- [x] Original RF/IF serialized artifacts retained.
- [x] Local pytest suite and API regression runner.
- [x] Render Blueprint, Dockerfile and environment examples.
- [x] Historical evaluation comparing RF, GAM and MoE, with conditional paired bootstrap and activity subgroups.

## NOT verified / blocked

- [ ] Independent external labeled Ethereum wallet population, required for valid generalization and live calibration assessment.
- [ ] Real InterpretML EBM fitted and deployed. Only optional executable experiment is provided; active GAM is not EBM.
- [ ] Demonstrable incremental learned IF/rule contributions: current MoE gate gives ~99.9987% weight to RF.
- [ ] Render deployment of v0.7 under the user's account; only local running and tests verified.
- [ ] Docker image built on this host (Docker not available in evaluation runtime).
- [ ] Full rule input coverage for contracts/ERC20 (source dataset missing fields).
- [ ] Source of truth transaction-level timestamps for real burst/dormancy reasoning.
- [ ] Independent causal or forensic validation of explanation statements.
- [ ] Authentication, rate limiting, production monitoring and model drift intervention.

## Scientific limitations

- [ ] Report the D1 results as a *historical/internal* evaluation, not an untouched external holdout.
- [ ] Explain that D1 has been repeatedly inspected in prior experiments, and the original RF feature selection and calibration artifacts were not newly generated for an untouched experiment.
- [ ] Report model selection and cost-sensitive decision policies on separate datasets if available. The current .5 decision boundary assumes equal costs.
- [ ] Do not claim statistically significant Brier improvement: the bootstrap interval crosses zero.
- [ ] Do not present the original IF anomaly score as a fraud probability; only the separately fitted supervised IF evidence adapter outputs such an estimate.
