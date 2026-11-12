# Known limitations — v0.7.0

1. **No independent external fraud confirmation.** D1 has been reused for years of development and evaluation. The 924-record historical test is separated from this version's fitting roles but is not pristine evidence of generalization.
2. **Original RF dominates the MoE.** Mean RF gate weight on the historical D1 test is about 0.999987. IF/rule adapters add effectively negligible weight. This provides no evidence that three-engine routing currently outperforms RF.
3. **No statistically established gain.** MoE Brier minus RF Brier paired 95% CI includes zero; GAM performs worse than RF on historical test ROC-AUC, PR-AUC and Brier.
4. **IF is anomaly detection, not fraud classification.** The IF-derived logistic adapter uses D1 labels; its scores must not be confused with pure IF fraud probabilities.
5. **Partial rule features and thresholds.** The original contract and ERC20-sent-address rules frequently cannot be evaluated from D1 aggregate inputs. Absence of triggered alerts is not proof of legitimacy.
6. **Feature-space and coverage shift.** D2 contains no transaction counts >200. The IF artifact does not directly consume transaction count, and its applicability penalty is a conservative proxy, not calibrated OOD probability.
7. **Temporal evidence limitations.** Average transaction rate over the observed first-to-last transaction span is not an actual burst detector or a dormant-wallet behavior guarantee.
8. **Old RF model artifact compatibility.** RF/IF serialized under sklearn 1.9; the local test image used sklearn 1.8 and produced warnings. The Render Docker image pins sklearn 1.9. Docker and Render runtime were not executed in the build environment.
9. **EBM optional.** `interpret-core` could not be installed offline, so InterpretML EBM was not trained. The deployed fitted GAM is **not** EBM; a reproducible optional training script is included.
10. **Calibration scope.** Independent logistic calibration of the learned model was fitted on original D1 calibration indices; conformal quantile on disjoint original conformal indices. The original RF calibration used the same original calibration labels, but meta predictions are formed from raw RF scores. None guarantees transfer under prevalence drift or live-wallet shift.
11. **No causal fraud proof.** Explanations describe model computations and associations, not verified fraud identity or a forensic proof.
12. **D1-specific label-missingness artifacts.** Earlier experiments found missing ERC20 fields associated only with a substantial subset of fraudulent records, potentially inflating apparent discrimination. Correct cross-population validation remains essential.
13. **No production-grade security.** Add authentication, quotas, secrets management, privacy controls and live input schema monitoring before public exposure.
