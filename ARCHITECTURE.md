# NeuroLink v0.7.0 architecture

```text
Wallet feature JSON (strict Pydantic schema)
               |
       Feature normalization
               |
     +---------+----------+------------------+
     |                    |                  |
 Original 20-feature RF   Original D2 IF     Eight rule modules
 raw & calibrated scores  anomaly percentile  fired/unfired/unavailable
 path group attributions  shift / missingness  exact predicates
     |                    |                  |
     +--------------------+------------------+
                          |
                  Evidence feature vector
                (raw scores, attributions,
                 coverage, activity context,
                 individual rule activations)
                          |
         +----------------+---------------+
         |                                |
     Trained MoE (default)         Trained spline GAM
     3 fraud-evidence experts      additive functions +
     contextual softmax gate       3 learned pairwise terms
         |                                |
   Independent label-held-out calibration (original D1 calibration partition)
                          |
      Separate split-conformal threshold (original D1 conformal partition)
                          |
       Fraud probability + class decision
       Gating weights or GAM attribution
       Rule and distribution caveats
       Recommended review advisory (no forced rule-based override)
                          |
                    POST /predict JSON
```

The original IF is unsupervised and never reports its anomaly percentile as fraud probability. MoE trains a **separate supervised mapping** of IF evidence to D1 fraud labels, as well as a supervised mapping from rule alerts. The RF baseline remains untouched. Routing and sigmoid calibrator coefficients were learned from labels, not chosen manually.

## Training partitions

The pre-existing D1 split metadata is reused, with exact all-feature duplicates removed across roles:

- 6,514 unique original RF train records -> fivefold RF out-of-fold meta-training.
- 923 unique original calibration records -> meta probability sigmoid fitting.
- 927 unique original conformal records -> nominal 90% prediction-set threshold.
- 924 unique original test records -> historical evaluation only.

The original RF was trained on the original RF train split and remains in deployment. Cross-fitted RF copies use the same 20 selected features but exclude validation-fold labels. These partitions are distinct after de-duplication, but **not externally independent**, and the original 20-feature selection and D1 exploratory work predate this implementation.

## Evidence dependence and limitations

- RF path contributions are exact tree-path probability differences, not SHAP values and not causal explanations.
- IF descriptions are reference-distribution *marginal* comparisons, not IF causal attributions.
- Rules report predicate outputs and missing dependencies, not negative fraud votes.
- Dependency and context annotations continue to be returned, but they do NOT manually override learned weights.
- The MoE might learn to ignore a weak expert, as it currently does on D1. This is not a defect that should be fixed by manually forcing expert votes.
