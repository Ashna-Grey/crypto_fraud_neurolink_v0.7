# Verified research sources and relevance

These sources ground the **established** mathematical components, not a claim that the NeuroLink combination is published or newly discovered.

- **[R1] David H. Wolpert (1992)**. *Stacked Generalization*. Neural Networks 5(2), 241–259. DOI: https://doi.org/10.1016/S0893-6080(05)80023-1 . Out-of-fold model outputs as second-stage learning inputs.
- **[R2] Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, Geoffrey E. Hinton (1991)**. *Adaptive Mixtures of Local Experts*. Neural Computation 3(1), 79–87. DOI: https://doi.org/10.1162/neco.1991.3.1.79 . Learned softmax gating and expert mixtures.
- **[R3] scikit-learn developers**. *Probability Calibration* / CalibratedClassifierCV documentation. https://scikit-learn.org/stable/modules/calibration.html . Calibration must avoid using in-sample training predictions; this implementation uses a held-out original D1 calibration split.
- **[R4] Yin Lou, Rich Caruana, Johannes Gehrke, Giles Hooker (2013)**. *Accurate Intelligible Models with Pairwise Interactions*. KDD 2013, 623–631. DOI: https://doi.org/10.1145/2487575.2487579 . Additive models with pairwise effects. The NeuroLink spline GAM is a separate implementation, not the authors' original training algorithm or InterpretML EBM.
- **[R5] Vladimir Vovk, Alexander Gammerman, Glenn Shafer (2005)**. *Algorithmic Learning in a Random World*. Springer. DOI: https://doi.org/10.1007/b106715 . Conformal prediction under relevant exchangeability assumptions.
- **InterpretML (software reference)**. `ExplainableBoostingClassifier` documentation: https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html . Optional alternative training script `scripts/train_optional_ebm.py`, not included as a pretrained model.

## Scientific caution

Stacking, MoE, GAN/GAM, IF, Random Forest, conformal prediction and score calibration are established ideas. The project provides an engineering combination and an experimental hypothesis, **not** a demonstrated algorithmic novelty or proof of fraud. The current D1 evaluation source has been repeatedly explored and cannot be described as pristine external holdout validation.

## v0.7 learned stacking and gating (publisher metadata verified 2026-11-12)

6. **David H. Wolpert (1992). _Stacked generalization_.** *Neural Networks*, 5(2), 241–259. DOI: [10.1016/S0893-6080(05)80023-1](https://doi.org/10.1016/S0893-6080(05)80023-1). Establishes use of out-of-sample base predictions for a second-level learner. Our exact evidence feature design is a **project-specific extension**, not attributed to Wolpert.
7. **Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, Geoffrey E. Hinton (1991). _Adaptive Mixtures of Local Experts_.** *Neural Computation*, 3(1), 79–87. DOI: [10.1162/neco.1991.3.1.79](https://doi.org/10.1162/neco.1991.3.1.79). Establishes experts with a learned gating network. Our sigmoid-calibrated heterogeneous evidence adapters are a project implementation, not a copied validated fraud method.
8. **Vladimir Vovk, Alexander Gammerman, Glenn Shafer (2005). _Algorithmic Learning in a Random World_.** Springer. DOI: [10.1007/b106715](https://doi.org/10.1007/b106715). Foundational conformal prediction. Our held-out classification sets rely on exchangeability assumptions; the research results **do not validate their coverage under D1/D2 drift**.
