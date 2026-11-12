# NeuroLink v0.7 | Calibrated learned evidence arbitration

A **Render-ready FastAPI research prototype** with the original deployed Random Forest, D2-trained Isolation Forest, eight deterministic rules, and two **label-trained** arbitration options:

* **Contextual Mixture of Experts (MoE), default:** the IF and rule engine scores are mapped to labeled fraud evidence by supervised adapters; a learned softmax gate combines those adapters with original RF predictions. A held-out sigmoid fits the final historical-D1 probability scale.
* **Regularized spline logistic GAM (`FUSION_MODE=gam`):** a learned, interpretable additive logistic meta-model with three interactions, out-of-fold RF predictions and a separately held-out sigmoid calibrator. This is **not** the proprietary InterpretML Explainable Boosting Machine (EBM); an optional actual EBM experiment is in `scripts/train_optional_ebm.py`.
* **Legacy (`FUSION_MODE=legacy`):** earlier provisional hand-weighted fusion, included for reproducibility only and **not** recommended for predictions.

The three base engines still report their own outputs and evidence. **IF anomaly scores are not fraud probabilities**, and nonfiring or missing rules are not evidence of legitimacy. The learned MoE's supervised IF and rule adapters are explicitly separate from the original IF and rule outputs.

## Important empirical finding

On 924 reused, duplicate-screened D1 historical evaluation wallets:

| Model | ROC-AUC | PR-AUC | Brier score (lower better) |
| --- | ---: | ---: | ---: |
| Original calibrated RF | 0.997733 | 0.990848 | 0.014295 |
| Learned spline GAM | 0.996895 | 0.987221 | 0.017840 |
| Contextual MoE | 0.997713 | 0.990848 | 0.013886 |

MoE assigned **99.9987% mean gating weight to RF**. The IF and rule adapters did not produce meaningful incremental contribution on these historical D1 cases. MoE Brier improvement vs original RF was **not statistically established** by paired record-level bootstrap (95% CI for MoE minus RF: [-0.00138804, 0.00039696]). Thus **no novelty or improvement is claimed**. These data were repeatedly used for earlier exploratory work and are **not independent external validation**. The estimated probabilities should not be treated as proven calibrated on new blockchain populations.

## Quick local start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m uvicorn fraud_reasoning.api:app --host 127.0.0.1 --port 8000
```

From another terminal:

```bash
python scripts/test_backend.py --url http://127.0.0.1:8000
```

The local Python interpreter should match the serialized RF's **scikit-learn 1.9.0** (pinned in requirements). `Dockerfile` uses Python 3.13 and installs these exact package dependencies. Never load a model pickle from an untrusted source.

## Docker / Render

```bash
docker compose up --build
```

Push this project root, including saved trained artifacts, to GitHub (private repository recommended). Render: New > Blueprint > select repo. The `render.yaml` sets `runtime: docker` and health check `/health`, with `PORT` auto-provided. A full account-level deployment cannot be verified here. `FUSION_MODE` defaults to `moe`; set `FUSION_MODE=gam` to compare the learned GAM without redeploying artifacts.

API: `/health`, `/schema`, `/models/status`, `/predict`, `/predict/batch`, `/learned/metrics`, `/learned/evidence-schema`, `/reasoning/rules`, `/research/methodology`, `/docs`.

## Reproduce training and historical calibration

Input: original labeled CSV at `01_Dataset/processed/ethereum_fraud_25_features.csv`, original RF training/calibration/test split metadata and D2-trained IF artifacts.

```bash
python scripts/train_learned_arbitration.py
python scripts/train_contextual_moe.py
python scripts/validate_learned_models.py
pytest -q
```

The first script uses 5-fold cross-fitted RF predictions on the original RF **training** portion to construct meta-training data, then evaluates separate original RF calibration, conformal and historical test partitions. It removes exact duplicate profiles across these roles. The IF uses unlabeled D2. A **historical internal check** is not external validation or proof that the learned arbitration generalizes.

For locally repeatable individual requests: `python scripts/test_backend.py --url http://127.0.0.1:8000`. For Render: supply the Render service URL instead. For optional real EBM training, see `requirements-ebm.txt` and `scripts/train_optional_ebm.py`, which is an **experiment only** and not a deployed model.

## Files

- `fraud_reasoning/`: actual API, adapters, feature attribution, reasoning, learned GAM/MoE
- `fraud_reasoning/artifacts/`: trained portable models and historical evaluation report
- `tests/`: unit and integration tests (default pytest collection)
- `scripts/test_backend.py`: live API health and inference checks
- `README_DEPLOY.md`, `LEARNED_ARBITRATION_V07.md`, `MATHEMATICAL_FOUNDATIONS.md`, `TEST_REPORT.md`: detailed methodology and limitations

**Do not deploy this publicly without authentication, rate limiting, and model monitoring. A fraud-risk prediction is not forensic proof of wrongdoing.**
