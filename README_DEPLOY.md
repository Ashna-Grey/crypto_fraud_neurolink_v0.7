# NeuroLink v0.7, Render and local deployment

## Version and mode

This package includes **two fitted learned models**. The default `FUSION_MODE=moe` uses the contextual mixture-of-experts trained on original D1 partitions; `FUSION_MODE=gam` uses a separately trained regularized spline GAM. `FUSION_MODE=legacy` loads the previous manually tuned v0.6 policy for scientific comparison only. The optional InterpretML EBM training script is not fitted or deployed by default.

## Render deployment

1. Unzip the project and put the **contents of `Crypto-fraud-intelligence-main`** at the root of a Git repository (or select that folder as Render root).
2. Create a Render Blueprint for that repository; `render.yaml` configures Docker and `/health`.
3. Render supplies `PORT`, so no environment variables are required. Optional `FUSION_MODE=moe` or `FUSION_MODE=gam`.
4. Deploy, then visit `https://YOUR-SERVICE.onrender.com/docs` and `/models/status`.
5. Confirm `version: 0.7.0` and `learned_arbitration.status: loaded`.
6. Run `python scripts/test_backend.py --url https://YOUR-SERVICE.onrender.com --output render_v07_results.json` from your laptop.

The saved RF uses scikit-learn 1.9.0. Docker pins this version. A prior local session loaded it with 1.8.0, with compatibility warnings; production must use the pinned version. No actual Docker build or Render deployment is asserted here.

## Local testing

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m uvicorn fraud_reasoning.api:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
python scripts/test_backend.py --url http://127.0.0.1:8000
```

Or use Docker: `docker compose up --build`.

## Included artifacts

`fraud_reasoning/artifacts/evidence_gam_v07.json` and `contextual_moe_v07.json` are portable pretrained artifacts; no training is required at startup. `learned_evaluation.json`, `historical_comparison_report.json`, and the partitioned meta-feature CSVs support reproducibility of historical checks.

## Warnings

- Internal D1 test partition is **not untouched independent external validation**, due extensive prior analysis.
- Live deployed wallets may exhibit significant distribution and prevalence shift; **historical D1 calibration is not guaranteed on Render live data**.
- Gating chose RF weight approximately 99.9987%; do not claim independent corroboration from IF or rules.
- The eight rule modules may be partially evaluable, depending on supplied inputs and threshold configuration.
- IF returns anomaly evidence, not fraud probability. Learned IF adapter is **label-supervised** and must not be described as unsupervised fraud prediction.
- Predictions, confidence sets, and textual explanations do not prove criminal fraud. Conformal guarantees need exchangeability and appropriate calibration.
- Add authentication, rate limiting, access logging controls, and drift monitoring before making a public API available.
