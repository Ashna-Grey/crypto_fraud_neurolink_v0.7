# NeuroLink v0.7 verification report

Run `pytest -q` from the project root. `pytest.ini` scopes test discovery to `tests/` because legacy scripts contain machine-specific hard-coded Windows paths and should not be collected as active tests.

Local project tests were run in a Python 3.13 environment with scikit-learn 1.8.0 and passed; the original RF pickle requires scikit-learn 1.9.0 for supported production compatibility. The Docker image pins scikit-learn 1.9.0, but a local Docker/Render build was not available in the testing environment.

Covered checks include: RF/IF artifact inference; additive RF attribution consistency; original rules; contextual logic and missing inputs; portable GAM calibration equivalence with its training artifact; MoE probability normalization; learned gating weights and response schema; HTTP validation and batch inference; and historical evaluation/report integrity.

`python scripts/test_backend.py --url http://127.0.0.1:8000` runs live HTTP smoke cases against the local Uvicorn server. For Render supply your live URL. The returned fraud labels are never sent in requests.

## Historical D1 internal evaluation

See `fraud_reasoning/artifacts/historical_comparison_report.json` and `LEARNED_ARBITRATION_V07.md`. The reported D1 historical comparison is **not independently externally validated**; this limitation applies to all metrics and probability-calibration assertions.

## Known limitations

- The learned gate overwhelmingly selects RF, providing no empirical evidence for extra expert value.
- Contextual model training and calibration rely on D1 labels and are supervised.
- The IF and rule engine alone do not produce fraud probabilities.
- Conformal marginal coverage requires distribution assumptions not verified on live wallets.
- Serialized base models may be incompatible with unsupported scikit-learn versions.

## Latest verified local run

`pytest -q`: **31 passed**, 29 warnings (mostly scikit-learn serialized-version compatibility and feature-name warnings). Four API GET checks returned HTTP 200; five valid POST predictions returned HTTP 200, and one intentionally incomplete payload returned HTTP 422. See `examples/local_v07_api_smoke_summary.json`. These tests were performed locally, not on Render.
