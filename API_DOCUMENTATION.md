# NeuroLink v0.7 API

Base: `http://localhost:8000` locally, or the Render service URL. Swagger docs: `/docs`.

- `GET /`, `GET /health`, `GET /models/status`, `GET /schema`.
- `GET /learned/metrics`: active learned model's training/calibration metadata and historical evaluation.
- `GET /learned/evidence-schema`: exact features used by learned GAM and adapter pipeline.
- `GET /research/methodology`, `GET /reasoning/rules`.
- `POST /predict`: JSON `{ "wallet_id": "example", "input_format": "raw-live", "features": { "<RF feature name>": 0, ... } }`. See `/schema` for all required RF feature names and `examples/sample_wallet_request.json` for a complete request.
- `POST /predict/batch`: `{ "wallets": [<predict-payload>, ...] }`; maximum 100.

Prediction response retains `random_forest`, `isolation_forest`, `rule_based_engine`, `reasoning_engine`, `learned_arbitration`, `belief_fusion`, and `final_output`. The `learned_arbitration` object contains trained expert estimates/gating or GAM additive terms, calibration metadata, and evidence features. The `final_output.fraud_probability` is calibrated on the historical D1 calibration role only. A `review_recommended` advisory **does not manually override** the supervised binary decision. `prediction_set_90pct_nominal` has no verified coverage on deployment data.

Set `FUSION_MODE=moe` (default), `gam`, or `legacy` as a deployment environment variable. `legacy` is the previous heuristic policy, for research ablation only.

`isolation_forest.anomaly_percentile` is *not* a fraud probability; `rule_based_engine.risk_score` is *not* calibrated. The learned IF/rule expert estimates arise from separate supervised adapters trained against D1 fraud labels.
