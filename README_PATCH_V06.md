# Render v0.6.0 Decision-Layer Patch

This ZIP is the **complete project**, not a small delta. Replace the prior v0.5.0 repository contents with the extracted project root, commit and push to redeploy Render.

## Fixes

- No negative vote from untriggered or unavailable fraud rules.
- No direct arithmetic disagreement between calibrated RF and uncalibrated rule risk.
- Incomplete optional rules are output as `final_output.advisories`, not unconditional manual-review blockers.
- Strong, conformal-consistent RF outputs can produce `FRAUD` or `LEGITIMATE` if input and OOD gates are satisfactory.
- Risk scores are RF-anchored, rule alerts only add positive risk support, and IF only assesses distributional applicability.
- The contextual adjustment uses **all 20 RF tree-path contributions** to recognize transaction-count dependence, not the truncated top-eight display list.
- Qualitative rule-vs-RF tension, poor IF training applicability, missing important RF inputs and extreme anomalies still require review.
- Version bumped to `0.6.0`.

## Deploy

1. Extract ZIP, open `Crypto-fraud-intelligence-main` as the repository root, replace old tracked files with contents of this folder and push the commit.
2. Render Docker service should automatically rebuild on the new commit; if not, use **Manual Deploy > Deploy latest commit**.
3. No new environment variables are required. The existing Dockerfile uses `${PORT:-8000}`.
4. Confirm the service responds to `GET /models/status` with version `0.6.0`.
5. Run `python run_render_tests.py https://crypto-fraud-neurolink.onrender.com` from your previous `neurolink_render_tests` package. Compare outputs with `examples/local_v06_regression_summary.json`.
6. Read `MATHEMATICAL_FOUNDATIONS.md`, `TEST_REPORT.md` and `KNOWN_LIMITATIONS.md` before using results in a paper.

**Important:** Deployed v0.6 has not been verified from this build environment. Classification decisions are experimental RF-led predictions, not ground-truth fraud determinations. Policy indices remain uncalibrated.
