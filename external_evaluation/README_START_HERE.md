# NeuroLink v0.7, Poison-Hunter external evaluation

**Append-only folder:** no existing model code, requirements, Render configuration, Dockerfile, or endpoints need to be edited. This kit does NOT retrain or tune the model.

## What has actually been evaluated

- The dataset audit establishes 5,890 Poison-Hunter phishing addresses and 1,154 benign addresses. The 200-wallet manifest has 100 verified labels of each class, zero duplicate addresses and zero overlaps with the **10,000 known D2 training wallet addresses**.
- The original package's **924-row D1 historical evaluation** was recomputed from the frozen model's saved data. See `results/historical_audit.json`. This is NOT external testing.
- The previous **five-wallet Poison-Hunter pilot** had four `MANUAL_REVIEW` abstentions and one failed collection, with **zero measurable fraud probabilities**. See `results/pilot_audit.json`. This is NOT evidence of model correctness or failure.
- The 200-wallet external predictions are **not yet measured**. They require access to an Alchemy key and the publicly deployed or locally installed NeuroLink model. The dataset contains addresses and labels, not model feature vectors.

## Exact folder structure to copy

Place this **entire `external_evaluation` directory** in your existing `Crypto-fraud-intelligence-main` repository root, next to `fraud_reasoning/`, `scripts/`, `tests/`, `render.yaml`, and `requirements.txt`.

```
Crypto-fraud-intelligence-main/
├── fraud_reasoning/                 (unchanged)
├── 03_Random_Forest/                (unchanged)
├── 04_Isolation_Forest/             (unchanged)
├── scripts/                         (unchanged)
├── tests/                           (unchanged)
├── render.yaml                      (unchanged)
├── requirements.txt                 (unchanged)
└── external_evaluation/             (NEW, copy entire folder)
    ├── README_START_HERE.md
    ├── manifest.csv                 (100 phishing, 100 benign)
    ├── Poison-Hunter-main.zip       (ground-truth source)
    ├── D2_training_addresses.txt    (overlap checks)
    ├── reference_snapshot.txt       (same block as five-wallet pilot)
    ├── network.py                   (retry and HTTP 401 handling)
    ├── feature_builder.py           (features and unknown-value handling)
    ├── metrics.py                   (accuracy, F1, confusion, ROC-AUC, PR-AUC)
    ├── run.py                       (collector, inference, reports, caching)
    ├── dataset_audit.py
    ├── historical_audit.py
    ├── pilot_audit.py
    ├── local_smoke.py
    ├── import_previous_cache.py
    ├── RUN_FIRST_5.ps1
    ├── RUN_ALL_200.ps1
    ├── baseline_pilot/              (five-wallet archived CSV and JSON)
    ├── tests/
    └── results/                     (audited reports; new reports go here)
```

## Instructions for Windows PowerShell

Open PowerShell **inside the `Crypto-fraud-intelligence-main` repository root**.

**1. Validate the frozen artifacts and dataset WITHOUT any API key:**

```powershell
python -m external_evaluation.dataset_audit
python -m external_evaluation.historical_audit
python -m external_evaluation.pilot_audit
python -m unittest discover -s external_evaluation/tests -v
```

These offline commands use Python's standard library. They work without installing extra packages. The frozen-model smoke test (`python -m external_evaluation.local_smoke`) has also been executed here and passed all 6 shipped cases; it requires the original model dependencies locally.

**2. Set your full Alchemy Ethereum Mainnet URL privately, in the terminal only:**

```powershell
$env:ALCHEMY_RPC_URL = "https://eth-mainnet.g.alchemy.com/v2/YOUR_REAL_PRIVATE_KEY"
```

Never paste the URL or key into ChatGPT, code files, CSVs, screenshots, Git commits, or `README`s. If an old `ALCHEMY_API_KEY` is configured, `ALCHEMY_RPC_URL` takes precedence.

**3. Confirm the deployed model is actually this v0.7 API and emits probability outputs:**

```powershell
python -m external_evaluation.run --probe-only
```

This tests `/health`, `/schema`, `/models/status`, and a known non-Poison-Hunter smoke fixture at `/predict`. No Alchemy requests and no labels sent. If it reports `Render FUSION_MODE=legacy`, open the Render dashboard, select your service, go to **Environment**, set `FUSION_MODE` to `moe`, and redeploy. The `legacy` setting is a likely cause of your four `MANUAL_REVIEW` outcomes. If it reports a **DEPLOYED /predict CONTRACT MISMATCH**, your Render deployment is not returning the outputs of the uploaded v0.7 MoE code; redeploy this exact repository. The archived `MANUAL_REVIEW` outcome is not an output that this v0.7 `fraud_reasoning/service.py` produces in `moe` mode. Do NOT run 200 until the probe succeeds.

**Optional: reuse the four successfully downloaded wallets from your earlier `Testing` folder:**

```powershell
python -m external_evaluation.import_previous_cache --from-dir "C:\Users\ASHNA\Desktop\Ashna\Projects\Crypto\Neurolink\Testing\results\cache"
```

Run this only if that directory exists. Four completed wallets may then be reused without paying for their Alchemy requests again. It won't copy failed/incomplete histories.

**4. Evaluate just the five pilot wallets:**

```powershell
python -m external_evaluation.run --max-wallets 5
```

**5. If a numerical fraud probability appears, run the 200-wallet sample:**

```powershell
python -m external_evaluation.run --max-wallets 200
```

PowerShell shortcuts `RUN_FIRST_5.ps1` and `RUN_ALL_200.ps1` are also included. The runner caches each Alchemy page in `results/cache`, resumes partial downloads, preserves the first evaluation block, and retries transient 429/5xx/`IncompleteRead` failures. It does not include the private API key in the cache or reports. A very popular wallet can require many requests and use Alchemy credits. The default cap is 100 pages *per direction* at 1000 events/page; if a wallet reaches that limit, it is marked as incomplete and excluded from the complete-history results.

## Results you should open

- `external_evaluation/results/report.html`: a browser-readable dashboard of newly scored wallets, produced after running the external test.
- `external_evaluation/results/predictions.csv`: each wallet's true label, model decision, numeric score (if provided), feature missingness, and any error or abstention.
- `external_evaluation/results/report.json`: provisional confusion matrix, metrics, exact coverage counts, and methodological limitations.
- `external_evaluation/results/historical_audit.json`: D1 internal historical metrics, verified now.
- `external_evaluation/results/pilot_audit.json`: archival five-wallet pilot, verified now.
- `external_evaluation/results/local_smoke.json`: all six frozen-model inference fixtures passed locally, without testing Poison-Hunter accuracy.
- `external_evaluation/VERIFIED_RESULTS.html`: browser dashboard with completed historical and five-wallet audit results.

**Do not interpret a `MANUAL_REVIEW` response as a true negative, correct positive, or numeric fraud prediction.** This kit explicitly counts it as an abstention. A binary prediction without a numerical probability is also excluded from probability metrics.

### Optional fallback: use the bundled model directly rather than Render

If the Render endpoint keeps returning `MANUAL_REVIEW` but the ZIP actually contains v0.7, and the repository dependencies are installed under a compatible Python version, run:

```powershell
python -m external_evaluation.run --engine local --probe-only
python -m external_evaluation.run --engine local --max-wallets 5
python -m external_evaluation.run --engine local --max-wallets 200
```

The repo's Docker image uses Python 3.13 and pins scikit-learn 1.9.0. **Local mode may NOT work with Python 3.15**, but remote mode needs no model dependencies and is the recommended option for your Windows machine.

## Git push

Execute from your existing Git repository root, after copying the folder:

```powershell
git status
git add external_evaluation
git commit -m "Add reproducible Poison-Hunter external evaluation"
git push origin HEAD
```

A git push to your Render-linked repo redeploys your **existing** NeuroLink service if auto-deploy is enabled. **It does not automatically execute the Alchemy evaluation.** Run the above PowerShell commands from your PC to generate the metrics, then push generated non-secret `results/report.json`, `results/predictions.csv`, and `results/report.html` with another commit if you want them saved to GitHub. Cached raw transaction pages are gitignored and must stay local.

## Scientific boundaries

- This is a retrospective benchmark of address-poisoning phishing wallets versus selected popular Etherscan benign wallets. It is **not** a sample of real-world fraud prevalence.
- `Alchemy` transfer counts and token quantity semantics may not match original Kaggle D1 training feature definitions. Missing ERC-20 token amounts are recorded as `null`, not fabricated, and the original RF imputer can still yield biased predictions. Any reported external numbers are therefore **provisional diagnostics**, not publication-grade estimates.
- D1 did not provide wallet addresses, so possible D1/Poison-Hunter overlap remains unknown, even though D2 overlap is excluded.
- Neither training nor decision thresholds are tuned based on the external labels.
- The accurate final external metric might be lower than the historical D1 score. No code should force the results to look high.
