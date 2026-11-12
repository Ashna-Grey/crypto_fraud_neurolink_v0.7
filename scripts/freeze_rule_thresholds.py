"""Freeze ALL three original threshold-dependent rules from a genuinely training-only CSV.

Example:
    python scripts/freeze_rule_thresholds.py --training-only-csv /path/to/TRAIN_ONLY_45features.csv
Input must be verified training-only. Do not run against a benchmark's held-out rows.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--training-only-csv',required=True)
args=parser.parse_args()
df=pd.read_csv(args.training_only_csv)
needed=['total_eth_received','total_eth_sent','avg_value_sent','total_eth_sent_contracts']
missing=[col for col in needed if col not in df.columns]
if missing:raise SystemExit('Training feature CSV missing: '+', '.join(missing))
for key in needed:df[key]=pd.to_numeric(df[key],errors='raise')
config={
  'high_value_threshold':float((df.total_eth_received+df.total_eth_sent).quantile(0.90)),
  'dust_threshold':float(df.avg_value_sent.quantile(0.10)),
  'contract_threshold':float(df.total_eth_sent_contracts.quantile(0.90)),
}
if any(not np.isfinite(v) for v in config.values()):raise SystemExit('Nonfinite quantiles. Refusing to store invalid thresholds.')
path=root/'02_Rule_Engine/artifacts/rule_thresholds.json'
path.write_text(json.dumps(config,indent=2)+'\n')
(root/'02_Rule_Engine/artifacts/threshold_provenance.json').write_text(json.dumps({
    'source_training_only_csv':str(Path(args.training_only_csv).resolve()),
    'source_rows':len(df),'warning':'User must verify training-only provenance. Not independently validated.'
},indent=2)+'\n')
print('Frozen:', path)
