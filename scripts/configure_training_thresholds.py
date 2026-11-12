"""Freeze TWO supported rule thresholds from the original RF TRAIN partition ONLY.

Does NOT fit contract threshold: source dataset lacks `total_eth_sent_contracts`.
Do not claim these development thresholds are independently validated.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
raw=pd.read_csv(ROOT/'01_Dataset/processed/ethereum_fraud_25_features.csv')
indices=joblib.load(ROOT/'03_Random_Forest/artifacts/rf_data_split_indices.joblib')['train_indices']
train=raw.iloc[indices]
volume=pd.to_numeric(train['total Ether sent'],errors='coerce')+pd.to_numeric(train['total ether received'],errors='coerce')
value=pd.to_numeric(train['avg val sent'],errors='coerce')
config={
  'high_value_threshold':float(volume.quantile(0.90)),
  'dust_threshold':float(value.quantile(0.10))
}
assert all(np.isfinite(x) for x in config.values())
path=ROOT/'02_Rule_Engine/artifacts/rule_thresholds.json'
path.write_text(json.dumps(config,indent=2)+'\n')
(ROOT/'02_Rule_Engine/artifacts/threshold_provenance.json').write_text(json.dumps({
  'provenance':'RF training indices, D1 dataset only; development configuration, not new holdout validation',
  'rows':int(len(train)),
  'high_value_threshold':'90th percentile of total Ether sent + total ether received',
  'dust_threshold':'10th percentile of avg val sent',
  'contract_threshold':'UNAVAILABLE: D1 processed data lacks total_eth_sent_contracts and created_contracts',
  'note':'This is not independent research validation and must not be used to claim rule precision.'
},indent=2)+'\n')
print(path.read_text())
