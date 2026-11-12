"""Load existing saved artifacts, run guarded inference, return structured evidence."""
from __future__ import annotations
import importlib.util
import json
import math
import sys
from pathlib import Path
import pandas as pd
from .fusion import combine
ROOT=Path(__file__).resolve().parents[1]

def _module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class FraudService:
    def __init__(self):
        self.rf=_module(ROOT/'03_Random_Forest/inference/predict.py','original_rf_predict')
        if_dir=ROOT/'04_Isolation_Forest'
        sys.path.insert(0,str(if_dir)) if str(if_dir) not in sys.path else None
        self.if_predict=_module(if_dir/'inference/predict.py','original_if_predict')
        from src.isolation_forest_utils import load_bundle
        self.bundle=load_bundle(if_dir/'artifacts/isolation_forest_b6b.joblib')
        self.rf_features=json.loads((ROOT/'03_Random_Forest/artifacts/feature_schema.json').read_text())['feature_order']
        self.if_features=list(self.bundle['feature_columns'])
        self.rule_dir=ROOT/'02_Rule_Engine'
        self.rules=[]
        for i in range(1,9):
            path=next((self.rule_dir/'rules').glob(f'rule_{i:02d}_*.py'))
            self.rules.append(_module(path,f'fraud_rule_{i}'))
        self.weights=[2.80,1.10,6.43,0.24,0.98,0.45,4.07,0.62]
        self.rule_required=['avg_min_between_sent','avg_min_between_received','wallet_lifetime_minutes','sent_transactions','received_transactions','created_contracts','unique_sources','unique_destinations','avg_value_sent','total_transactions','total_eth_sent','total_eth_received','total_eth_sent_contracts','erc20_transactions','erc20_unique_sent_addresses','erc20_unique_sent_tokens']
        self.threshold_path=self.rule_dir/'artifacts/rule_thresholds.json'
        try: self.thresholds=json.loads(self.threshold_path.read_text())
        except (ValueError,OSError): self.thresholds={}

    def rule_score(self, row):
        missing=[x for x in self.rule_required if x not in row or row[x] is None]
        if missing:return None,{'status':'UNAVAILABLE','reason':'Missing rule features','missing':missing}
        # Quantile thresholds must be frozen from training data; never infer from the request.
        threshold_keys=['high_value_threshold','dust_threshold','contract_threshold']
        if not all(isinstance(self.thresholds.get(k),(int,float)) for k in threshold_keys):
            return None,{'status':'UNAVAILABLE','reason':'Training-derived rule thresholds not configured','required_keys':threshold_keys}
        args=[(),(),(),(self.thresholds['high_value_threshold'],),(self.thresholds['dust_threshold'],),(self.thresholds['contract_threshold'],),(),()]
        try:
            fired=[int(bool(rule.evaluate(row,*extra)['triggered'])) for rule,extra in zip(self.rules,args)]
            score=sum(w*f for w,f in zip(self.weights,fired))/sum(self.weights)
            return score,{'status':'OK','fired':fired,'score_semantics':'heuristic_support_not_calibrated_probability'}
        except (ValueError,TypeError,KeyError,OverflowError) as exc:
            return None,{'status':'UNAVAILABLE','reason':str(exc)}

    def predict(self, row, input_format='preprocessed'):
        if not isinstance(row,dict) or not row: raise ValueError('Expected a nonempty feature dictionary')
        if input_format not in ('preprocessed','raw-live'):raise ValueError('input_format must be preprocessed or raw-live')
        row={str(k).strip():v for k,v in row.items()}
        if len(row)!=len(set(row)):raise ValueError('Duplicate feature keys')
        for key,value in row.items():
            if value is not None and isinstance(value,(float,int)) and not math.isfinite(value):raise ValueError(f'Nonfinite value for {key}')
        rf_missing=[x for x in self.rf_features if x not in row]
        if rf_missing:raise ValueError('Missing RF features: '+', '.join(rf_missing))
        rf_input=pd.DataFrame([{f:row.get(f) for f in self.rf_features}])
        for f in self.rf_features:rf_input[f]=pd.to_numeric(rf_input[f],errors='raise')
        rf_result=self.rf.predict_dataframe(rf_input).iloc[0]
        p=float(rf_result['rf_mu_calibrated'])
        rf_quality=sum(pd.notna(row.get(f)) for f in self.rf_features)/len(self.rf_features)
        if_missing=[x for x in self.if_features if x not in row]
        if_result={'status':'UNAVAILABLE','missing_features':if_missing}
        anomaly=None
        applicability=0.0 # fail closed if IF coverage cannot be assessed
        if not if_missing:
            if_frame=pd.DataFrame([{f:row.get(f) for f in self.if_features}])
            scored=self.if_predict.score_dataframe(if_frame,self.bundle,input_format=input_format).iloc[0]
            if pd.notna(scored['if_mu']):
                anomaly=float(scored['if_mu'])
                out_of_range=int(scored['outside_training_minmax_count'])
                missing_ratio=float(scored['missing_feature_ratio'])
                # Policy proxy, not validated density or likelihood ratio.
                applicability=max(0.0,1-missing_ratio-out_of_range/len(self.if_features))
                if_result={'status':'OK','anomaly_percentile':round(anomaly*100,3),
                    'applicability_proxy':round(applicability,4),
                    'missing_feature_count':int(scored['missing_feature_count']),
                    'outside_training_minmax_count':out_of_range,
                    'score_semantics':'anomaly_percentile_not_fraud_probability'}
            else:if_result={'status':'UNAVAILABLE','reason':'All IF features missing'}
        rule_support,rule_info=self.rule_score(row)
        fusion=combine(p,rule_support,rf_quality=rf_quality,applicability=applicability,
                       if_anomaly=anomaly,conformal_decision=str(rf_result['rf_conformal_decision']))
        return {'rf':{'status':'OK','calibrated_fraud_probability':p,
                     'conformal_decision':str(rf_result['rf_conformal_decision']),
                     'tree_variance':float(rf_result['rf_sigma_squared']),
                     'input_quality':rf_quality},
                'if':if_result,'rules':rule_info,'fusion':fusion,
                'notice':'Research prototype. No independently validated fusion calibration or live fraud guarantee.'}
