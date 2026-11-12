"""Context-conditioned mixture of calibrated supervised evidence adapters.

The IF engine remains an UNSUPERVISED anomaly detector. Its score is mapped to
fraud evidence by a separate LABEL-TRAINED logistic adapter. Rule alert flags
are similarly mapped by a supervised adapter. Routing weights are learned by
minimizing proper log loss, not manually chosen or evidence of expert truth.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, softmax, logit

IF_COLUMNS=['if_anomaly_percentile','if_applicability','if_bootstrap_variance',
            'log_activity_rate','log_total_transactions','if_training_activity_unsupported']
RULE_COLUMNS=['rule_fired_'+r for r in ('rapid_transaction_activity','fund_splitting',
              'fund_consolidation','new_wallet_high_value','dust_spam_activity',
              'contract_activity','erc20_activity','high_activity')]+[
              'rule_unavailable_'+r for r in ('rapid_transaction_activity','fund_splitting',
              'fund_consolidation','new_wallet_high_value','dust_spam_activity',
              'contract_activity','erc20_activity','high_activity')]+[
              'rule_coverage','rule_trigger_count','log_activity_rate']
GATE_COLUMNS=['rf_raw_probability','if_anomaly_percentile','if_applicability',
              'log_activity_rate','log_total_transactions','rule_trigger_count',
              'rule_coverage','rf_volume_path_contribution',
              'if_training_activity_unsupported']

class PortableContextualMoE:
    def __init__(self, payload):
        self.payload=payload
        self.metadata=payload['metadata']

    @classmethod
    def from_file(cls,path):
        return cls(json.loads(Path(path).read_text()))

    @staticmethod
    def _matrix(rows,names):
        return np.array([[float(x.get(name,0) or 0) for name in names] for x in rows],dtype=float)

    def _expert(self,rows,key,columns):
        obj=self.payload[key]
        x=self._matrix(rows,columns)
        x=(x-np.asarray(obj['mean']))/np.asarray(obj['scale'])
        return expit(x @ np.asarray(obj['coef'])+obj['intercept'])

    def _predict(self,rows):
        if isinstance(rows,dict):rows=[rows]
        e=np.column_stack([self._matrix(rows,['rf_raw_probability']).reshape(-1),
                           self._expert(rows,'if_adapter',IF_COLUMNS),
                           self._expert(rows,'rule_adapter',RULE_COLUMNS)])
        gate=self._matrix(rows,GATE_COLUMNS)
        info=self.payload['gate_scaler']
        gate=(gate-np.asarray(info['mean']))/np.asarray(info['scale'])
        gate=np.column_stack([np.ones(len(gate)),gate])
        weights=softmax(gate @ np.asarray(self.payload['gate_weights']),axis=1)
        raw=np.sum(weights*e,axis=1)
        raw=np.clip(raw,1e-7,1-1e-7)
        p=expit(self.payload['calibration_b']+self.payload['calibration_a']*logit(raw))
        return p,e,weights,raw

    def predict_proba(self,rows):
        p,_,_,_=self._predict(rows)
        return np.column_stack([1-p,p])

    def explain(self,row,top_k=14):
        p,expert,weight,raw=self._predict([row])
        labels=['original_RF_raw','D1_label_mapped_IF_evidence','D1_label_mapped_rule_evidence']
        cx=self._matrix([row],GATE_COLUMNS)[0]
        gate_obj=self.payload['gate_scaler']
        scaled=(cx-np.asarray(gate_obj['mean']))/np.asarray(gate_obj['scale'])
        gate_design=np.concatenate([[1.],scaled])
        gate_weights=np.asarray(self.payload['gate_weights'],float)
        gate_logits=gate_design@gate_weights
        gate_details=[]
        for k,label in enumerate(labels):
            units=scaled*gate_weights[1:,k]
            contributors=[{'context_feature':key,'logit_contribution':round(float(z),8)}
                          for key,z in sorted(zip(GATE_COLUMNS,units),key=lambda t:-abs(t[1]))[:5]]
            gate_details.append({'engine':label,'unnormalized_gate_logit':float(gate_logits[k]),
                                 'top_gate_logit_terms':contributors})
        details=[{'engine':label,'learned_expert_fraud_estimate':float(e),
                  'learned_contextual_weight':float(w),
                  'raw_weighted_probability_contribution':float(e*w)}
                 for label,e,w in zip(labels,expert[0],weight[0])]
        return {'fraud_probability':float(p[0]),'uncalibrated_probability':float(raw[0]),
                'weights':details,'gate_reasoning':gate_details,
                'calibration':{'slope':self.payload['calibration_a'],
                               'intercept':self.payload['calibration_b']},
                'explanation_semantics':'Learned mixture contributions BEFORE final sigmoid calibration; gating weights are not proof of correctness',
                'algorithm':'supervised_contextual_mixture_of_experts'}
