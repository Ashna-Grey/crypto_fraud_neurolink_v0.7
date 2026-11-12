"""Learned evidence arbitration, replacing manually selected RF/IF/rule weights.

A regularized generalized additive logistic model with learned spline main
terms and selected tensor-spline pairwise interactions. This is NOT an EBM.
Term contributions add exactly in calibrated log-odds space.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LogisticRegression

from .rules import SPEC
from .logic import facts_from_input

RULE_NAMES = [row[0] for row in SPEC]
CONTINUOUS = [
    'rf_raw_probability', 'rf_volume_path_contribution',
    'rf_temporal_path_contribution', 'rf_erc20_path_contribution',
    'if_anomaly_percentile', 'if_bootstrap_variance', 'if_applicability',
    'log_total_transactions', 'log_activity_rate', 'log_observation_days',
    'rule_trigger_count', 'rule_coverage'
]
INTERACTIONS = [
    ('rf_raw_probability', 'if_anomaly_percentile'),
    ('rf_raw_probability', 'log_activity_rate'),
    ('rf_raw_probability', 'rule_trigger_count')
]
BINARY = ['activity_rate_missing', 'if_unavailable', 'if_training_activity_unsupported', 'rf_missing_fraction'] + [
    name for rule in RULE_NAMES for name in ('rule_fired_' + rule, 'rule_unavailable_' + rule)
]
FEATURE_NAMES = CONTINUOUS + BINARY


def clip(x, a=0., b=1.):
    return max(a, min(b, float(x)))


def _safe_log(x):
    try:
        y = float(x)
        return math.log1p(max(0., y)) if math.isfinite(y) else 0.
    except (ValueError, TypeError):
        return 0.


def group_of_rf_feature(name: str) -> str:
    low = name.lower()
    if ('total transactions' in low or low in ('sent tnx', 'received tnx')):
        return 'rf_volume_path_contribution'
    if ('min between' in low or 'time diff' in low):
        return 'rf_temporal_path_contribution'
    if 'erc20' in low:
        return 'rf_erc20_path_contribution'
    return 'rf_other_path_contribution'


def rf_group_contributions(model, imputer, frame: pd.DataFrame, feature_names: list[str]):
    """Efficient exact raw-RF tree-path contributions, for any number of rows.

    The sum of group contributions + root-probability baseline reconstructs
    raw RF probabilities. This decomposition is not SHAP or causal effect.
    """
    x = np.asarray(imputer.transform(frame[feature_names]), dtype=float)
    groups = ['rf_volume_path_contribution', 'rf_temporal_path_contribution',
              'rf_erc20_path_contribution', 'rf_other_path_contribution']
    n = len(frame)
    out = {g: np.zeros(n, dtype=float) for g in groups}
    baseline = 0.
    pos = list(model.classes_).index(1)
    for tree in model.estimators_:
        t = tree.tree_
        count = t.node_count
        vals = np.asarray(t.value[:, 0, :], dtype=float)
        probs = vals[:, pos] / np.maximum(vals.sum(axis=1), 1e-12)
        baseline += float(probs[0])
        contrib = {g: np.zeros(count, dtype=float) for g in groups}
        stack = [0]
        while stack:
            parent = stack.pop()
            feature_index = int(t.feature[parent])
            if feature_index < 0:
                continue
            group = group_of_rf_feature(feature_names[feature_index])
            for child in (int(t.children_left[parent]), int(t.children_right[parent])):
                for g in groups:
                    contrib[g][child] = contrib[g][parent]
                contrib[group][child] += probs[child] - probs[parent]
                stack.append(child)
        leaves = tree.apply(x)
        for g in groups:
            out[g] += contrib[g][leaves]
    num_trees = len(model.estimators_)
    for g in groups:
        out[g] /= num_trees
    baseline /= num_trees
    raw = model.predict_proba(x)[:, pos]
    residual = raw - (baseline + np.sum(list(out.values()), axis=0))
    if np.max(np.abs(residual)) > 1e-5:
        raise ArithmeticError('RF group path decomposition failed to reconstruct raw predictions')
    return out


def evidence_row(raw_row: Mapping, rf_raw: float, rf_contributions: Mapping,
                 if_output: Mapping, rule_output: Mapping, rf_features: list[str]):
    """Identical deterministic evidence schema for training and deployment."""
    facts = facts_from_input(dict(raw_row))
    total = facts.get('total_transactions')
    rate = facts.get('average_transactions_per_day_over_observed_span')
    span = facts.get('wallet_lifetime_minutes')
    flags = {r['rule']: r.get('triggered') for r in rule_output.get('evaluated', [])}
    missing = {r['rule'] for r in rule_output.get('unavailable_rules', [])}
    fired = sum(bool(v) for v in flags.values())
    good_if = if_output.get('status') == 'OK'
    applicability = if_output.get('applicability') or {}
    out = {
        'rf_raw_probability': clip(rf_raw),
        'rf_volume_path_contribution': float(rf_contributions.get('rf_volume_path_contribution', 0.)),
        'rf_temporal_path_contribution': float(rf_contributions.get('rf_temporal_path_contribution', 0.)),
        'rf_erc20_path_contribution': float(rf_contributions.get('rf_erc20_path_contribution', 0.)),
        'if_anomaly_percentile': clip(float(if_output.get('anomaly_percentile') or 50.) / 100.) if good_if else .5,
        'if_bootstrap_variance': max(0., float(if_output.get('bootstrap_anomaly_variance') or 0.)) if good_if else 0.,
        'if_applicability': clip(applicability.get('score') or 0.) if good_if else 0.,
        'log_total_transactions': _safe_log(total),
        'log_activity_rate': _safe_log(rate),
        'log_observation_days': _safe_log(span / 1440. if span is not None else None),
        'rule_trigger_count': float(fired),
        'rule_coverage': clip(rule_output.get('coverage') or 0.),
        'activity_rate_missing': float(rate is None),
        'if_unavailable': float(not good_if),
        'if_training_activity_unsupported': float(total is not None and total > 200),
        'rf_missing_fraction': float(sum(raw_row.get(f) is None or pd.isna(raw_row.get(f)) for f in rf_features) / len(rf_features))
    }
    for rule in RULE_NAMES:
        out['rule_fired_' + rule] = float(flags.get(rule) is True)
        out['rule_unavailable_' + rule] = float(rule in missing)
    return out


@dataclass
class Attribution:
    probability: float
    uncalibrated_probability: float
    calibrated_log_odds: float
    intercept_log_odds: float
    terms: list
    reconstruction_error: float


class SplineEvidenceGAM:
    """Trainable sparse, regularized additive logistic model with pairwise terms.

    No manually weighted model vote. Main effect/pairwise terms are created
    before fitting; all coefficients are estimated from labels. The validation
    calibrator is fitted on disjoint, held-out labeled records.
    """
    def __init__(self, c: float = 0.35, n_knots: int = 5, seed: int = 42):
        self.c, self.n_knots, self.seed = c, n_knots, seed

    def _df(self, rows):
        if isinstance(rows, pd.DataFrame):
            data = rows.reindex(columns=FEATURE_NAMES, fill_value=0.).copy()
        else:
            data = pd.DataFrame(rows).reindex(columns=FEATURE_NAMES, fill_value=0.)
        data = data.apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan).fillna(0.)
        return data.astype(float)

    def _matrix(self, rows, fit=False):
        frame = self._df(rows)
        if fit:
            self.main_splines_ = {}
            for col in CONTINUOUS:
                transformer = SplineTransformer(n_knots=self.n_knots, degree=2, include_bias=False,
                                               knots='quantile', extrapolation='constant')
                transformer.fit(frame[[col]])
                self.main_splines_[col] = transformer
        blocks, names, owners = [], [], []
        main = {}
        for col in CONTINUOUS:
            vals = self.main_splines_[col].transform(frame[[col]])
            main[col] = vals
            blocks.append(vals)
            names.extend([f'{col}[spline_{i}]' for i in range(vals.shape[1])])
            owners.extend([col] * vals.shape[1])
        for col in BINARY:
            blocks.append(frame[[col]].to_numpy(dtype=float))
            names.append(col)
            owners.append(col)
        for left,right in INTERACTIONS:
            a,b = main[left],main[right]
            product = (a[:,:,None] * b[:,None,:]).reshape(len(frame),-1)
            blocks.append(product)
            names.extend([f'{left}*{right}[basis_{i}]' for i in range(product.shape[1])])
            owners.extend([f'{left} × {right}'] * product.shape[1])
        if fit:
            self.design_names_=names
            self.design_owners_=owners
        return np.hstack(blocks)

    def fit(self, rows, labels):
        x = self._matrix(rows, fit=True)
        self.meta_ = LogisticRegression(C=self.c, max_iter=2500, solver='lbfgs', random_state=self.seed)
        self.meta_.fit(x, np.asarray(labels,dtype=int))
        self.calibration_ = None
        self.calibration_qualified_ = False
        return self

    def calibrate(self, rows, labels):
        scores = self.raw_logit(rows)
        labels = np.asarray(labels,dtype=int)
        if len(set(labels)) < 2 or len(labels) < 100:
            raise ValueError('Calibration requires at least 100 records with both labels')
        # Independent Platt sigmoid on meta-model LOGITS, not on fitted labels.
        self.calibration_ = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
        self.calibration_.fit(scores.reshape(-1,1), labels)
        self.calibration_qualified_ = True
        return self

    def raw_logit(self, rows):
        return self.meta_.decision_function(self._matrix(rows))

    def predict_proba(self, rows):
        logits = self.raw_logit(rows)
        if self.calibration_ is None:
            raise RuntimeError('Evidence GAM has no held-out calibration artifact')
        return self.calibration_.predict_proba(logits.reshape(-1,1))

    def explain(self, row, top_k=12):
        matrix = self._matrix([row])[0]
        coef = self.meta_.coef_[0]
        original_intercept = float(self.meta_.intercept_[0])
        a = float(self.calibration_.coef_[0][0]); b = float(self.calibration_.intercept_[0])
        aggregate = {}
        for owner,value in zip(self.design_owners_, matrix * coef):
            aggregate[owner] = aggregate.get(owner,0.) + a * float(value)
        intercept = b + a * original_intercept
        final_logit = intercept + sum(aggregate.values())
        probability = float(expit(final_logit))
        check = float(self.predict_proba([row])[0,1])
        return {'fraud_probability':probability, 'uncalibrated_probability':float(expit(original_intercept+np.dot(coef,matrix))),
                'calibrated_log_odds':final_logit,'intercept_log_odds':intercept,
                'terms':[{'term':key,'calibrated_log_odds_contribution':round(value,8),
                          'direction':'toward_fraud' if value>0 else 'away_from_fraud' if value<0 else 'neutral'}
                         for key,value in sorted(aggregate.items(),key=lambda kv:-abs(kv[1]))[:top_k]],
                'all_term_contributions':aggregate,'reconstruction_error':abs(probability-check),
                'explanation_semantics':'Exact additive contributions to calibrated log-odds; observational not causal',
                'algorithm':'regularized_tensor_spline_GAM_not_InterpretML_EBM'}


class PortableEvidenceGAM:
    """Version-neutral JSON inference for the trained GAM, using scipy BSplines.

    Avoids deserializing sklearn 1.8 meta artifacts on a sklearn 1.9 Render image.
    Existing original RF / IF remain loaded by the original sklearn-compatible code.
    """
    def __init__(self, payload):
        from scipy.interpolate import BSpline
        self.payload=payload
        self.columns=payload['features']
        self.splines={}
        for name,info in payload['splines'].items():
            if 'constant_basis' in info:
                self.splines[name]=('constant',np.asarray(info['constant_basis'],dtype=float))
                continue
            knots=np.asarray(info['knots'],float)
            num=int(info['basis_count'])
            bs=BSpline(knots,np.eye(num,dtype=float),int(info['degree']),extrapolate=False)
            self.splines[name]=(bs,knots[int(info['degree'])],knots[-int(info['degree'])-1],int(info['output_count']))
        self.coefficients=np.asarray(payload['logistic_coefficients'],dtype=float)
        self.intercept=float(payload['logistic_intercept'])
        self.calibration_a=float(payload['calibration_a'])
        self.calibration_b=float(payload['calibration_b'])
        self.owners=payload['design_owners']
        self.metadata=payload['metadata']

    @classmethod
    def from_file(cls,path):
        import json
        from pathlib import Path
        return cls(json.loads(Path(path).read_text()))

    def _design(self, row):
        main={}
        for name in CONTINUOUS:
            spl=self.splines[name]
            if spl[0]=='constant':
                main[name]=spl[1].copy()
            else:
                bs,lo,hi,nout=spl
                value=float(row.get(name,0.) or 0.)
                if not math.isfinite(value):value=0.
                main[name]=np.asarray(bs(clip(value,lo,hi)),float).reshape(-1)[:nout]
        parts=[main[k] for k in CONTINUOUS]
        parts.extend([np.asarray([float(row.get(name,0.) or 0.)],float) for name in BINARY])
        for a,b in INTERACTIONS:
            parts.append(np.outer(main[a],main[b]).flatten())
        design=np.concatenate(parts)
        if len(design)!=len(self.coefficients):
            raise RuntimeError('Portable GAM coefficient/design mismatch')
        return design

    def predict_proba(self, rows):
        if isinstance(rows,dict):rows=[rows]
        vals=[]
        for row in rows:
            x=self._design(row)
            logit=self.calibration_b+self.calibration_a*(self.intercept+np.dot(x,self.coefficients))
            p=float(expit(logit))
            vals.append([1-p,p])
        return np.asarray(vals,dtype=float)

    def explain(self,row,top_k=14):
        x=self._design(row)
        aggregates={}
        for owner,unit in zip(self.owners,x*self.coefficients*self.calibration_a):
            aggregates[owner]=aggregates.get(owner,0.)+float(unit)
        intercept=self.calibration_b+self.calibration_a*self.intercept
        logit=intercept+sum(aggregates.values())
        p=float(expit(logit))
        validation=float(self.predict_proba([row])[0,1])
        return {'fraud_probability':p,'uncalibrated_probability':float(expit(self.intercept+np.dot(self.coefficients,x))),
                'calibrated_log_odds':logit,'intercept_log_odds':intercept,
                'terms':[{'term':key,'calibrated_log_odds_contribution':round(v,8),
                          'direction':'toward_fraud' if v>0 else 'away_from_fraud' if v<0 else 'neutral'}
                         for key,v in sorted(aggregates.items(),key=lambda kv:-abs(kv[1]))[:top_k]],
                'all_term_contributions':aggregates,'reconstruction_error':abs(validation-p),
                'explanation_semantics':'Exact additive contributions to calibrated log-odds; observational not causal',
                'algorithm':'regularized_tensor_spline_GAM_not_InterpretML_EBM'}


def export_portable_gam(model:SplineEvidenceGAM, metadata:dict, output_path):
    import json
    from pathlib import Path
    portable={'version':'0.7.0','features':FEATURE_NAMES,'metadata':metadata,
              'splines':{},'design_owners':model.design_owners_,
              'design_names':model.design_names_,
              'logistic_coefficients':model.meta_.coef_[0].tolist(),
              'logistic_intercept':float(model.meta_.intercept_[0]),
              'calibration_a':float(model.calibration_.coef_[0,0]),
              'calibration_b':float(model.calibration_.intercept_[0])}
    for name,transformer in model.main_splines_.items():
        bs=transformer.bsplines_[0]
        if len(set(bs.t)) < 2:
            portable['splines'][name]={'constant_basis':transformer.transform(pd.DataFrame({name:[float(bs.t[0])]}))[0].tolist()}
        else:
            portable['splines'][name]={'knots':bs.t.tolist(),
                                      'degree':int(bs.k),
                                      'basis_count':int(bs.c.shape[0]),
                                      'output_count':int(transformer.n_features_out_)}
    artifact=PortableEvidenceGAM(portable)
    for features in [{name:0. for name in FEATURE_NAMES},
                     {name:float(np.random.default_rng(42).random()) for name in FEATURE_NAMES}]:
        expected=float(model.predict_proba([features])[0,1]);actual=float(artifact.predict_proba([features])[0,1])
        if abs(expected-actual)>1e-10:
            raise ArithmeticError('Portable GAM inference differs from sklearn model')
    Path(output_path).write_text(json.dumps(portable,indent=2))
    return artifact
