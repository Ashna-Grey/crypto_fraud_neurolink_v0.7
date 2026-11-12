"""Adapter for the original eight rules; rejects unavailable inputs rather than substituting zeros."""
import importlib.util
import json
from pathlib import Path
from .logic import facts_from_input

SPEC = [
 ('rapid_transaction_activity', ['avg_min_between_sent','avg_min_between_received','total_transactions'], None),
 ('fund_splitting', ['sent_transactions','unique_destinations','avg_min_between_sent'], None),
 ('fund_consolidation',['received_transactions','unique_sources','avg_min_between_received'], None),
 ('new_wallet_high_value',['wallet_lifetime_minutes','total_eth_received','total_eth_sent','total_transactions'],'high_value_threshold'),
 ('dust_spam_activity',['sent_transactions','unique_destinations','avg_value_sent'],'dust_threshold'),
 ('contract_activity',['total_eth_sent_contracts','created_contracts'],'contract_threshold'),
 ('erc20_activity',['erc20_transactions','erc20_unique_sent_addresses','erc20_unique_sent_tokens'], None),
 ('high_activity',['total_transactions','wallet_lifetime_minutes'],None)
]
WEIGHTS=[2.80,1.10,6.43,0.24,0.98,0.45,4.07,0.62]

class RuleAdapter:
    def __init__(self, root:Path):
        self.root=Path(root)
        config=self.root/'02_Rule_Engine/artifacts/rule_thresholds.json'
        try:self.thresholds=json.loads(config.read_text())
        except (OSError,ValueError):self.thresholds={}
        self.modules=[]
        for i in range(1,9):
            path=next((self.root/'02_Rule_Engine/rules').glob(f'rule_{i:02d}_*.py'))
            spec=importlib.util.spec_from_file_location(f'original_rule_{i}',path)
            module=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.modules.append(module)

    def run(self,row):
        facts=facts_from_input(row)
        evaluated=[]; unavailable=[]
        for (name,deps,threshold_key),weight,module in zip(SPEC,WEIGHTS,self.modules):
            missing=[k for k in deps if k not in facts]
            if missing:
                unavailable.append({'rule':name,'reason':'missing_features','missing':missing})
                continue
            if threshold_key and (not isinstance(self.thresholds.get(threshold_key),(int,float)) or isinstance(self.thresholds.get(threshold_key),bool)):
                unavailable.append({'rule':name,'reason':'missing_frozen_threshold','threshold_key':threshold_key})
                continue
            args=(float(self.thresholds[threshold_key]),) if threshold_key else ()
            try:
                triggered=bool(module.evaluate(facts,*args)['triggered'])
            except (ValueError,TypeError,KeyError,OverflowError) as exc:
                unavailable.append({'rule':name,'reason':'evaluation_error','detail':str(exc)})
                continue
            evaluated.append({'rule':name,'triggered':triggered,'dependencies':deps,
                              'observed':{k:facts[k] for k in deps},'threshold':self.thresholds.get(threshold_key) if threshold_key else None,
                              'weight':weight, 'explanation':f'{name}: original deterministic rule {"triggered" if triggered else "not triggered"}; no causal fraud claim.'})
        # Do not renormalize on only available rules, which would hide missing strong rules.
        total=sum(WEIGHTS)
        support=sum(item['weight'] for item in evaluated if item['triggered'])/total if evaluated else None
        status='OK' if not unavailable else 'PARTIAL' if evaluated else 'UNAVAILABLE'
        return {'status':status,'risk_score':round(support,6) if support is not None else None,
                'score_semantics':'weighted_sum_of_fired_original_rules_not_calibrated_probability',
                'evaluated':evaluated, 'triggered_rules':[r['rule'] for r in evaluated if r['triggered']],
                'non_triggered_rules':[r['rule'] for r in evaluated if not r['triggered']],
                'unavailable_rules':unavailable,'coverage':len(evaluated)/len(SPEC),
                'weights_from_original_rule_f1_not_revalidated':True,
                'warning':'A partial score is a lower bound on weighted triggered-rule support, not a complete risk estimate.' if unavailable else None}
