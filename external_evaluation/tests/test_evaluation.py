import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from external_evaluation.feature_builder import build_features,make_payload,transfer_key,RF,IF
from external_evaluation.metrics import calculate,summarize
from external_evaluation.run import interpret_response,fetch_direction,read_manifest

class TestEvaluation(unittest.TestCase):
    def test_manifest_valid(self):
        rows=read_manifest(Path(__file__).parents[1]/'manifest.csv')
        self.assertEqual(len(rows),200)
        self.assertEqual(sum(x['label']=='1' for x in rows),100)
    def test_no_labels_sent(self):
        payload=make_payload('0x'+'1'*40,{k:None for k in set(RF+IF)})
        self.assertEqual(payload['input_format'],'raw-live')
        self.assertNotIn('label',json.dumps(payload))
        self.assertTrue(all(k in payload['features'] for k in RF))
    def test_genuine_transfer_id_prevents_hash_collision(self):
        tx={'hash':'0xabc','uniqueId':'0xabc:log:1','from':'a','to':'b'}
        ty={**tx,'uniqueId':'0xabc:log:2'}
        self.assertNotEqual(transfer_key(tx),transfer_key(ty))
    def test_missing_token_values_are_not_fake_zeros(self):
        tx={'uniqueId':'x','category':'erc20','asset':'ABC','from':'x','to':'y',
            'metadata':{'blockTimestamp':'2020-01-01T00:00:00Z'},'rawContract':{'address':'0x1'}}
        f,q=build_features('0x'+'2'*40,[tx],[],0.0)
        self.assertIsNone(f['ERC20 max val rec'])
        self.assertEqual(q['erc20_events'],1)
    def test_zero_token_history_can_have_zero_token_aggregates(self):
        f,q=build_features('0x'+'2'*40,[],[],0.0)
        self.assertEqual(f['ERC20 max val rec'],0)
        self.assertIsNone(f['total ether balance'])
        self.assertIsNone(f['Avg min between received tnx'])
    def test_manual_review_is_abstention(self):
        out=interpret_response({'final_output':{'decision':'MANUAL_REVIEW'}})
        self.assertEqual(out['status'],'abstained_manual_review')
        self.assertIsNone(calculate([{'label':1,**out}]))
    def test_decision_and_probability(self):
        out=interpret_response({'final_output':{'decision':'FRAUD','fraud_probability':0.93}})
        self.assertEqual(out['status'],'classified')
        self.assertFalse(out['decision_mismatch'])
    def test_probability_inconsistency_is_flagged(self):
        out=interpret_response({'final_output':{'decision':'FRAUD','fraud_probability':0.03}})
        self.assertTrue(out['decision_mismatch'])
        self.assertEqual(out['status'],'invalid_inconsistent_decision')
    def test_metrics_confusion(self):
        rows=[{'label':1,'decision':'FRAUD','probability':.9},
              {'label':0,'decision':'LEGITIMATE','probability':.1},
              {'label':0,'decision':'FRAUD','probability':.8},
              {'label':1,'decision':'LEGITIMATE','probability':.2}]
        result=calculate(rows)
        self.assertEqual((result['TP'],result['TN'],result['FP'],result['FN']),(1,1,1,1))
        self.assertEqual(result['accuracy'],0.5)
    def test_transfer_pagination_resume_after_network_failure(self):
        wallet='0x'+'a'*40
        calls=[]
        def mock_rpc(url,method,params):
            calls.append(params[0].get('pageKey'))
            if len(calls)==1:return {'transfers':[{'uniqueId':'1'}], 'pageKey':'page2'}
            if len(calls)==2:raise ConnectionError('test failure')
            self.assertEqual(params[0].get('pageKey'),'page2')
            return {'transfers':[{'uniqueId':'2'}]}
        with tempfile.TemporaryDirectory() as d,patch('external_evaluation.run.rpc',side_effect=mock_rpc):
            with self.assertRaises(ConnectionError):
                fetch_direction('url',wallet,'incoming','0x100',10,1000,Path(d))
            state=fetch_direction('url',wallet,'incoming','0x100',10,1000,Path(d))
            self.assertTrue(state['complete'])
            self.assertEqual([x['uniqueId'] for x in state['transfers']],['1','2'])
            self.assertEqual(calls,[None,'page2','page2'])
    def test_unfinished_histories_excluded(self):
        report=summarize([{'label':1,'status':'classified_partial_history',
                           'decision':'FRAUD','probability':0.99,'complete':False}],1)
        self.assertIsNone(report['provisional_metrics_complete_collections'])

if __name__=='__main__':unittest.main()

# End of regression checks.
