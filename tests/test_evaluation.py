import copy
import json
from pathlib import Path
import tempfile
import unittest
from evaluation import make_bundle, score_bundle, validate_record, summarize, ranking_metrics


def annotated():
    return {'qid':'q','reviewer':'test reviewer','status':'complete',
            'evidence':[{'evidence_id':'E1'}],
            'claims':[{'text':'Needs AA batteries','verdict':'supported','evidence_ids':['E1'],'reason':'Explicit'},
                      {'text':'Batteries are included','verdict':'insufficient','evidence_ids':[],'reason':'Absent'}],
            'required_facts':[{'text':'Needs AA batteries','evidence_ids':['E1'],'expressed_correctly':True,'explicitly_abstained':False}],
            'unanswerable_parts':[{'question':'Are batteries included?','handled_correctly':False}],
            'has_off_topic_content':False,'conflict_present':False,'conflict_handled_correctly':None,
            'response_type':'answer','context_relevance':None}


class EvaluationTests(unittest.TestCase):
    def test_partial_evidence_with_hallucination(self):
        r=annotated();validate_record(r)
        metrics=summarize([r])
        self.assertEqual(metrics['faithfulness']['micro'],.5)
        self.assertEqual(metrics['ucr']['micro'],.5)
        self.assertEqual(metrics['answerable_coverage']['micro'],1)
        self.assertEqual(metrics['unknown_handling_accuracy']['micro'],0)
        self.assertIsNone(metrics['conflict_handling_accuracy']['micro'])

    def test_refusal_does_not_get_perfect_faithfulness_or_coverage(self):
        r=annotated();r['claims']=[];r['required_facts'][0].update(expressed_correctly=False,explicitly_abstained=True)
        r['response_type']='abstain';validate_record(r)
        metrics=summarize([r])
        self.assertIsNone(metrics['faithfulness']['micro'])
        self.assertEqual(metrics['answerable_coverage']['micro'],0)
        self.assertEqual(metrics['false_abstention_rate']['micro'],1)
        r['required_facts']=[]
        self.assertIsNone(summarize([r])['answerable_coverage']['micro'])

    def test_micro_macro_and_empty_report(self):
        a=annotated();b=annotated();b['claims']=b['claims'][:1]
        m=summarize([a,b])
        self.assertAlmostEqual(m['faithfulness']['micro'],2/3)
        self.assertEqual(m['faithfulness']['macro'],.75)
        self.assertIsNone(summarize([])['faithfulness']['micro'])

    def test_invalid_annotations(self):
        r=annotated();r['claims'][0]['evidence_ids']=['E99']
        with self.assertRaises(ValueError):validate_record(r)
        r=annotated();r['required_facts'][0]['explicitly_abstained']=True
        with self.assertRaises(ValueError):validate_record(r)
        r=annotated();r['conflict_present']=True
        with self.assertRaises(ValueError):validate_record(r)

    def test_export_pending_binding_and_complete_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'predictions.jsonl'
            prediction={'qid':'q','question':'Batteries?','answer':'AA and included',
                        'retrieved':[{'evidence_id':'E1','chunk_id':'x','text':'Needs AA batteries','sources':['description']}]}
            path.write_text(json.dumps(prediction)+'\n')
            bundle=make_bundle(path)
            self.assertEqual(score_bundle(bundle)['reviewed_samples'],0)
            changed=copy.deepcopy(bundle);changed['records'][0]['answer']='Edited'
            with self.assertRaises(ValueError):score_bundle(changed)
            record=bundle['records'][0]
            for key,value in annotated().items():
                if key not in ('qid','evidence'):record[key]=value
            report=score_bundle(bundle)
            self.assertEqual(report['reviewed_samples'],1)
            self.assertEqual(report['by_evidence_condition']['partial']['samples'],1)
            path.write_text(json.dumps({**prediction,'answer':'Changed'})+'\n')
            with self.assertRaises(ValueError):score_bundle(bundle)

    def test_reject_retrieval_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'predictions.jsonl';path.write_text(json.dumps({'qid':'x','answer':None})+'\n')
            with self.assertRaises(ValueError):make_bundle(path)

    def test_ranking_known_values_and_unjudged_rejection(self):
        r={'fully_judged':True,'judgments':{'a':2,'b':1,'c':0},'ranking':['b','a','c']}
        m=ranking_metrics(r,1)
        self.assertEqual(m['hit@1'],0)
        self.assertEqual(m['mrr'],.5)
        self.assertAlmostEqual(m['ndcg@1'],1/3)
        r['ranking'].append('unjudged')
        with self.assertRaises(ValueError):ranking_metrics(r,3)
        r={'fully_judged':True,'judgments':{'a':0},'ranking':['a']}
        self.assertTrue(all(v is None for v in ranking_metrics(r,3).values()))

if __name__=='__main__':unittest.main()
