import unittest
from reranking import rank_candidates

class RerankTests(unittest.TestCase):
    def test_reranker_can_promote_beyond_initial_top_three(self):
        candidates=[{'chunk_id':str(i),'asin':'A','score':1/(60+i+1), 'score_type':'rrf', 'evidence_id':f'E{i+1}'} for i in range(5)]
        selected,trace=rank_candidates(candidates,[-2,0,1,7,1],3)
        self.assertEqual([c['chunk_id'] for c in selected],['3','2','4'])
        self.assertEqual(selected[0]['pre_rerank_rank'],4)
        self.assertEqual([c['evidence_id'] for c in selected],['E1','E2','E3'])
        self.assertEqual(len(trace),5)
        self.assertEqual(candidates[3]['evidence_id'],'E4')
        self.assertTrue(all(c['asin']=='A' for c in selected))
    def test_empty_and_bad_scores(self):
        self.assertEqual(rank_candidates([],[],3),([],[]))
        with self.assertRaises(ValueError): rank_candidates([{}],[],3)
        with self.assertRaises(ValueError): rank_candidates([{}],[float('nan')],3)

if __name__=='__main__': unittest.main()
