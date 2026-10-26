import unittest
from hybrid_retrieval import BM25, tokenize, reciprocal_rank_fusion, retrieve

class HybridTests(unittest.TestCase):
    def test_model_numbers_and_no_overlap(self):
        self.assertEqual(tokenize('FRS20QRB #240337103'), ['frs20qrb','240337103'])
        scorer=BM25(['fits FRS20QRB', 'fits FRT18', ''])
        self.assertGreater(scorer.scores('FRS20QRB')[0], 0)
        self.assertEqual(scorer.scores('unmatched'), [0,0,0])
        self.assertEqual(BM25(['','']).scores('x'), [0,0])

    def test_rrf_rewards_agreement_and_empty_branch(self):
        order,scores=reciprocal_rank_fusion([0,1],[1,2],60)
        self.assertEqual(order[0],1)
        self.assertAlmostEqual(scores[1],1/62+1/61)
        self.assertEqual(reciprocal_rank_fusion([2,0],[])[0],[2,0])

    def test_product_mapping_fusion_and_dense_control(self):
        import faiss
        import numpy as np
        chunks=[{'asin':'B','text':'FRS20QRB'}, {'asin':'A','text':'other'}, {'asin':'A','text':'FRS20QRB'}]
        index=faiss.IndexFlatIP(2)
        index.add(np.array([[1,0],[0.8,0.6]],dtype='float32'))
        kwargs=dict(index=index, positions=[1,2], chunks=chunks,vector=[1,0],
                    question='FRS20QRB',lexical_index=BM25(['other','FRS20QRB']),top_k=1,recall_k=2)
        hybrid=retrieve(**kwargs)
        self.assertEqual(hybrid[0]['text'],'FRS20QRB')
        self.assertEqual(hybrid[0]['asin'],'A')
        self.assertEqual(hybrid[0]['score_type'],'rrf')
        self.assertEqual(retrieve(**kwargs,mode='dense')[0]['text'],'other')
        kwargs['question']='nooverlap'
        self.assertEqual(retrieve(**kwargs)[0]['text'],'other')

if __name__=='__main__':
    unittest.main()
