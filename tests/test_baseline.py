import csv
from pathlib import Path
import tempfile
import unittest
from baseline import read_data, chunk_documents

class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()
    def decode(self, ids, skip_special_tokens=True):
        return ' '.join(ids)

class DataTests(unittest.TestCase):
    def test_corpus_excludes_targets_and_deduplicates_within_product(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'dev.csv'
            rows = [
                ['q1','Question?','A','some evidence','review','2','SECRET ANSWER'],
                ['q1','Question?','A','some  evidence','bullet','0','SECRET ANSWER'],
                ['q2','Other?','B','some evidence','review','1','OTHER SECRET'],
                ['q1','Question?','A','  ','review','0','SECRET ANSWER'],
            ]
            with path.open('w', newline='') as f:
                w=csv.writer(f); w.writerow(['qid','question','ASIN','candidate','source','label','answer']); w.writerows(rows)
            questions, products=read_data(path)
            self.assertEqual(len(questions),2)
            self.assertEqual(len(products['A']),1)
            self.assertEqual(len(products['B']),1)
            self.assertNotIn('SECRET',str(products))
            self.assertNotIn('label',str(products))
            self.assertNotEqual(next(iter(products['A'])), next(iter(products['B'])))

    def test_chunk_boundaries_preserve_tail_and_product(self):
        doc={'doc_id':'d','asin':'A','text':'0 1 2 3 4 5 6','sources':['review']}
        chunks=chunk_documents({'A':{'d':doc}},WordTokenizer(),size=4,overlap=1)
        self.assertEqual([c['text'] for c in chunks], ['0 1 2 3','3 4 5 6'])
        self.assertTrue(all(c['asin']=='A' for c in chunks))
        self.assertEqual(chunks[-1]['token_end'],7)

if __name__=='__main__':
    unittest.main()
