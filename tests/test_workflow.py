import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from product_qa.evaluation import score
from product_qa.retrieval import ProductRetriever
from prepare_epqa import convert
from evaluate_retrieval import evaluate


class WorkflowTests(unittest.TestCase):
    def test_partial_candidate_has_no_supervised_answer(self):
        rows = list(convert(ROOT / "examples" / "toy_epqa.tsv", "dev"))
        self.assertEqual([row["evidence_label"] for row in rows], [1, 2, 0, 2])
        self.assertIsNone(rows[0]["reference_answer"])
        self.assertEqual(rows[1]["reference_answer"], "It requires two AA batteries.")

    def test_retrieval_keeps_products_separate(self):
        rows = list(convert(ROOT / "examples" / "toy_epqa.tsv", "dev"))
        retriever = ProductRetriever(rows)
        passages = retriever.search("toy-001", "Which batteries are needed?", top_k=3)
        self.assertTrue(passages)
        self.assertTrue(all("cotton" not in passage["text"] for passage in passages))
        self.assertEqual(retriever.search("missing-product", "batteries"), [])

    def test_retrieval_metric_uses_answerable_questions(self):
        rows = list(convert(ROOT / "examples" / "toy_epqa.tsv", "dev"))
        metrics = evaluate(rows, top_k=1)
        self.assertEqual(metrics["questions"], 4)
        self.assertEqual(metrics["questions_with_label_2"], 2)
        # "What material ...?" has no lexical overlap with "100% cotton".
        self.assertEqual(metrics["label_2_hit_count"], 1)

    def test_claim_and_coverage_denominators(self):
        row = {
            "id": "dev:1",
            "complete": True,
            "evidence_label": 1,
            "claims": [{"text": "two AA", "supported": True},
                       {"text": "included", "supported": False}],
            "answerable_units": [{"text": "two AA", "covered": True},
                                 {"text": "unknown inclusion", "covered": False}],
        }
        self.assertEqual(score([row])["overall"]["unsupported_claim_rate"], 0.5)
        self.assertEqual(score([row])["overall"]["answerable_coverage"], 0.5)
        empty = dict(row, claims=[], answerable_units=[])
        self.assertIsNone(score([empty])["overall"]["unsupported_claim_rate"])
        self.assertIsNone(score([empty])["overall"]["answerable_coverage"])
        with self.assertRaises(ValueError):
            score([dict(row, complete=False)])


if __name__ == "__main__":
    unittest.main()
