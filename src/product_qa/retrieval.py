"""A transparent lexical retriever restricted to the same product."""

import math
import re
from collections import Counter, defaultdict

from .data import evidence_item


def tokenize(text):
    # ePQA questions and candidates are predominantly English.
    return re.findall(r"[a-z0-9]+", text.lower())


class ProductRetriever:
    def __init__(self, records):
        grouped = defaultdict(dict)
        for row in records:
            product_id = str(row.get("product_id", ""))
            for item in row.get("evidence", []):
                text = item["text"]
                source = item.get("source", "unknown")
                grouped[product_id][(text, source)] = evidence_item(text, source)

        self.documents = {product: list(items.values()) for product, items in grouped.items()}
        self.idf = {}
        self.vectors = {}
        for product, documents in self.documents.items():
            tokens = [Counter(tokenize(item["text"])) for item in documents]
            document_frequency = Counter()
            for counts in tokens:
                document_frequency.update(counts.keys())
            n = len(documents)
            idf = {
                term: math.log((n + 1) / (frequency + 1)) + 1
                for term, frequency in document_frequency.items()
            }
            self.idf[product] = idf
            self.vectors[product] = [
                {term: (1 + math.log(count)) * idf[term] for term, count in counts.items()}
                for counts in tokens
            ]

    def search(self, product_id, question, top_k=3):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        product_id = str(product_id)
        documents = self.documents.get(product_id, [])
        idf = self.idf.get(product_id, {})
        query_counts = Counter(tokenize(question))
        query = {
            term: (1 + math.log(count)) * idf[term]
            for term, count in query_counts.items()
            if term in idf
        }
        query_norm = math.sqrt(sum(weight * weight for weight in query.values()))
        if not query_norm:
            return []
        scored = []
        for index, document in enumerate(documents):
            vector = self.vectors[product_id][index]
            norm = math.sqrt(sum(weight * weight for weight in vector.values()))
            score = sum(weight * vector.get(term, 0) for term, weight in query.items())
            if score > 0 and norm:
                scored.append((score / (norm * query_norm), index, document))
        scored.sort(key=lambda result: (-result[0], result[1]))
        return [document for _, _, document in scored[:top_k]]
