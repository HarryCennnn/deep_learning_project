"""Small product-local BM25 and equal-weight reciprocal rank fusion.

BM25 uses log(1 + (N-df+0.5)/(df+0.5)), k1=1.5, b=0.75.
No stemming/stop-word removal. Alphanumeric model IDs stay intact; punctuation
separates tokens. Zero-overlap documents do not receive a lexical RRF vote.
"""
from collections import Counter
import math
import re


def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.counts = [Counter(tokenize(text)) for text in texts]
        self.lengths = [sum(count.values()) for count in self.counts]
        self.avg_length = sum(self.lengths) / len(texts) if texts else 0
        df = Counter(term for count in self.counts for term in count)
        self.idf = {term: math.log1p((len(texts)-freq+0.5)/(freq+0.5)) for term, freq in df.items()}

    def scores(self, question):
        terms = set(tokenize(question))
        result = []
        for count, length in zip(self.counts, self.lengths):
            norm = self.k1 * (1-self.b+self.b*length/(self.avg_length or 1))
            score = sum(self.idf.get(term, 0) * count[term] * (self.k1+1)/(count[term]+norm)
                        for term in terms if count[term])
            result.append(score)
        return result


def reciprocal_rank_fusion(dense_order, lexical_order, c=60):
    """Inputs contain unique local document indices, best first. Ties use index."""
    if c <= 0:
        raise ValueError('RRF c must be positive')
    scores = {}
    for order in (dense_order, lexical_order):
        if len(order) != len(set(order)):
            raise ValueError('Duplicate document in ranking')
        for rank, doc in enumerate(order, 1):
            scores[doc] = scores.get(doc, 0.0) + 1.0/(c+rank)
    return sorted(scores, key=lambda doc: (-scores[doc], doc)), scores


def retrieve(index, positions, chunks, vector, question, lexical_index,
             mode='hybrid', top_k=3, recall_k=10, rrf_c=60):
    import numpy as np
    # Pools are small: rank the full product pool to make ties deterministic,
    # then truncate each branch to recall_k before fusion.
    values, neighbors = index.search(np.asarray([vector], dtype=np.float32), index.ntotal)
    dense_scores = {int(i): float(score) for i, score in zip(neighbors[0], values[0])}
    dense_all = sorted(dense_scores, key=lambda i: (-dense_scores[i], i))
    dense_order = dense_all[:recall_k] if mode == 'hybrid' else dense_all[:top_k]
    dense_ranks = {i: rank for rank, i in enumerate(dense_order, 1)}
    lexical_scores, lexical_ranks, fused_scores = None, {}, {}
    if mode == 'hybrid':
        lexical_scores = lexical_index.scores(question)
        lexical_order = sorted((i for i, score in enumerate(lexical_scores) if score > 0),
                               key=lambda i: (-lexical_scores[i], i))[:recall_k]
        lexical_ranks = {i: rank for rank, i in enumerate(lexical_order, 1)}
        order, fused_scores = reciprocal_rank_fusion(dense_order, lexical_order, rrf_c)
    else:
        order = dense_order
    return [{**chunks[positions[i]], 'evidence_id': f'E{rank+1}',
             'score': fused_scores[i] if mode == 'hybrid' else dense_scores[i],
             'score_type': 'rrf' if mode == 'hybrid' else 'cosine',
             'dense_score': dense_scores[i], 'dense_rank': dense_ranks.get(i),
             'bm25_score': lexical_scores[i] if lexical_scores is not None else None,
             'bm25_rank': lexical_ranks.get(i), 'rrf_score': fused_scores.get(i)}
            for rank, i in enumerate(order[:top_k])]
