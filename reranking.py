"""Optional frozen cross-encoder reranking of retrieved evidence."""
import math


def rank_candidates(candidates, scores, top_k):
    if len(candidates) != len(scores):
        raise ValueError('One reranker score is required per candidate')
    if not all(math.isfinite(float(s)) for s in scores):
        raise ValueError('Non-finite reranker score')
    # Stable ties preserve retrieval order. Retain all scores for inspection.
    order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), i))
    ranked = []
    for rank, i in enumerate(order, 1):
        candidate = candidates[i]
        ranked.append({**candidate, 'pre_rerank_rank': i+1,
                       'pre_rerank_score': candidate['score'],
                       'pre_rerank_score_type': candidate['score_type'],
                       'reranker_score': float(scores[i]), 'reranker_rank': rank,
                       'score': float(scores[i]), 'score_type': 'cross_encoder_logit',
                       'evidence_id': f'E{rank}'})
    return ranked[:top_k], ranked


class EvidenceReranker:
    def __init__(self, model_name, revision):
        import torch
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, revision=revision, device='cpu',
                                  max_length=512, activation_fn=torch.nn.Identity())

    def rerank(self, question, candidates, top_k):
        if not candidates:
            return [], []
        pairs = [(question, c['text']) for c in candidates]
        # Fail explicitly rather than score a truncated passage and send a different
        # full passage to the generator. Chunking can be adjusted if needed.
        lengths = [len(self.model.tokenizer(q, d, truncation=False)['input_ids']) for q, d in pairs]
        if max(lengths) > 512:
            raise ValueError('Reranker input exceeds 512 tokens; reduce chunk size or question length explicitly')
        scores = self.model.predict(pairs, batch_size=16, show_progress_bar=False)
        return rank_candidates(candidates, scores, top_k)
