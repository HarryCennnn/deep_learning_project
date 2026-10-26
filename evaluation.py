"""Export auditable annotation forms and score reviewed product-QA outputs.
No model calls, automatic semantic judgments, or fabricated gold labels.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

VERSION = 'product-qa-eval-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_predictions(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({str(r['qid']) for r in rows}) != len(rows):
        raise ValueError('Duplicate qid in predictions')
    if not rows or any(not isinstance(r.get('answer'), str) for r in rows):
        raise ValueError('Need generated answers; retrieval-only outputs cannot be evaluated')
    return rows


def make_bundle(path):
    rows = load_predictions(path)
    return {
        'schema_version': VERSION,
        'source_predictions': str(path.resolve()),
        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'annotation_method': 'human',
        'records': [{
            'qid': str(r['qid']), 'prediction_fingerprint': digest(r),
            'question': r['question'],
            'evidence': [{'evidence_id': e['evidence_id'], 'chunk_id': e['chunk_id'],
                          'text': e['text'], 'sources': e.get('sources', [])} for e in r['retrieved']],
            'required_facts': None, 'unanswerable_parts': None,
            'conflict_present': None, 'conflict_handled_correctly': None,
            'answer': r['answer'], 'claims': None,
            'response_type': None, 'has_off_topic_content': None,
            'context_relevance': None,
            'reviewer': '', 'status': 'pending', 'notes': ''
        } for r in rows]
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_record(r):
    prefix = f"qid={r['qid']}: "
    require(bool(r.get('reviewer', '').strip()), prefix+'reviewer required')
    ids = {e['evidence_id'] for e in r['evidence']}
    require(len(ids) == len(r['evidence']), prefix+'duplicate evidence IDs')
    for key in ('claims', 'required_facts', 'unanswerable_parts'):
        require(isinstance(r.get(key), list), prefix+key+' must be a reviewed list (empty is allowed)')
    require(type(r.get('has_off_topic_content')) is bool, prefix+'has_off_topic_content must be boolean')
    require(type(r.get('conflict_present')) is bool, prefix+'conflict_present must be boolean')
    require(r.get('response_type') in ('answer', 'partial_answer', 'abstain'), prefix+'invalid response_type')
    if r['conflict_present']:
        require(type(r.get('conflict_handled_correctly')) is bool, prefix+'conflict handling required')
    else:
        require(r.get('conflict_handled_correctly') is None, prefix+'conflict handling must be null if no conflict')

    def refs(item, needed):
        value = item.get('evidence_ids')
        require(isinstance(value, list) and set(value) <= ids, prefix+'invalid evidence_ids')
        require(not needed or bool(value), prefix+'supporting evidence required')

    for claim in r['claims']:
        require(bool(claim.get('text','').strip()), prefix+'empty claim')
        require(claim.get('verdict') in ('supported', 'contradicted', 'insufficient'), prefix+'invalid claim verdict')
        require(bool(claim.get('reason','').strip()), prefix+'claim reason required')
        refs(claim, claim['verdict'] in ('supported', 'contradicted'))
    for fact in r['required_facts']:
        require(bool(fact.get('text','').strip()), prefix+'empty required fact')
        refs(fact, True)
        for key in ('expressed_correctly', 'explicitly_abstained'):
            require(type(fact.get(key)) is bool, prefix+key+' must be boolean')
        require(not (fact['expressed_correctly'] and fact['explicitly_abstained']), prefix+'fact cannot be both covered and refused')
    for part in r['unanswerable_parts']:
        require(bool(part.get('question','').strip()), prefix+'empty unknown subquestion')
        require(type(part.get('handled_correctly')) is bool, prefix+'unknown handling must be boolean')
    require(r['required_facts'] or r['unanswerable_parts'], prefix+'identify at least one requested answerable/unknown unit')
    if r['response_type'] == 'abstain':
        require(not any(f['expressed_correctly'] for f in r['required_facts']), prefix+'abstain conflicts with covered required facts')
    context = r.get('context_relevance')
    if context is not None:
        require(isinstance(context,list), prefix+'context_relevance must be null or a list')
        require(len(context)==len(ids) and {x.get('evidence_id') for x in context}==ids,
                prefix+'context relevance must cover each presented evidence once')
        require(all(type(x.get('relevant')) is bool for x in context), prefix+'context relevance requires booleans')


def counts_for(r):
    claims, facts, unknowns = r['claims'], r['required_facts'], r['unanswerable_parts']
    context = r.get('context_relevance')
    n, supported = len(claims), sum(c['verdict']=='supported' for c in claims)
    return {
        'faithfulness': (supported, n), 'ucr': (n-supported, n),
        'answerable_coverage': (sum(f['expressed_correctly'] for f in facts), len(facts)),
        'false_abstention_rate': (sum(f['explicitly_abstained'] for f in facts), len(facts)),
        'unknown_handling_accuracy': (sum(u['handled_correctly'] for u in unknowns), len(unknowns)),
        'on_topic_rate': (int(not r['has_off_topic_content']), 1),
        'conflict_handling_accuracy': (int(r['conflict_handled_correctly'] is True), int(r['conflict_present'])),
        'full_abstention_rate': (int(r['response_type']=='abstain'), 1),
        'relevant_chunk_fraction': (sum(x['relevant'] for x in context), len(context)) if context is not None else (0,0)
    }


def summarize(records):
    metrics = {}
    counts = [counts_for(r) for r in records]
    names = list(counts[0]) if counts else list(counts_for({
        'claims':[], 'required_facts':[], 'unanswerable_parts':[], 'has_off_topic_content':False,
        'conflict_present':False, 'conflict_handled_correctly':None, 'response_type':'abstain'}))
    for name in names:
        nums = [c[name] for c in counts]
        numerator, denominator = sum(n for n,d in nums), sum(d for n,d in nums)
        valid = [n/d for n,d in nums if d]
        metrics[name] = {'micro': numerator/denominator if denominator else None,
                         'macro': mean(valid) if valid else None,
                         'numerator': numerator, 'denominator': denominator,
                         'eligible_samples': len(valid), 'not_applicable_samples': len(records)-len(valid)}
    return metrics


def score_bundle(bundle):
    require(bundle.get('schema_version')==VERSION, 'Unsupported annotation version')
    require(bundle.get('annotation_method')=='human', 'This scorer is configured for human annotations')
    path = Path(bundle['source_predictions'])
    require(hashlib.sha256(path.read_bytes()).hexdigest()==bundle['source_sha256'], 'Source file changed')
    original = {str(r['qid']): r for r in load_predictions(path)}
    require(isinstance(bundle.get('records'),list), 'records must be a list')
    require(len(bundle['records'])==len(original) and {r['qid'] for r in bundle['records']}==set(original),
            'Annotation qids must match source exactly; no duplicate/missing samples')
    valid, pending = [], []
    for r in bundle['records']:
        source = original[r['qid']]
        require(r['prediction_fingerprint']==digest(source), 'Prediction fingerprint mismatch')
        expected_evidence = [{'evidence_id':e['evidence_id'],'chunk_id':e['chunk_id'],
                              'text':e['text'],'sources':e.get('sources',[])} for e in source['retrieved']]
        require(r['question']==source['question'] and r['answer']==source['answer'] and r['evidence']==expected_evidence,
                f"qid={r['qid']}: do not edit question, answer, or evidence")
        require(r['status'] in ('pending','complete'), 'Invalid annotation status')
        if r['status']=='pending':
            pending.append(r['qid']); continue
        validate_record(r)
        valid.append(r)
    groups = {}
    for r in valid:
        condition = ('partial' if r['unanswerable_parts'] else 'complete') if r['required_facts'] else 'insufficient'
        groups.setdefault(condition, []).append(r)
    return {'schema_version':VERSION, 'annotation_method':'human',
            'total_samples':len(original), 'reviewed_samples':len(valid), 'pending_qids':pending,
            'metrics':summarize(valid),
            'by_evidence_condition': {k:{'samples':len(v),'metrics':summarize(v)} for k,v in groups.items()},
            'per_sample': [{'qid':r['qid'],'metrics':{k:n/d if d else None for k,(n,d) in counts_for(r).items()}}
                           for r in valid],
            'limitations': [
                'Human-judged faithfulness inspired by Ragas; not an execution of the Ragas library.',
                'On-topic rate and relevant-chunk fraction are custom diagnostics, not paper Answer/Context Relevance.',
                'Coverage is conditioned on actually retrieved evidence; denominators may change across retrievers.',
                'Pending annotations are excluded; a partial report is not a full-run evaluation.',
                'Null means not applicable, not perfect or zero performance.'
            ]}


def ranking_metrics(record, k):
    """Only for an explicitly fully judged, fixed candidate pool. Not inferred from ePQA."""
    require(record.get('fully_judged') is True, 'Ranking pool must be explicitly fully judged')
    judgments, ranking = record['judgments'], record['ranking']
    require(bool(judgments), 'Empty judgment pool')
    require(all(type(v) is int and v in (0,1,2) for v in judgments.values()), 'Expected grades 0/1/2')
    require(len(ranking)==len(set(ranking)) and set(ranking)==set(judgments), 'Need complete ranking of judged pool')
    top=ranking[:k]
    relevant={i for i,g in judgments.items() if g==2}
    dcg=sum((2**judgments[i]-1)/math.log2(rank+2) for rank,i in enumerate(top))
    ideal=sum((2**g-1)/math.log2(rank+2) for rank,g in enumerate(sorted(judgments.values(),reverse=True)[:k]))
    first=next((rank for rank,i in enumerate(ranking,1) if i in relevant),None)
    return {f'hit@{k}':int(bool(set(top)&relevant)) if relevant else None,
            f'recall@{k}':len(set(top)&relevant)/len(relevant) if relevant else None,
            'mrr':1/first if first else None,
            f'ndcg@{k}':dcg/ideal if ideal else None}


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
    export=commands.add_parser('export')
    export.add_argument('--run',type=Path,required=True)
    export.add_argument('--output',type=Path,required=True)
    score=commands.add_parser('score')
    score.add_argument('--annotations',type=Path,required=True)
    score.add_argument('--output',type=Path,required=True)
    score.add_argument('--allow-partial',action='store_true')
    ranking=commands.add_parser('ranking')
    ranking.add_argument('--input',type=Path,required=True)
    ranking.add_argument('--output',type=Path,required=True)
    ranking.add_argument('--k',type=int,default=3)
    ragas=commands.add_parser('export-ragas')
    ragas.add_argument('--run',type=Path,required=True)
    ragas.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    try:
        if args.command=='export':
            result=make_bundle(args.run/'predictions.jsonl')
        elif args.command=='score':
            result=score_bundle(json.loads(args.annotations.read_text()))
            require(args.allow_partial or not result['pending_qids'], 'Pending annotations remain. Finish them, or explicitly use --allow-partial.')
        elif args.command=='ranking':
            require(args.k>0,'k must be positive')
            data=json.loads(args.input.read_text())
            require(isinstance(data,list) and bool(data),'Expected nonempty list of ranking cases')
            require(len({r['qid'] for r in data})==len(data),'Duplicate ranking qid')
            cases=[{'qid':r['qid'],'metrics':ranking_metrics(r,args.k)} for r in data]
            result={'scope':'explicit fully judged fixed pools; grade 2 is binary positive; nDCG gain=2^grade-1',
                    'per_sample':cases,'metrics':{name:{'macro':mean(vals) if vals else None,'eligible_samples':len(vals)}
                      for name in cases[0]['metrics']
                      for vals in [[c['metrics'][name] for c in cases if c['metrics'][name] is not None]]}}
        else:
            result=[{'user_input':r['question'],'response':r['answer'],
                     'retrieved_contexts':[e['text'] for e in r['retrieved']]}
                    for r in load_predictions(args.run/'predictions.jsonl')]
        write_new(args.output,result)
        print(f'Saved: {args.output}')
        if args.command=='score':
            print(f"Reviewed {result['reviewed_samples']}/{result['total_samples']}; pending {len(result['pending_qids'])}")
            print(json.dumps(result['metrics'],indent=2))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        p.exit(2,f'Evaluation error: {exc}\n')

if __name__=='__main__':
    main()
