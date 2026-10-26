"""Inspect the provided CSV files without modifying them. Standard library only."""
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    report = {'splits': {}, 'overlap': {}, 'notes': [
        'Candidate labels are not evidence-set sufficiency labels.',
        'Product overlap means the official test split is not product-disjoint.',
        'Dataset release/provenance has not been independently verified.',
        'Exact evidence overlap uses ASIN plus whitespace-normalized candidate text.'
    ]}
    sets = {}
    for split in ('train', 'dev', 'test'):
        path = ROOT / f'{split}.csv'
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            columns = reader.fieldnames
        required = {'ASIN', 'qid', 'question', 'candidate', 'label'}
        if not required.issubset(columns or []):
            raise ValueError(f'{path}: missing {required - set(columns or [])}')
        products = defaultdict(set)
        questions = defaultdict(set)
        for row in rows:
            text = ' '.join(row['candidate'].split())
            products[row['ASIN']].add(text)
            questions[row['qid']].add((row['ASIN'], row['question']))
        sizes = sorted(len(texts) for texts in products.values())
        sets[split] = {
            'products': set(products), 'questions': set(questions),
            'evidence': {(asin, text) for asin, texts in products.items() for text in texts}
        }
        report['splits'][split] = {
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'rows': len(rows), 'questions': len(questions), 'products': len(products),
            'columns': columns, 'labels': dict(Counter(row['label'] for row in rows)),
            'sources': dict(Counter(row.get('source', '') for row in rows)),
            'rows_per_question_histogram': dict(Counter(Counter(row['qid'] for row in rows).values())),
            'unique_candidates_per_product': {
                'min': sizes[0], 'median': (sizes[(len(sizes)-1)//2]+sizes[len(sizes)//2])/2,
                'max': sizes[-1], 'at_most_three': sum(size <= 3 for size in sizes)
            },
            'inconsistent_question_ids': sum(len(values) != 1 for values in questions.values()),
            'empty_candidates': sum(not row['candidate'].strip() for row in rows)
        }
    for left, right in itertools.combinations(sets, 2):
        report['overlap'][f'{left}__{right}'] = {
            key: len(sets[left][key] & sets[right][key]) for key in sets[left]
        }
    output = ROOT / 'outputs' / 'data_audit.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'\nSaved: {output}')

if __name__ == '__main__':
    main()
