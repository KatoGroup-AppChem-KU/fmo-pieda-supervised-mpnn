"""Non-destructive dry run of policy 2.5; execute only in aurum-g01 Slurm."""
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from statistics import median
ROOT = Path(__import__('os').environ['FMO_WORKDIR'])
RANK = ('Kd', 'Ki', 'IC50', 'EC50')
BASE = ROOT / 'results/ic50_coverage_20261006'
SOURCES = [BASE / 'Table_S1_BindingDB_structures.csv', BASE / 'Table_S1_BindingDB_affinity_records.csv', ROOT / 'dataset/fixedtest_cv5_20260925/assignments.csv']

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def parse(raw):
    raw = raw.strip().replace('≤', '<=').replace('≥', '>=').replace('≈', '~')
    m = re.fullmatch('(<=|>=|<|>|=|~)?\\s*([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?)', raw)
    if not m:
        raise ValueError('unparsed')
    value = Decimal(m[2])
    if value <= 0 or not math.isfinite(float(value)):
        raise ValueError('nonpositive_or_nonfinite')
    return (m[1] or '=', value)

def px(value):
    return 9 - math.log10(float(value))

def representative(rows, weighted=False):
    counts = Counter()
    for r in rows:
        counts[r['value']] += int(r['source_row_count'] or 1) if weighted else 1
    highest = max(counts.values())
    tied = sorted((v for v, n in counts.items() if n == highest))
    reason = 'single_record' if len(rows) == 1 else 'unique_mode' if len(tied) == 1 else 'tied_mode_median_px'
    return (median((px(v) for v in tied)), reason, counts, tied)

def violates(point, sign, threshold):
    b = px(threshold)
    close = math.isclose(point, b, rel_tol=0, abs_tol=1e-10)
    if sign == '>':
        return point > b or close
    if sign == '>=':
        return point > b and (not close)
    if sign == '<':
        return point < b or close
    if sign == '<=':
        return point < b and (not close)
    raise ValueError(sign)

def dumpcsv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def main():
    assert True
    assert parse('>1E4') == ('>', Decimal('10000'))
    assert parse('≈ 10') == ('~', Decimal(10))
    assert px(Decimal(1000)) == 6
    assert violates(7, '>', Decimal(1000)) and (not violates(5, '>', Decimal(1000)))
    assert violates(6, '>', Decimal(1000)) and (not violates(6, '>=', Decimal(1000)))
    example = [{'value': Decimal(v), 'source_row_count': '1'} for v in ('10', '1000')]
    assert representative(example)[0] == 7
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}
    structures, records, assignments = map(read, SOURCES)
    metadata = {r['pdb_id'].upper(): r for r in structures}
    split = {r['pdb'].upper(): r for r in assignments if r['source'] == 'bdb'}
    assert len(metadata) == len(structures) == len(split) == 3714
    assert set(metadata) == set(split)
    assert all((r['subset'] == split[p]['subset'] for p, r in metadata.items()))
    groups = defaultdict(lambda: defaultdict(list))
    invalid, parsed = ([], [])
    for i, r in enumerate(records, 2):
        r = dict(r, record_line=i)
        assert r['pdb_id'].upper() in metadata
        if r['affinity_type'] not in RANK:
            invalid.append(dict(r, reason='unsupported_type'))
            continue
        try:
            sign, value = parse(r['affinity_value_raw'])
        except ValueError as e:
            invalid.append(dict(r, reason=str(e)))
            continue
        r.update(sign=sign, value=value, kind='point' if sign in ('=', '~') else 'bound')
        parsed.append(r)
        groups[r['pdb_id'].upper()][r['affinity_type']].append(r)
    labels, evidence, target_examples = ([], [], [])
    for pdb, meta in sorted(metadata.items()):
        bytype = groups[pdb]
        points = [t for t in RANK if any((r['kind'] == 'point' for r in bytype[t]))]
        available = [t for t in RANK if bytype[t]]
        t = (points or available or [''])[0]
        selected = bytype[t] if t else []
        pointrows = [r for r in selected if r['kind'] == 'point']
        bounds = [r for r in selected if r['kind'] == 'bound']
        row = dict(pdb_id=pdb, subset=split[pdb]['subset'], affinity_type=t, status='', label_kind='', candidate_px='', px_relation='', reason='', selected_record_count=len(selected), point_count=len(pointrows), bound_count=len(bounds), point_bound_conflict_count=0, approximate_present=any((r['sign'] == '~' for r in pointrows)), mode_counts_nm='', tied_candidates_nm='', weighted_candidate_px='', source_weight_changes_label=False, target_id_mismatch=False, target_relation='', lower_point_over_upper_bound=bool(points and available and (points[0] != available[0])), unit_assumed='nM', graph_paths_exist=all(((ROOT / split[pdb][k]).is_file() for k in ('v6_path', 'v7_path'))))
        if pointrows:
            value, reason, counts, tied = representative(pointrows)
            weighted = representative(pointrows, weighted=True)[0]
            conflicts = [r for r in bounds if violates(value, r['sign'], r['value'])]
            row.update(status='review_point_bound_conflict' if conflicts else 'candidate_point', label_kind='point', candidate_px=value, px_relation='=', reason=reason, point_bound_conflict_count=len(conflicts), mode_counts_nm=json.dumps({str(v): n for v, n in sorted(counts.items())}), tied_candidates_nm=';'.join(map(str, tied)), weighted_candidate_px=weighted, source_weight_changes_label=not math.isclose(value, weighted, abs_tol=1e-10))
        elif bounds:
            unique = {(r['sign'], r['value']) for r in bounds}
            row.update(label_kind='bound', reason='single_unique_bound' if len(unique) == 1 else 'multiple_bounds_unresolved')
            if len(unique) == 1:
                sign, value = next(iter(unique))
                row.update(status='candidate_bound', candidate_px=px(value), px_relation={'>': '<', '>=': '<=', '<': '>', '<=': '>='}[sign])
            else:
                row['status'] = 'review_multiple_bounds'
        else:
            row.update(status='no_valid_value', reason='no_positive_supported_measurement')
        row['target_id_mismatch'] = any((r['affinity_uniprot'].strip() != meta['uniprot'].strip() for r in selected))

        def ids(s):
            return set((x for x in re.split('[;,|\\s]+', s.strip()) if x))
        structure_ids = ids(meta['uniprot'])
        relations = set()
        for r in selected:
            assay_ids = ids(r['affinity_uniprot'])
            relations.add('missing' if not structure_ids or not assay_ids else 'same_set' if structure_ids == assay_ids else 'overlap' if structure_ids & assay_ids else 'disjoint')
        row['target_relation'] = ';'.join(sorted(relations))
        if row['target_id_mismatch'] and len(target_examples) < 20:
            target_examples.append(dict(pdb_id=pdb, structure_uniprot=meta['uniprot'], assay_uniprot=sorted({r['affinity_uniprot'] for r in selected}), relation=row['target_relation']))
        for r in selected:
            evidence.append(dict(pdb_id=pdb, affinity_type=t, record_line=r['record_line'], value_raw=r['affinity_value_raw'], sign=r['sign'], value_nm=str(r['value']), px=px(r['value']), source_row_count=r['source_row_count'], doi=r['doi'], pmid=r['pmid'], affinity_uniprot=r['affinity_uniprot'], bindingdb_monomer_id=r['bindingdb_monomer_id']))
        labels.append(row)
    assert len(labels) == 3714 and all((r['graph_paths_exist'] for r in labels))
    assert all((hashes[str(p)] == hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES))
    out = ROOT / 'results' / ('affinity_flow_dryrun_20261007_' + os.environ.get('SLURM_JOB_ID', 'standalone'))
    out.mkdir(exist_ok=False)
    dumpcsv(out / 'candidate_labels.csv', labels)
    dumpcsv(out / 'selected_evidence.csv', evidence)
    dumpcsv(out / 'invalid_records.csv', invalid)
    dumpcsv(out / 'review_cases.csv', [r for r in labels if not r['status'].startswith('candidate_') or r['source_weight_changes_label'] or r['target_id_mismatch']])
    summary = dict(total_structures=len(labels), input_records=len(records), valid_records=len(parsed), invalid_records=len(invalid), status=dict(Counter((r['status'] for r in labels))), selected_types=dict(Counter((r['affinity_type'] for r in labels))), representative_reasons=dict(Counter((r['reason'] for r in labels))), approximate_structures=sum((r['approximate_present'] for r in labels)), lower_point_over_upper_bound=sum((r['lower_point_over_upper_bound'] for r in labels)), source_weight_changes_label=sum((r['source_weight_changes_label'] for r in labels)), target_id_mismatch=sum((r['target_id_mismatch'] for r in labels)), target_relations=dict(Counter((r['target_relation'] for r in labels))), target_examples=target_examples, conservative_candidates=sum((r['status'].startswith('candidate_') and (not r['source_weight_changes_label']) and (r['target_relation'] == 'same_set') for r in labels)), subset_status={s: dict(Counter((r['status'] for r in labels if r['subset'] == s))) for s in sorted({r['subset'] for r in labels})}, source_hashes=hashes, input_files_unchanged=True, graph_paths_verified=True, caveats=['dry run, not final labels', 'one CSV row = one vote; source_row_count sensitivity only', 'no additional independent-assay deduplication', 'target matching inherited from source CSV', 'no training performed; no evaluation scores computed'], output=str(out))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
