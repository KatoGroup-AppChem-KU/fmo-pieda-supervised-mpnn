"""Freeze agreed rescue decisions; no normalization-search candidates adopted."""
import csv, json, os, math, shutil
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_affinity_v2_20261016 import readcsv, canonical, sha, CSV, MAN, SMI
from validate_affinity_flow_20261007 import parse, representative, violates, RANK
assert True
ROOT = Path(__import__('os').environ['FMO_WORKDIR'])
BASE = ROOT / 'results/affinity_v2_20261016'
TRACE = ROOT / 'results/stereo_measurement_trace_7562'
OUT = ROOT / 'results' / ('affinity_v2_adopted_' + os.environ.get('SLURM_JOB_ID', 'standalone'))
OUT.mkdir(exist_ok=False)
shutil.copy2(__file__, OUT / Path(__file__).name)

def dump(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding='utf-8-sig')

def ids(s):
    return set(s.split())
excluded = ids('1cma 1cmc 2g70 2g72 1rhj 1xh5 2cma 2clx 3tn8 3tnh 3tnw 6i10 7sza')
name_ok = ids('1nfw 1xlz 2qfo 2bhe 2zm3 3py0 3py1 6guc')
mixture = ids('4v2y 5amh 5ami 5amj 5amk 5oh1 6r13 6r19 6r1a 7pso 8bc7 1xm6 2vl4 3v36')
missing = {r['pdb_id']: r for r in readcsv(BASE / 'recovery_audit/structures.csv')}
trace_summary = {r['pdb_id']: r for r in readcsv(TRACE / 'structure_summary.csv')}
assert len(trace_summary) == 306
decisions = {}
for p, r in trace_summary.items():
    if p in excluded:
        continue
    rel = r['relations'].split(';') if r['relations'] else []
    if int(r['same_value_exact_structure_records']):
        group = 'official_exact'
        flag = 'official_exact_structure_value_candidate'
    elif p in name_ok:
        group = 'name_EZ'
        flag = 'name_EZ_consistent'
    elif p in mixture:
        group = 'mixture'
        flag = 'stereo_mixture_approximation'
    elif rel == ['official_stereo_unspecified']:
        group = 'source_unspecified'
        flag = 'stereo_unspecified_allowed'
    elif rel == ['target_stereo_unspecified']:
        group = 'target_unspecified'
        flag = 'fmo_smiles_stereo_unspecified'
    elif not int(r['same_value_records']):
        group = 'measurement_unconfirmed'
        flag = 'official_measurement_unconfirmed_csv_used'
    elif rel == ['stereo_conflict_or_mapping_review']:
        group = 'conflict_allowed'
        flag = 'stereo_conflict_or_mapping_unresolved_allowed'
    else:
        group = 'mixed'
        flag = 'mixed_candidate_status_allowed'
    if p == '3inf':
        flag += ';stereo_mixture_approximation'
    decisions[p] = dict(group=group, flag=flag)
expected = dict(official_exact=86, name_EZ=8, mixture=14, source_unspecified=115, target_unspecified=26, measurement_unconfirmed=2, conflict_allowed=29, mixed=13)
assert dict(Counter((d['group'] for d in decisions.values()))) == expected
assert len(decisions) == 293
dump('adoption_decisions.csv', [dict(pdb_id=p, **d) for p, d in sorted(decisions.items())])
evidence = defaultdict(list)
with (TRACE / 'measurement_structure_trace.csv').open(encoding='utf-8-sig', newline='') as f:
    for r in csv.DictReader(f):
        if r['pdb_id'] not in decisions or r['same_value'] != 'True':
            continue
        d = decisions[r['pdb_id']]
        wanted = 'exact_isomeric' if d['group'] == 'official_exact' else 'official_stereo_unspecified'
        if r['structure_relation'] == wanted:
            evidence[r['pdb_id'], r['source_line']].append(r)
review = [r for r in readcsv(BASE / 'recovery_audit/candidates.csv') if r['pdb_id'] in decisions]
eligible_keys = set()
for r in review:
    p = r['pdb_id']
    d = decisions[p]
    if r['positive_point'] != 'True':
        continue
    restricted = d['group'] in ('official_exact', 'source_unspecified', 'name_EZ', 'mixture')
    if not restricted or evidence[p, r['source_line']]:
        eligible_keys.add((p, r['monomerid'], canonical(r['smiles'])[0]))
selected_lines = {int(r['source_line']): r for r in review if (r['pdb_id'], r['monomerid'], canonical(r['smiles'])[0]) in eligible_keys}
assert {r['pdb_id'] for r in selected_lines.values()} == set(decisions)
groups = defaultdict(list)
seen = {}
saved_evidence = []
old_audit = {r['pdb_id']: r for r in readcsv(BASE / 'structure_audit_v2.csv')}
for r in readcsv(BASE / 'matched_measurements_v2.csv'):
    if r['kind'] != 'invalid':
        r['value'] = Decimal(r['value_nm'])
    r.update(acceptance_flags='strict_isomeric_match', candidate_relation='exact_isomeric', official_rsids='', official_article_dois='', official_smiles='')
    groups[r['pdb_id']].append(r)
with CSV.open(newline='') as f:
    reader = csv.reader(f)
    assert next(reader) == ['pdb_id', 'uniprot', 'monomerid', 'smile', 'affinity_type', 'affinity', 'pmid', 'doi', 'pdb_id']
    for line, v in enumerate(reader, 2):
        if line not in selected_lines:
            continue
        a = selected_lines[line]
        p = v[8].lower()
        assert len(v) == 9 and p == a['pdb_id'] == v[0].lower()
        assert v[2] == a['monomerid'] and v[3] == a['smiles'] and (v[4] == a['affinity_type']) and (v[5] == a['affinity'])
        iso, ns = canonical(v[3])
        target_iso, target_ns = canonical(old_audit[p]['ligand_smiles'])
        assert ns == target_ns
        key = (p, v[1], v[2], iso, v[4], v[5], v[6], v[7])
        ev = evidence.get((p, str(line)), [])
        saved_evidence.extend(ev)
        if key in seen:
            r = seen[key]
            r['source_row_count'] += 1
            r['source_lines'] += ';' + str(line)
            continue
        flags = decisions[p]['flag'] + ';original_csv_relation_' + a['category']
        r = dict(pdb_id=p, uniprot=v[1], monomerid=v[2], smiles=v[3], canonical_smiles=iso, affinity_type=v[4], affinity_raw=v[5], pmid=v[6], doi=v[7], source_lines=str(line), source_row_count=1, sign='', value_nm='', kind='invalid', parse_error='', selected_for_label=False, acceptance_flags=flags, candidate_relation=a['category'], official_rsids=';'.join(sorted({e['official_rsid'] for e in ev})), official_article_dois=';'.join(sorted({e['article_doi'] for e in ev if e['article_doi']})), official_smiles=';'.join(sorted({e['official_smiles'] for e in ev})))
        try:
            assert v[4] in RANK, 'unsupported_type'
            sign, val = parse(v[5])
            r.update(sign=sign, value_nm=str(val), value=val, kind='point' if sign in ('=', '~') else 'bound')
        except (ValueError, AssertionError) as e:
            r['parse_error'] = str(e)
        groups[p].append(r)
        seen[key] = r
print('Approved measurement records loaded; selecting representatives', flush=True)
labels = []
audits = []
for p, original in sorted(old_audit.items()):
    a = dict(original)
    rr = groups[p]
    a['acceptance_flags'] = 'strict_isomeric_match' if p not in decisions else decisions[p]['flag']
    a['rescue_group'] = decisions.get(p, {}).get('group', '')
    if p in decisions:
        a.update(matching_rows=sum((int(r['source_row_count']) for r in rr)), matching_unique_measurements=len(rr))
        bytype = {t: [r for r in rr if r['affinity_type'] == t and r['kind'] != 'invalid'] for t in RANK}
        available = [t for t in RANK if any((r['kind'] == 'point' for r in bytype[t]))]
        if not available:
            a['status'] = 'no_valid_point_value'
        else:
            t = available[0]
            pts = [r for r in bytype[t] if r['kind'] == 'point']
            bounds = [r for r in bytype[t] if r['kind'] == 'bound']
            val, reason, counts, tied = representative(pts)
            conflicts = [r for r in bounds if violates(val, r['sign'], r['value'])]
            a.update(status='point_bound_conflict' if conflicts else 'accepted', affinity_type=t, value_nm=10 ** (9 - val), target_px=val, selection_method=reason, selected_point_records=len(pts), selected_bound_records=len(bounds), conflict_count=len(conflicts), approximate_present=any((r['sign'] == '~' for r in pts)), monomer_ids=';'.join(sorted({r['monomerid'] for r in pts})), source_lines=';'.join((r['source_lines'] for r in pts)), doi=';'.join(sorted({r['doi'] for r in pts if r['doi']})), pmid=';'.join(sorted({r['pmid'] for r in pts if r['pmid']})), acceptance_flags=';'.join(sorted({flag for r in pts for flag in r['acceptance_flags'].split(';')})))
            for r in bytype[t]:
                r['selected_for_label'] = not conflicts
    elif p in missing:
        a['acceptance_flags'] = ''
        a['status'] = 'deferred_no_bond_graph_match' if missing[p]['has_stereo_candidate'] == 'False' else 'excluded_by_review' if p in excluded else 'no_valid_point_value'
    audits.append(a)
    if a['status'] == 'accepted':
        labels.append(a)
    assert (a['split'], a['subset']) == (original['split'], original['subset'])
assert len(audits) == 3706 and len({r['pdb_id'] for r in labels}) == len(labels)
assert sum((r['status'] == 'deferred_no_bond_graph_match' for r in audits)) == 400
assert all((math.isfinite(float(r['target_px'])) for r in labels))
old_labels = {r['pdb_id']: r for r in readcsv(BASE / 'affinity_labels_v2.csv')}
new_labels = {r['pdb_id']: r for r in labels}
assert len(old_labels) == 2847 and set(old_labels) <= set(new_labels)
assert all((float(r['target_px']) == float(new_labels[p]['target_px']) and r['affinity_type'] == new_labels[p]['affinity_type'] for p, r in old_labels.items()))
dump('affinity_labels_v2_adopted.csv', labels)
dump('structure_audit.csv', audits)
dump('excluded_and_deferred.csv', [r for r in audits if r['status'] != 'accepted'])
dump('matched_measurements.csv', [{k: v for k, v in r.items() if k != 'value'} for rr in groups.values() for r in rr])
dump('official_confirmation_evidence.csv', saved_evidence)
dump('additional_labels.csv', [r for r in labels if r['pdb_id'] not in old_labels])
df = pd.DataFrame(labels)
df['target_px'] = df.target_px.astype(float)
stats = []
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
bins = np.linspace(math.floor(df.target_px.min()), math.ceil(df.target_px.max()), 25)
for ax, t in zip(axes.flat, ['All', *RANK]):
    sub = df if t == 'All' else df[df.affinity_type == t]
    for split in ['All', 'train', 'test']:
        z = sub if split == 'All' else sub[sub.split == split]
        v = z.target_px.to_numpy()
        stats.append(dict(indicator=t, split=split, n=len(v), mean=float(v.mean()) if len(v) else None, sd=float(v.std(ddof=1)) if len(v) > 1 else None, **{name: float(np.quantile(v, q)) if len(v) else None for name, q in [('min', 0), ('q25', 0.25), ('median', 0.5), ('q75', 0.75), ('max', 1)]}))
    ax.hist([sub[sub.split == 'train'].target_px, sub[sub.split == 'test'].target_px], bins=bins, stacked=True, label=['Train', 'Test'])
    ax.set(title=f'{t} (n={len(sub)})', xlabel='pX', ylabel='Structures')
    ax.legend()
axes.flat[-1].axis('off')
fig.tight_layout()
fig.savefig(OUT / 'affinity_distribution.png', dpi=180)
fig.savefig(OUT / 'affinity_distribution.pdf')
plt.close(fig)
dump('distribution_summary.csv', stats)
summary = dict(complete=True, labels=len(labels), additional_labels=len(labels) - 2847, status=dict(Counter((r['status'] for r in audits))), types=dict(Counter((r['affinity_type'] for r in labels))), splits=dict(Counter((r['split'] for r in labels))), adoption_groups=expected, rescue_final_status=dict(Counter((a['status'] for a in audits if a['pdb_id'] in decisions))), source_hashes={str(p): sha(p) for p in [CSV, MAN, SMI, BASE / 'summary.json', TRACE / 'structure_summary.csv', Path(__file__)]}, caveats=['nM inherited; source CSV has no units column', 'stereo uncertainty explicitly allowed by recorded decisions', 'no assay filtering; heterogeneous endpoints represented as pX', '400 normalized-search structures not adopted', 'not a new graph build or model training'], output=str(OUT))
(OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != 'source_hashes'}, ensure_ascii=False, indent=2), flush=True)
