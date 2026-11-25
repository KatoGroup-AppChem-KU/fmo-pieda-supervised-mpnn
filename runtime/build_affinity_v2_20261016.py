"""Build ligand-matched point-pX candidates and distributions; Slurm CPU only."""
import csv, hashlib, json, math, os, sys, shutil, time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validate_affinity_flow_20261007 import parse, px, representative, violates, RANK
ROOT = Path(__import__('os').environ['FMO_WORKDIR'])
OUT = ROOT / 'results/affinity_v2_20261016'
CSV = Path(__import__('os').environ['FMO_BINDINGDB_CSV'])
MAN = ROOT / 'dataset/fpv2_prepared_20261016/final/manifest.json'
SMI = ROOT / 'dataset/bdb_fp_v2_20261016_r2_full/smiles_manifest.json'
OLD = ROOT / 'results/affinity_flow_dryrun_20261007_7176/candidate_labels.csv'

def readcsv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def dump(name, rows, columns=None):
    pd.DataFrame(rows, columns=columns).to_csv(OUT / name, index=False, encoding='utf-8-sig')

@lru_cache(maxsize=100000)
def canonical(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return (None, None)
    m = Chem.RemoveHs(m)
    return (Chem.MolToSmiles(m, isomericSmiles=True), Chem.MolToSmiles(m, isomericSmiles=False))

def main():
    assert True
    RDLogger.DisableLog('rdApp.*')
    OUT.mkdir(exist_ok=False)
    shutil.copy2(__file__, OUT / Path(__file__).name)
    assert canonical('CCO')[0] == canonical('OCC')[0]
    assert canonical('F[C@H](Cl)Br')[0] != canonical('F[C@@H](Cl)Br')[0]
    assert parse('≈10')[0] == '~' and px(1000) == 6
    assert violates(6, '>', 1000) and (not violates(6, '>=', 1000))
    mf = json.loads(MAN.read_text())
    smiles = json.loads(SMI.read_text())
    records = {r['pdb'].lower(): r for r in mf['records'] if r['source'] == 'bdb'}
    assert len(records) == 3706 and len(smiles) == 3706
    targets = {p: canonical(smiles[p]['smiles']) for p in records}
    assert all((v[0] for v in targets.values())), 'Unparseable accepted SMILES: do not silently bypass'
    old = {r['pdb_id'].lower(): r for r in readcsv(OLD)}
    groups = defaultdict(list)
    seen = {}
    raw_count = Counter()
    match_count = Counter()
    stereo_count = Counter()
    bad = Counter()
    review = []
    mismatch_pdb_columns = 0
    start = time.monotonic()
    with CSV.open(newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ['pdb_id', 'uniprot', 'monomerid', 'smile', 'affinity_type', 'affinity', 'pmid', 'doi', 'pdb_id']
        for line, values in enumerate(reader, 2):
            assert len(values) == 9, (line, 'bad row')
            if values[0].lower() != values[8].lower():
                mismatch_pdb_columns += 1
            p = values[8].lower()
            if p not in records:
                continue
            raw_count[p] += 1
            iso, nostereo = canonical(values[3])
            if iso is None:
                bad[p] += 1
                continue
            if iso != targets[p][0]:
                if nostereo == targets[p][1]:
                    stereo_count[p] += 1
                    review.append(dict(pdb_id=p, source_line=line, smiles=values[3], monomerid=values[2], affinity_type=values[4], affinity=values[5], doi=values[7], reason='same_bond_order_graph_but_stereochemistry_differs_or_unspecified'))
                continue
            match_count[p] += 1
            key = (p, values[1], values[2], iso, values[4], values[5], values[6], values[7])
            if key in seen:
                seen[key]['source_row_count'] += 1
                seen[key]['source_lines'] += ';' + str(line)
                continue
            r = dict(pdb_id=p, uniprot=values[1], monomerid=values[2], smiles=values[3], canonical_smiles=iso, affinity_type=values[4], affinity_raw=values[5], pmid=values[6], doi=values[7], source_lines=str(line), source_row_count=1, sign='', value_nm='', kind='invalid', parse_error='', selected_for_label=False)
            try:
                assert r['affinity_type'] in RANK, 'unsupported_type'
                sign, val = parse(values[5])
                r.update(sign=sign, value_nm=str(val), value=val, kind='point' if sign in ('=', '~') else 'bound')
            except (ValueError, AssertionError) as e:
                r['parse_error'] = str(e)
            seen[key] = r
            groups[p].append(r)
            if line % 100000 == 0:
                print(f'line {line}, elapsed {time.monotonic() - start:.0f}s', flush=True)
    labels = []
    audit = []
    comparisons = []
    for p, meta in sorted(records.items()):
        rr = groups[p]
        bytype = {t: [r for r in rr if r['affinity_type'] == t and r['kind'] != 'invalid'] for t in RANK}
        available = [t for t in RANK if any((r['kind'] == 'point' for r in bytype[t]))]
        a = dict(pdb_id=p, graph_key=meta['key'], ligand_smiles=smiles[p]['smiles'], ligand_source=smiles[p]['source'], split=meta['split'], subset=meta['subset'], status='', affinity_type='', value_nm='', target_px='', selection_method='', source_candidate_rows=raw_count[p], matching_rows=match_count[p], matching_unique_measurements=len(rr), stereochemistry_review_rows=stereo_count[p], invalid_smiles_rows=bad[p], selected_point_records=0, selected_bound_records=0, conflict_count=0, approximate_present=False, unit='nM', unit_basis='inherited source convention; source CSV has no unit column', monomer_ids='', source_lines='', doi='', pmid='')
        if meta.get('affinity_excluded_by_structure'):
            a['status'] = 'excluded_by_structure'
        elif not rr:
            a['status'] = 'no_exact_ligand_measurement'
        elif not available:
            a['status'] = 'no_valid_point_value'
        else:
            t = available[0]
            pts = [r for r in bytype[t] if r['kind'] == 'point']
            bounds = [r for r in bytype[t] if r['kind'] == 'bound']
            value, reason, counts, tied = representative(pts)
            conflicts = [r for r in bounds if violates(value, r['sign'], r['value'])]
            a.update(status='point_bound_conflict' if conflicts else 'accepted', affinity_type=t, value_nm=10 ** (9 - value), target_px=value, selection_method=reason, selected_point_records=len(pts), selected_bound_records=len(bounds), conflict_count=len(conflicts), approximate_present=any((r['sign'] == '~' for r in pts)), monomer_ids=';'.join(sorted({r['monomerid'] for r in pts})), source_lines=';'.join((r['source_lines'] for r in pts)), doi=';'.join(sorted({r['doi'] for r in pts if r['doi']})), pmid=';'.join(sorted({r['pmid'] for r in pts if r['pmid']})))
            for r in bytype[t]:
                r['selected_for_label'] = not conflicts
        audit.append(a)
        if a['status'] == 'accepted':
            labels.append(a)
        o = old[p]
        oa = o['status'] == 'candidate_point'
        na = a['status'] == 'accepted'
        category = ('retained_same_type_value' if o['affinity_type'] == a['affinity_type'] and math.isclose(float(o['candidate_px']), float(a['target_px']), abs_tol=1e-09) else 'changed_type_or_value') if oa and na else 'lost_label' if oa else 'gained_label' if na else 'neither_accepted'
        comparisons.append(dict(pdb_id=p, comparison=category, old_status=o['status'], new_status=a['status'], old_type=o['affinity_type'], new_type=a['affinity_type'], old_px=o['candidate_px'], new_px=a['target_px'], subset=a['subset']))
    assert len(audit) == 3706 and len({r['pdb_id'] for r in labels}) == len(labels)
    assert not mismatch_pdb_columns, 'Duplicated PDB columns disagree'
    assert all((math.isfinite(r['target_px']) for r in labels))
    dump('affinity_labels_v2.csv', labels)
    dump('structure_audit_v2.csv', audit)
    dump('excluded_v2.csv', [r for r in audit if r['status'] != 'accepted'])
    dump('matched_measurements_v2.csv', [{k: v for k, v in r.items() if k != 'value'} for rr in groups.values() for r in rr])
    dump('stereochemistry_review_v2.csv', review)
    dump('comparison_to_v1.csv', comparisons)
    df = pd.DataFrame(labels)
    statistics = []
    for kind in ['All', *RANK]:
        sub = df if kind == 'All' else df[df.affinity_type == kind]
        for split in ['All', 'train', 'test']:
            z = sub if split == 'All' else sub[sub.split == split]
            v = z.target_px.to_numpy(float)
            stats = dict(group=kind, split=split, n=len(v))
            for name, quant in [('min', 0), ('q25', 0.25), ('median', 0.5), ('q75', 0.75), ('max', 1)]:
                stats[name] = float(np.quantile(v, quant)) if len(v) else None
            stats.update(mean=float(v.mean()) if len(v) else None, sd=float(v.std(ddof=1)) if len(v) > 1 else None)
            statistics.append(stats)
    dump('distribution_summary_v2.csv', statistics)
    bins = np.linspace(math.floor(df.target_px.min()), math.ceil(df.target_px.max()), 25)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    hist = []
    for ax, kind in zip(axes.flat, ['All', *RANK]):
        sub = df if kind == 'All' else df[df.affinity_type == kind]
        ax.hist([sub[sub.split == 'train'].target_px, sub[sub.split == 'test'].target_px], bins=bins, stacked=True, label=['Train', 'Test'], color=['#4777a8', '#e08d43'])
        ax.set(title=f'{kind} (n={len(sub)})', xlabel='pX', ylabel='Structures')
        ax.legend(fontsize=8)
        for split in ['train', 'test']:
            counts, _ = np.histogram(sub[sub.split == split].target_px, bins=bins)
            for lo, hi, n in zip(bins[:-1], bins[1:], counts):
                hist.append(dict(group=kind, split=split, bin_left=lo, bin_right=hi, n=int(n)))
    axes.flat[-1].axis('off')
    fig.tight_layout()
    fig.savefig(OUT / 'affinity_distribution_v2.png', dpi=180)
    fig.savefig(OUT / 'affinity_distribution_v2.pdf')
    plt.close(fig)
    dump('distribution_histogram_v2.csv', hist)
    summary = dict(complete=True, structures=3706, labels=len(labels), status=dict(Counter((r['status'] for r in audit))), types=dict(Counter((r['affinity_type'] for r in labels))), splits=dict(Counter((r['split'] for r in labels))), comparison=dict(Counter((r['comparison'] for r in comparisons))), source_hashes={str(p): sha(p) for p in [CSV, MAN, SMI, OLD, Path(__file__)]}, matching='same PDB and RDKit canonical isomeric SMILES after RemoveHs; no salt removal, tautomer or charge normalization; relaxed matches not adopted', selection='positive point values incl approximate; Kd>Ki>IC50>EC50; one distinct source measurement key one vote; tied modes median pX; same-type bound contradictions excluded', limitations=['nM inherited from original convention, not independently verified for every record', 'assay conditions not filtered', 'target UniProt not an additional selection filter', 'no model training or performance evaluation; distribution only', 'strict stereo matching may leave resolvable molecules unlabelled'])
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'source_hashes'}, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    main()
