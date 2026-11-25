"""Prepare only: retain fixed split, replace graphs, recompute train-only stats.

Independent audit approval is deliberately NOT granted by this script.
No affinity labels or training jobs are created.
"""
import copy
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
assert True
ROOT = Path(__import__('os').environ['FMO_WORKDIR'])
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import torch
from prepare_geo_pieda_matched_20260911 import load, fingerprint, raw_targets, DISTANCES
torch.set_num_threads(2)
OUT = ROOT / 'dataset/fpv2_prepared_20261016'
NEW = ROOT / 'dataset/bdb_fp_v2_20261016_r2_full'
WATER = ROOT / 'dataset/waterfp_20261014'
EXCLUDED = set('1kyn 1qbv 2ity 1nde 2xno 4eil 4mdn 5xvg'.split())

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, obj):
    with Path(p).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
done = read(NEW / 'COMPLETE.json')
assert done['all_verified'] and done['structures'] == 3706 and (done['graphs'] == 7412)
assert read(WATER / 'COMPLETE.json')['complete']
new = {r['key']: r for r in read(NEW / 'graph_manifest.json')['records']}
water_manifest = read(WATER / 'final/manifest.json')
assert sha(WATER / 'final/manifest.json') == read(WATER / 'final/COMPLETE.json')['files_sha256']['manifest.json']
water = {r['key']: r for r in water_manifest['records']}
source = ROOT / 'dataset/final_train_20261008/manifest.json'
old = read(source)
removed = [r for r in old['records'] if r['source'] == 'bdb' and r['pdb'] in EXCLUDED]
assert len(removed) == 8
records = []
for original in old['records']:
    if original['source'] == 'bdb' and original['pdb'] in EXCLUDED:
        continue
    r = copy.deepcopy(original)
    replacement = new[r['key']] if r['source'] == 'bdb' else water[r['key']]
    assert replacement['pdb'] == r['pdb'] and replacement['subset'] == r['subset']
    for field in ('paths', 'fingerprints'):
        r[field] = copy.deepcopy(replacement[field])
    if r['source'] == 'bdb':
        r['fp_source'] = replacement['fp_source']
        r['affinity_excluded_by_structure'] = replacement['affinity_excluded_by_structure']
    records.append(r)
assert len(records) == 8983 and len({r['key'] for r in records}) == 8983
groups = defaultdict(set)
for r in records:
    groups[r['pdb']].add(r['subset'])
assert all((len(v) == 1 for v in groups.values())), 'PDB crosses subsets'
OUT.mkdir(exist_ok=False)
sums = {f'subset{i}': [0, torch.zeros(7, dtype=torch.float64), torch.zeros(7, dtype=torch.float64)] for i in range(1, 6)}
for i, r in enumerate(records, 1):
    gs = {v: load(r['paths'][v]) for v in ('v6', 'v7')}
    for v, g in gs.items():
        assert fingerprint(g) == r['fingerprints'][v], (r['key'], v, 'fingerprint')
        assert torch.isfinite(g.x).all() and torch.isfinite(g.edge_attr).all()
        names = list(g.node_feature_names)
        cols = [names.index('small_mol_morgan_' + str(k)) for k in range(512)]
        mask = g.x[:, names.index('is_water')] > 0.5
        expected = torch.zeros(512, dtype=g.x.dtype)
        expected[278] = 1
        assert torch.equal(g.x[mask][:, cols], expected.expand(int(mask.sum()), 512)), (r['key'], v, 'water')
    a, b = (gs['v6'], gs['v7'])
    assert torch.equal(a.edge_index, b.edge_index) and torch.equal(a.frag_id, b.frag_id)
    assert torch.equal(a.pos, b.pos)
    assert torch.allclose(raw_targets(a), raw_targets(b), atol=0.0002, rtol=1e-05)
    if r['subset'] != 'test':
        y = torch.cat([b.edge_attr[:, [b.edge_attr_names.index(k) for k in DISTANCES]].double(), raw_targets(b)], 1)
        assert torch.isfinite(y).all()
        st = sums[r['subset']]
        st[0] += len(y)
        st[1] += y.sum(0)
        st[2] += (y * y).sum(0)
    if i % 500 == 0:
        print(f'Validated graphs and accumulated training statistics: {i}/{len(records)}', flush=True)
    (OUT / 'progress.json').write_text(json.dumps(dict(verified_structures=i, total=len(records))))
audits = {}
for fold in [1, 2, 3, 4, 5, None]:
    src = ROOT / f'dataset/fixedtest_cv5_20260925_prepared/fold{fold}.json' if fold else source
    m = read(src)
    original_by_key = {r['key']: r for r in m['records']}
    rr = copy.deepcopy(records)
    for r in rr:
        r['split'] = 'test' if r['subset'] == 'test' else 'valid' if r['subset'] == f'subset{fold}' else 'train'
        assert r['split'] == original_by_key[r['key']]['split']
    tr = [s for k, s in sums.items() if k != f'subset{fold}']
    n = sum((s[0] for s in tr))
    total = sum((s[1] for s in tr))
    square = sum((s[2] for s in tr))
    mean = total / n
    std = (square / n - mean.square()).clamp_min(0).sqrt()
    assert (std > 1e-08).all()
    counts = {sp: dict(Counter((r['source'] for r in rr if r['split'] == sp))) for sp in ('train', 'valid', 'test')}
    m.update(records=rr, counts=counts, train_stats=dict(n=n, mean=mean.tolist(), std=std.tolist()), normalization_status='recomputed after BDB exclusions: train only, population std, canonical v7 raw targets', normalization_train_keys_sha256=hashlib.sha256('\n'.join((r['key'] for r in rr if r['split'] == 'train')).encode()).hexdigest(), all_verified=False, independent_audit_required=True, training_authorized=False, fpv2_provenance=dict(source_manifest=str(src), source_sha256=sha(src), graph_manifest_sha256=sha(NEW / 'graph_manifest.json'), water_manifest_sha256=sha(WATER / 'final/manifest.json'), excluded_bdb=sorted(EXCLUDED), split_assignments_preserved=True, affinity_labels='not created; must rebuild against corrected ligand identities'))
    m['graph_validation'] = 'preparation checked fingerprints, water FP, cross-version topology/raw targets; independent audit pending'
    folder = OUT / 'cv' if fold else OUT / 'final'
    folder.mkdir(exist_ok=True)
    name = f'fold{fold}.json' if fold else 'manifest.json'
    write(folder / name, m)
    audits[name] = dict(counts=counts, normalization_edges=n, normalization_pairs=n // 2)
for folder in (OUT / 'cv', OUT / 'final'):
    hashes = {p.name: sha(p) for p in folder.glob('*.json')}
    write(folder / 'PREPARED.json', dict(prepared=True, training_authorized=False, files_sha256=hashes))
write(OUT / 'audit.json', dict(manifests=audits, removed_records=removed, structures=len(records), no_pdb_subset_overlap=True, normalization_excludes_validation_test=True))
write(OUT / 'PREPARED.json', dict(prepared=True, training_authorized=False, independent_audit_required=True, structures=len(records), script_sha256=sha(__file__), affinity_labels_ready=False))
print('PREPARED: independent audit approval and training launch remain pending.', flush=True)
