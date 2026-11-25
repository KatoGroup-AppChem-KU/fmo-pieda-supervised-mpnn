"""Audit final original heads and frozen probes on standard-AA--ligand edges.

Existing predictions only: no training/inference/model selection. All compute
and graph/row-alignment checks are confined to an aurum-g01 allocation.
"""
import os, sys, json, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
assert True
ROOT = Path(__import__('os').environ['FMO_WORKDIR'])
sys.path[:0] = [str(ROOT / 'outputs'), str(ROOT)]
import fpv2_sg_b_20261016 as fp
import audit_fpv2_original_heads_20261021 as audit
from final_pieda_ci_20261012 import pooled
torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
torch.set_num_interop_threads(1)
OUT = ROOT / 'results/ligand_protein_final_audit_20261109'
OUT.mkdir(exist_ok=False)
TARGETS = ['PIEDA_sum', 'ES', 'EX', 'CTmix', 'DI']
hashes = {}

def sha(path):
    path = Path(path)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes[str(path)] = h
    return h

def loadnp(path):
    sha(path)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
base = ROOT / 'results/fpv2_pieda_headcv_7571/model1_v6'
records_path = base / 'geo_test_records.json'
records = json.loads(records_path.read_text())
sha(records_path)
assert len(records) == 898 and len({r['pdb'] for r in records}) == 874
assert records == json.loads((base / 'geo_pieda_test_records.json').read_text())
pdbs = sorted({r['pdb'] for r in records})
lookup = {p: i for i, p in enumerate(pdbs)}
graph_pdb = np.array([lookup[r['pdb']] for r in records])
origbase = ROOT / 'results/fpv2_original_head_audit_20261021'
orig = loadnp(origbase / 'final/metadata.npz')
assert orig['keys'].tolist() == [r['key'] for r in records]
original_indices = loadnp(origbase / 'pdb_bootstrap_indices.npz')
assert original_indices['pdbids'].tolist() == pdbs
probe_indices = loadnp(base / 'pdb_bootstrap_indices.npz')
assert probe_indices['pdbids'].tolist() == pdbs
mfpath = fp.PREP / 'final/manifest.json'
mf = json.loads(mfpath.read_text())
sha(mfpath)
bags = {k: [] for k in ['y', 'graph_ids', 'source', 'kind', 'pair']}
selection_hash = hashlib.sha256()
offset = 0
begin = time.monotonic()
for gi, r in enumerate(records):
    g = fp.cv.load(r, mf, 'v6')
    raw = fp.base.load(r['paths']['v6'])
    typ = audit.flags(raw)
    sel = fp.base.probe_selection(g)
    u, v = g['ei'][:, sel]
    a, b = (typ[u], typ[v])
    bags['y'].append(g['raw'][sel].numpy())
    bags['graph_ids'].append(np.full(len(sel), gi, np.int32))
    bags['source'].append(np.full(len(sel), int(r['source'] == 'scop2'), np.int8))
    bags['kind'].append((torch.minimum(a, b) * 4 + torch.maximum(a, b)).numpy().astype(np.int8))
    bags['pair'].append(g['inverse'][sel].numpy() + offset)
    offset += g['pairs']
    selection_hash.update(g['key'].encode())
    selection_hash.update(sel.numpy().tobytes())
    if (gi + 1) % 100 == 0:
        print(f'Align frozen-probe edges {gi + 1}/898; {time.monotonic() - begin:.1f}s', flush=True)
probe = {k: np.concatenate(v) for k, v in bags.items()}
probe['selection_sha256'] = selection_hash.hexdigest()
assert len(probe['y']) == 1579974
assert probe['selection_sha256'] == '3d62065a32c2f66aa580f8056559ece204528265bad3daade5d8dd548e37dcba'
np.savez_compressed(OUT / 'probe_edge_metadata.npz', **{k: v for k, v in probe.items() if isinstance(v, np.ndarray)})
points = []
intervals = []
counts = []

def evaluate(series, rep, condition, meta, p, indices, reference=None):
    y = meta['y'][:, 2:] if series == 'original_head' else meta['y']
    y = y.astype(np.float64)
    p = p.astype(np.float64)
    assert y.shape == p.shape and y.shape[1] == 5 and np.isfinite(y).all() and np.isfinite(p).all()
    if series == 'frozen_probe':
        expected = {('GEO', 'node'): 0.8512, ('GEO', 'pair'): 0.7289, ('C0', 'node'): 0.9184, ('C0', 'pair'): 0.9315, ('C1', 'node'): 0.9415, ('C1', 'pair'): 0.8964}
    else:
        expected = {('C0', 'node'): 0.928, ('C0', 'pair'): 0.9256, ('C1', 'node'): 0.9564, ('C1', 'pair'): 0.9552}
    allr = float(np.corrcoef(y[:, 0], p[:, 0])[0, 1])
    assert abs(allr - expected[condition, rep]) < 5.1e-05, (series, rep, condition, allr)
    take = meta['kind'] == 1
    assert take.any()
    gids = meta['graph_ids'][take]
    cluster = graph_pdb[gids]
    yy, pp = (y[take], p[take])
    N = len(yy)
    n_graph = len(np.unique(gids))
    n_pdb = len(np.unique(cluster))
    n_pairs = len(np.unique(meta['pair'][take]))
    assert N == 2 * n_pairs, (series, N, n_pairs)
    counts.append(dict(series=series, representation=rep, condition=condition, n_records=N, n_unordered_pairs=n_pairs, n_graphs=n_graph, n_pdbs=n_pdb, all_records=len(y)))
    stat = np.zeros((len(pdbs), 5, 8), dtype=np.float64)
    for j in range(5):
        a, b = (yy[:, j], pp[:, j])
        for k, values in enumerate([np.ones(N), a, b, a * a, b * b, a * b, (a - b) ** 2, np.abs(a - b)]):
            stat[:, j, k] = np.bincount(cluster, weights=values, minlength=len(pdbs))
    point = pooled(stat.sum(0))
    draws = []
    for chunk in np.array_split(indices, 100):
        mult = np.stack([np.bincount(row, minlength=len(pdbs)) for row in chunk])
        draws.append(pooled(np.einsum('bg,gck->bck', mult, stat)))
    draws = np.concatenate(draws)
    assert np.isfinite(draws).all()
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    for j, target in enumerate(TARGETS):
        met = audit.metric(yy[:, j], pp[:, j])
        assert abs(met['pearson_r'] - point[j, 0]) < 1e-10
        points.append(dict(series=series, representation=rep, condition=condition, target=target, **met, n_records=N, n_unordered_pairs=n_pairs, n_graphs=n_graph, n_pdbs=n_pdb))
        for k, name in enumerate(['pearson_r', 'rmse', 'mae', 'r2']):
            intervals.append(dict(series=series, representation=rep, condition=condition, target=target, metric=name, estimate=point[j, k], ci_low=lo[j, k], ci_high=hi[j, k], n_records=N, n_unordered_pairs=n_pairs, n_graphs=n_graph, n_pdbs=n_pdb))
    if reference is not None:
        refpoint, refdraw, refgids, refy = reference
        np.testing.assert_array_equal(gids, refgids)
        np.testing.assert_array_equal(yy, refy)
        diff = point - refpoint
        dc = draws - refdraw
        lo, hi = np.quantile(dc, [0.025, 0.975], axis=0)
        for j, target in enumerate(TARGETS):
            for k, name in enumerate(['pearson_r', 'rmse', 'mae', 'r2']):
                intervals.append(dict(series=series, representation=rep, condition='delta_' + condition, target=target, metric=name, estimate=diff[j, k], ci_low=lo[j, k], ci_high=hi[j, k], n_records=N, n_unordered_pairs=n_pairs, n_graphs=n_graph, n_pdbs=n_pdb))
    print(series, rep, condition, 'edges', N, 'pairs', n_pairs, 'graphs', n_graph, 'PDBs', n_pdb, 'sum r', point[0, 0], flush=True)
    return (point, draws, gids, yy)
for rep in ['node', 'pair']:
    for cond, path in [('C0', origbase / 'final' / f'{rep}_predictions.npz'), ('C1', ROOT / 'results/c1_original_heads_20261107' / f'{rep}_predictions.npz')]:
        data = loadnp(path)
        evaluate('original_head', rep, cond, orig, data['prediction'], original_indices['indices'])
    ref = None
    for cond, path in [('GEO', base / 'R3' / f'{rep}_raw_probe/test_predictions.npz'), ('C0', base / 'R1' / f'{rep}_raw_probe/test_predictions.npz'), ('C1', ROOT / 'results/pieda_context_v2_20261103/model1_v6/pieda/model1_v6/R1' / f'{rep}_raw_probe/test_predictions.npz')]:
        data = loadnp(path)
        for key in ['y', 'graph_ids', 'source']:
            np.testing.assert_array_equal(data[key], probe[key])
        au = json.loads((path.parent / 'test_probe_audit.json').read_text())
        assert au['selection_sha256'] == probe['selection_sha256']
        assert au['raw_targets_sha256'] == hashlib.sha256(probe['y'].tobytes()).hexdigest()
        result = evaluate('frozen_probe', rep, cond, probe, data['prediction'], probe_indices['indices'], ref)
        if cond == 'GEO':
            ref = result
pd.DataFrame(points).to_csv(OUT / 'point_metrics.csv', index=False)
pd.DataFrame(intervals).to_csv(OUT / 'confidence_intervals.csv', index=False)
pd.DataFrame(counts).to_csv(OUT / 'evaluation_counts.csv', index=False)
sha(Path(__file__))
(OUT / 'COMPLETE.json').write_text(json.dumps(dict(complete=True, partition=os.environ.get('SLURM_JOB_PARTITION', 'standalone'), job_id=os.environ.get('SLURM_JOB_ID'), series=['original_head', 'frozen_probe'], condition_scope='C0/C1 original heads; GEO/C0/C1 final frozen probes', edge_group='standard-AA to FMO target-ligand, excludes water and other fragments', original_selection='historical 15% masked pairs, both directions', probe_selection='production probe_selection; all ligand-involving edges retained; nonligand cap unchanged', probe_row_selection_sha256=probe['selection_sha256'], all_conditions_exact_target_and_row_alignment=True, bootstrap='10000 paired PDB-cluster resamples, same source indices as original series; all 874 Test PDBs sampled, empty subgroup clusters carry zero edges', no_training=True, no_inference=True, no_checkpoint_changes=True, input_sha256=hashes), indent=2))
print('COMPLETE', OUT, flush=True)
