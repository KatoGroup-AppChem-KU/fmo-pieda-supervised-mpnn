"""SG/ESM-FP FPv2 original-head audit. Run only in aurum-g01 allocations.

CV uses saved pair-head predictions; final uses frozen epoch300 CPU inference.
No optimizer, fitting, altered masks, or new model selection.
"""
import argparse
import contextlib
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
ROOT = Path(__import__('os').environ['FMO_WORKDIR'])
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'outputs'))
import fpv2_sg_b_20261016 as fp
from final_pieda_ci_20261012 import pooled
OUT = ROOT / 'results/fpv2_original_head_audit_20261021'
CV = ROOT / 'results/fpv2_C_full_20261017/model1_v6/cv'
FINAL = ROOT / 'results/fpv2_sg_b_unitloss_20261016/final/geo_pieda'
TARGETS = ['PIEDA_sum', 'ES', 'EX', 'CTmix', 'DI']
TYPES = ['AA', 'ligand', 'water', 'other']

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False))

def flags(raw):
    names = list(raw.node_feature_names)
    fs = []
    for name in ('is_standard_aa', 'is_ligand', 'is_water'):
        x = raw.x[:, names.index(name)]
        assert ((x == 0) | (x == 1)).all(), name
        val = x.bool()
        if hasattr(raw, name):
            assert torch.equal(val, getattr(raw, name).reshape(-1).bool()), name
        fs.append(val)
    assert (sum((v.int() for v in fs)) <= 1).all()
    result = torch.full((len(raw.x),), 3, dtype=torch.long)
    for i, val in enumerate(fs):
        result[val] = i
    return result

def records_manifest(fold):
    run = CV / f'fold{fold}/geo_pieda' if fold else FINAL
    mfpath = run / 'train_manifest.json'
    mf = json.loads(mfpath.read_text())
    cfg = json.loads((run / 'config.json').read_text())
    assert cfg['model_number'] == 1 and cfg['version'] == 'v6'
    assert cfg['loss_weights'] == dict(node_distance=1, pair_distance=1, node_pieda=1, pair_pieda=1)
    assert sha(mfpath) == cfg['manifest_sha256']
    records = [r for r in mf['records'] if r['split'] == 'test']
    assert len(records) == 898 and len(mf['records']) == 8983
    assert not {r['pdb'] for r in records} & {r['pdb'] for r in mf['records'] if r['split'] != 'test'}
    return (run, mfpath, mf, cfg, records)

@contextlib.contextmanager
def cpu_factory():
    old = torch.nn.Module.cuda
    torch.nn.Module.cuda = lambda module, *a, **kw: module.cpu()
    try:
        yield
    finally:
        torch.nn.Module.cuda = old

def model_for(ckpath):
    ck = fp.base.load(ckpath)
    with cpu_factory():
        m = fp.make_model('geo_pieda')
    m.load_state_dict(ck['model_state'], strict=True)
    m.eval().requires_grad_(False)
    assert next(m.parameters()).device.type == 'cpu'
    return (m, ck['epoch'])

def extract(fold, smoke=False):
    run, mfpath, mf, cfg, records = records_manifest(fold)
    dest = OUT / ('smoke' if smoke else f'fold{fold}' if fold else 'final')
    dest.mkdir(parents=True, exist_ok=False)
    saved = None
    hashes = {str(mfpath): sha(mfpath), str(run / 'config.json'): sha(run / 'config.json')}
    if fold:
        path = run / 'original_head_test_predictions.npz'
        with np.load(path, allow_pickle=False) as z:
            saved = {k: z[k] for k in z.files}
        assert saved['keys'].tolist() == [r['key'] for r in records]
        hashes[str(path)] = sha(path)
    m = None
    if not fold or smoke:
        ckpath = run / ('best_model.pt' if fold else 'epoch_300.pt')
        hashes[str(ckpath)] = sha(ckpath)
        m, epoch = model_for(ckpath)
        if not fold:
            assert epoch == 300 and cfg['train_graphs'] == 8085
        else:
            selected = json.loads((run / 'selected_checkpoint.json').read_text())
            assert selected['sha256'] == sha(ckpath) and selected['epoch'] == epoch
    begin = time.monotonic()
    mean, std = (torch.tensor(mf['train_stats'][k]) for k in ('mean', 'std'))
    parts = {k: [] for k in ('y', 'graph_ids', 'kind', 'source', 'pair', 'distance', 'selected_edge')}
    preds = {'pair': [], 'node': []} if m is not None else {}
    mh = hashlib.sha256()
    cursor = 0
    offset = 0
    used_records = records[:8] if smoke else records
    for start in range(0, len(used_records), 8):
        batch_records = used_records[start:start + 8]
        gs, bs = ([], [])
        for gi, r in enumerate(batch_records, start):
            g = fp.cv.load(r, mf, 'v6')
            raw = fp.base.load(r['paths']['v6'])
            typ = flags(raw)
            b = fp.prior.masked(g, 'test12345', mh)
            sel = b.decode_mask.nonzero().flatten()
            assert torch.count_nonzero(b.edge_attr[:, g['fmo']]) == 0
            assert torch.count_nonzero(b.edge_attr[sel][:, g['geom']]) == 0
            pair_counts = torch.bincount(g['inverse'][sel], minlength=g['pairs'])
            assert ((pair_counts == 0) | (pair_counts == 2)).all()
            assert int((pair_counts == 2).sum()) == max(1, int(np.ceil(0.15 * g['pairs'])))
            u, v = g['ei'][:, sel]
            a, c = (typ[u], typ[v])
            y = b.y[sel, :7] * std + mean
            if saved is not None:
                n = len(sel)
                np.testing.assert_array_equal(saved['y'][cursor:cursor + n], y.numpy())
                np.testing.assert_array_equal(saved['graph_ids'][cursor:cursor + n], np.full(n, gi))
                cursor += n
            parts['y'].append(y.numpy())
            parts['graph_ids'].append(np.full(len(sel), gi, np.int32))
            parts['kind'].append((torch.minimum(a, c) * 4 + torch.maximum(a, c)).numpy().astype(np.int8))
            parts['source'].append(np.full(len(sel), int(r['source'] == 'scop2'), np.int8))
            parts['pair'].append(g['inverse'][sel].numpy() + offset)
            parts['distance'].append(raw.edge_attr[sel, raw.edge_attr_names.index('DIST_raw')].numpy())
            parts['selected_edge'].append(sel.numpy())
            offset += g['pairs']
            gs.append(g)
            bs.append(b)
        if m is not None:
            b = fp.base.Batch.from_data_list(bs)
            with torch.inference_mode():
                h = m.encode_nodes(b.x, b.edge_index, b.edge_attr)
                sel = b.decode_mask.nonzero().flatten()
                u, v = b.edge_index[:, sel]
                outputs = dict(pair=m.decode_edges(h, b.edge_index, None, sel), node=m.node_pieda_decoder(torch.cat([h[u], h[v]], -1)))
                for name, z in outputs.items():
                    preds[name].append((z * std[2:] + mean[2:]).numpy())
            if smoke and saved is not None:
                p = preds['pair'][-1]
                expected = saved['prediction'][:len(p), 2:]
                np.testing.assert_allclose(p, expected, atol=0.0003, rtol=3e-05)
                print('CPU/saved-GPU max abs difference', np.max(abs(p - expected)), flush=True)
        if start // 8 % 10 == 0 or start + 8 >= len(used_records):
            print(f'fold={fold} {start + len(batch_records)}/{len(used_records)} graphs, seconds={time.monotonic() - begin:.1f}', flush=True)
    meta = {k: np.concatenate(v) for k, v in parts.items()}
    if saved is not None:
        assert cursor == (len(meta['y']) if smoke else len(saved['y']))
        if not smoke:
            preds = {'pair': [saved['prediction'][:, 2:]]}
    meta['keys'] = np.array([r['key'] for r in used_records])
    meta['pdbids'] = np.array([r['pdb'] for r in used_records])
    for name, ps in preds.items():
        prediction = np.concatenate(ps)
        assert prediction.shape == meta['y'][:, 2:].shape and np.isfinite(prediction).all()
        np.savez_compressed(dest / f'{name}_predictions.npz', prediction=prediction)
    np.savez_compressed(dest / 'metadata.npz', **meta)
    if saved is not None:
        reference = json.loads((run / 'original_head_test.json').read_text())
        p = np.concatenate(preds['pair'])
        if not smoke:
            for j, target in enumerate(TARGETS):
                met = metric(meta['y'][:, j + 2].astype(float), p[:, j].astype(float))
                for key in ('pearson_r', 'mae', 'r2'):
                    assert np.isclose(met[key], reference['raw'][target][key], atol=2e-07, rtol=2e-06)
    for path, h in hashes.items():
        assert sha(path) == h
    write(dest / 'COMPLETE.json', dict(complete=True, fold=fold, smoke=smoke, trained_head=True, input_hashes=hashes, heads=list(preds), mask_sha256=mh.hexdigest(), graphs=len(used_records), records=len(meta['y']), unique_pairs=len(np.unique(meta['pair'])), cpu_inference=m is not None, saved_prediction_exact_alignment=saved is not None, source_sha256={str(Path(__file__).resolve()): sha(__file__)}))

def metric(y, p):
    if not len(y):
        return dict(pearson_r=np.nan, spearman_r=np.nan, rmse=np.nan, mae=np.nan, r2=np.nan)
    d = p - y
    var = np.sum((y - y.mean()) ** 2)
    return dict(pearson_r=float(np.corrcoef(y, p)[0, 1]) if var > 0 and np.ptp(p) > 0 else np.nan, spearman_r=float(spearmanr(y, p).statistic) if var > 0 and np.ptp(p) > 0 else np.nan, rmse=float(np.sqrt(np.mean(d * d))), mae=float(np.mean(abs(d))), r2=float(1 - np.sum(d * d) / var) if var > 0 else np.nan)

def groups(meta):
    kind = meta['kind']
    lig = (kind // 4 == 1) | (kind % 4 == 1)
    result = [('all', np.ones(len(kind), bool)), ('ligand_involving', lig), ('non_ligand', ~lig)]
    result += [(TYPES[a] + '--' + TYPES[b], kind == a * 4 + b) for a, b in itertools.combinations_with_replacement(range(4), 2)]
    result += [(f'DIST_[{lo},{hi})_A', (meta['distance'] >= lo) & (meta['distance'] < hi)) for lo, hi in ((0, 3), (3, 4), (4, 5), (5, np.inf))]
    return result

def bootstrap(meta, prediction, out, head):
    pdbs = sorted(set(meta['pdbids'].tolist()))
    lookup = {p: i for i, p in enumerate(pdbs)}
    gi = np.array([lookup[p] for p in meta['pdbids']])[meta['graph_ids']]
    rng = np.random.default_rng(20261012)
    indices = rng.integers(len(pdbs), size=(10000, len(pdbs)), dtype=np.int32)
    np.savez_compressed(out / 'pdb_bootstrap_indices.npz', indices=indices, pdbids=np.array(pdbs))
    rows = []
    y, p = (meta['y'][:, 2:].astype(float), prediction.astype(float))
    for group, take in groups(meta):
        if group not in ('all', 'AA--ligand') or not take.any():
            continue
        stats = np.zeros((len(pdbs), 5, 8))
        for j in range(5):
            a, b = (y[take, j], p[take, j])
            for k, values in enumerate((np.ones(len(a)), a, b, a * a, b * b, a * b, (a - b) ** 2, abs(a - b))):
                stats[:, j, k] = np.bincount(gi[take], weights=values, minlength=len(pdbs))
        point = pooled(stats.sum(0))
        draws = []
        for chunk in np.array_split(indices, 100):
            counts = np.stack([np.bincount(row, minlength=len(pdbs)) for row in chunk])
            draws.append(pooled(np.einsum('bg,gck->bck', counts, stats)))
        draws = np.concatenate(draws)
        assert np.isfinite(draws).all()
        lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
        mult = np.bincount(indices[0], minlength=len(pdbs))
        ii = np.repeat(np.flatnonzero(take), mult[gi[take]])
        for j, target in enumerate(TARGETS):
            direct = metric(y[take, j], p[take, j])
            np.testing.assert_allclose(point[j, :3], [direct[n] for n in ('pearson_r', 'rmse', 'mae')], atol=1e-10)
            np.testing.assert_allclose(draws[0, j, 0], np.corrcoef(y[ii, j], p[ii, j])[0, 1], atol=1e-10)
            for k, name in enumerate(('pearson_r', 'rmse', 'mae', 'r2')):
                rows.append(dict(head=head, edge_group=group, target=target, metric=name, estimate=point[j, k], ci_low=lo[j, k], ci_high=hi[j, k], evaluation_pdbs=len(np.unique(gi[take])), bootstrap_pdbs=len(pdbs), evaluation_graphs=len(np.unique(meta['graph_ids'][take])), directed_records=int(take.sum()), unique_pairs=len(np.unique(meta['pair'][take]))))
    return rows

def summarize():
    rows, cis, audits = ([], [], [])
    selection = None
    for fold in range(6):
        path = OUT / (f'fold{fold}' if fold else 'final')
        done = json.loads((path / 'COMPLETE.json').read_text())
        assert done['complete'] and (not done['smoke'])
        with np.load(path / 'metadata.npz', allow_pickle=False) as z:
            md = {k: z[k] for k in z.files}
        comparable = {k: sha_bytes(md[k]) for k in ('keys', 'pdbids', 'graph_ids', 'kind', 'source', 'pair', 'selected_edge')}
        if selection is None:
            selection = comparable
        else:
            assert selection == comparable
        for head in done['heads']:
            with np.load(path / f'{head}_predictions.npz') as z:
                p = z['prediction'].astype(float)
            y = md['y'][:, 2:].astype(float)
            for source, code in [('all', None), ('bdb', 0), ('scop2', 1)]:
                sm = np.ones(len(y), bool) if code is None else md['source'] == code
                for group, take0 in groups(md):
                    take = sm & take0
                    count = dict(directed_records=int(take.sum()), unique_pairs=len(np.unique(md['pair'][take])), graphs=len(np.unique(md['graph_ids'][take])), pdbs=len(set(md['pdbids'][md['graph_ids'][take]].tolist())))
                    for j, target in enumerate(TARGETS):
                        rows.append(dict(fold=fold, head=head, source=source, edge_group=group, target=target, **count, **metric(y[take, j], p[take, j])))
            if not fold:
                cis += bootstrap(md, p, OUT, head)
        audits.append(done)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'all_metrics.csv', index=False)
    columns = ['head', 'source', 'edge_group', 'target']
    mets = ['pearson_r', 'spearman_r', 'rmse', 'mae', 'r2', 'graphs', 'pdbs', 'unique_pairs', 'directed_records']
    agg = df[df.fold > 0].groupby(columns)[mets].agg(['mean', 'std', 'count'])
    agg.columns = ['_'.join(c) for c in agg.columns]
    agg.reset_index().to_csv(OUT / 'cv_mean_sd.csv', index=False)
    pd.DataFrame(cis).to_csv(OUT / 'final_confidence_intervals.csv', index=False)
    write(OUT / 'COMPLETE.json', dict(complete=True, model='SG/ESM-FP', condition='GEO+PIEDA C', primary_head='pair', cv_heads=['pair'], final_heads=['pair', 'node'], evaluation='15% unordered pairs masked in both directions, all FMO/q inputs zero, pooled directed predictions', bootstrap='10000 PDB-cluster resamples over all Test PDBs; same PDB carries all its structures and selected edges', interval='percentile 2.5/97.5; fixed fitted model', units='PIEDA errors kcal/mol; correlations dimensionless', cv_sd='sample SD ddof=1 over five validation-selected encoders on the same Test', exact_selection_hashes=selection, runs=audits, script_sha256=sha(__file__)))
    main = df[(df.source == 'all') & df.edge_group.isin(['all', 'AA--ligand'])]
    main.to_csv(OUT / 'main_table_metrics.csv', index=False)
    print(main.to_string(index=False), flush=True)

def sha_bytes(a):
    return hashlib.sha256(a.tobytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, choices=range(6), default=0)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--summarize', action='store_true')
    args = ap.parse_args()
    assert True and Path.cwd() == ROOT
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    torch.set_num_interop_threads(1)
    OUT.mkdir(exist_ok=True)
    if args.summarize:
        summarize()
    else:
        extract(args.fold, args.smoke)
if __name__ == '__main__':
    main()
