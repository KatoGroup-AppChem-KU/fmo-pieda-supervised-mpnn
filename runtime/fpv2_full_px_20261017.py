"""Frozen FPv2 final encoder -> affinity MLP head CV/refit; unchanged fit routine."""
import argparse, hashlib, json, os, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import fpv2_sg_b_20261016 as fp
from run_px_headcv_20261012 import fit, sha, write
from bootstrap_px_selected_20261012 import metrics
ROOT = Path(__import__('os').environ['FMO_WORKDIR'])
LABEL = ROOT / 'results/affinity_v2_adopted_7579/affinity_labels_v2_adopted.csv'
KINDS = ['node', 'pair', 'concat']

def run(a, out):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available()
    condition = ['geo', 'geo_pieda'][a.task]
    number = int(os.environ['FPV2_MODEL'])
    version = os.environ['FPV2_VERSION']
    dest = out / condition
    dest.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, dest / Path(__file__).name)
    df = pd.read_csv(LABEL).sort_values('pdb_id').reset_index(drop=True)
    assert len(df) == 3135 and df.pdb_id.is_unique and (df.status == 'accepted').all()
    df['pdbid'] = df.pdb_id
    df['split_pdbid'] = df.split
    test = df.split.eq('test')
    assert test.sum() == 309 and (~test).sum() == 2826
    manifest = json.loads((fp.PREP / 'final/manifest.json').read_text())
    records = {r['pdb']: r for r in manifest['records'] if r['source'] == 'bdb'}
    for row in df.itertuples():
        assert records[row.pdbid]['subset'] == row.subset and records[row.pdbid]['split'] == row.split
    for k in range(1, 6):
        mf = json.loads((fp.PREP / f'cv/fold{k}.json').read_text())
        rec = {r['pdb']: r for r in mf['records'] if r['source'] == 'bdb'}
        df[f'fold{k}'] = np.where(test, 'test', np.where(df.subset.eq(f'subset{k}'), 'valid', 'train'))
        assert all((rec[r.pdbid]['split'] == getattr(r, f'fold{k}') for r in df.itertuples()))
    if a.smoke:
        df = pd.concat([df[df.subset == s].head(12) for s in df.subset.unique()]).sort_values('pdbid').reset_index(drop=True)
    df.to_csv(dest / 'cohort_and_folds.csv', index=False)
    hashes = {str(p): sha(p) for p in [LABEL, Path(__file__), Path(fit.__code__.co_filename), fp.PREP / 'final/manifest.json']}
    ckpath = fp.RUNS / 'final' / condition / 'epoch_300.pt'
    done = json.loads((ckpath.parent / 'COMPLETE.json').read_text())
    assert done['complete'] and done['checkpoint_sha256'] == sha(ckpath)
    hashes[str(ckpath)] = sha(ckpath)
    pre = json.loads((ckpath.parent / 'train_manifest.json').read_text())
    assert not set(df.loc[df.split == 'test', 'pdbid']) & {r['pdb'] for r in pre['records'] if r['split'] == 'train'}
    ck = fp.base.load(ckpath)
    assert ck['epoch'] == 300 and ck['config']['train_graphs'] == 8085
    model = fp.make_model(condition)
    model.load_state_dict(ck['model_state'], strict=True)
    del ck
    model.eval().requires_grad_(False)
    before = fp.base.digest_state(model.state_dict())
    vectors = {k: [] for k in KINDS}
    ih = hashlib.sha256()
    with torch.inference_mode():
        for i, row in enumerate(df.itertuples(), 1):
            rec = records[row.pdbid]
            g = fp.base.load(rec['paths'][version])
            assert fp.base.fingerprint(g) == rec['fingerprints'][version]
            geom, _, fmo, names = fp.base.pm.infer_feature_groups(g)
            assert (len(geom), len(fmo), len(names)) == (66, 6, 72)
            ea = g.edge_attr.clone().float()
            ea[:, fmo] = 0
            ei = g.edge_index.long()
            lig = g.is_ligand.bool().reshape(-1)
            assert int(lig.sum()) == 1
            selected = lig.clone()
            selected[ei[:, lig[ei[0]] | lig[ei[1]]].reshape(-1)] = True
            edges = torch.where(lig[ei[0]] & ~lig[ei[1]])[0]
            neighbors = torch.where(selected & ~lig)[0]
            assert len(edges) > 0 and torch.equal(torch.sort(ei[1, edges]).values, neighbors)
            ih.update(row.pdbid.encode())
            for t in [g.x, ei, ea, selected, edges]:
                ih.update(t.numpy().tobytes())
            h = model.encode_nodes(g.x.float().cuda(), ei.cuda(), ea.cuda())
            node = h[selected.cuda()].mean(0).cpu()
            pair = model.pair_state[edges.cuda()].mean(0).cpu()
            assert node.shape == (256,) and pair.shape == (256 if number == 1 else 512,)
            for k, v in [('node', node), ('pair', pair), ('concat', torch.cat([node, pair]))]:
                assert torch.isfinite(v).all()
                vectors[k].append(v)
            if i % 250 == 0:
                print(f'{condition}: extracted {i}/{len(df)}', flush=True)
    assert before == fp.base.digest_state(model.state_dict())
    assert all((not p.requires_grad and p.grad is None for p in model.parameters()))
    del model
    torch.cuda.empty_cache()
    config = dict(condition=condition, smoke=a.smoke, seed=42, hidden_dim=256, layers=2, dropout=0.2, batch_size=64, optimizer='AdamW', lr=0.001, weight_decay=0.001, max_epochs=500, patience=50, pooling='node: ligand and 1-hop neighbors mean; pair: ligand-touch outgoing edges mean; concat: both', inputs='all geometry retained; all FMO/q zero', encoder=f'fixed FPv2 model{number}/{version} final epoch300', cv='head-only; existing five subsets; final epochs median validation-best; Test not used for selection')
    write(dest / 'config.json', config)
    rows = []
    for kind in KINDS:
        x = torch.stack(vectors[kind])
        torch.save(dict(embeddings=x, pdbids=df.pdbid.tolist()), dest / (kind + '.pt'))
        mapping = dict(zip(df.pdbid, x.numpy()))
        epochs = []
        for k in range(1, 6):
            part = df[df.split != 'test'].copy()
            part['split_pdbid'] = part[f'fold{k}']
            result, pred = fit(part, mapping, x.shape[1], dest / kind / f'fold{k}', epochs=2 if a.smoke else 500)
            epochs.append(result['best_epoch'])
            rows.append(dict(condition=condition, representation=kind, fold=k, best_epoch=result['best_epoch'], **result['valid']))
            pd.DataFrame(rows).to_csv(dest / 'cv_metrics.csv', index=False)
        final_epochs = int(np.median(epochs))
        part = df.copy()
        part['split_pdbid'] = part.split
        fit(part, mapping, x.shape[1], dest / kind / 'alltrain', epochs=final_epochs, validation=False)
        write(dest / kind / 'epoch_selection.json', dict(fold_best_epochs=epochs, selected_epoch=final_epochs, test_used=False))
    for p, h in hashes.items():
        assert sha(p) == h
    write(dest / 'COMPLETE.json', dict(complete=True, smoke=a.smoke, source_hashes=hashes, input_sha256=ih.hexdigest(), config=config))

def aggregate(out):
    done = [json.loads((out / c / 'COMPLETE.json').read_text()) for c in ['geo', 'geo_pieda']]
    assert all((d['complete'] and (not d['smoke']) for d in done))
    assert done[0]['input_sha256'] == done[1]['input_sha256']
    frames = [pd.read_csv(out / c / 'cohort_and_folds.csv') for c in ['geo', 'geo_pieda']]
    pd.testing.assert_frame_equal(*frames)
    cv = pd.concat([pd.read_csv(out / c / 'cv_metrics.csv') for c in ['geo', 'geo_pieda']], ignore_index=True)
    assert len(cv) == 30
    cv.to_csv(out / 'cv_metrics.csv', index=False)
    names = ['pearson_r', 'spearman_r', 'rmse', 'mae']
    cv.groupby(['condition', 'representation'])[names].agg(['mean', 'std']).to_csv(out / 'cv_mean_sd.csv')
    idx = np.random.default_rng(20261012).integers(309, size=(10000, 309), dtype=np.int32)
    rows = []
    bytype = []
    improvements = []
    for kind in KINDS:
        pts = {}
        draws = {}
        ref = None
        for c in ['geo', 'geo_pieda']:
            df = pd.read_csv(out / c / kind / 'alltrain/predictions.csv')
            df = df[df.split_pdbid == 'test'].sort_values('pdbid').reset_index(drop=True)
            assert len(df) == 309 and df.pdbid.is_unique
            if ref is None:
                ref = df
            else:
                assert df.pdbid.tolist() == ref.pdbid.tolist()
                np.testing.assert_array_equal(df.target_px, ref.target_px)
            y = df.target_px.to_numpy()
            p = df.prediction.to_numpy()
            pts[c] = metrics(y, p)
            draws[c] = np.concatenate([metrics(y[j], p[j]) for j in np.array_split(idx, 40)])
            for t, z in df.groupby('affinity_type'):
                bytype.append(dict(condition=c, representation=kind, indicator=t, n=len(z), **dict(zip(names, metrics(z.target_px.to_numpy(), z.prediction.to_numpy())))))
        np.savez_compressed(out / 'bootstrap_indices.npz', indices=idx, pdbids=ref.pdbid.to_numpy(dtype=str))
        for label, point, draw in [('GEO', pts['geo'], draws['geo']), ('GEO+PIEDA', pts['geo_pieda'], draws['geo_pieda']), ('difference', pts['geo_pieda'] - pts['geo'], draws['geo_pieda'] - draws['geo'])]:
            assert np.isfinite(draw).all()
            lo, hi = np.quantile(draw, [0.025, 0.975], axis=0)
            for k, n in enumerate(names):
                rows.append(dict(representation=kind, condition=label, metric=n, estimate=point[k], ci_low=lo[k], ci_high=hi[k]))
        z = cv[cv.representation == kind].pivot(index='fold', columns='condition', values='pearson_r')
        improvements.append(dict(representation=kind, improved_folds=int((z.geo_pieda > z.geo).sum()), folds=5))
    pd.DataFrame(rows).to_csv(out / 'confidence_intervals.csv', index=False)
    pd.DataFrame(bytype).to_csv(out / 'test_by_indicator.csv', index=False)
    pd.DataFrame(improvements).to_csv(out / 'cv_improvement_counts.csv', index=False)
    write(out / 'COMPLETE.json', dict(complete=True, head_cv=True, encoder_cv=False, n_train=2826, n_test=309, bootstrap=10000, conditions=done))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', type=int, choices=[0, 1])
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--aggregate', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    assert True and Path.cwd() == ROOT
    out = Path(a.out)
    if a.aggregate:
        aggregate(out)
    else:
        run(a, out)
