"""Frozen SG/v6: existing five subsets -> head CV -> all-development refit."""
import os, json, time, hashlib, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import predict_pic50_mlp_probe as probe
from bootstrap_px_selected_20261012 import metrics as boot_metrics

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, obj):
    p.write_text(json.dumps(obj, indent=2))

def fit(df, mapping, dim, dest, epochs=500, validation=True):
    """Reuse original model, dataset, scaler, optimizer and minibatch settings."""
    dest.mkdir(parents=True)
    probe.seed_all(42)
    train = df[df.split_pdbid == 'train'].sort_values('pdbid')
    std = probe.Standardizer(np.stack([mapping[p] for p in train.pdbid]))
    loaders = {}
    for sp in ['train', 'valid'] if validation else ['train']:
        part = df[df.split_pdbid == sp].sort_values('pdbid')
        rows = [dict(pdbid=r.pdbid, pic50=float(r.target_px)) for r in part.itertuples()]
        ds = probe.GraphProbeDataset(rows, mapping, 'pretrained', dim, {}, std)
        loaders[sp] = DataLoader(ds, batch_size=64, shuffle=sp == 'train', num_workers=2, pin_memory=True)
    model = probe.GraphMLPProbe(dim, 256, 0.2, 2).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    best, best_epoch, history = (float('inf'), 0, [])
    start = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for x, y in loaders['train']:
            x, y = (x.cuda(), y.cuda())
            pred = model(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(y)
            count += len(y)
        row = dict(epoch=epoch, train_loss=total / count)
        if validation:
            va = probe.eval_graph_probe(model, loaders['valid'], torch.device('cuda'))
            row['valid'] = va
            if va['loss'] < best:
                best, best_epoch = (va['loss'], epoch)
                torch.save(dict(model=model.state_dict()), dest / 'best_model.pt')
        else:
            best_epoch = epoch
        history.append(row)
        if epoch == 1 or epoch % 10 == 0:
            extra = f" | valid loss {va['loss']:.4f} r {va['pearson_r']:.4f} | best@{best_epoch}" if validation else ''
            print(f'{dest.name} epoch {epoch:03d}/{epochs} | train loss {total / count:.4f}{extra} | elapsed {time.monotonic() - start:.1f}s', flush=True)
        if validation and epoch - best_epoch >= 50:
            break
    if validation:
        model.load_state_dict(torch.load(dest / 'best_model.pt', map_location='cuda', weights_only=False)['model'])
    else:
        torch.save(dict(model=model.state_dict()), dest / 'final_model.pt')
    write(dest / 'history.json', history)
    np.savez(dest / 'scaler.npz', mean=std.mean, std=std.std)
    model.eval()
    x = std.transform(np.stack([mapping[p] for p in df.pdbid]))
    with torch.inference_mode():
        pred = np.concatenate([model(torch.from_numpy(a).cuda()).cpu().numpy() for a in np.array_split(x, max(1, (len(x) + 63) // 64))])
    predictions = df[['pdbid', 'split_pdbid', 'affinity_type', 'target_px']].copy()
    predictions['prediction'] = pred
    predictions.to_csv(dest / 'predictions.csv', index=False)
    result = {sp: probe.regression_metrics(g.target_px, g.prediction) for sp, g in predictions.groupby('split_pdbid')}
    result['best_epoch'] = best_epoch
    write(dest / 'metrics.json', result)
    print(f'DONE {dest}: {time.monotonic() - start:.1f}s', flush=True)
    return (result, predictions)

def main():
    assert True and torch.cuda.is_available()
    assert Path.cwd() == Path(__import__('os').environ['FMO_WORKDIR'])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    assert sha(probe.__file__) == 'bea288b0b088affda8c970ad66e40dcbdec68ae65c008c857e66dabbfaef4afe'
    src = Path(os.environ.get('PX_CACHE_OUT', 'results/frozen_px_full_7296'))
    assert json.loads((src / 'COMPLETE.json').read_text())['complete']
    out = Path(os.environ.get('PX_CV_OUT', 'results/px_headcv_' + os.environ.get('SLURM_JOB_ID', 'standalone')))
    out.mkdir()
    shutil.copy2(__file__, out / Path(__file__).name)
    df = pd.read_csv(src / 'cohort.csv')
    assert len(df) == 3397 and df.pdbid.is_unique
    test = df.split_pdbid.eq('test')
    assert test.sum() == 341 and (~test).sum() == 3056
    assert set(df.loc[~test, 'subset']) == {f'subset{i}' for i in range(1, 6)}
    for i in range(1, 6):
        manifest = json.loads(Path(f'dataset/fixedtest_cv5_20260925_prepared/fold{i}.json').read_text())
        rec = {r['pdb']: r for r in manifest['records'] if r['source'] == 'bdb'}
        for row in df.itertuples():
            expected = 'test' if row.split_pdbid == 'test' else 'valid' if row.subset == f'subset{i}' else 'train'
            assert rec[row.pdbid]['split'] == expected
        df[f'fold{i}'] = np.where(test, 'test', np.where(df.subset.eq(f'subset{i}'), 'valid', 'train'))
    df.to_csv(out / 'cohort_and_folds.csv', index=False)
    config = dict(model=os.environ.get('EVAL_MODEL_LABEL', 'SG/ESM-FP'), conditions=['R3', 'R1'], representations=['node', 'pair', 'concat'], seed=42, hidden_dim=256, layers=2, dropout=0.2, batch_size=64, num_workers=2, lr=0.001, weight_decay=0.001, max_epochs=500, patience=50, final_epoch_rule='median of five validation-best epochs, separately per condition/representation', fixed_test_n=341, development_n=3056, split='existing subset1-5; no reassignment', caveat='CV validation selects early stopping and is not nested CV. Frozen encoder saw development structures, not pX labels. CI conditional on fitted models; PDB bootstrap does not account for shared proteins.')
    write(out / 'config.json', config)
    hashes = {str(p): sha(p) for p in [src / 'cohort.csv', Path(probe.__file__), Path(__file__)]}
    rows = []
    final = {}
    cv_predictions = []
    for cond in ['R3', 'R1']:
        for kind in ['node', 'pair', 'concat']:
            cache = src / cond / (kind + '.pt')
            hashes[str(cache)] = sha(cache)
            obj = torch.load(cache, map_location='cpu', weights_only=False)
            assert obj['pdbids'] == df.pdbid.tolist()
            x = obj['embeddings'].numpy()
            assert np.isfinite(x).all()
            mapping = dict(zip(df.pdbid, x))
            best_epochs = []
            for fold in range(1, 6):
                part = df.loc[~test].copy()
                part['split_pdbid'] = part[f'fold{fold}']
                result, pred = fit(part, mapping, x.shape[1], out / cond / kind / f'fold{fold}')
                best_epochs.append(result['best_epoch'])
                rows.append(dict(condition=cond, representation=kind, fold=fold, best_epoch=result['best_epoch'], **result['valid']))
                pp = pred[pred.split_pdbid == 'valid'].copy()
                pp['condition'] = cond
                pp['representation'] = kind
                pp['fold'] = fold
                cv_predictions.append(pp)
                pd.DataFrame(rows).to_csv(out / 'cv_metrics.csv', index=False)
            epochs = int(np.median(best_epochs))
            part = df.copy()
            part['split_pdbid'] = np.where(test, 'test', 'train')
            result, pred = fit(part, mapping, x.shape[1], out / cond / kind / 'alltrain', epochs=epochs, validation=False)
            final[cond, kind] = pred[pred.split_pdbid == 'test'].sort_values('pdbid').reset_index(drop=True)
    pd.concat(cv_predictions).to_csv(out / 'cv_validation_predictions.csv', index=False)
    pd.DataFrame(rows).groupby(['condition', 'representation'])[['pearson_r', 'spearman_r', 'rmse', 'mae']].agg(['mean', 'std']).to_csv(out / 'cv_mean_sd.csv')
    idx = np.random.default_rng(20261012).integers(0, 341, size=(10000, 341), dtype=np.int32)
    np.savez_compressed(out / 'bootstrap_indices.npz', indices=idx)
    ci = []
    names = ['pearson_r', 'spearman_r', 'rmse', 'mae']
    for kind in ['node', 'pair', 'concat']:
        pts = {}
        draws = {}
        for cond in ['R3', 'R1']:
            pred = final[cond, kind]
            ref = final['R3', kind]
            assert pred.pdbid.tolist() == ref.pdbid.tolist()
            np.testing.assert_array_equal(pred.target_px, ref.target_px)
            y = pred.target_px.to_numpy()
            p = pred.prediction.to_numpy()
            pts[cond] = boot_metrics(y, p)
            draws[cond] = np.concatenate([boot_metrics(y[j], p[j]) for j in np.array_split(idx, 40)])
        for label, point, draw in [('GEO', pts['R3'], draws['R3']), ('GEO+PIEDA', pts['R1'], draws['R1']), ('difference', pts['R1'] - pts['R3'], draws['R1'] - draws['R3'])]:
            assert np.isfinite(draw).all()
            lo, hi = np.quantile(draw, [0.025, 0.975], axis=0)
            for k, n in enumerate(names):
                ci.append(dict(representation=kind, condition=label, metric=n, estimate=point[k], ci_low=lo[k], ci_high=hi[k]))
    pd.DataFrame(ci).to_csv(out / 'confidence_intervals.csv', index=False)
    for p, h in hashes.items():
        assert sha(p) == h
    write(out / 'COMPLETE.json', dict(complete=True, source_hashes=hashes, config=config, bootstrap_n=10000, confidence=0.95, bootstrap_unit='PDB', paired=True))
    print('COMPLETE: 30 CV fits + 6 all-development refits; paired PDB bootstrap 10000', flush=True)
if __name__ == '__main__':
    main()
