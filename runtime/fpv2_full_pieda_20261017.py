"""Frozen final encoders: fivefold probe selection, development refit, paired CI."""
import argparse, copy, gc, json, os, shutil, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import run_cv5_20260925 as cv
import cv5_loops_20260925 as loops
import geopair_cfg_20261006 as gp
import fpv2_sg_b_20261016 as fp
from final_pieda_ci_20261012 import bootstrap, write
TARGETS = ['PIEDA_sum', 'ES', 'EX', 'CTmix', 'DI']

def predict(model, x, ym, ys):
    model.eval()
    p = torch.empty((len(x), 5))
    with torch.no_grad():
        for i in range(0, len(x), 4096):
            p[i:i + 4096] = model(x[i:i + 4096].cuda()).cpu() * ys + ym
    assert torch.isfinite(p).all()
    return p

def metrics(y, p):
    ans = loops.metrics(y.numpy(), p.numpy())
    for j, name in enumerate(TARGETS):
        ans[name]['rmse'] = float((y[:, j] - p[:, j]).square().mean().sqrt())
    return ans

def fit(factory, train, valid, out, epochs=80, fixed=False):
    out.mkdir(parents=True, exist_ok=True)
    x, y = train[:2]
    xm = x.mean(0)
    xs = x.std(0, unbiased=False).clamp_min(1e-08)
    ym = y.mean(0)
    ys = y.std(0, unbiased=False).clamp_min(1e-08)
    x = (x - xm) / xs
    yn = (y - ym) / ys
    vx = (valid[0] - xm) / xs if valid is not None else None
    torch.manual_seed(42)
    m = factory().cuda()
    initial = loops.digest_state(m.state_dict())
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    best = float('inf')
    best_ep = 0
    state = None
    stale = 0
    for ep in range(1, epochs + 1):
        begin = time.monotonic()
        m.train()
        sums = torch.zeros(5, dtype=torch.float64)
        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(loops.seed_for('probe_order', 42, ep)))
        for i in range(0, len(x), 4096):
            ids = order[i:i + 4096]
            opt.zero_grad(set_to_none=True)
            losses = (m(x[ids].cuda()) - yn[ids].cuda()).square().mean(0)
            loss = losses.mean()
            assert torch.isfinite(loss)
            loss.backward()
            opt.step()
            sums += losses.detach().cpu().double() * len(ids)
        v = None
        if valid is not None:
            pred = predict(m, vx, ym, ys)
            v = ((pred - valid[1]) / ys).square().mean(0)
            score = float(v.mean())
            if score < best:
                best = score
                best_ep = ep
                state = copy.deepcopy(m.state_dict())
                stale = 0
            else:
                stale += 1
        else:
            best_ep = ep
        train_loss = (sums / len(x)).tolist()
        loops.json_line(out / 'training.jsonl', dict(epoch=ep, train_components=train_loss, valid_components=None if v is None else v.tolist()))
        fmt = lambda a: ' '.join((f'{k}={float(t):.6f}' for k, t in zip(TARGETS, a)))
        print(f'Epoch {ep}/{epochs} {time.monotonic() - begin:.1f}s Train {fmt(train_loss)}' + (' | Valid ' + fmt(v) if v is not None else ' | refit: no validation'), flush=True)
        if not fixed and stale >= 10:
            break
    if state is not None:
        m.load_state_dict(state)
    torch.save(dict(model_state=m.state_dict(), x_mean=xm, x_std=xs, y_mean=ym, y_std=ys, best_epoch=best_ep, initial_sha256=initial), out / 'probe.pt')
    result = dict(best_epoch=best_ep, epochs_run=ep, best_valid_mse=None if fixed else best, initial_sha256=initial)
    if valid is not None:
        p = predict(m, vx, ym, ys)
        result['validation'] = metrics(valid[1], p)
        np.savez_compressed(out / 'validation_predictions.npz', y=valid[1].numpy(), prediction=p.numpy(), source=valid[2], graph_ids=valid[3])
    write(out / 'fit.json', result)
    return (m, (xm, xs, ym, ys), result)

def save_test(m, scale, data, out):
    xm, xs, ym, ys = scale
    x, y, s, gids = data
    p = predict(m, (x - xm) / xs, ym, ys)
    np.savez_compressed(out / 'test_predictions.npz', y=y.numpy(), prediction=p.numpy(), source=s, graph_ids=gids)
    result = metrics(y, p)
    write(out / 'test_metrics.json', result)
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', type=int, required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    assert True and torch.cuda.is_available()
    assert Path.cwd() == Path(__import__('os').environ['FMO_WORKDIR']) and 0 <= a.task < 2
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    number, version = (int(os.environ['FPV2_MODEL']), os.environ['FPV2_VERSION'])
    condition = ['geo', 'geo_pieda'][a.task]
    cv.PREP = fp.PREP / 'cv'
    label = ('SG' if number == 1 else 'CG') + '/' + ('ESM-FP' if version == 'v6' else 'Unified-FP')
    out = Path(os.environ['PIEDA_HEADCV_ROOT']) / f'model{number}_{version}'
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, out / (condition + '_' + Path(__file__).name))
    manifests = [json.loads((cv.PREP / f'fold{k}.json').read_text()) for k in range(1, 6)]
    prep = json.loads((cv.PREP / 'PREPARED.json').read_text())
    hashes = {}
    final_path = fp.PREP / 'final/manifest.json'
    final = json.loads(final_path.read_text())
    final_prep = json.loads((fp.PREP / 'final/PREPARED.json').read_text())
    assert cv.sha(final_path) == final_prep['files_sha256']['manifest.json']
    hashes[str(final_path)] = cv.sha(final_path)
    for k, mf in enumerate(manifests, 1):
        path = cv.PREP / f'fold{k}.json'
        hashes[str(path)] = cv.sha(path)
        assert hashes[str(path)] == prep['files_sha256'][path.name]
        write(out / (condition + '_' + path.name), mf)
    key = lambda r: (r['source'], r['key'])
    records = manifests[0]['records']
    test_records = [r for r in records if r['split'] == 'test']
    development = [r for r in records if r['split'] != 'test']
    assert len(test_records) == 898 and len(development) == 8085
    test_keys = [key(r) for r in test_records]
    all_keys = {key(r) for r in records}
    assert len(all_keys) == len(records)
    valid_sets = []
    for mf in manifests:
        assert [key(r) for r in mf['records'] if r['split'] == 'test'] == test_keys
        assert {key(r) for r in mf['records']} == all_keys
        valid_sets.append({key(r) for r in mf['records'] if r['split'] == 'valid'})
    assert sum(map(len, valid_sets)) == 8085 and len(set.union(*valid_sets)) == 8085
    for mf in manifests:
        sets = [{r['pdb'] for r in mf['records'] if r['split'] == sp} for sp in ['train', 'valid', 'test']]
        assert not sets[0] & sets[1] and (not sets[0] & sets[2]) and (not sets[1] & sets[2])
    config = dict(model=label, encoder='frozen all8085 epoch300', seed=42, folds=5, epochs=80, patience=10, optimizer='Adam', lr=0.001, batch_size=4096, train_edge_cap=800000, valid_edge_cap=150000, final_epoch='median five validation-best epochs separately for each condition/readout', final_training='all8085 structures as candidate pool; existing train edge cap retained', caveat='Head-only CV, encoder fixed and saw development structures. Validation used for early stopping, not nested CV. Test never used for selection.', smoke=a.smoke)
    config.update(condition=condition, normalization='fixed final encoder training statistics', inputs='unmasked geometry, all PIEDA/q zero', series='FPv2 C full series')
    write(out / (condition + '_config.json'), config)
    sources = fp.source_hashes()
    hashes[__file__] = cv.sha(__file__)
    hashes[fit.__code__.co_filename] = cv.sha(fit.__code__.co_filename)
    needed = records
    if a.smoke:
        chosen = set()
        for mf in manifests:
            for sp in ['train', 'valid', 'test']:
                chosen.update((key(r) for r in [r for r in mf['records'] if r['split'] == sp][:4]))
        needed = [r for r in records if key(r) in chosen]
    graph_map = {}
    for i, r in enumerate(needed, 1):
        graph_map[key(r)] = cv.load(r, final, version)
        if i % 1000 == 0:
            print(f'preload {i}/{len(needed)}', flush=True)
    select = lambda mf, sp: [r for r in mf['records'] if r['split'] == sp][:4] if a.smoke else [r for r in mf['records'] if r['split'] == sp]
    audits = {}
    rows = []
    for cond in [['R3', 'R1'][a.task]]:
        ckpath = fp.RUNS / 'final' / condition / 'epoch_300.pt'
        hashes[str(ckpath)] = cv.sha(ckpath)
        done = json.loads((ckpath.parent / 'COMPLETE.json').read_text())
        assert done['complete'] and done['checkpoint_sha256'] == hashes[str(ckpath)]
        pre = json.loads((ckpath.parent / 'train_manifest.json').read_text())
        assert not {r['pdb'] for r in test_records} & {r['pdb'] for r in pre['records'] if r['split'] == 'train'}
        ck = cv.base.load(ckpath)
        assert ck['epoch'] == 300 and ck['config']['train_graphs'] == 8085
        assert done['source_hashes'] == fp.source_hashes()
        assert ck['config']['manifest_sha256'] == cv.sha(final_path)
        model = fp.make_model(condition)
        model.load_state_dict(ck['model_state'], strict=True)
        del ck
        model.eval().requires_grad_(False)
        before = loops.digest_state(model.state_dict())
        for kind in ['node', 'pair']:
            print(f'BEGIN {label} {cond} {kind}', flush=True)
            extractor = cv.prior.ORIGINAL_EXTRACT if kind == 'node' else cv.extract_pair
            factory = cv.prior.ORIGINAL_PROBE if kind == 'node' or number == 2 else cv.prior.PairProbe

            def extract(recs, sp, dest, tag):
                data = extractor(model, [graph_map[key(r)] for r in recs], sp, dest)
                audit = json.loads((dest / f'{sp}_probe_audit.json').read_text())
                if tag in audits:
                    assert audit == audits[tag]
                else:
                    audits[tag] = audit
                return data
            dest = out / cond / (kind + '_raw_probe')
            dest.mkdir(parents=True)
            test = extract(select(manifests[0], 'test'), 'test', dest, 'test')
            epochs = []
            for fold, mf in enumerate(manifests, 1):
                fd = out / 'cv' / cond / kind / f'fold{fold}'
                fd.mkdir(parents=True)
                train = extract(select(mf, 'train'), 'train', fd, f'{fold}/train')
                valid = extract(select(mf, 'valid'), 'valid', fd, f'{fold}/valid')
                m, scale, result = fit(factory, train, valid, fd, epochs=2 if a.smoke else 80)
                epochs.append(result['best_epoch'])
                tm = save_test(m, scale, test, fd)
                for split, ms in [('validation', result['validation']), ('test', tm)]:
                    for target, vals in ms.items():
                        rows.append(dict(model=label, condition=cond, representation=kind, fold=fold, split=split, target=target, best_epoch=result['best_epoch'], **vals))
                pd.DataFrame(rows).to_csv(out / (condition + '_cv_metrics.csv'), index=False)
                del train, valid, m, scale
                gc.collect()
                torch.cuda.empty_cache()
            chosen = int(np.median(epochs))
            assert 1 <= chosen <= 80
            dev = development if not a.smoke else needed
            if a.smoke:
                dev = [r for r in dev if r['split'] != 'test']
            train = extract(dev, 'train', dest, 'final/train')
            write(dest / 'epoch_selection.json', dict(fold_best_epochs=epochs, selected_epoch=chosen, test_used=False))
            m, scale, result = fit(factory, train, None, dest, epochs=chosen, fixed=True)
            save_test(m, scale, test, dest)
            assert loops.digest_state(model.state_dict()) == before
            del train, test, m, scale
            gc.collect()
            torch.cuda.empty_cache()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.groupby(['model', 'condition', 'representation', 'split', 'target'])[['pearson_r', 'rmse', 'mae', 'r2']].agg(['mean', 'std']).to_csv(out / (condition + '_cv_mean_sd.csv'))
    write(out / (condition + '_test_records.json'), test_records)
    for path, h in hashes.items():
        assert cv.sha(path) == h
    assert sources == fp.source_hashes()
    write(out / (condition + '_COMPLETE.json'), dict(complete=True, config=config, hashes=hashes, sources=sources))
    print('COMPLETE', out, flush=True)
if __name__ == '__main__':
    main()
