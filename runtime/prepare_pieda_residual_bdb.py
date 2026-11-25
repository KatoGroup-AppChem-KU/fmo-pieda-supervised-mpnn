"""Train-only PDB-group OOF local baseline; no FMO input or graph message passing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
import screen_g_p1_bdb_20260914 as screen
base = screen.base
ROOT = screen.ROOT
DATA = screen.DATA_ROOT

def records_manifest(mode):
    raw = (DATA / 'manifest.json').read_bytes()
    manifest = json.loads(raw)
    records = [r for r in manifest['records'] if r['source'] == 'bdb']
    base.require({s: sum((r['split'] == s for r in records)) for s in ('train', 'valid', 'test')} == dict(train=2970, valid=372, test=372), 'unexpected split')
    if mode == 'smoke':
        records = sum(([r for r in records if r['split'] == s][:n] for s, n in [('train', 10), ('valid', 2), ('test', 2)]), [])
    return (records, manifest, hashlib.sha256(raw).hexdigest())

def features(g, manifest):
    names = manifest['schemas']['v6'][0]
    base.require(len(names) == 1026 and names[505:510] == ['is_standard_aa', 'is_hetero', 'is_ligand', 'is_water', 'is_other_hetero'], 'local feature schema changed')
    u, v = g['ei']
    local = g['x'][:, 480:514]
    enames = manifest['schemas']['v6'][1]
    geom = g['ea'][:, [enames.index('DIST_raw'), enames.index('centroid_distance_raw')]]
    return torch.cat([geom, local[u] + local[v], (local[u] - local[v]).abs()], 1)

def local_model():
    return nn.Sequential(nn.Linear(70, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 4)).cuda()

def fit(xs, ys, out, epochs):
    x, y = (torch.cat(xs), torch.cat(ys))
    xm, sd = (x.mean(0), x.std(0, unbiased=False))
    sd = torch.where(sd < 1e-06, torch.ones_like(sd), sd)
    ym, yd = (y.mean(0), y.std(0, unbiased=False).clamp_min(1e-08))
    x, y = (((x - xm) / sd).cuda(), ((y - ym) / yd).cuda())
    torch.manual_seed(42)
    model = local_model()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    for ep in range(1, epochs + 1):
        order = torch.randperm(len(x), device='cuda')
        total = 0.0
        for ids in order.split(4096):
            opt.zero_grad(set_to_none=True)
            loss = (model(x[ids]) - y[ids]).square().mean()
            base.require(torch.isfinite(loss), 'baseline loss nonfinite')
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(ids)
        if ep % 10 == 0 or ep == epochs:
            print(f'Baseline {out.stem} epoch={ep}/{epochs} MSE={total / len(x):.6f}', flush=True)
    state = dict(model=model.cpu().state_dict(), xm=xm, xs=sd, ym=ym, ys=yd, epochs=epochs, fit_rows=len(x))
    torch.save(state, out)
    return state

def predict(state, x):
    model = local_model()
    model.load_state_dict(state['model'])
    model.eval()
    with torch.no_grad():
        parts = [model(((a - state['xm']) / state['xs']).cuda()).cpu() * state['ys'] + state['ym'] for a in x.split(8192)]
    components = torch.cat(parts)
    return torch.cat([components.sum(1, keepdim=True), components], 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['smoke', 'full'], required=True)
    args = ap.parse_args()
    base.require(True and True, 'wrong execution site')
    os.chdir(ROOT)
    torch.set_num_threads(4)
    records, manifest, mhash = records_manifest(args.mode)
    out = ROOT / 'results' / f'pieda_residual_bdb_{args.mode}'
    out.mkdir(exist_ok=False)
    (out / 'graphs').mkdir()
    k = 2 if args.mode == 'smoke' else 5
    pdbs = sorted({r['pdb'] for r in records if r['split'] == 'train'})
    np.random.default_rng(42).shuffle(pdbs)
    folds = {p: i % k for i, p in enumerate(pdbs)}
    xs, ys = ({}, {})
    for r in records:
        if r['split'] != 'train':
            continue
        g = base.load_graph(r, 'v6', manifest)
        f = features(g, manifest)
        ids = (g['ei'][0] < g['ei'][1]).nonzero().flatten()
        gen = torch.Generator().manual_seed(base.seed_for('baseline_cap', 42, r['key']))
        ids = ids[torch.randperm(len(ids), generator=gen)[:256]]
        xs[r['key']], ys[r['key']] = (f[ids], g['raw'][ids, 1:])
    assignments = []
    for fold in list(range(k)) + [-1]:
        fit_rows = [r for r in records if r['split'] == 'train' and folds[r['pdb']] != fold]
        apply_rows = [r for r in records if r['split'] == 'train' and folds[r['pdb']] == fold or (fold == -1 and r['split'] != 'train')]
        fit_pdb = sorted({r['pdb'] for r in fit_rows})
        apply_pdb = sorted({r['pdb'] for r in apply_rows})
        base.require(not set(fit_pdb) & set(apply_pdb), 'OOF PDB leakage')
        tag = 'full_train' if fold == -1 else f'fold{fold}'
        state = fit([xs[r['key']] for r in fit_rows], [ys[r['key']] for r in fit_rows], out / (tag + '.pt'), 2 if args.mode == 'smoke' else 50)
        assignments.append(dict(model=tag, fit_pdb=fit_pdb, predict_pdb=apply_pdb))
        for r in apply_rows:
            g = base.load_graph(r, 'v6', manifest)
            pred = predict(state, features(g, manifest))
            residual = g['raw'] - pred
            base.require(torch.isfinite(residual).all(), 'residual nonfinite')
            torch.save(dict(key=r['key'], split=r['split'], pdb=r['pdb'], baseline=tag, fingerprint=r['fingerprints']['v6'], raw_sha256=hashlib.sha256(g['raw'].numpy().tobytes()).hexdigest(), residual=residual, prediction=pred), out / 'graphs' / (r['key'].replace(':', '_') + '.pt'))
        print(f'BASELINE_COMPLETE {tag} graphs={len(apply_rows)}', flush=True)
    total = torch.zeros(5, dtype=torch.float64)
    squares = total.clone()
    count = 0
    metric = {s: [[], []] for s in ('train', 'valid', 'test')}
    for r in records:
        a = torch.load(out / 'graphs' / (r['key'].replace(':', '_') + '.pt'), weights_only=False)
        v = a['residual'].double()
        if r['split'] == 'train':
            total += v.sum(0)
            squares += v.square().sum(0)
            count += len(v)
        metric[r['split']][0].append(a['prediction'] + a['residual'])
        metric[r['split']][1].append(a['prediction'])
    mean = total / count
    std = (squares / count - mean.square()).clamp_min(1e-16).sqrt()
    report = {s: base.metrics(torch.cat(a).numpy(), torch.cat(b).numpy()) for s, (a, b) in metric.items()}
    base.write_json(out / 'manifest.json', dict(manifest_sha256=mhash, folds=folds, assignments=assignments, records=records, residual_mean=mean.tolist(), residual_std=std.tolist(), baseline='70->128->64->4 SiLU; sum derived from 4 outputs', features='distance2 + sum/abs-diff of AA25+flags9; no ESM/FP/FMO/q/formal charge', fit_sampling='max256 unordered pairs per train graph, seed42', epochs=2 if args.mode == 'smoke' else 50, metrics=report, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    base.write_json(out / 'COMPLETE.json', dict(complete=True, graphs=len(records), oof_folds=k))
if __name__ == '__main__':
    main()
