"""Model1/2 x v6/v7 x GEO/GP x fold: old model/loops, new split and best selection."""
import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
import train_pair_residual_bdb as prior
import pair_gate_stages_20260916 as model1
import pair_gate_wide_20260924 as model2
import train_pair_gate_wide_20260924 as wide_tools
import cv5_loops_20260925 as loops
import cv5_pic50_20260925 as downstream
base = prior.base
PREP = Path('dataset/fixedtest_cv5_20260925_prepared')
RUNS = Path('results/cv5_20260925')
ORIGINAL_FACTORY = prior.screen.ORIGINAL_MAKE_MODEL

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def source_hashes():
    return {Path(m.__file__).name: sha(m.__file__) for m in list(sys.modules.values()) if getattr(m, '__file__', None) and Path(m.__file__).resolve().parent == Path.cwd() and str(m.__file__).endswith('.py') and Path(m.__file__).is_file()}

def make_model(number, version, condition):
    prior.screen.ORIGINAL_MAKE_MODEL = lambda unused, task: ORIGINAL_FACTORY(version, task)
    try:
        return model1.make_model(condition, 1) if number == 1 else model2.make_model(condition, 3)
    finally:
        prior.screen.ORIGINAL_MAKE_MODEL = ORIGINAL_FACTORY

def load(r, manifest, version):
    g = base.load_graph(r, version, manifest)
    g['residual_z'] = torch.zeros_like(g['raw'])
    return g

def extract_pair(model, graphs, label, out):
    method = prior.extract_pair if model.stage == 1 else wide_tools.extract_pair
    x, y, s, ids = method(model, graphs, label, out)
    s[:] = np.array([int(graphs[int(i)]['source'] == 'scop2') for i in ids], dtype=np.int8)
    return (x, y, s, ids)

def reconstruction(model, graphs, manifest, out, condition):
    ys = []
    ps = []
    groups = []
    mean = torch.tensor(manifest['train_stats']['mean'])
    std = torch.tensor(manifest['train_stats']['std'])
    with torch.no_grad():
        for start in range(0, len(graphs['test']), 8):
            b = base.Batch.from_data_list([prior.masked(g, 'test12345') for g in graphs['test'][start:start + 8]]).cuda()
            h = model.encode_nodes(b.x, b.edge_index, b.edge_attr)
            sel = b.decode_mask.nonzero().flatten()
            pred = torch.cat([model.decode_distance(h, b.edge_index, sel), model.decode_edges(h, b.edge_index, None, sel)], 1)
            ys.append(b.y[sel, :7].cpu() * std + mean)
            ps.append(pred.cpu() * std + mean)
            groups.append(b.batch[b.edge_index[0, sel]].cpu() + start)
    y, p = (torch.cat(ys).numpy(), torch.cat(ps).numpy())
    assert np.isfinite(p).all()
    geometry = {name: dict(rmse=float(np.sqrt(np.mean((y[:, k] - p[:, k]) ** 2))), pearson_r=float(np.corrcoef(y[:, k], p[:, k])[0, 1])) for k, name in enumerate(('DIST', 'centroid'))}
    base.write_json(out / 'original_head_test.json', dict(geometry=geometry, raw=base.metrics(y[:, 2:], p[:, 2:]), pieda_head_trained=condition == 'R1', note='All parameters frozen for evaluation; see config condition for trained PIEDA head'))
    np.savez_compressed(out / 'original_head_test_predictions.npz', y=y, prediction=p, graph_ids=torch.cat(groups).numpy(), keys=np.array([g['key'] for g in graphs['test']]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', type=int, required=True)
    ap.add_argument('--mode', choices=['smoke', 'full'], required=True)
    args = ap.parse_args()
    assert True and True
    assert Path.cwd() == Path(__import__('os').environ['FMO_WORKDIR']) and torch.cuda.is_available()
    assert 0 <= args.task < 40
    args.fold = args.task // 8 + 1
    combo = args.task % 8
    args.model_number = combo // 4 + 1
    args.version = ('v6', 'v7')[combo % 4 // 2]
    args.condition = ('R0', 'R1')[combo % 2]
    args.smoke = args.mode == 'smoke'
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    done = json.loads((PREP / 'COMPLETE.json').read_text())
    assert done['complete']
    mp = PREP / f'fold{args.fold}.json'
    cp = PREP / f'pic50_fold{args.fold}.csv'
    assert sha(mp) == done['files_sha256'][mp.name] and sha(cp) == done['files_sha256'][cp.name]
    manifest = json.loads(mp.read_text())
    records = manifest['records']
    sources = source_hashes()
    tag = f'model{args.model_number}_{args.version}_{args.condition}_fold{args.fold}'
    out = RUNS / args.mode / tag
    if not args.smoke:
        smoke = RUNS / 'smoke' / f'model{args.model_number}_{args.version}_{args.condition}_fold1' / 'COMPLETE.json'
        gate = json.loads(smoke.read_text())
        assert gate['complete'] and gate['source_hashes'] == sources, 'smoke source mismatch'
    out.mkdir(parents=True, exist_ok=False)
    snap = out / 'code'
    snap.mkdir()
    for name in sources:
        shutil.copy2(name, snap / name)
    shutil.copy2(mp, out / 'fold_manifest.json')
    shutil.copy2(cp, out / 'pic50_cohort.csv')
    if args.smoke:
        largest = sorted(records, key=lambda r: r['edges'], reverse=True)[:8]
        batch = base.Batch.from_data_list([prior.masked(load(r, manifest, args.version), 1) for r in largest]).cuda()
        m = make_model(args.model_number, args.version, args.condition)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=0.001, weight_decay=0.0001)
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            v, _ = loops.components(m, batch)
            loss = prior.total_loss(v, args.condition)
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0, error_if_nonfinite=True)
            opt.step()
        torch.cuda.synchronize()
        base.write_json(out / 'resource.json', dict(keys=[r['key'] for r in largest], batch_size=8, peak_GiB=torch.cuda.max_memory_allocated() / 2 ** 30))
        del m, opt, batch, v, loss
        torch.cuda.empty_cache()
        records = [r for sp, n in [('train', 5), ('valid', 2), ('test', 2)] for src in ('bdb', 'scop2') for r in [r for r in records if r['split'] == sp and r['source'] == src][:n]]
    graphs = {s: [] for s in ('train', 'valid', 'test')}
    for i, r in enumerate(records, 1):
        graphs[r['split']].append(load(r, manifest, args.version))
        if i % 500 == 0:
            print(f'{tag}: preload {i}/{len(records)}', flush=True)
    if args.smoke:
        original = base.make_model
        base.make_model = lambda version, condition: make_model(args.model_number, version, 'R0' if condition == 'geo' else 'R1')
        try:
            audit = base.smoke_checks(graphs, args.version)
        finally:
            base.make_model = original
        m = make_model(args.model_number, args.version, 'R1')
        b = base.Batch.from_data_list([prior.masked(graphs['train'][0], 1)]).cuda()
        v, _ = loops.components(m, b)
        v[2:7].mean().backward()
        for name in ('node_gates', 'pair_messages', 'pair_updates'):
            assert any((p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(m, name).parameters())), name
        if args.model_number == 2:
            p = torch.randn(3, 512, device='cuda', requires_grad=True)
            m.pair_state = p
            pred = m.decode_edges(None, None, None, torch.arange(3, device='cuda'))
            for k in range(4):
                grad = torch.autograd.grad(pred[:, k + 1].sum(), p, retain_graph=True)[0]
                assert torch.count_nonzero(grad * (1 - m.component_masks[k])) == 0 and grad.abs().sum() > 0
        audit['pieda_reaches_feedback'] = True
        base.write_json(out / 'checks.json', audit)
        del m, b, v
        torch.cuda.empty_cache()
    model = make_model(args.model_number, args.version, args.condition)
    initial = base.digest_state(model.state_dict())
    control = make_model(args.model_number, args.version, 'R1' if args.condition == 'R0' else 'R0')
    assert initial == base.digest_state(control.state_dict())
    del control
    epochs = 2 if args.smoke else 300
    config = dict(model_number=args.model_number, legacy='S1' if args.model_number == 1 else 'S3-128', version=args.version, condition=args.condition, fold=args.fold, epochs=epochs, seed=42, head_seed=42, node_dim=256, pair_dim=256 if args.model_number == 1 else 512, layers=4, batch_size=8, optimizer='AdamW', lr=0.001, weight_decay=0.0001, scheduler=None, grad_clip=5.0, dropout=0.1, validation_every=5, checkpoint_every=25, checkpoint_selection='minimum validation total loss', mask_ratio=0.15, input_policy='all FMO/q zero; selected pair both-direction geometry zero', loss_weights=[1, int(args.condition == 'R1'), 0], residual='unused frozen legacy branch; loss zero', normalization='this fold train only; population std; canonical v7 raw targets', train_stats=manifest['train_stats'], manifest_sha256=sha(mp), cohort_sha256=sha(cp), initial_sha256=initial, source_hashes=sources, smoke=args.smoke, parameters=sum((p.numel() for p in model.parameters())), updated_parameters=sum((p.numel() for p in model.parameters() if p.requires_grad)), cohort={s: dict(Counter((g['source'] for g in gs))) for s, gs in graphs.items()}, probe_config=dict(seed=42, edge_selection_seed=123, batch_size=4096, epochs=80, patience=10, optimizer='Adam', lr=0.001))
    base.write_json(out / 'config.json', config)
    loops.pretrain(model, graphs, args.condition, out, config, epochs)
    last = base.load(out / f'epoch_{epochs:03d}.pt')
    assert base.digest_state(last['model_state']) == base.digest_state(model.state_dict())
    del last
    ckpath = out / 'best_model.pt'
    best = base.load(ckpath)
    model.load_state_dict(best['model_state'], strict=True)
    model.eval().requires_grad_(False)
    assert best['epoch'] <= epochs and np.isfinite(best['valid_loss'])
    base.write_json(out / 'selected_checkpoint.json', dict(path=str(ckpath), sha256=sha(ckpath), epoch=best['epoch'], valid_loss=best['valid_loss']))
    del best
    reconstruction(model, graphs, manifest, out, args.condition)
    for readout in ('node', 'pair'):
        dest = out / f'{readout}_raw_probe'
        dest.mkdir()
        loops.extract = prior.ORIGINAL_EXTRACT if readout == 'node' else extract_pair
        loops.Probe = prior.ORIGINAL_PROBE if readout == 'node' or args.model_number == 2 else prior.PairProbe
        loops.run_probe(model, graphs, args, dest)
    for sp in graphs:
        assert json.loads((out / 'node_raw_probe' / f'{sp}_probe_audit.json').read_text()) == json.loads((out / 'pair_raw_probe' / f'{sp}_probe_audit.json').read_text())
    del graphs
    gc.collect()
    torch.cuda.empty_cache()
    downstream.evaluate(model, manifest, cp, args, out / 'pic50', config, ckpath, base)
    assert source_hashes() == sources, 'code changed during run'
    base.write_json(out / 'COMPLETE.json', dict(complete=True, source_hashes=sources, model_number=args.model_number, version=args.version, condition=args.condition, fold=args.fold, head_seed=42, best_checkpoint_sha256=sha(ckpath), frozen_node_pair_and_pic50=True))
    print(f'ALL_COMPLETE {out}', flush=True)
if __name__ == '__main__':
    main()
