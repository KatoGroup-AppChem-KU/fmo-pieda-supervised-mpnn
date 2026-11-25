"""BindingDB-only G/P1 head comparison on corrected v6 graphs.

The established matched training and frozen common-probe implementation is
reused. Only the source subset and, for P1, the PIEDA decoder are changed.
"""
import argparse
import copy
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
import train_geo_pieda_matched_20260911 as base
ROOT = Path(__import__('os').environ['FMO_WORKDIR'])
DATA_ROOT = ROOT / 'dataset' / 'geo_pieda_matched_20260911'
ORIGINAL_MAKE_MODEL = base.make_model

def make_model(version, condition, variant):
    """Preserve the established GNN/distance head; optionally shrink PIEDA head."""
    model = ORIGINAL_MAKE_MODEL(version, condition)
    if variant == 'P1':
        model.edge_decoder_trunk = nn.Sequential(nn.Linear(512, 32), nn.LayerNorm(32), nn.SiLU(), nn.Dropout(0.1)).cuda()
        model.component_heads = nn.ModuleList([nn.Linear(32, 1) for _ in range(5)]).cuda()
        if condition == 'geo':
            model.edge_decoder_trunk.requires_grad_(False)
            model.component_heads.requires_grad_(False)
    return model

def run_bdb_probe(model, graphs, args, out):
    """Established common nonlinear probe, restricted to BindingDB."""
    encoder_before = base.digest_state(model.state_dict())
    datasets = {label: base.extract(model, graphs[label], label, out) for label in ('train', 'valid', 'test')}
    x, y = datasets['train'][:2]
    xm, xs = (x.mean(0), x.std(0, unbiased=False).clamp_min(1e-08))
    ym, ys = (y.mean(0), y.std(0, unbiased=False).clamp_min(1e-08))
    for xx, _, _, _ in datasets.values():
        xx.sub_(xm).div_(xs)
    yn = (y - ym) / ys
    torch.manual_seed(123)
    probe = base.Probe().cuda()
    initial = base.digest_state(probe.state_dict())
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.001)
    best, best_state, best_epoch, stale = (float('inf'), None, 0, 0)
    epochs = 2 if args.mode == 'smoke' else 80
    for epoch in range(1, epochs + 1):
        probe.train()
        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(base.seed_for('probe_order', 123, epoch)))
        for start in range(0, len(x), 4096):
            ids = order[start:start + 4096]
            optimizer.zero_grad(set_to_none=True)
            loss = (probe(x[ids].cuda()) - yn[ids].cuda()).square().mean()
            base.require(torch.isfinite(loss), 'nonfinite probe training loss')
            loss.backward()
            optimizer.step()
        probe.eval()
        vx, vy = datasets['valid'][:2]
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(vx), 4096):
                prediction = probe(vx[start:start + 4096].cuda()).cpu()
                target = (vy[start:start + 4096] - ym) / ys
                total += ((prediction - target) ** 2).sum().item()
        valid = total / (len(vx) * 5)
        base.require(math.isfinite(valid), 'nonfinite validation loss')
        if valid < best:
            best, best_state, best_epoch, stale = (valid, copy.deepcopy(probe.state_dict()), epoch, 0)
        else:
            stale += 1
        base.json_line(out / 'probe_training.jsonl', {'epoch': epoch, 'valid_mse': valid})
        if epoch % 10 == 0 or args.mode == 'smoke':
            print(f'probe epoch={epoch} valid={valid:.6f} best={best:.6f}', flush=True)
        if stale >= 10:
            break
    probe.load_state_dict(best_state)
    probe.eval()
    tx, ty, source, graph_ids = datasets['test']
    base.require(bool((source == 0).all()), 'non-BindingDB probe row')
    pred = torch.empty_like(ty)
    with torch.no_grad():
        for start in range(0, len(tx), 4096):
            pred[start:start + 4096] = probe(tx[start:start + 4096].cuda()).cpu() * ys + ym
    base.require(torch.isfinite(pred).all(), 'nonfinite test prediction')
    base.require(base.digest_state(model.state_dict()) == encoder_before, 'encoder updated by probe')
    yy, pp = (ty.numpy(), pred.numpy())
    test_metrics = base.metrics(yy, pp)
    result = {'initial_sha256': initial, 'best_epoch': best_epoch, 'epochs': epoch, 'best_valid': best, 'encoder_frozen': True, 'metrics_weighting': 'directed selected BindingDB edges', 'parameters': sum((p.numel() for p in probe.parameters())), 'pooled': test_metrics, 'bdb': {'edges': int(len(yy)), 'metrics': test_metrics}}
    for gi in np.unique(graph_ids):
        take = graph_ids == gi
        base.json_line(out / 'test_per_graph.jsonl', {'key': graphs['test'][int(gi)]['key'], 'edges': int(take.sum()), 'metrics': base.metrics(yy[take], pp[take])})
    np.savez_compressed(out / 'test_predictions.npz', y=yy, prediction=pp, source=source, graph_ids=graph_ids)
    torch.save({'model_state': best_state, 'x_mean': xm, 'x_std': xs, 'y_mean': ym, 'y_std': ys, 'best_epoch': best_epoch, 'init_sha256': initial}, out / 'probe.pt')
    base.write_json(out / 'probe_results.json', result)
    print('PROBE_COMPLETE ' + json.dumps(result), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['G', 'P1'], required=True)
    ap.add_argument('--condition', choices=['geo', 'geo_pieda'], required=True)
    ap.add_argument('--mode', choices=['smoke', 'full'], default='full')
    ap.add_argument('--stage', choices=['train_probe', 'probe_only'], default='train_probe')
    ap.add_argument('--checkpoint', type=Path)
    args = ap.parse_args()
    base.require(True, 'wrong user')
    base.require(True, 'wrong partition')
    base.require(torch.cuda.is_available(), 'GPU required')
    os.chdir(ROOT)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    base.require((DATA_ROOT / 'COMPLETE.json').exists(), 'graph audit incomplete')
    manifest_bytes = (DATA_ROOT / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    records = [r for r in manifest['records'] if r['source'] == 'bdb']
    counts = {s: sum((r['split'] == s for r in records)) for s in ('train', 'valid', 'test')}
    base.require(counts == {'train': 2970, 'valid': 372, 'test': 372}, f'unexpected BindingDB split counts: {counts}')
    if args.mode == 'smoke':
        records = [next((r for r in records if r['split'] == s)) for s in ('train', 'valid', 'test')]
    out = ROOT / 'results' / f'head_screen_bdb_20260914_{args.mode}' / f'{args.variant}_{args.condition}'
    if args.stage == 'train_probe':
        base.require(args.checkpoint is None, 'checkpoint is only valid with probe_only')
        out.mkdir(parents=True, exist_ok=False)
    else:
        base.require(args.mode == 'full', 'probe_only requires full mode')
        base.require(args.checkpoint is not None, 'probe_only requires checkpoint')
        base.require(out.is_dir(), f'missing output directory: {out}')
        base.require(args.checkpoint.resolve().parent == out.resolve(), 'checkpoint must belong to the selected output directory')
        base.require(args.checkpoint.name == 'epoch_100.pt', 'pilot comparison must use epoch_100.pt')
    graphs = {}
    for split in ('train', 'valid', 'test'):
        rows = [r for r in records if r['split'] == split]
        graphs[split] = []
        for i, row in enumerate(rows, 1):
            graphs[split].append(base.load_graph(row, 'v6', manifest))
            if i % 1000 == 0:
                print(f'preload {split} {i}/{len(rows)}', flush=True)
        print(f"preloaded {split} graphs={len(rows)} edges={sum((g['ei'].shape[1] for g in graphs[split]))}", flush=True)
    base.make_model = lambda version, condition: make_model(version, condition, args.variant)
    args.version = 'v6'
    args.epochs = 2 if args.mode == 'smoke' else 100
    if args.stage == 'train_probe':
        base.write_json(out / 'architecture.json', {'variant': args.variant, 'condition': args.condition, 'graphs': 'corrected v6 BindingDB-only', 'split_counts': counts, 'input_policy': 'noctx: all FMO/q zero; target-pair geometry masked; other geometry retained', 'gnn': 'established 4-layer MPNN hidden256', 'distance_decoder': 'established 512->256->128->2', 'pieda_decoder': 'established 512->256->128->5' if args.variant == 'G' else 'one-hidden-layer 512->32->5 with LN/SiLU/dropout0.1', 'probe': 'frozen GNN; common 512->256->128->5 MLP', 'comparison_budget': 'fixed 100 epochs for full pilot', 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'base_sha256': hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(), 'job': os.environ.get('SLURM_JOB_ID', 'standalone'), 'partition': os.environ.get('SLURM_JOB_PARTITION', 'standalone')})
        base.write_json(out / 'smoke_checks.json', base.smoke_checks(graphs, 'v6'))
        model = base.pretrain(graphs, args, out, hashlib.sha256(manifest_bytes).hexdigest())
    else:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        base.require(checkpoint.get('epoch') == 100, 'checkpoint epoch is not 100')
        config = checkpoint.get('config', {})
        base.require(config.get('version') == 'v6', 'checkpoint graph version mismatch')
        base.require(config.get('condition') == args.condition, 'checkpoint condition mismatch')
        model = make_model('v6', args.condition, args.variant)
        model.load_state_dict(checkpoint['model_state'], strict=True)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        base.write_json(out / 'probe_source_epoch100.json', {'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_epoch': 100, 'variant': args.variant, 'condition': args.condition, 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'job': os.environ.get('SLURM_JOB_ID', 'standalone'), 'partition': os.environ.get('SLURM_JOB_PARTITION', 'standalone')})
    run_bdb_probe(model, graphs, args, out)
    base.write_json(out / 'COMPLETE.json', {'all_finished': True, 'variant': args.variant, 'condition': args.condition, 'pretrain_epochs': args.epochs, 'frozen_common_probe': True, 'job': os.environ.get('SLURM_JOB_ID', 'standalone'), 'partition': 'aurum-g01'})
    print(f'COMPLETE {out}', flush=True)
if __name__ == '__main__':
    main()
