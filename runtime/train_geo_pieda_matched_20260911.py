"""Matched geometry vs geometry+PIEDA training followed by frozen common probe.

Separate experiment: never modifies original graphs, code, splits or runs.
All condition-dependent behavior is confined to the PIEDA loss/head update.
CUDA scatter reductions can still have normal floating-point nondeterminism.
"""
import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data, Batch
import pretrain_masked_edge_multigraph_v17r_rawdist as pm
from prepare_geo_pieda_matched_20260911 import ROOT, TARGETS, DISTANCES, require, load, write_json, raw_targets, fingerprint

def seed_for(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'little') % (2 ** 63 - 1)

def digest_state(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def json_line(path, obj):
    with path.open('a') as f:
        f.write(json.dumps(obj, allow_nan=False) + '\n')

def load_graph(r, version, manifest):
    g = load(r['paths'][version])
    require(fingerprint(g) == r['fingerprints'][version], r['key'] + ': graph changed since audit')
    canonical = g if version == 'v7' else load(r['paths']['v7'])
    if version != 'v7':
        require(fingerprint(canonical) == r['fingerprints']['v7'], 'canonical teacher changed')
    y = torch.cat([canonical.edge_attr[:, [canonical.edge_attr_names.index(n) for n in DISTANCES]].double(), raw_targets(canonical)], 1).float()
    mean = torch.tensor(manifest['train_stats']['mean'])
    std = torch.tensor(manifest['train_stats']['std'])
    geom, targets, fmo, names = pm.infer_feature_groups(g)
    require(len(geom) == 66 and len(fmo) == 6 and (len(names) == 72), 'unexpected schema')
    ea = g.edge_attr.clone().float()
    ea[:, fmo] = 0
    ei = g.edge_index.long()
    lo, hi = (ei.min(0).values, ei.max(0).values)
    require(bool((lo != hi).all()), 'self edge')
    pair, inverse, counts = torch.unique(lo * g.x.shape[0] + hi, sorted=True, return_inverse=True, return_counts=True)
    require(bool((counts == 2).all()), 'expected two directed edges per pair')
    require(torch.unique(ei[0] * g.x.shape[0] + ei[1]).numel() == ei.shape[1], 'duplicate direction')
    require(hasattr(g, 'is_ligand'), 'missing ligand metadata')
    is_lig = g.is_ligand.bool().reshape(-1)
    require(len(is_lig) == len(g.x), 'bad ligand metadata')
    lig_edges = is_lig[ei[0]] | is_lig[ei[1]]
    return dict(key=r['key'], source=r['source'], x=g.x.float(), ei=ei, ea=ea, y=(y - mean) / std, raw=y[:, 2:], geom=geom, fmo=fmo, inverse=inverse, pairs=len(pair), ligand=lig_edges)

def masked_graph(g, epoch, audit=None):
    gen = torch.Generator().manual_seed(seed_for('mask', 42, g['key'], epoch))
    ids = torch.randperm(g['pairs'], generator=gen)[:max(1, math.ceil(0.15 * g['pairs']))]
    chosen = torch.zeros(g['pairs'], dtype=torch.bool)
    chosen[ids] = True
    sel = chosen[g['inverse']]
    ea = g['ea'].clone()
    ea[sel.nonzero().flatten()[:, None], torch.tensor(g['geom'])] = 0
    if audit is not None:
        audit.update(g['key'].encode())
        audit.update(sel.numpy().tobytes())
    return Data(x=g['x'], edge_index=g['ei'], edge_attr=ea, y=g['y'], decode_mask=sel)

def make_model(version, condition):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    m = pm.EdgeMLPMPNN(node_in_dim=1026 if version == 'v6' else 546, edge_in_dim=72, geom_dim=66, target_dim=5, hidden_dim=256, num_layers=4, dropout=0.1, decoder_head='hybrid_nogeom', component_head_hidden=0, distance_target_dim=2, node_target_dim=0)
    if condition == 'geo':
        for mod in (m.edge_decoder_trunk, m.component_heads):
            for p in mod.parameters():
                p.requires_grad_(False)
    return m.cuda()

def losses(m, b):
    h = m.encode_nodes(b.x, b.edge_index, b.edge_attr)
    sel = b.decode_mask.nonzero().flatten()
    geom_pred = m.decode_distance(h, b.edge_index, sel)
    pieda_pred = m.decode_edges(h, b.edge_index, None, sel)
    lg = (geom_pred - b.y[sel, :2]).square().mean()
    lp = (pieda_pred - b.y[sel, 2:]).square().mean()
    return (lg, lp, len(sel))

def smoke_checks(graphs, version):
    g = graphs['train'][0]
    h1, h2 = (hashlib.sha256(), hashlib.sha256())
    a, b = (masked_graph(g, 1, h1), masked_graph(g, 1, h2))
    require(torch.equal(a.edge_attr, b.edge_attr) and h1.hexdigest() == h2.hexdigest(), 'mask mismatch')
    require(torch.equal(a.edge_index, g['ei']), 'topology changed')
    require(torch.count_nonzero(a.edge_attr[:, g['fmo']]) == 0, 'FMO leak')
    require(torch.count_nonzero(a.edge_attr[a.decode_mask][:, g['geom']]) == 0, 'distance leak')
    require(torch.equal(a.edge_attr[~a.decode_mask], g['ea'][~a.decode_mask]), 'incident mask changed')
    geo, joint = (make_model(version, 'geo'), make_model(version, 'geo_pieda'))
    initial = digest_state(geo.state_dict())
    require(initial == digest_state(joint.state_dict()), 'initialization differs')
    batch = Batch.from_data_list([a]).cuda()
    for model, weight in ((geo, 0), (joint, 1)):
        torch.manual_seed(777)
        lg, lp, _ = losses(model, batch)
        (lg + weight * lp).backward()
        require(any((p.grad is not None and p.grad.abs().sum() > 0 for p in model.node_encoder.parameters())), 'no encoder gradient')
        head = list(model.edge_decoder_trunk.parameters()) + list(model.component_heads.parameters())
        require(all((p.grad is None for p in head)) if weight == 0 else any((p.grad is not None and p.grad.abs().sum() > 0 for p in head)), 'PIEDA gradient mismatch')
    geo.zero_grad(set_to_none=True)
    torch.manual_seed(888)
    lg, lp, _ = losses(geo, batch)
    (lg + 0 * lp).backward()
    before = [p.grad.clone() for p in geo.node_encoder.parameters()]
    batch.y[:, 2:] += 100
    geo.zero_grad(set_to_none=True)
    torch.manual_seed(888)
    lg, lp, _ = losses(geo, batch)
    (lg + 0 * lp).backward()
    require(all((torch.allclose(x, p.grad, atol=1e-06, rtol=1e-05) for x, p in zip(before, geo.node_encoder.parameters()))), 'GEO depends on PIEDA teacher')
    del geo, joint, batch
    torch.cuda.empty_cache()
    return dict(mask_both_directions=True, all_fmo_zero=True, topology_unchanged=True, no_incident_mask=True, initialization_matched=True, gradient_checks=True, initial_sha256=initial)

def pretrain(graphs, args, out, manifest_hash):
    model = make_model(args.version, args.condition)
    initial_hash = digest_state(model.state_dict())
    frozen_before = digest_state({k: v for k, v in model.state_dict().items() if k.startswith(('edge_decoder_trunk.', 'component_heads.'))})
    config = dict(version=args.version, condition=args.condition, epochs=args.epochs, seed=42, hidden_dim=256, layers=4, dropout=0.1, batch_size=8, optimizer='AdamW', lr=0.001, weight_decay=0.0001, scheduler=None, grad_clip=5.0, mask_ratio=0.15, incident_mask=False, loss='mean standardized component MSE; Lgeom + lambda*Lpieda', pieda_lambda=int(args.condition == 'geo_pieda'), checkpoint_selection='fixed final epoch', total_parameters=sum((p.numel() for p in model.parameters())), updated_parameters=sum((p.numel() for p in model.parameters() if p.requires_grad)), initial_sha256=initial_hash, manifest_sha256=manifest_hash, job=os.environ.get('SLURM_JOB_ID', 'standalone'), partition=os.environ.get('SLURM_JOB_PARTITION', 'standalone'))
    write_json(out / 'config.json', config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0.0001)
    timings, steps = ([], 0)
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        model.train()
        order = np.random.default_rng(seed_for('order', 42, epoch)).permutation(len(graphs['train']))
        mask_hash = hashlib.sha256()
        sums, count = (np.zeros(2), 0)
        for bi, start in enumerate(range(0, len(order), 8)):
            batch = Batch.from_data_list([masked_graph(graphs['train'][int(i)], epoch, mask_hash) for i in order[start:start + 8]]).cuda()
            torch.manual_seed(seed_for('dropout', 42, epoch, bi))
            optimizer.zero_grad(set_to_none=True)
            lg, lp, n = losses(model, batch)
            loss = lg + config['pieda_lambda'] * lp
            require(torch.isfinite(loss), 'nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            sums += np.array([lg.item(), lp.item()]) * n
            count += n
            steps += 1
        valid = None
        if epoch % 5 == 0 or epoch == args.epochs or args.mode == 'smoke':
            model.eval()
            vs, vn = (np.zeros(2), 0)
            with torch.no_grad():
                for start in range(0, len(graphs['valid']), 8):
                    batch = Batch.from_data_list([masked_graph(g, 'valid12345') for g in graphs['valid'][start:start + 8]]).cuda()
                    lg, lp, n = losses(model, batch)
                    vs += np.array([lg.item(), lp.item()]) * n
                    vn += n
            valid = (vs / vn).tolist()
        elapsed = time.monotonic() - started
        timings.append(elapsed)
        record = dict(epoch=epoch, steps=steps, train=(sums / count).tolist(), valid=valid, seconds=elapsed, mask_sha256=mask_hash.hexdigest())
        json_line(out / 'training.jsonl', record)
        estimate = float(np.mean(timings[-10:])) * (args.epochs - epoch)
        print(json.dumps(dict(**record, pretrain_remaining_seconds=estimate, estimated_pretrain_end=dt.datetime.now().astimezone().isoformat() if epoch == args.epochs else (dt.datetime.now().astimezone() + dt.timedelta(seconds=estimate)).isoformat())), flush=True)
        if epoch % 25 == 0 or epoch == args.epochs:
            torch.save(dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), config=config, epoch=epoch, steps=steps), out / f'epoch_{epoch:03d}.pt')
    if args.condition == 'geo':
        frozen_after = digest_state({k: v for k, v in model.state_dict().items() if k.startswith(('edge_decoder_trunk.', 'component_heads.'))})
        require(frozen_before == frozen_after, 'GEO PIEDA head updated')
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model

class Probe(nn.Module):

    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.SiLU())
        self.heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(5)])

    def forward(self, x):
        h = self.trunk(x)
        return torch.cat([head(h) for head in self.heads], 1)

def probe_selection(g):
    lig = g['ligand'].nonzero().flatten()
    nonlig = (~g['ligand']).nonzero().flatten()
    gen = torch.Generator().manual_seed(seed_for('probe', 123, g['key']))
    if len(nonlig) > 2000:
        nonlig = nonlig[torch.randperm(len(nonlig), generator=gen)[:2000]]
    return torch.cat([lig, nonlig]).sort().values

def extract(model, graphs, label, out):
    selected = [probe_selection(g) for g in graphs]
    n = sum((len(s) for s in selected))
    cap = {'train': 800000, 'valid': 150000, 'test': n}[label]
    kept = np.sort(np.random.default_rng(seed_for('probe_cap', 123, label)).choice(n, min(n, cap), replace=False))
    x = torch.empty((len(kept), 512))
    y = torch.empty((len(kept), 5))
    source = np.empty(len(kept), dtype=np.int8)
    graph_ids = np.empty(len(kept), dtype=np.int32)
    offset, position = (0, 0)
    selection_hash = hashlib.sha256()
    with torch.no_grad():
        for gi, (g, sel) in enumerate(zip(graphs, selected)):
            end = np.searchsorted(kept, offset + len(sel))
            rows = sel[torch.from_numpy(kept[position:end] - offset)]
            selection_hash.update(g['key'].encode())
            selection_hash.update(rows.numpy().tobytes())
            if len(rows):
                h = model.encode_nodes(g['x'].cuda(), g['ei'].cuda(), g['ea'].cuda()).cpu()
                x[position:end] = torch.cat([h[g['ei'][0, rows]], h[g['ei'][1, rows]]], 1)
                y[position:end] = g['raw'][rows]
                source[position:end] = int(g['source'] == 'scop2')
                graph_ids[position:end] = gi
            position, offset = (end, offset + len(sel))
            if (gi + 1) % 1000 == 0:
                print(f'probe extract {label} {gi + 1}/{len(graphs)}', flush=True)
    require(position == len(kept) and torch.isfinite(x).all() and torch.isfinite(y).all(), 'probe extraction')
    audit = dict(candidate_edges=n, retained_edges=len(kept), graphs=len(graphs), selection_sha256=selection_hash.hexdigest(), raw_targets_sha256=hashlib.sha256(y.numpy().tobytes()).hexdigest())
    write_json(out / (label + '_probe_audit.json'), audit)
    return (x, y, source, graph_ids)

def metrics(y, pred):
    out = {}
    for j, name in enumerate(TARGETS):
        a, b = (y[:, j].astype(np.float64), pred[:, j].astype(np.float64))
        ss = ((a - a.mean()) ** 2).sum()
        r = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else None
        out[name.removesuffix('_z')] = dict(pearson_r=float(r) if r is not None else None, r2=float(1 - ((a - b) ** 2).sum() / ss) if ss > 0 else None, mae=float(np.abs(a - b).mean()))
    return out

def run_probe(model, graphs, args, out):
    encoder_before = digest_state(model.state_dict())
    datasets = {l: extract(model, graphs[l], l, out) for l in ('train', 'valid', 'test')}
    x, y = datasets['train'][:2]
    xm, xs = (x.mean(0), x.std(0, unbiased=False).clamp_min(1e-08))
    ym, ys = (y.mean(0), y.std(0, unbiased=False).clamp_min(1e-08))
    for l, (xx, yy, _, _) in datasets.items():
        xx.sub_(xm).div_(xs)
    yn = (y - ym) / ys
    torch.manual_seed(123)
    probe = Probe().cuda()
    init = digest_state(probe.state_dict())
    opt = torch.optim.Adam(probe.parameters(), lr=0.001)
    best, best_state, best_epoch, stale = (float('inf'), None, 0, 0)
    epochs = 2 if args.mode == 'smoke' else 80
    for epoch in range(1, epochs + 1):
        probe.train()
        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed_for('probe_order', 123, epoch)))
        for start in range(0, len(x), 4096):
            ids = order[start:start + 4096]
            opt.zero_grad(set_to_none=True)
            loss = (probe(x[ids].cuda()) - yn[ids].cuda()).square().mean()
            require(torch.isfinite(loss), 'nonfinite probe training loss')
            loss.backward()
            opt.step()
        probe.eval()
        vx, vy = datasets['valid'][:2]
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(vx), 4096):
                p = probe(vx[start:start + 4096].cuda()).cpu()
                total += ((p - (vy[start:start + 4096] - ym) / ys) ** 2).sum().item()
        val = total / (len(vx) * 5)
        require(math.isfinite(val), 'nonfinite validation loss')
        if val < best:
            best, best_state, best_epoch, stale = (val, copy.deepcopy(probe.state_dict()), epoch, 0)
        else:
            stale += 1
        json_line(out / 'probe_training.jsonl', dict(epoch=epoch, valid_mse=val))
        if epoch % 10 == 0 or args.mode == 'smoke':
            print(f'probe epoch={epoch} valid={val:.6f} best={best:.6f}', flush=True)
        if stale >= 10:
            break
    probe.load_state_dict(best_state)
    probe.eval()
    tx, ty, source, graph_ids = datasets['test']
    pred = torch.empty_like(ty)
    with torch.no_grad():
        for start in range(0, len(tx), 4096):
            pred[start:start + 4096] = probe(tx[start:start + 4096].cuda()).cpu() * ys + ym
    require(torch.isfinite(pred).all(), 'nonfinite test prediction')
    require(digest_state(model.state_dict()) == encoder_before, 'encoder updated by probe')
    yy, pp = (ty.numpy(), pred.numpy())
    result = dict(initial_sha256=init, best_epoch=best_epoch, epochs=epoch, best_valid=best, encoder_frozen=True, metrics_weighting='directed selected edges; pooled and source-separated', parameters=sum((p.numel() for p in probe.parameters())), pooled=metrics(yy, pp))
    for s, code in [('bdb', 0), ('scop2', 1)]:
        take = source == code
        require(take.any(), 'missing source in probe test')
        result[s] = dict(edges=int(take.sum()), metrics=metrics(yy[take], pp[take]))
    for gi in np.unique(graph_ids):
        take = graph_ids == gi
        json_line(out / 'test_per_graph.jsonl', dict(key=graphs['test'][int(gi)]['key'], edges=int(take.sum()), metrics=metrics(yy[take], pp[take])))
    np.savez_compressed(out / 'test_predictions.npz', y=yy, prediction=pp, source=source, graph_ids=graph_ids)
    torch.save(dict(model_state=best_state, x_mean=xm, x_std=xs, y_mean=ym, y_std=ys, best_epoch=best_epoch, init_sha256=init), out / 'probe.pt')
    write_json(out / 'probe_results.json', result)
    print('PROBE_COMPLETE ' + json.dumps(result), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['smoke', 'full'], required=True)
    ap.add_argument('--version', choices=['v6', 'v7'], required=True)
    ap.add_argument('--condition', choices=['geo', 'geo_pieda'], required=True)
    args = ap.parse_args()
    require(True, 'g01 only')
    require(torch.cuda.is_available(), 'GPU required')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    args.epochs = 2 if args.mode == 'smoke' else 300
    require((ROOT / 'COMPLETE.json').exists(), 'audit incomplete')
    manifest_bytes = (ROOT / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    out = Path(f'results/geo_pieda_matched_20260911_{args.mode}/{args.version}_{args.condition}')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'code_hashes.json', {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (__file__, pm.__file__, 'prepare_geo_pieda_matched_20260911.py')})
    graphs = {}
    for label in ('train', 'valid', 'test'):
        rows = [r for r in manifest['records'] if r['split'] == label]
        if args.mode == 'smoke':
            rows = [r for s in ('bdb', 'scop2') for r in [x for x in rows if x['source'] == s][:2]]
        graphs[label] = []
        for i, r in enumerate(rows):
            graphs[label].append(load_graph(r, args.version, manifest))
            if (i + 1) % 1000 == 0:
                print(f'preload {label} {i + 1}/{len(rows)}', flush=True)
        print(f"preloaded {label} graphs={len(rows)} edges={sum((g['ei'].shape[1] for g in graphs[label]))}", flush=True)
    if args.mode == 'smoke':
        write_json(out / 'smoke_checks.json', smoke_checks(graphs, args.version))
    model = pretrain(graphs, args, out, hashlib.sha256(manifest_bytes).hexdigest())
    run_probe(model, graphs, args, out)
    write_json(out / 'COMPLETE.json', dict(all_finished=True, version=args.version, condition=args.condition, job=os.environ.get('SLURM_JOB_ID', 'standalone'), partition=__import__('os').environ.get('SLURM_JOB_PARTITION', 'standalone')))
if __name__ == '__main__':
    main()
