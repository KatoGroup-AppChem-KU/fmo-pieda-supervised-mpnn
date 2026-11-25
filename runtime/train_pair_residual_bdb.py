"""R0/R1/R2 controlled training, with common node/pair raw/residual probes.

Training loop mirrors screen_d2_linear_bdb_20260914.pretrain; only the
architecture and the optional independently standardized residual loss change.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import screen_d2_linear_bdb_20260914 as reference
from pair_feedback_model import make_model
from prepare_pieda_residual_bdb import records_manifest
screen, base = (reference.screen, reference.screen.base)
ROOT = reference.ROOT
ORIGINAL_EXTRACT, ORIGINAL_PROBE = (base.extract, base.Probe)

def masked(g, epoch, audit=None):
    b = base.masked_graph(g, epoch, audit)
    b.y = torch.cat([b.y, g['residual_z']], 1)
    return b

def components(model, b):
    h = model.encode_nodes(b.x, b.edge_index, b.edge_attr)
    sel = b.decode_mask.nonzero().flatten()
    geo = model.decode_distance(h, b.edge_index, sel)
    raw = model.decode_edges(h, b.edge_index, None, sel)
    res = model.residual_decoder(model.pair_state[sel])
    return ((torch.cat([geo, raw, res], 1) - b.y[sel]).square().mean(0), len(sel))

def total_loss(v, condition):
    return v[:2].mean() + int(condition != 'R0') * v[2:7].mean() + int(condition == 'R2') * v[7:].mean()

def record(v, condition):
    r = reference.metric_record(v[:7], int(condition != 'R0'))
    r['residual'] = float(np.mean(v[7:]))
    r['residual_components'] = v[7:].tolist()
    r['total'] += int(condition == 'R2') * r['residual']
    return r

def checks(graphs):
    base.make_model = lambda version, condition: make_model('R0' if condition == 'geo' else 'R1')
    audit = base.smoke_checks(graphs, 'v6')
    b = base.Batch.from_data_list([masked(graphs['train'][0], 1)]).cuda()
    initial = []
    for condition in ('R0', 'R1', 'R2'):
        m = make_model(condition)
        initial.append(base.digest_state(m.state_dict()))
        torch.manual_seed(777)
        v, _ = components(m, b)
        total_loss(v, condition).backward()
        for name in ('node_encoder', 'pair_init', 'pair_updates', 'pair_messages'):
            base.require(any((p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(m, name).parameters())), f'{condition}: no gradient {name}')
        for name, active in [('residual_decoder', condition == 'R2'), ('component_heads', condition != 'R0')]:
            ps = list(getattr(m, name).parameters())
            base.require(any((p.grad is not None and p.grad.abs().sum() > 0 for p in ps)) if active else all((p.grad is None for p in ps)), 'inactive/active head gradient mismatch')
        if condition != 'R2':
            saved = b.y.clone()
            b.y[:, 7 if condition == 'R1' else 2:] += 100
            before = [p.grad.clone() for p in m.node_encoder.parameters()]
            m.zero_grad(set_to_none=True)
            torch.manual_seed(777)
            w, _ = components(m, b)
            total_loss(w, condition).backward()
            base.require(all((torch.allclose(a, p.grad, atol=1e-06, rtol=1e-05) for a, p in zip(before, m.node_encoder.parameters()))), 'inactive teacher affected encoder')
            b.y = saved
        del m
    base.require(len(set(initial)) == 1, 'R0/R1/R2 initial weights differ')
    m = make_model('R2')
    m.eval()
    v, _ = components(m, b)
    v[2:7].mean().backward()
    base.require(any((p.grad is not None and p.grad.abs().sum() > 0 for p in m.pair_messages.parameters())), 'PIEDA does not reach pair-node feedback')
    with torch.no_grad():
        h = m.encode_nodes(b.x, b.edge_index, b.edge_attr).clone()
        hooks = [a.register_forward_hook(lambda mod, inp, out: torch.zeros_like(out)) for a in m.pair_messages]
        other = m.encode_nodes(b.x, b.edge_index, b.edge_attr)
        for hook in hooks:
            hook.remove()
        base.require(not torch.allclose(h, other), 'feedback has no effect on node state')
    audit.update(initial_all_conditions_matched=True, feedback_changes_nodes=True, residual_gradient_checks=True, pieda_reaches_feedback=True)
    return audit

def pretrain(model, graphs, condition, out, config, epochs):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0.0001)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    timings = []
    steps = 0
    print(f"Loss weights: GEO=1 PIEDA={int(condition != 'R0')} residual={int(condition == 'R2')}; weight-0 losses are diagnostic only", flush=True)
    for ep in range(1, epochs + 1):
        started = time.monotonic()
        model.train()
        order = np.random.default_rng(base.seed_for('order', 42, ep)).permutation(len(graphs['train']))
        mh = hashlib.sha256()
        sums = np.zeros(12)
        count = 0
        for bi, start in enumerate(range(0, len(order), 8)):
            b = base.Batch.from_data_list([masked(graphs['train'][int(i)], ep, mh) for i in order[start:start + 8]]).cuda()
            torch.manual_seed(base.seed_for('dropout', 42, ep, bi))
            opt.zero_grad(set_to_none=True)
            v, n = components(model, b)
            loss = total_loss(v, condition)
            base.require(torch.isfinite(loss), 'nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            opt.step()
            sums += v.detach().cpu().numpy() * n
            count += n
            steps += 1
        valid = None
        if ep % 5 == 0 or ep == epochs:
            model.eval()
            vs = np.zeros(12)
            vn = 0
            with torch.no_grad():
                for start in range(0, len(graphs['valid']), 8):
                    b = base.Batch.from_data_list([masked(g, 'valid12345') for g in graphs['valid'][start:start + 8]]).cuda()
                    v, n = components(model, b)
                    vs += v.cpu().numpy() * n
                    vn += n
            valid = record(vs / vn, condition)
        elapsed = time.monotonic() - started
        timings.append(elapsed)
        train = record(sums / count, condition)
        base.json_line(out / 'training.jsonl', dict(epoch=ep, steps=steps, train=train, valid=valid, seconds=elapsed, mask_sha256=mh.hexdigest()))
        reference.pretty.print_epoch(ep, epochs, train, valid, elapsed, np.mean(timings[-10:]) * (epochs - ep))
        for label, r in [('Train', train), ('Valid', valid)]:
            if r is not None:
                vals = ' '.join((f'{name}={value:.6f}' for name, value in zip(['sum', 'ES', 'EX', 'CTmix', 'DI'], r['residual_components'])))
                print(f"  {label} residual={r['residual']:.6f} ({vals})", flush=True)
        if ep % 25 == 0 or ep == epochs:
            torch.save(dict(model_state=model.state_dict(), optimizer_state=opt.state_dict(), config=config, epoch=ep, steps=steps), out / f'epoch_{ep:03d}.pt')
    base.require(all((torch.equal(p, frozen[n]) for n, p in model.named_parameters() if n in frozen)), 'inactive head changed')
    model.requires_grad_(False)
    model.eval()

class PairProbe(ORIGINAL_PROBE):

    def __init__(self):
        super().__init__()
        self.trunk[0] = nn.Linear(256, 256)

def extract_pair(model, graphs, label, out):
    selected = [base.probe_selection(g) for g in graphs]
    n = sum((len(s) for s in selected))
    cap = {'train': 800000, 'valid': 150000, 'test': n}[label]
    kept = np.sort(np.random.default_rng(base.seed_for('probe_cap', 123, label)).choice(n, min(n, cap), replace=False))
    x, y = (torch.empty((len(kept), 256)), torch.empty((len(kept), 5)))
    source = np.empty(len(kept), np.int8)
    gids = np.empty(len(kept), np.int32)
    offset = position = 0
    sha = hashlib.sha256()
    with torch.no_grad():
        for gi, (g, sel) in enumerate(zip(graphs, selected)):
            end = np.searchsorted(kept, offset + len(sel))
            rows = sel[torch.from_numpy(kept[position:end] - offset)]
            sha.update(g['key'].encode())
            sha.update(rows.numpy().tobytes())
            if len(rows):
                model.encode_nodes(g['x'].cuda(), g['ei'].cuda(), g['ea'].cuda())
                x[position:end] = model.pair_state[rows.cuda()].cpu()
                y[position:end] = g['raw'][rows]
                source[position:end] = 0
                gids[position:end] = gi
            position, offset = (end, offset + len(sel))
            if (gi + 1) % 1000 == 0:
                print(f'pair extract {label} {gi + 1}/{len(graphs)}', flush=True)
    base.require(position == len(kept) and torch.isfinite(x).all() and torch.isfinite(y).all(), 'bad extraction')
    base.write_json(out / (label + '_probe_audit.json'), dict(candidate_edges=n, retained_edges=len(kept), graphs=len(graphs), selection_sha256=sha.hexdigest(), raw_targets_sha256=hashlib.sha256(y.numpy().tobytes()).hexdigest()))
    return (x, y, source, gids)

def evaluate(model, graphs, args, out):
    for target in ('raw', 'residual'):
        gs = graphs if target == 'raw' else {s: [dict(g, raw=g['residual']) for g in rows] for s, rows in graphs.items()}
        for readout in ('node', 'pair'):
            folder = out / f'{readout}_{target}_probe'
            folder.mkdir()
            base.extract = ORIGINAL_EXTRACT if readout == 'node' else extract_pair
            base.Probe = ORIGINAL_PROBE if readout == 'node' else PairProbe
            try:
                screen.run_bdb_probe(model, gs, args, folder)
            finally:
                base.extract, base.Probe = (ORIGINAL_EXTRACT, ORIGINAL_PROBE)
        for split in graphs:
            a = json.loads((out / f'node_{target}_probe' / (split + '_probe_audit.json')).read_text())
            b = json.loads((out / f'pair_{target}_probe' / (split + '_probe_audit.json')).read_text())
            base.require(a == b, 'node/pair probe cohort mismatch')

def reconstruction(model, graphs, manifest, out):
    """Final original heads on fixed masked test pairs, separate from probes."""
    true = []
    pred = []
    res_true = []
    res_pred = []
    mean = torch.tensor(manifest['train_stats']['mean'][2:])
    std = torch.tensor(manifest['train_stats']['std'][2:])
    with torch.no_grad():
        for start in range(0, len(graphs['test']), 8):
            rows = graphs['test'][start:start + 8]
            b = base.Batch.from_data_list([masked(g, 'test12345') for g in rows]).cuda()
            h = model.encode_nodes(b.x, b.edge_index, b.edge_attr)
            sel = b.decode_mask.nonzero().flatten()
            pred.append(model.decode_edges(h, b.edge_index, None, sel).cpu() * std + mean)
            true.append(b.y[sel, 2:7].cpu() * std + mean)
            res_pred.append(model.residual_decoder(model.pair_state[sel]).cpu())
            res_true.append(b.y[sel, 7:].cpu())
    base.write_json(out / 'original_head_test.json', dict(selection='test12345 fixed 15% pair mask, both directions, noctx', raw=base.metrics(torch.cat(true).numpy(), torch.cat(pred).numpy()), residual_standardized=base.metrics(torch.cat(res_true).numpy(), torch.cat(res_pred).numpy()), note='Inactive heads are untrained diagnostics, not comparable reconstruction models'))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition', choices=['R0', 'R1', 'R2'], required=True)
    ap.add_argument('--mode', choices=['smoke', 'full'], required=True)
    args = ap.parse_args()
    base.require(True and True, 'wrong site')
    base.require(torch.cuda.is_available(), 'GPU required')
    os.chdir(ROOT)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    records, manifest, mhash = records_manifest(args.mode)
    residual_root = ROOT / 'results' / f'pieda_residual_bdb_{args.mode}'
    base.require((residual_root / 'COMPLETE.json').exists(), 'residual preparation incomplete')
    rr = (residual_root / 'manifest.json').read_bytes()
    rm = json.loads(rr)
    base.require(rm['manifest_sha256'] == mhash and rm['records'] == records, 'residual source mismatch')
    mean, std = (torch.tensor(rm['residual_mean']), torch.tensor(rm['residual_std']))
    graphs = {s: [] for s in ('train', 'valid', 'test')}
    for i, r in enumerate(records, 1):
        g = base.load_graph(r, 'v6', manifest)
        a = torch.load(residual_root / 'graphs' / (r['key'].replace(':', '_') + '.pt'), weights_only=False)
        base.require(a['key'] == r['key'] and a['fingerprint'] == r['fingerprints']['v6'] and (a['raw_sha256'] == hashlib.sha256(g['raw'].numpy().tobytes()).hexdigest()), 'residual graph alignment')
        g['residual'] = a['residual']
        g['residual_z'] = (a['residual'] - mean) / std
        base.require(g['residual'].shape == g['raw'].shape and torch.isfinite(g['residual_z']).all(), 'bad residual')
        graphs[r['split']].append(g)
        if i % 1000 == 0:
            print(f'preload {i}/{len(records)}', flush=True)
    out = ROOT / 'results' / f'pair_residual_bdb_{args.mode}' / args.condition
    out.mkdir(parents=True, exist_ok=False)
    base.write_json(out / 'checks.json', checks(graphs))
    m = make_model(args.condition)
    epochs = 2 if args.mode == 'smoke' else 300
    config = dict(condition=args.condition, epochs=epochs, node_dim=256, pair_dim=256, layers=4, pair_init='768->256->256', pair_update='residual LN(1024->256->256)', node_update='sum Linear(pair)->GRU->residual dropout LN', distance_head='node concat512->256->128->2', pieda_head='pair256->256->128->5', residual_head='independent pair256->256->128->5, present in all conditions', seed=42, batch_size=8, optimizer='AdamW', lr=0.001, weight_decay=0.0001, grad_clip=5, validation_every=5, checkpoint_every=25, mask_ratio=0.15, input_policy='existing noctx', loss_weights=[1, int(args.condition != 'R0'), int(args.condition == 'R2')], raw_normalization='unchanged manifest train_stats, originally BDB+SCOP2 train', residual_normalization='all BDB train OOF edges only', probe_seed=123, manifest_sha256=mhash, residual_manifest_sha256=hashlib.sha256(rr).hexdigest(), initial_sha256=base.digest_state(m.state_dict()), parameters=sum((p.numel() for p in m.parameters())), source_hashes={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['train_pair_residual_bdb.py', 'pair_feedback_model.py', 'prepare_pieda_residual_bdb.py', 'screen_d2_linear_bdb_20260914.py', 'screen_g_p1_bdb_20260914.py', 'train_geo_pieda_matched_20260911.py']})
    base.write_json(out / 'config.json', config)
    pretrain(m, graphs, args.condition, out, config, epochs)
    reconstruction(m, graphs, manifest, out)
    evaluate(m, graphs, args, out)
    base.write_json(out / 'COMPLETE.json', dict(complete=True, epochs=epochs, condition=args.condition, node_pair_raw_residual_probes=True))
    print(f'ALL_COMPLETE {out}', flush=True)
if __name__ == '__main__':
    main()
