"""SG/ESM-FP design B: matched node/pair geometry, optional node/pair PIEDA.

Reuse existing encoder, loading, masking, optimizer schedule and epoch logger.
Separate output series; final epoch 300 fixed before Test evaluation.
"""
import argparse
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from torch import nn
ROOT = Path(__import__('os').environ['FMO_WORKDIR'])
sys.path.insert(0, str(ROOT))
import run_cv5_20260925 as cv
import geopair_cfg_20261006 as gp
base, prior = (cv.base, cv.prior)
PREP = ROOT / 'dataset/fpv2_prepared_20261016'
RUNS = ROOT / 'results/fpv2_sg_b_unitloss_20261016'

def make_model(condition):
    assert condition in ('geo', 'geo_pieda')
    m = gp.make_model(1, 'v6')
    torch.manual_seed(base.seed_for('node_pieda_head', 42))
    m.node_pieda_decoder = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 5)).cuda()
    for module in (m.edge_decoder_trunk, m.component_heads, m.node_pieda_decoder):
        module.requires_grad_(condition == 'geo_pieda')
    return m

def components(m, b):
    h = m.encode_nodes(b.x, b.edge_index, b.edge_attr)
    sel = b.decode_mask.nonzero().flatten()
    u, v = b.edge_index[:, sel]
    ng = m.decode_distance(h, b.edge_index, sel)
    pp = m.decode_edges(h, b.edge_index, None, sel)
    pg = m.pair_distance_decoder(m.pair_state[sel])
    npred = m.node_pieda_decoder(torch.cat([h[u], h[v]], -1))
    target = b.y[sel]
    loss = torch.cat([(ng - target[:, :2]).square().mean(0), (pg - target[:, :2]).square().mean(0), (npred - target[:, 2:7]).square().mean(0), (pp - target[:, 2:7]).square().mean(0)])
    return (loss, len(sel))

def total_loss(v, condition):
    loss = v[:2].mean() + v[2:4].mean()
    if condition == 'geo_pieda':
        loss = loss + v[4:9].mean() + v[9:14].mean()
    return loss

def record(v, condition):
    r = prior.reference.metric_record(np.concatenate([v[:2] + v[2:4], v[4:9] + v[9:14]]), int(condition == 'geo_pieda'))
    r.update(total=float(total_loss(v, condition)), node_geometry=float(np.mean(v[:2])), pair_geometry=float(np.mean(v[2:4])), node_pieda=float(np.mean(v[4:9])), pair_pieda=float(np.mean(v[9:14])), pair_geometry_components=v[2:4].tolist(), node_pieda_components=v[4:9].tolist(), pair_pieda_components=v[9:14].tolist())
    return r

def source_hashes():
    d = cv.source_hashes()
    d[Path(__file__).name] = cv.sha(__file__)
    return d

def has_grad(module):
    return any((p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters()))

def smoke(graphs, out):
    for sl in (slice(0, 4), slice(4, 14)):
        probe = torch.zeros(14, device='cuda', requires_grad=True)
        with torch.no_grad():
            probe[sl] = 1
        assert torch.equal(total_loss(probe, 'geo_pieda'), torch.tensor(2.0, device='cuda'))
        total_loss(probe, 'geo_pieda').backward()
        assert torch.allclose(probe.grad[:4].sum(), probe.grad[4:].sum())
    for sl in (slice(0, 2), slice(2, 4), slice(4, 9), slice(9, 14)):
        probe = torch.zeros(14, device='cuda')
        probe[sl] = 1
        assert torch.equal(total_loss(probe, 'geo_pieda'), torch.tensor(1.0, device='cuda'))
    a, b = (make_model('geo'), make_model('geo_pieda'))
    assert base.digest_state(a.state_dict()) == base.digest_state(b.state_dict())
    initial = base.digest_state(a.state_dict())
    batches = []
    digests = []
    for _ in range(2):
        mh = hashlib.sha256()
        batches.append(base.Batch.from_data_list([prior.masked(g, 1, mh) for g in graphs]).cuda())
        digests.append(mh.hexdigest())
    assert digests[0] == digests[1]
    batch = batches[0]
    assert torch.equal(batch.edge_attr, batches[1].edge_attr)
    assert torch.count_nonzero(batch.edge_attr[:, graphs[0]['fmo']]) == 0
    assert torch.count_nonzero(batch.edge_attr[batch.decode_mask][:, graphs[0]['geom']]) == 0
    for g in graphs:
        masked = prior.masked(g, 1)
        sums = torch.zeros(g['pairs'], dtype=torch.long)
        sums.index_add_(0, g['inverse'], masked.decode_mask.long())
        assert ((sums == 0) | (sums == 2)).all()
    torch.manual_seed(991)
    vals, _ = components(a, batch)
    la = total_loss(vals, 'geo')
    la.backward()
    grad = {n: p.grad.detach().clone() for n, p in a.named_parameters() if p.grad is not None}
    original = batch.y.clone()
    batch.y[:, 2:7] += 100
    a.zero_grad(set_to_none=True)
    torch.manual_seed(991)
    vals, _ = components(a, batch)
    lb = total_loss(vals, 'geo')
    lb.backward()
    assert torch.equal(la, lb)
    assert all((torch.allclose(grad[n], p.grad, atol=1e-06, rtol=1e-05) for n, p in a.named_parameters() if n in grad))
    assert all((p.grad is None for module in (a.node_pieda_decoder, a.edge_decoder_trunk, a.component_heads) for p in module.parameters()))
    batch.y = original
    b.eval()
    values, _ = components(b, batch)
    values[4:9].mean().backward()
    assert has_grad(b.node_gates[-1]) and has_grad(b.pair_messages[-1]) and has_grad(b.node_encoder)
    b.zero_grad(set_to_none=True)
    values, _ = components(b, batch)
    values[9:14].mean().backward()
    assert has_grad(b.node_encoder) and has_grad(b.pair_updates) and has_grad(b.node_gates[0])
    assert not has_grad(b.node_gates[-1])
    for condition in ('geo', 'geo_pieda'):
        model = make_model(condition)
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0.0001)
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        torch.manual_seed(123)
        optimizer.zero_grad(set_to_none=True)
        v, _ = components(model, batch)
        assert torch.isfinite(v).all()
        rec = record(v.detach().cpu().numpy(), condition)
        assert np.isfinite(rec['total'])
        prior.reference.pretty.print_epoch(1, 2, rec, None, 1.0, 1.0)
        base.write_json(out / f'{condition}_record_check.json', rec)
        total_loss(v, condition).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        ck = out / f'{condition}_roundtrip.pt'
        torch.save(dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict()), ck)
        second = make_model(condition)
        second.load_state_dict(base.load(ck)['model_state'])
        second.train()
        second_opt = torch.optim.AdamW([p for p in second.parameters() if p.requires_grad], lr=0.001, weight_decay=0.0001)
        second_opt.load_state_dict(base.load(ck)['optimizer_state'])
        for mm, oo in ((model, optimizer), (second, second_opt)):
            torch.manual_seed(456)
            oo.zero_grad(set_to_none=True)
            v, _ = components(mm, batch)
            total_loss(v, condition).backward()
            nn.utils.clip_grad_norm_(mm.parameters(), 5.0, error_if_nonfinite=True)
            oo.step()
        assert all((torch.allclose(t, second.state_dict()[n], atol=1e-06, rtol=1e-05) for n, t in model.state_dict().items()))
        assert all((torch.equal(p, frozen[n]) for n, p in model.named_parameters() if n in frozen))
    return dict(equal_task_weight_verified=True, initial_sha256=initial, mask_sha256=digests[0], matched_initialization=True, fmo_q_zero=True, bidirectional_geometry_mask=True, geo_invariant_to_pieda_teacher=True, node_pieda_reaches_final_gate=True, pair_pieda_reaches_previous_nodes=True, checkpoint_continuation_verified=True)

def train(model, graphs, condition, out, config):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0.0001)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    timings = []
    steps = 0
    print('GEO(node)=1 GEO(pair)=1 PIEDA(node)=PIEDA(pair)=' + ('1' if condition == 'geo_pieda' else '0 (diagnostic only)'), flush=True)
    for ep in range(1, 301):
        started = time.monotonic()
        model.train()
        order = np.random.default_rng(base.seed_for('order', 42, ep)).permutation(len(graphs))
        mh = hashlib.sha256()
        sums = np.zeros(14)
        count = 0
        for bi, start in enumerate(range(0, len(order), 8)):
            batch = base.Batch.from_data_list([prior.masked(graphs[int(i)], ep, mh) for i in order[start:start + 8]]).cuda()
            torch.manual_seed(base.seed_for('dropout', 42, ep, bi))
            opt.zero_grad(set_to_none=True)
            v, n = components(model, batch)
            loss = total_loss(v, condition)
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            opt.step()
            sums += v.detach().cpu().numpy() * n
            count += n
            steps += 1
        elapsed = time.monotonic() - started
        timings.append(elapsed)
        tr = record(sums / count, condition)
        base.json_line(out / 'training.jsonl', dict(epoch=ep, steps=steps, train=tr, valid=None, seconds=elapsed, mask_sha256=mh.hexdigest()))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            prior.reference.pretty.print_epoch(ep, 300, tr, None, elapsed, np.mean(timings[-10:]) * (300 - ep))
        print(buf.getvalue().replace('evaluated every 5 epochs', 'not performed: all subsets are Train'), end='', flush=True)
        print(f"  GEO(node)={tr['node_geometry']:.6f} GEO(pair)={tr['pair_geometry']:.6f} pair DIST={tr['pair_geometry_components'][0]:.6f} centroid={tr['pair_geometry_components'][1]:.6f}", flush=True)
        for branch in ('node', 'pair'):
            values = ' '.join((f'{k}={v:.6f}' for k, v in zip(('sum', 'ES', 'EX', 'CTmix', 'DI'), tr[f'{branch}_pieda_components'])))
            print(f"  PIEDA({branch})={tr[f'{branch}_pieda']:.6f} {values}", flush=True)
        if ep % 25 == 0 or ep == 300:
            torch.save(dict(model_state=model.state_dict(), optimizer_state=opt.state_dict(), config=config, epoch=ep, steps=steps), out / f'epoch_{ep:03d}.pt')
    assert all((torch.equal(p, frozen[n]) for n, p in model.named_parameters() if n in frozen))
    model.eval().requires_grad_(False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('smoke', 'final'), required=True)
    p.add_argument('--condition', choices=('geo', 'geo_pieda'), default='geo')
    args = p.parse_args()
    assert True and Path.cwd() == ROOT and torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    mp = PREP / 'final/manifest.json'
    prepared = json.loads((PREP / 'final/PREPARED.json').read_text())
    assert prepared['files_sha256']['manifest.json'] == cv.sha(mp)
    mf = json.loads(mp.read_text())
    records = [r for r in mf['records'] if r['split'] == 'train']
    assert len(records) == 8085 and all((r['subset'] != 'test' for r in records))
    hashes = source_hashes()
    out = RUNS / ('smoke_r2' if args.mode == 'smoke' else f'final/{args.condition}')
    if args.mode == 'final':
        gate = json.loads((RUNS / 'smoke_r2/COMPLETE.json').read_text())
        assert gate['complete'] and gate['source_hashes'] == hashes and (gate['manifest_sha256'] == cv.sha(mp))
        approval = json.loads((RUNS / 'APPROVAL.json').read_text())
        assert approval['training_authorized'] and approval['manifest_sha256'] == cv.sha(mp)
        assert approval['runner_sha256'] == cv.sha(__file__)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'code').mkdir()
    for name in hashes:
        src = Path(__file__) if name == Path(__file__).name else ROOT / name
        shutil.copy2(src, out / 'code' / name)
    shutil.copy2(mp, out / 'train_manifest.json')
    if args.mode == 'smoke':
        records = sorted(records, key=lambda r: r['edges'], reverse=True)[:8]
    graphs = []
    for i, r in enumerate(records, 1):
        graphs.append(cv.load(r, mf, 'v6'))
        if i % 500 == 0:
            print('Preload', i, '/', len(records), flush=True)
    if args.mode == 'smoke':
        audit = smoke(graphs, out)
        assert source_hashes() == hashes
        base.write_json(out / 'COMPLETE.json', dict(complete=True, checks=audit, source_hashes=hashes, manifest_sha256=cv.sha(mp)))
        print('SMOKE PASSED', flush=True)
        return
    model = make_model(args.condition)
    config = dict(condition=args.condition, design='B', model_number=1, version='v6', seed=42, epochs=300, batch_size=8, optimizer='AdamW', lr=0.001, weight_decay=0.0001, grad_clip=5.0, scheduler=None, loss_weights=dict(node_distance=1.0, pair_distance=1.0, node_pieda=1.0 if args.condition == 'geo_pieda' else 0, pair_pieda=1.0 if args.condition == 'geo_pieda' else 0), node_pieda_head='512->256(LayerNorm/SiLU/dropout0.1)->128(SiLU)->5', train_graphs=len(graphs), test_graphs=sum((r['split'] == 'test' for r in mf['records'])), test_loaded=False, train_stats=mf['train_stats'], manifest_sha256=cv.sha(mp), source_hashes=hashes, checkpoint_selection='fixed epoch 300; no validation/test selection', initial_sha256=base.digest_state(model.state_dict()), parameters=sum((p.numel() for p in model.parameters())), updated_parameters=sum((p.numel() for p in model.parameters() if p.requires_grad)))
    base.write_json(out / 'config.json', config)
    train(model, graphs, args.condition, out, config)
    ck = out / 'epoch_300.pt'
    assert base.digest_state(base.load(ck)['model_state']) == base.digest_state(model.state_dict())
    assert source_hashes() == hashes
    base.write_json(out / 'COMPLETE.json', dict(complete=True, checkpoint_sha256=cv.sha(ck), source_hashes=hashes, train_only=True))
    print('FINAL TRAINING COMPLETE', flush=True)
if __name__ == '__main__':
    main()
