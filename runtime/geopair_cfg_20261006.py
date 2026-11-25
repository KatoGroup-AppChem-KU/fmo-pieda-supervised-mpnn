"""GEO + pair-distance control (R3), generalized to FMO-Net-SG/CG x ESM-FP/Unified-FP (copy of geopair_20261005).

Same encoder, masks, seeds and optimizer as cv5_20260925 GEO (R0).  One extra head predicts
the two normalized distances (closest, centroid) of the masked pairs from the final pair state,
so that the pair representation receives direct supervision without PIEDA.
Loss = mean(node-distance losses) + mean(pair-distance losses); PIEDA heads stay frozen as in R0.
"""
import hashlib
import time
import numpy as np
import torch
from torch import nn
import train_pair_residual_bdb as prior
import cv5_loops_20260925 as loops
import run_cv5_20260925 as cv5
base, reference = (prior.base, prior.reference)
HEAD = 'pair_distance_decoder'

def make_model(number, version):
    m = cv5.make_model(number, version, 'R0')
    pair_dim = 256 if number == 1 else 512
    torch.manual_seed(base.seed_for('pair_distance_head', 42))
    setattr(m, HEAD, nn.Sequential(nn.Linear(pair_dim, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 2)).cuda())
    return m

def shared_digest(model):
    return base.digest_state({k: v for k, v in model.state_dict().items() if not k.startswith(HEAD + '.')})

def components(model, b):
    v, n = loops.components(model, b)
    sel = b.decode_mask.nonzero().flatten()
    pair = getattr(model, HEAD)(model.pair_state[sel])
    return (torch.cat([v, (pair - b.y[sel][:, :2]).square().mean(0)]), n)

def total_loss(v, condition=None):
    return v[:2].mean() + v[12:14].mean()

def record(v, condition=None):
    r = prior.record(v[:12], 'R0')
    r['pair_DIST'], r['pair_centroid'] = (float(v[12]), float(v[13]))
    r['pair_geometry'] = float(np.mean(v[12:14]))
    r['total'] += r['pair_geometry']
    return r

def pretrain(model, graphs, out, config, epochs):
    """loops.pretrain with the R3 loss/record; optimizer, batching, seeds and checkpointing unchanged."""
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0.0001)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    timings = []
    steps = 0
    best = float('inf')
    print('Loss weights: GEO(node)=1 GEO(pair)=1 PIEDA=0 residual=0; weight-0 losses are diagnostic only', flush=True)
    for ep in range(1, epochs + 1):
        started = time.monotonic()
        model.train()
        order = np.random.default_rng(base.seed_for('order', 42, ep)).permutation(len(graphs['train']))
        mh = hashlib.sha256()
        sums = np.zeros(14)
        count = 0
        for bi, start in enumerate(range(0, len(order), 8)):
            b = base.Batch.from_data_list([prior.masked(graphs['train'][int(i)], ep, mh) for i in order[start:start + 8]]).cuda()
            torch.manual_seed(base.seed_for('dropout', 42, ep, bi))
            opt.zero_grad(set_to_none=True)
            v, n = components(model, b)
            loss = total_loss(v)
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
            vs = np.zeros(14)
            vn = 0
            with torch.no_grad():
                for start in range(0, len(graphs['valid']), 8):
                    b = base.Batch.from_data_list([prior.masked(g, 'valid12345') for g in graphs['valid'][start:start + 8]]).cuda()
                    v, n = components(model, b)
                    vs += v.cpu().numpy() * n
                    vn += n
            valid = record(vs / vn)
        if valid is not None and valid['total'] < best:
            best = valid['total']
            torch.save(dict(model_state=model.state_dict(), optimizer_state=opt.state_dict(), config=config, epoch=ep, steps=steps, valid_loss=best), out / 'best_model.pt')
        elapsed = time.monotonic() - started
        timings.append(elapsed)
        train = record(sums / count)
        base.json_line(out / 'training.jsonl', dict(epoch=ep, steps=steps, train=train, valid=valid, seconds=elapsed, mask_sha256=mh.hexdigest()))
        reference.pretty.print_epoch(ep, epochs, train, valid, elapsed, np.mean(timings[-10:]) * (epochs - ep))
        for label, r in [('Train', train), ('Valid', valid)]:
            if r is not None:
                print(f"  {label} GEO(pair)={r['pair_geometry']:.6f} (DIST={r['pair_DIST']:.6f}, centroid={r['pair_centroid']:.6f}) | total incl. pair={r['total']:.6f}", flush=True)
        if ep % 25 == 0 or ep == epochs:
            torch.save(dict(model_state=model.state_dict(), optimizer_state=opt.state_dict(), config=config, epoch=ep, steps=steps), out / f'epoch_{ep:03d}.pt')
    base.require(all((torch.equal(p, frozen[n]) for n, p in model.named_parameters() if n in frozen)), 'inactive head changed')
    model.requires_grad_(False)
    model.eval()

def pair_distance_reconstruction(model, graphs, manifest, out):
    """Test-set prediction of the masked distances from the pair head (original-head evaluation analogue)."""
    ys = []
    ps = []
    mean = torch.tensor(manifest['train_stats']['mean'][:2])
    std = torch.tensor(manifest['train_stats']['std'][:2])
    with torch.no_grad():
        for start in range(0, len(graphs['test']), 8):
            b = base.Batch.from_data_list([prior.masked(g, 'test12345') for g in graphs['test'][start:start + 8]]).cuda()
            model.encode_nodes(b.x, b.edge_index, b.edge_attr)
            sel = b.decode_mask.nonzero().flatten()
            p = getattr(model, HEAD)(model.pair_state[sel])
            ys.append(b.y[sel, :2].cpu() * std + mean)
            ps.append(p.cpu() * std + mean)
    y, p = (torch.cat(ys).numpy(), torch.cat(ps).numpy())
    assert np.isfinite(p).all()
    base.write_json(out / 'pair_distance_head_test.json', {name: dict(rmse=float(np.sqrt(np.mean((y[:, k] - p[:, k]) ** 2))), pearson_r=float(np.corrcoef(y[:, k], p[:, k])[0, 1])) for k, name in enumerate(('DIST', 'centroid'))})
