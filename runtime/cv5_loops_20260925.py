"""Existing training/probe loops, copied with narrowly scoped CV changes.
Pretrain: add validation-best checkpoint; optimizer/batching/logging unchanged.
Probe: seed123->42 for initialization/order; selection seed123 unchanged.
"""
import copy
import math
import time
import numpy as np
import torch
from torch import nn
import train_pair_residual_bdb as prior
from train_geo_pieda_matched_20260911 import *
base, reference = (prior.base, prior.reference)
masked, total_loss, record = (prior.masked, prior.total_loss, prior.record)

def components(model, batch):
    values, n = prior.components(model, batch)
    return (torch.cat([values[:7], torch.zeros_like(values[7:])]), n)

def pretrain(model, graphs, condition, out, config, epochs):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0.0001)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    timings = []
    steps = 0
    best = float('inf')
    best_epoch = 0
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
        if valid is not None and valid['total'] < best:
            best, best_epoch = (valid['total'], ep)
            torch.save(dict(model_state=model.state_dict(), optimizer_state=opt.state_dict(), config=config, epoch=ep, steps=steps, valid_loss=best), out / 'best_model.pt')
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

def run_probe(model, graphs, args, out):
    encoder_before = digest_state(model.state_dict())
    datasets = {l: extract(model, graphs[l], l, out) for l in ('train', 'valid', 'test')}
    x, y = datasets['train'][:2]
    xm, xs = (x.mean(0), x.std(0, unbiased=False).clamp_min(1e-08))
    ym, ys = (y.mean(0), y.std(0, unbiased=False).clamp_min(1e-08))
    for l, (xx, yy, _, _) in datasets.items():
        xx.sub_(xm).div_(xs)
    yn = (y - ym) / ys
    torch.manual_seed(42)
    probe = Probe().cuda()
    init = digest_state(probe.state_dict())
    opt = torch.optim.Adam(probe.parameters(), lr=0.001)
    best, best_state, best_epoch, stale = (float('inf'), None, 0, 0)
    epochs = 2 if args.mode == 'smoke' else 80
    for epoch in range(1, epochs + 1):
        probe.train()
        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed_for('probe_order', 42, epoch)))
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
    print(f'PROBE_COMPLETE best_epoch={best_epoch} valid={best:.6f} output={out}', flush=True)
