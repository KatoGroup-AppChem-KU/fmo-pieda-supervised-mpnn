"""FPv2 C: 4 configurations, encoder CV and final, frozen downstream evaluation.

Existing C objective, fitters, masking and optimizer are reused without changes.
CV encoder uses validation-best as the historical CV series; final uses epoch300.
No nested head CV for a CV encoder: its held-out subset selects the probe.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch import nn
import fpv2_sg_b_20261016 as fp
import fpv2_full_pieda_20261017 as ph
import fpv2_full_px_20261017 as px
ROOT = fp.ROOT
SERIES = ROOT / 'results/fpv2_C_full_20261017'
NUMBER = 1
VERSION = 'v6'
BASE_HASH = fp.source_hashes
SOURCE_PATHS = None

def sources():
    return {str(p): fp.cv.sha(p) for p in SOURCE_PATHS}

def make_model(condition):
    m = fp.gp.make_model(NUMBER, VERSION)
    torch.manual_seed(fp.base.seed_for('node_pieda_head', 42))
    m.node_pieda_decoder = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 5)).cuda()
    for module in (m.edge_decoder_trunk, m.component_heads, m.node_pieda_decoder):
        module.requires_grad_(condition == 'geo_pieda')
    return m

def manifest(fold):
    path = fp.PREP / (f'cv/fold{fold}.json' if fold else 'final/manifest.json')
    prepared = json.loads((path.parent / 'PREPARED.json').read_text())
    assert fp.cv.sha(path) == prepared['files_sha256'][path.name]
    mf = json.loads(path.read_text())
    assert len(mf['records']) == 8983
    sets = [{r['pdb'] for r in mf['records'] if r['split'] == sp} for sp in ('train', 'valid', 'test')]
    assert not sets[0] & sets[1] and (not sets[0] & sets[2]) and (not sets[1] & sets[2])
    assert sum((r['split'] == 'test' for r in mf['records'])) == 898
    assert all((r['subset'] != f'subset{fold}' for r in mf['records'] if r['split'] == 'train'))
    return (path, mf)

def smoke():
    out = SERIES / 'smoke' / f'model{NUMBER}_{VERSION}'
    out.mkdir(parents=True, exist_ok=False)
    path, mf = manifest(1)
    records = sorted([r for r in mf['records'] if r['split'] == 'train'], key=lambda r: r['edges'], reverse=True)[:8]
    graphs = [fp.cv.load(r, mf, VERSION) for r in records]
    checks = fp.smoke(graphs, out)
    if NUMBER == 1 and VERSION == 'v6':
        old = json.loads((ROOT / 'results/fpv2_sg_b_unitloss_20261016/final/geo_pieda/config.json').read_text())
        assert checks['initial_sha256'] == old['initial_sha256']
    m = make_model('geo_pieda')
    if NUMBER == 2:
        p = torch.randn(3, 512, device='cuda', requires_grad=True)
        m.pair_state = p
        prediction = m.decode_edges(None, None, None, torch.arange(3, device='cuda'))
        for k in range(4):
            grad = torch.autograd.grad(prediction[:, k + 1].sum(), p, retain_graph=True)[0]
            assert torch.count_nonzero(grad * (1 - m.component_masks[k])) == 0 and grad.abs().sum() > 0
    del m
    gc.collect()
    torch.cuda.empty_cache()
    records = [r for sp in ('train', 'valid', 'test') for src in ('bdb', 'scop2') for r in [r for r in mf['records'] if r['split'] == sp and r['source'] == src][:2]]
    gs = {sp: [fp.cv.load(r, mf, VERSION) for r in records if r['split'] == sp] for sp in ('train', 'valid', 'test')}
    model = make_model('geo_pieda').eval().requires_grad_(False)
    pieda_probe(model, gs, out / 'probe_smoke', True)
    affinity(model, mf, out / 'affinity_smoke', True)
    fp.base.write_json(out / 'COMPLETE.json', dict(complete=True, checks=checks, source_hashes=sources(), manifest_sha256=fp.cv.sha(path)))
    print('SMOKE COMPLETE', NUMBER, VERSION, flush=True)

def print_branches(tr, va):
    for label, r in [('Train', tr), ('Valid', va)]:
        if r is None:
            continue
        print(f"  {label} GEO(node)={r['node_geometry']:.6f} GEO(pair)={r['pair_geometry']:.6f}", flush=True)
        for branch in ('pair_geometry', 'node_pieda', 'pair_pieda'):
            names = ('DIST', 'centroid') if branch == 'pair_geometry' else ('sum', 'ES', 'EX', 'CTmix', 'DI')
            print('  ' + label + ' ' + branch + ' ' + ' '.join((f'{k}={v:.6f}' for k, v in zip(names, r[branch + '_components']))), flush=True)

def train_cv(model, graphs, condition, out, config):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0.0001)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    times = []
    steps = 0
    best = float('inf')
    print('C: node distance=pair distance=1; node PIEDA=pair PIEDA=' + str(int(condition == 'geo_pieda')), flush=True)
    for ep in range(1, 301):
        begin = time.monotonic()
        model.train()
        sums = np.zeros(14)
        count = 0
        mh = hashlib.sha256()
        order = np.random.default_rng(fp.base.seed_for('order', 42, ep)).permutation(len(graphs['train']))
        for bi, start in enumerate(range(0, len(order), 8)):
            b = fp.base.Batch.from_data_list([fp.prior.masked(graphs['train'][int(i)], ep, mh) for i in order[start:start + 8]]).cuda()
            torch.manual_seed(fp.base.seed_for('dropout', 42, ep, bi))
            opt.zero_grad(set_to_none=True)
            v, n = fp.components(model, b)
            loss = fp.total_loss(v, condition)
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            opt.step()
            sums += v.detach().cpu().numpy() * n
            count += n
            steps += 1
        va = None
        if ep % 5 == 0:
            model.eval()
            vs = np.zeros(14)
            vn = 0
            with torch.no_grad():
                for start in range(0, len(graphs['valid']), 8):
                    b = fp.base.Batch.from_data_list([fp.prior.masked(g, 'valid12345') for g in graphs['valid'][start:start + 8]]).cuda()
                    v, n = fp.components(model, b)
                    vs += v.cpu().numpy() * n
                    vn += n
            va = fp.record(vs / vn, condition)
            if va['total'] < best:
                best = va['total']
                torch.save(dict(model_state=model.state_dict(), optimizer_state=opt.state_dict(), config=config, epoch=ep, steps=steps, valid_loss=best), out / 'best_model.pt')
        elapsed = time.monotonic() - begin
        times.append(elapsed)
        tr = fp.record(sums / count, condition)
        fp.base.json_line(out / 'training.jsonl', dict(epoch=ep, steps=steps, train=tr, valid=va, seconds=elapsed, mask_sha256=mh.hexdigest()))
        fp.prior.reference.pretty.print_epoch(ep, 300, tr, va, elapsed, np.mean(times[-10:]) * (300 - ep))
        print_branches(tr, va)
        if ep % 25 == 0:
            torch.save(dict(model_state=model.state_dict(), optimizer_state=opt.state_dict(), config=config, epoch=ep, steps=steps), out / f'epoch_{ep:03d}.pt')
    assert all((torch.equal(p, frozen[n]) for n, p in model.named_parameters() if n in frozen))
    model.eval().requires_grad_(False)

def pieda_probe(model, graphs, out, smoke=False):
    before = fp.base.digest_state(model.state_dict())
    out.mkdir(parents=True)
    for kind in ('node', 'pair'):
        dest = out / (kind + '_raw_probe')
        dest.mkdir()
        extractor = fp.cv.prior.ORIGINAL_EXTRACT if kind == 'node' else fp.cv.extract_pair
        factory = fp.cv.prior.ORIGINAL_PROBE if kind == 'node' or NUMBER == 2 else fp.cv.prior.PairProbe
        data = {sp: extractor(model, graphs[sp], sp, dest) for sp in ('train', 'valid', 'test')}
        m, scale, result = ph.fit(factory, data['train'], data['valid'], dest, epochs=2 if smoke else 80)
        ph.save_test(m, scale, data['test'], dest)
        del data, m, scale
        gc.collect()
        torch.cuda.empty_cache()
    assert before == fp.base.digest_state(model.state_dict())

def affinity(model, mf, out, smoke=False):
    out.mkdir(parents=True)
    df = pd.read_csv(px.LABEL).sort_values('pdb_id').reset_index(drop=True)
    assert len(df) == 3135 and df.pdb_id.is_unique
    records = {r['pdb']: r for r in mf['records'] if r['source'] == 'bdb'}
    df['pdbid'] = df.pdb_id
    df['split_pdbid'] = [records[p]['split'] for p in df.pdbid]
    assert all((records[r.pdbid]['subset'] == r.subset for r in df.itertuples()))
    if smoke:
        df = pd.concat([df[df.split_pdbid == sp].head(8) for sp in ('train', 'valid', 'test')]).reset_index(drop=True)
    df.to_csv(out / 'cohort.csv', index=False)
    before = fp.base.digest_state(model.state_dict())
    vectors = {k: [] for k in px.KINDS}
    with torch.inference_mode():
        for i, row in enumerate(df.itertuples(), 1):
            rec = records[row.pdbid]
            g = fp.base.load(rec['paths'][VERSION])
            assert fp.base.fingerprint(g) == rec['fingerprints'][VERSION]
            geom, _, fmo, names = fp.base.pm.infer_feature_groups(g)
            assert (len(geom), len(fmo), len(names)) == (66, 6, 72)
            ea = g.edge_attr.clone().float()
            ea[:, fmo] = 0
            ei = g.edge_index.long()
            lig = g.is_ligand.bool().reshape(-1)
            assert lig.sum() == 1
            selected = lig.clone()
            selected[ei[:, lig[ei[0]] | lig[ei[1]]].reshape(-1)] = True
            edges = torch.where(lig[ei[0]] & ~lig[ei[1]])[0]
            assert torch.equal(torch.sort(ei[1, edges]).values, torch.where(selected & ~lig)[0]) and len(edges) > 0
            h = model.encode_nodes(g.x.float().cuda(), ei.cuda(), ea.cuda())
            node = h[selected.cuda()].mean(0).cpu()
            pair = model.pair_state[edges.cuda()].mean(0).cpu()
            assert node.shape == (256,) and pair.shape == (256 if NUMBER == 1 else 512,)
            for k, v in [('node', node), ('pair', pair), ('concat', torch.cat([node, pair]))]:
                assert torch.isfinite(v).all()
                vectors[k].append(v)
            if i % 500 == 0:
                print('Affinity extraction', i, len(df), flush=True)
    assert before == fp.base.digest_state(model.state_dict())
    for kind in px.KINDS:
        x = torch.stack(vectors[kind])
        torch.save(dict(embeddings=x, pdbids=df.pdbid.tolist()), out / (kind + '.pt'))
        px.fit(df, dict(zip(df.pdbid, x.numpy())), x.shape[1], out / kind, epochs=2 if smoke else 500)
    fp.base.write_json(out / 'COMPLETE.json', dict(complete=True, head_seed=42, encoder_cv=True, nested_cv=False, label_sha256=fp.cv.sha(px.LABEL)))

def run(fold, condition):
    gate = json.loads((SERIES / 'smoke' / f'model{NUMBER}_{VERSION}' / 'COMPLETE.json').read_text())
    assert gate['complete'] and gate['source_hashes'] == sources()
    path, mf = manifest(fold)
    out = fp.RUNS / (f'cv/fold{fold}/{condition}' if fold else f'final/{condition}')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'code').mkdir()
    hashes = sources()
    for p in SOURCE_PATHS:
        shutil.copy2(p, out / 'code' / p.name)
    shutil.copy2(path, out / 'train_manifest.json')
    graphs = {sp: [] for sp in ('train', 'valid', 'test')}
    for i, r in enumerate([r for r in mf['records'] if r['split'] != 'test'], 1):
        graphs[r['split']].append(fp.cv.load(r, mf, VERSION))
        if i % 500 == 0:
            print('Preload', i, flush=True)
    model = make_model(condition)
    initial = fp.base.digest_state(model.state_dict())
    config = dict(model_number=NUMBER, version=VERSION, condition=condition, design='C', fold=fold, seed=42, epochs=300, batch_size=8, optimizer='AdamW', lr=0.001, weight_decay=0.0001, grad_clip=5.0, scheduler=None, loss_weights=dict(node_distance=1, pair_distance=1, node_pieda=int(condition == 'geo_pieda'), pair_pieda=int(condition == 'geo_pieda')), train_graphs=len(graphs['train']), test_loaded=False, train_stats=mf['train_stats'], manifest_sha256=fp.cv.sha(path), checkpoint_selection='minimum validation total loss' if fold else 'fixed epoch300', initial_sha256=initial, source_hashes=hashes, parameters=sum((p.numel() for p in model.parameters())), inputs='all FMO/q zero; selected pair geometry masked in both directions')
    fp.base.write_json(out / 'config.json', config)
    if fold:
        train_cv(model, graphs, condition, out, config)
    else:
        fp.train(model, graphs['train'], condition, out, config)
    ckpath = out / ('best_model.pt' if fold else 'epoch_300.pt')
    ck = fp.base.load(ckpath)
    model.load_state_dict(ck['model_state'], strict=True)
    model.eval().requires_grad_(False)
    fp.base.write_json(out / 'selected_checkpoint.json', dict(path=str(ckpath), sha256=fp.cv.sha(ckpath), epoch=ck['epoch']))
    del ck
    fp.base.write_json(out / 'COMPLETE.json', dict(complete=True, checkpoint_sha256=fp.cv.sha(ckpath), source_hashes=hashes, train_only=True))
    graphs['test'] = [fp.cv.load(r, mf, VERSION) for r in mf['records'] if r['split'] == 'test']
    fp.cv.reconstruction(model, graphs, mf, out, 'R1' if condition == 'geo_pieda' else 'R0')
    fp.gp.pair_distance_reconstruction(model, graphs, mf, out)
    if fold:
        pieda_probe(model, graphs, out / 'pieda')
        del graphs
        gc.collect()
        torch.cuda.empty_cache()
        affinity(model, mf, out / 'px')
    else:
        del model, graphs
        gc.collect()
        torch.cuda.empty_cache()
        os.environ['PIEDA_HEADCV_ROOT'] = str(fp.RUNS / 'pieda')
        sys.argv = [ph.__file__, '--task', str(int(condition == 'geo_pieda'))]
        ph.main()
        px.run(SimpleNamespace(task=int(condition == 'geo_pieda'), smoke=False), fp.RUNS / 'px')
    assert hashes == sources()
    fp.base.write_json(out / 'EVALUATION_COMPLETE.json', dict(complete=True, encoder_fold=fold, source_hashes=hashes))

def main():
    global NUMBER, VERSION, SOURCE_PATHS
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['smoke', 'run', 'aggregate'], required=True)
    ap.add_argument('--task', type=int, required=True)
    a = ap.parse_args()
    assert True and Path.cwd() == ROOT
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    combos = [(1, 'v6'), (1, 'v7'), (2, 'v6'), (2, 'v7')]
    tasks = [(n, v, k, c) for n, v in combos for k in range(1, 6) for c in ('geo', 'geo_pieda')]
    tasks += [(n, v, 0, c) for n, v in combos[1:] for c in ('geo', 'geo_pieda')]
    if a.stage == 'run':
        NUMBER, VERSION, fold, condition = tasks[a.task]
    else:
        NUMBER, VERSION = combos[a.task]
    fp.RUNS = SERIES / f'model{NUMBER}_{VERSION}'
    fp.make_model = make_model
    os.environ['FPV2_MODEL'] = str(NUMBER)
    os.environ['FPV2_VERSION'] = VERSION
    SOURCE_PATHS = sorted({ROOT / n if (ROOT / n).is_file() else ROOT / 'outputs' / n for n in BASE_HASH()} | {Path(__file__).resolve(), Path(ph.__file__).resolve(), Path(px.__file__).resolve(), ROOT / 'outputs/aggregate_fpv2_full_pieda_20261017.py', Path(px.fit.__code__.co_filename).resolve(), ROOT / 'outputs/final_pieda_ci_20261012.py', ROOT / 'outputs/bootstrap_px_selected_20261012.py'})
    fp.source_hashes = sources
    if a.stage == 'smoke':
        smoke()
    elif a.stage == 'run':
        run(fold, condition)
    else:
        assert all(((fp.RUNS / f'cv/fold{k}/{c}/EVALUATION_COMPLETE.json').exists() for k in range(1, 6) for c in ('geo', 'geo_pieda')))
        if a.task != 0:
            sys.argv = ['aggregate', str(fp.RUNS / 'pieda')]
            runpy.run_path(str(ROOT / 'outputs/aggregate_fpv2_full_pieda_20261017.py'), run_name='__main__')
            px.aggregate(fp.RUNS / 'px')
        rows = []
        for k in range(1, 6):
            for c in ('geo', 'geo_pieda'):
                d = fp.RUNS / f'cv/fold{k}/{c}'
                for kind in ('node', 'pair'):
                    for target, metrics in json.loads((d / f'pieda/{kind}_raw_probe/test_metrics.json').read_text()).items():
                        rows.append(dict(task='PIEDA', condition=c, fold=k, representation=kind, target=target, **metrics))
                for kind in px.KINDS:
                    met = json.loads((d / f'px/{kind}/metrics.json').read_text())['test']
                    rows.append(dict(task='affinity', condition=c, fold=k, representation=kind, target='pX', **met))
        df = pd.DataFrame(rows)
        df.to_csv(fp.RUNS / 'encoder_cv_metrics.csv', index=False)
        df.groupby(['task', 'condition', 'representation', 'target'])[['pearson_r', 'rmse', 'mae']].agg(['mean', 'std']).to_csv(fp.RUNS / 'encoder_cv_mean_sd.csv')
        counts = []
        for keys, g in df.groupby(['task', 'representation', 'target']):
            p = g.pivot(index='fold', columns='condition', values='pearson_r')
            counts.append(dict(task=keys[0], representation=keys[1], target=keys[2], improved_folds=int((p.geo_pieda > p.geo).sum()), folds=5))
        pd.DataFrame(counts).to_csv(fp.RUNS / 'encoder_cv_improvements.csv', index=False)
        fp.base.write_json(fp.RUNS / 'COMPLETE.json', dict(complete=True, source_hashes=sources(), existing_SG_final_reused=a.task == 0))
if __name__ == '__main__':
    main()
