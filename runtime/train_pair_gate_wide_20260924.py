"""BDB v6 300 epochs: reuse established pretrain, mask and frozen raw probes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
import numpy as np
from torch import nn
import train_pair_residual_bdb as prior
import pair_gate_wide_20260924 as architecture
base = prior.base
ORIGINAL_COMPONENTS = prior.components

def components(model, batch):
    values, n = ORIGINAL_COMPONENTS(model, batch)
    return (torch.cat([values[:7], torch.zeros_like(values[7:])]), n)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(row, manifest):
    graph = base.load_graph(row, 'v6', manifest)
    graph['residual_z'] = torch.zeros_like(graph['raw'])
    return graph

def has_grad(module):
    return any((p.grad is not None and torch.isfinite(p.grad).all() and (p.grad.abs().sum() > 0) for p in module.parameters()))

def checks(graphs, stage):
    factory = base.make_model
    base.make_model = lambda version, condition: architecture.make_model('R0' if condition == 'geo' else 'R1', stage)
    try:
        audit = base.smoke_checks(graphs, 'v6')
    finally:
        base.make_model = factory
    batch = base.Batch.from_data_list([prior.masked(graphs['train'][0], 1)]).cuda()
    for condition in ('R0', 'R1'):
        model = architecture.make_model(condition, stage)
        values, _ = components(model, batch)
        prior.total_loss(values, condition).backward()
        for name in ('node_encoder', 'pair_init', 'pair_updates', 'pair_messages', 'node_gates'):
            assert has_grad(getattr(model, name)), (condition, name)
        if stage >= 2:
            for layer in model.component_projections:
                assert all((has_grad(branch) for branch in layer))
        assert all((p.grad is None for p in model.residual_decoder.parameters()))
        del model
    model = architecture.make_model('R1', stage).eval()
    values, _ = components(model, batch)
    values[2:7].mean().backward()
    assert has_grad(model.node_gates), 'PIEDA must reach pair->node gates'
    assert has_grad(model.pair_messages), 'PIEDA must reach pair->node messages'
    with torch.no_grad():
        original = model.encode_nodes(batch.x, batch.edge_index, batch.edge_attr).clone()
        handles = [layer.register_forward_hook(lambda mod, inp, out: out * 0) for gates in model.node_gates for layer in gates]
        changed = model.encode_nodes(batch.x, batch.edge_index, batch.edge_attr)
        for handle in handles:
            handle.remove()
        assert not torch.allclose(original, changed), 'gates do not affect node states'
    if stage >= 2:
        p = torch.randn(3, 512, device='cuda', requires_grad=True)
        model.pair_state = p
        pred = model.decode_edges(None, None, None, torch.arange(3, device='cuda'))
        for k in range(4):
            grad = torch.autograd.grad(pred[:, k + 1].sum(), p, retain_graph=True)[0]
            outside = grad * (1 - model.component_masks[k])
            assert torch.count_nonzero(outside) == 0
            assert grad.abs().sum() > 0
        if stage == 3:
            comparison = architecture.make_model('R1', 2).eval()
            state = comparison.state_dict()
            other = model.state_dict()
            assert all((torch.equal(value, other[name]) for name, value in state.items()))
            with torch.no_grad():
                a = comparison.encode_nodes(batch.x, batch.edge_index, batch.edge_attr)
                b = model.encode_nodes(batch.x, batch.edge_index, batch.edge_attr)
                assert torch.allclose(a, b, atol=0.0002, rtol=0.0002), 'equal gates differ at initialization'
            del comparison
    del model, batch
    torch.cuda.empty_cache()
    audit.update(gate_gradient=True, pieda_reaches_feedback=True, component_routing=stage >= 2, stage3_equal_gate_equivalence=stage == 3)
    return audit

class WidePairProbe(prior.ORIGINAL_PROBE):

    def __init__(self):
        super().__init__()
        assert self.trunk[0].in_features == 512

def extract_pair(model, graphs, label, out):
    selected = [base.probe_selection(g) for g in graphs]
    n = sum((len(s) for s in selected))
    cap = {'train': 800000, 'valid': 150000, 'test': n}[label]
    kept = np.sort(np.random.default_rng(base.seed_for('probe_cap', 123, label)).choice(n, min(n, cap), replace=False))
    x, y = (torch.empty((len(kept), 512)), torch.empty((len(kept), 5)))
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, choices=[2, 3], required=True)
    ap.add_argument('--condition', choices=['R0', 'R1'], required=True)
    ap.add_argument('--mode', choices=['smoke', 'full'], required=True)
    args = ap.parse_args()
    assert True
    assert True and torch.cuda.is_available()
    os.chdir(prior.ROOT)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    prior.components = components
    reference = json.loads(Path('results/pair_residual_bdb_full/R0/config.json').read_text())
    for path, digest in reference['source_hashes'].items():
        assert sha(path) == digest, path
    hashes = dict(reference['source_hashes'], **{name: sha(name) for name in [Path(__file__).name, Path(architecture.__file__).name]})
    records, manifest, manifest_hash = prior.records_manifest('full')
    assert manifest_hash == reference['manifest_sha256']
    out = Path(f'results/pair_gate_wide128_bdb_v6_20260924_{args.mode}/S{args.stage}_{args.condition}')
    out.mkdir(parents=True, exist_ok=False)
    if args.mode == 'full':
        smoke = Path(f'results/pair_gate_wide128_bdb_v6_20260924_smoke/S{args.stage}_{args.condition}')
        completion = json.loads((smoke / 'COMPLETE.json').read_text())
        assert completion['complete'] and completion['source_hashes'] == hashes
    else:
        largest = sorted([r for r in records if r['split'] == 'train'], key=lambda r: r['edges'], reverse=True)[:8]
        batch = base.Batch.from_data_list([prior.masked(load(r, manifest), 1) for r in largest]).cuda()
        model = architecture.make_model(args.condition, args.stage)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0.0001)
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            values, _ = components(model, batch)
            loss = prior.total_loss(values, args.condition)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
        torch.cuda.synchronize()
        base.write_json(out / 'resource.json', dict(batch_size=8, keys=[r['key'] for r in largest], peak_GiB=torch.cuda.max_memory_allocated() / 2 ** 30))
        print(f'S{args.stage} {args.condition}: largest batch8 resource check passed', flush=True)
        del model, optimizer, batch, values, loss
        torch.cuda.empty_cache()
        records, _, _ = prior.records_manifest('smoke')
    graphs = {s: [] for s in ('train', 'valid', 'test')}
    for i, row in enumerate(records, 1):
        graphs[row['split']].append(load(row, manifest))
        if i % 500 == 0:
            print(f'preload {i}/{len(records)}', flush=True)
    if args.mode == 'smoke':
        base.write_json(out / 'checks.json', checks(graphs, args.stage))
    model = architecture.make_model(args.condition, args.stage)
    config = dict(reference, condition=args.condition, stage=args.stage, epochs=2 if args.mode == 'smoke' else 300, dataset='BindingDB-only', version='v6', node_update='LN(h + dropout(sigmoid(Linear[h,m])*m)); no GRU', gate_bias=-2.0, gate_weight_init='zero', gate_input='node and aggregated message', component_latent='ES/EX/CTmix/DI 128 each (pair total 512)' if args.stage >= 2 else 'shared 256', component_update='shared pair update -> Linear256x128/SiLU per component -> residual LN128' if args.stage >= 2 else 'baseline shared pair update', gate_mode='per-component before sum' if args.stage == 3 else 'shared after sum', pieda_head='expanded input 512->256->128 trunk and five Linear outputs; ' + ('component input masked to matching 128; sum uses all 512' if args.stage >= 2 else 'all outputs use full 256'), initial_sha256=base.digest_state(model.state_dict()), parameters=sum((p.numel() for p in model.parameters())), updated_parameters=sum((p.numel() for p in model.parameters() if p.requires_grad)), source_hashes=hashes, loss_weights=[1, int(args.condition == 'R1'), 0], residual_normalization=None, residual_manifest_sha256=None, residual_note='disabled, logged zero; no residual teacher used')
    base.write_json(out / 'config.json', config)
    print(f"Stage {args.stage} {args.condition}: parameters={config['parameters']:,}; 300 epochs in full mode; residual disabled", flush=True)
    prior.pretrain(model, graphs, args.condition, out, config, config['epochs'])
    checkpoint = base.load(out / f"epoch_{config['epochs']:03d}.pt")
    restored = architecture.make_model(args.condition, args.stage)
    restored.load_state_dict(checkpoint['model_state'], strict=True)
    assert base.digest_state(restored.state_dict()) == base.digest_state(model.state_dict())
    del checkpoint, restored
    prior.reconstruction(model, graphs, manifest, out)
    for name in ('node', 'pair'):
        folder = out / f'{name}_raw_probe'
        folder.mkdir()
        base.extract = prior.ORIGINAL_EXTRACT if name == 'node' else extract_pair
        base.Probe = prior.ORIGINAL_PROBE if name == 'node' else WidePairProbe
        try:
            prior.screen.run_bdb_probe(model, graphs, args, folder)
        finally:
            base.extract, base.Probe = (prior.ORIGINAL_EXTRACT, prior.ORIGINAL_PROBE)
    for split in graphs:
        node = json.loads((out / 'node_raw_probe' / f'{split}_probe_audit.json').read_text())
        pair = json.loads((out / 'pair_raw_probe' / f'{split}_probe_audit.json').read_text())
        assert node == pair
    base.write_json(out / 'COMPLETE.json', dict(complete=True, stage=args.stage, condition=args.condition, epochs=config['epochs'], source_hashes=hashes, node_pair_raw_probes=True))
    print(f'ALL_COMPLETE {out}', flush=True)
if __name__ == '__main__':
    main()
