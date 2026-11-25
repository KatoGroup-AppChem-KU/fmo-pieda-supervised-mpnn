"""Visible PIEDA context; isolated adapters for the unchanged FPv2 C loops.

Input PIEDA uses canonical raw values and TRAIN statistics, never graph z-scores.
PIEDA probe targets are hidden in both directions, in <=15%-of-pairs chunks.
Probe geometry stays visible as in the historical frozen downstream task.
"""
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
import fpv2_sg_b_20261016 as fp
ORIGINAL_LOAD = fp.cv.load
ORIGINAL_MASK = fp.prior.masked
ORIGINAL_SMOKE = fp.smoke
ORIGINAL_NODE_EXTRACT = fp.cv.prior.ORIGINAL_EXTRACT
ORIGINAL_PAIR_EXTRACT = fp.cv.extract_pair

def load(r, manifest, version):
    g = ORIGINAL_LOAD(r, manifest, version)
    graph = fp.base.load(r['paths'][version])
    _, targets, fmo, names = fp.base.pm.infer_feature_groups(graph)
    cols = [names.index(n) for n in fp.base.TARGETS]
    assert set(cols) == set(targets) and len(cols) == 5
    q = sorted(set(fmo) - set(cols))
    assert len(q) == 1 and len(names) == 72
    g['ea'][:, cols] = g['y'][:, 2:7]
    assert torch.count_nonzero(g['ea'][:, q]) == 0
    assert torch.isfinite(g['ea']).all()
    g['context_pieda_cols'], g['context_q_cols'] = (cols, q)
    return g

def edge_attr(graph, rec, manifest, version):
    """Affinity pathway: all geometry and PIEDA visible, q still zero."""
    g = load(rec, manifest, version)
    assert torch.equal(g['x'], graph.x.float())
    assert torch.equal(g['ei'], graph.edge_index.long())
    return g['ea']

def hide_pairs(g, pair_ids):
    hidden = torch.zeros(g['pairs'], dtype=torch.bool)
    hidden[pair_ids] = True
    sel = hidden[g['inverse']]
    ea = g['ea'].clone()
    ea[sel.nonzero().flatten()[:, None], torch.tensor(g['context_pieda_cols'])] = 0
    return (ea, sel)

def masked(g, epoch, audit=None):
    b = ORIGINAL_MASK(g, epoch, audit)
    if 'context_pieda_cols' in g:
        b.edge_attr[b.decode_mask.nonzero().flatten()[:, None], torch.tensor(g['context_pieda_cols'])] = 0
    return b

def extract(model, graphs, label, out, kind):
    selected = [fp.base.probe_selection(g) for g in graphs]
    n = sum((len(s) for s in selected))
    cap = {'train': 800000, 'valid': 150000, 'test': n}[label]
    kept = np.sort(np.random.default_rng(fp.base.seed_for('probe_cap', 123, label)).choice(n, min(n, cap), replace=False))
    x = torch.empty((len(kept), 512 if kind == 'node' else 256))
    y = torch.empty((len(kept), 5))
    source, gids = (np.empty(len(kept), np.int8), np.empty(len(kept), np.int32))
    offset = position = forwards = 0
    selection_hash, mask_hash = (hashlib.sha256(), hashlib.sha256())
    with torch.no_grad():
        for gi, (g, sel) in enumerate(zip(graphs, selected)):
            end = np.searchsorted(kept, offset + len(sel))
            rows = sel[torch.from_numpy(kept[position:end] - offset)]
            selection_hash.update(g['key'].encode())
            selection_hash.update(rows.numpy().tobytes())
            if len(rows):
                pairs = torch.unique(g['inverse'][rows], sorted=True)
                order = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(fp.base.seed_for('probe_context_chunks', 42, g['key'])))
                pairs = pairs[order]
                chunk_size = max(1, math.ceil(0.15 * g['pairs']))
                assigned = torch.zeros(len(rows), dtype=torch.bool)
                ei, xx = (g['ei'].cuda(), g['x'].cuda())
                for start in range(0, len(pairs), chunk_size):
                    chunk = pairs[start:start + chunk_size]
                    ea, hidden = hide_pairs(g, chunk)
                    pick = hidden[rows]
                    assert not bool((assigned & pick).any())
                    assigned |= pick
                    rr, dest = (rows[pick], position + torch.where(pick)[0])
                    mask_hash.update(g['key'].encode())
                    mask_hash.update(chunk.numpy().tobytes())
                    h = model.encode_nodes(xx, ei, ea.cuda())
                    if kind == 'node':
                        value = torch.cat([h[ei[0, rr.cuda()]], h[ei[1, rr.cuda()]]], 1)
                    else:
                        value = model.pair_state[rr.cuda()]
                    x[dest] = value.cpu()
                    forwards += 1
                assert bool(assigned.all())
                y[position:end] = g['raw'][rows]
                source[position:end] = int(g['source'] == 'scop2')
                gids[position:end] = gi
            position, offset = (end, offset + len(sel))
            if (gi + 1) % 1000 == 0:
                print(f'context {kind} extract {label} {gi + 1}/{len(graphs)}', flush=True)
    assert position == len(kept) and torch.isfinite(x).all() and torch.isfinite(y).all()
    fp.base.write_json(out / (label + '_probe_audit.json'), dict(candidate_edges=n, retained_edges=len(kept), graphs=len(graphs), selection_sha256=selection_hash.hexdigest(), raw_targets_sha256=hashlib.sha256(y.numpy().tobytes()).hexdigest()))
    fp.base.write_json(out / (label + '_context_audit.json'), dict(target_pieda_hidden_both_directions=True, q_zero=True, geometry_retained=True, max_pairs_per_forward='ceil(0.15 * graph unordered pairs)', forwards=forwards, context_mask_sha256=mask_hash.hexdigest()))
    return (x, y, source, gids)

def extract_node(model, graphs, label, out):
    return extract(model, graphs, label, out, 'node')

def extract_pair(model, graphs, label, out):
    return extract(model, graphs, label, out, 'pair')

def smoke(graphs, out):
    plain = []
    for g in graphs:
        d = {k: v for k, v in g.items() if not k.startswith('context_')}
        d['ea'] = g['ea'].clone()
        d['ea'][:, g['fmo']] = 0
        plain.append(d)
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        checks = ORIGINAL_SMOKE(plain, out)
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
    checks['legacy_no_context_fmo_q_zero'] = checks.pop('fmo_q_zero')
    model = fp.make_model('geo_pieda').eval()
    batch = fp.base.Batch.from_data_list([masked(g, 'context_smoke') for g in graphs]).cuda()
    vals, _ = fp.components(model, batch)
    assert torch.isfinite(vals).all()
    fp.total_loss(vals, 'geo_pieda').backward()
    assert fp.has_grad(model.node_encoder) and fp.has_grad(model.pair_updates)
    for g in graphs:
        b = masked(g, 'context_smoke')
        selected = b.decode_mask
        cols, q = (g['context_pieda_cols'], g['context_q_cols'])
        assert torch.count_nonzero(b.edge_attr[selected][:, cols]) == 0
        assert torch.count_nonzero(b.edge_attr[selected][:, g['geom']]) == 0
        assert torch.count_nonzero(b.edge_attr[:, q]) == 0
        assert torch.equal(b.edge_attr[~selected], g['ea'][~selected])
        assert torch.equal(g['ea'][:, cols], g['y'][:, 2:7])
        assert torch.count_nonzero(g['ea'][~selected][:, cols]) > 0
        pairs = torch.unique(g['inverse'][selected])
        ea, hidden = hide_pairs(g, pairs)
        assert torch.equal(hidden, selected)
        assert torch.count_nonzero(ea[hidden][:, cols]) == 0
        assert torch.equal(ea[:, g['geom']], g['ea'][:, g['geom']])
        changed = dict(g, ea=g['ea'].clone())
        changed['ea'][hidden.nonzero().flatten()[:, None], torch.tensor(cols)] += 100
        other, _ = hide_pairs(changed, pairs)
        assert torch.equal(ea, other)
    g = graphs[0]
    with torch.no_grad():
        h = model.encode_nodes(g['x'].cuda(), g['ei'].cuda(), g['ea'].cuda()).clone()
        blank = g['ea'].clone()
        blank[:, g['fmo']] = 0
        z = model.encode_nodes(g['x'].cuda(), g['ei'].cuda(), blank.cuda())
        assert not torch.allclose(h, z), 'PIEDA context has no effect'
    import json
    for kind, old in [('node', ORIGINAL_NODE_EXTRACT), ('pair', ORIGINAL_PAIR_EXTRACT)]:
        baseline_dir, context_dir = (out / ('cohort_plain_' + kind), out / ('cohort_context_' + kind))
        baseline_dir.mkdir()
        context_dir.mkdir()
        old_data = old(model, [g], 'test', baseline_dir)
        new_data = extract(model, [g], 'test', context_dir, kind)
        assert torch.equal(old_data[1], new_data[1])
        assert np.array_equal(old_data[2], new_data[2]) and np.array_equal(old_data[3], new_data[3])
        assert json.loads((baseline_dir / 'test_probe_audit.json').read_text()) == json.loads((context_dir / 'test_probe_audit.json').read_text())
    checks.update(context_uses_canonical_train_z=True, context_visible=True, q_zero=True, target_pieda_hidden_both_directions=True, surrounding_geometry_pieda_unchanged=True, own_target_perturbation_invariant=True, visible_context_changes_encoder=True, probe_cohort_matches_original=True, legacy_exact_check_deterministic_only=True)
    del model, batch
    torch.cuda.empty_cache()
    return checks
