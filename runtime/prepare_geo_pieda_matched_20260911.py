"""Read-only graph audit and new shared PDB-group split; Slurm g01 only."""
import hashlib
import json
import os
import re
from pathlib import Path
import numpy as np
import torch
ROOT = Path('dataset/geo_pieda_matched_20260911')
SOURCES = {'bdb': {'v6': 'dataset/graph_v4_fmofrag_pretrain_strict_cached_qcfiltered_v06prov', 'v7': 'dataset/graph_v7_fmofrag_morganunified_strict_qcfiltered_v06prov'}, 'scop2': {'v6': 'dataset/scop2_bindingdb_v6_20260911/graphs', 'v7': 'dataset/scop2_bindingdb_v7_20260911/graphs'}}
TARGETS = ['PIEDA_sum_z', 'ES_z', 'EX_z', 'CTmix_z', 'DI_z']
DISTANCES = ['DIST_raw', 'centroid_distance_raw']

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)

def write_json(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)

def raw_targets(g):
    names = list(g.edge_attr_names)
    z = g.edge_attr[:, [names.index(n) for n in TARGETS]].double()
    return z * torch.tensor([g.fmo_std[n] for n in TARGETS], dtype=torch.float64) + torch.tensor([g.fmo_mean[n] for n in TARGETS], dtype=torch.float64)

def fingerprint(g):
    h = hashlib.sha256()
    for t in (g.x, g.edge_index, g.edge_attr, g.frag_id, g.pos):
        a = t.contiguous().numpy()
        h.update(str((a.shape, a.dtype)).encode())
        h.update(a.tobytes())
    h.update(json.dumps([g.fmo_mean, g.fmo_std], sort_keys=True).encode())
    return h.hexdigest()

def main():
    require(True, 'g01 only')
    torch.set_num_threads(4)
    ROOT.mkdir(exist_ok=False)
    records, excluded, schemas = ([], [], {})
    for source, dirs in SOURCES.items():
        paths = {v: {p.stem: p for p in Path(d).glob('*.pt')} for v, d in dirs.items()}
        missing = {v: sorted(set(paths['v7' if v == 'v6' else 'v6']) - set(paths[v])) for v in paths}
        write_json(ROOT / (source + '_missing.json'), missing)
        require(all((not x for x in missing.values())), source + ': unmatched key sets; consult user')
        require(len(paths['v6']) > 0, 'empty source')
        for i, key in enumerate(sorted(paths['v6']), 1):
            gs = {v: load(paths[v][key]) for v in paths}
            a, b = (gs['v6'], gs['v7'])
            require(re.match('^[0-9][a-zA-Z0-9]{3}', key) is not None, 'bad PDB key')
            for v, g in gs.items():
                require(g.x.shape[1] == (1026 if v == 'v6' else 546), key + ': node dim')
                require(g.edge_attr.shape[1] == 72, key + ': edge dim')
                require(torch.isfinite(g.x).all() and torch.isfinite(g.edge_attr).all(), key + ': finite')
                schema = [list(g.node_feature_names), list(g.edge_attr_names), str(g.graph_version)]
                if v not in schemas:
                    schemas[v] = schema
                require(schema == schemas[v], key + ': feature schema')
                require(g.edge_index.shape[1] > 0, key + ': no edges')
            require(torch.equal(a.edge_index, b.edge_index), key + ': different edge order/topology')
            require(torch.equal(a.frag_id, b.frag_id) and torch.equal(a.pos, b.pos), key + ': node mapping')
            require(torch.equal(a.x[:, 480:514], b.x[:, :34]), key + ': AA/flags mismatch')
            geom = [j for j, n in enumerate(a.edge_attr_names) if n in DISTANCES or n.startswith(('DIST_rbf_', 'centroid_distance_rbf_'))]
            require(len(geom) == 66, key + ': geometry')
            require(torch.equal(a.edge_attr[:, geom], b.edge_attr[:, geom]), key + ': geometry differs')
            y6, y7 = (raw_targets(a), raw_targets(b))
            require(torch.isfinite(y6).all() and torch.isfinite(y7).all(), key + ': raw finite')
            require(torch.allclose(y6, y7, atol=0.0002, rtol=1e-05), key + ': raw targets differ')
            n = a.edge_index.shape[1]
            rec = dict(key=source + ':' + key, source=source, pdb=key[:4].lower(), paths={v: str(paths[v][key]) for v in paths}, edges=n, fingerprints={v: fingerprint(g) for v, g in gs.items()}, max_raw_difference=float((y6 - y7).abs().max()))
            (excluded if n > 50000 else records).append(rec)
            if i % 250 == 0 or i == len(paths['v6']):
                print(f"audit {source} {i}/{len(paths['v6'])}", flush=True)
    write_json(ROOT / 'excluded_over_50000.json', excluded)
    groups = {}
    for r in records:
        groups.setdefault(r['pdb'], []).append(r)
    rng = np.random.default_rng(42)
    for sources in ({'bdb'}, {'scop2'}, {'bdb', 'scop2'}):
        keys = sorted((k for k, rs in groups.items() if {r['source'] for r in rs} == sources))
        rng.shuffle(keys)
        n = round(len(keys) * 0.1)
        for i, key in enumerate(keys):
            label = 'test' if i < n else 'valid' if i < 2 * n else 'train'
            for r in groups[key]:
                r['split'] = label
    counts = {s: {l: sum((r['source'] == s and r['split'] == l for r in records)) for l in ('train', 'valid', 'test')} for s in SOURCES}
    write_json(ROOT / 'counts.json', counts)
    for s, c in counts.items():
        for l, ratio in [('train', 0.8), ('valid', 0.1), ('test', 0.1)]:
            require(abs(c[l] / sum(c.values()) - ratio) < 0.01, f'{s}: split ratio deviation; consult user')
    require(all((len({r['split'] for r in rs}) == 1 for rs in groups.values())), 'PDB leakage')
    n, total, square = (0, torch.zeros(7, dtype=torch.float64), torch.zeros(7, dtype=torch.float64))
    for i, r in enumerate(records):
        if r['split'] != 'train':
            continue
        g = load(r['paths']['v7'])
        y = torch.cat([g.edge_attr[:, [g.edge_attr_names.index(k) for k in DISTANCES]].double(), raw_targets(g)], 1)
        n += len(y)
        total += y.sum(0)
        square += (y * y).sum(0)
    mean = total / n
    std = (square / n - mean * mean).clamp_min(0).sqrt()
    require(bool((std > 1e-08).all()), 'degenerate train target')
    manifest = dict(records=records, excluded=len(excluded), counts=counts, split_seed=42, schemas=schemas, train_stats=dict(n=n, mean=mean.tolist(), std=std.tolist()), target_order=DISTANCES + TARGETS, pdb_groups=len(groups), overlap_groups=sum((len({r['source'] for r in rs}) == 2 for rs in groups.values())), normalization='train-global population std, canonical v7 raw targets', max_edges=50000, all_verified=True)
    write_json(ROOT / 'manifest.json', manifest)
    write_json(ROOT / 'COMPLETE.json', dict(counts=counts, graphs=len(records), excluded=len(excluded), pdb_groups=len(groups), all_verified=True))
    print(json.dumps(manifest['counts']), flush=True)
    print('AUDIT_COMPLETE', flush=True)
if __name__ == '__main__':
    main()
