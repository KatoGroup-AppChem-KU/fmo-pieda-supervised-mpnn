"""Nonlinear probe on the exact audited 20260912 Ridge embedding caches.

Reuse the existing pIC50 MLP training loop unchanged. No encoder is loaded.
All settings follow slurm_predict_pic50_mlp.sh; three paired head seeds.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import predict_pic50_mlp_probe as base
SOURCE = Path('results/pic50_geo_pieda_same_split_20260912_full')
LABELS = ('v6_geo', 'v6_geo_pieda', 'v7_geo', 'v7_geo_pieda')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    assert True
    torch.set_num_threads(4)
    assert sha(base.__file__) == 'bea288b0b088affda8c970ad66e40dcbdec68ae65c008c857e66dabbfaef4afe'
    assert read(SOURCE / 'comparison.json')['all_matched_checks_passed']
    audit = read(SOURCE / 'audit.json')
    cohort_path = SOURCE / 'cohort.csv'
    assert sha(cohort_path) == audit['cohort_sha256']
    assert audit['cross_split_pdb_overlap'] == audit['test_pretrain_train_overlap'] == 0
    df = pd.read_csv(cohort_path)
    assert df.pdbid.is_unique
    assert df.split_pdbid.value_counts().to_dict() == {'train': 1582, 'valid': 194, 'test': 197}
    assert np.isfinite(df.reference_pIC50).all()
    maps, provenance = ({}, {})
    for label in LABELS:
        cache = SOURCE / label / 'embeddings.pt'
        done = read(SOURCE / label / 'COMPLETE.json')
        assert done['encoder_frozen'] and done['cohort_sha256'] == sha(cohort_path)
        assert done['embedding_sha256'] == sha(cache)
        obj = torch.load(cache, map_location='cpu', weights_only=False)
        assert obj['pooling'] == 'pocket_ligand_neighbor_mean'
        mapping, dim, meta = base.load_graph_cache(str(cache))
        assert dim == 256 and meta['n_cache'] == meta['n_unique'] == len(df)
        assert set(mapping) == set(df.pdbid)
        assert np.isfinite(np.stack(list(mapping.values()))).all()
        maps[label], provenance[label] = (mapping, done)
    for key in ('input_sha256', 'pooling_nodes_sha256', 'cohort_sha256'):
        assert len({v[key] for v in provenance.values()}) == 1
    mode = 'smoke' if args.smoke else 'full'
    out = Path(f"results/pic50_matched_mlp_{mode}_{os.environ.get('SLURM_JOB_ID', 'standalone')}")
    out.mkdir(exist_ok=False)
    config = dict(graph_mode='pretrained', target_col='reference_pIC50', split_col='split_pdbid', hidden_dim=256, mlp_layers=2, dropout=0.2, lr=0.001, weight_decay=0.001, batch_size=64, num_workers=2, epochs=2 if args.smoke else 500, patience=50, log_every=10 if not args.smoke else 1)
    write(out / 'audit.json', dict(source=str(SOURCE), provenance=provenance, counts=audit['counts'], config=config, code_sha256=sha(__file__), base_sha256=sha(base.__file__), encoder_loaded=False, selection='minimum validation MSE; test evaluated only after selection'))
    results = {}
    initial_hashes = {}
    for seed in [42] if args.smoke else [42, 43, 44]:
        for label in LABELS:
            dest = out / f'{label}_seed{seed}'
            dest.mkdir()
            base.seed_all(seed)
            initial = base.GraphMLPProbe(256, 256, 0.2, 2)
            initial_hash = hashlib.sha256(b''.join((x.detach().numpy().tobytes() for x in initial.state_dict().values()))).hexdigest()
            initial_hashes.setdefault(seed, initial_hash)
            assert initial_hashes[seed] == initial_hash
            base.seed_all(seed)
            print(f'\nSTART {label} seed={seed} | frozen cache -> MLP -> pIC50', flush=True)
            metrics = base.run_mlp_probe(SimpleNamespace(**config, seed=seed), df, maps[label], 256, dest)
            assert metrics['test']['n'] == 197
            assert all((np.isfinite(metrics[sp][k]) for sp in ('train', 'valid', 'test') for k in ('pearson_r', 'spearman_r', 'rmse', 'mae')))
            train_ids = sorted(df.loc[df.split_pdbid == 'train', 'pdbid'])
            scaler = base.Standardizer(np.stack([maps[label][p] for p in train_ids]))
            model = base.GraphMLPProbe(256, 256, 0.2, 2)
            model.load_state_dict(torch.load(dest / 'best_model.pt', map_location='cpu', weights_only=False)['model'])
            model.eval()
            with torch.no_grad():
                x = scaler.transform(np.stack([maps[label][p] for p in df.pdbid]))
                pred = model(torch.from_numpy(x)).numpy()
            predictions = df[['pdbid', 'split_pdbid', 'reference_pIC50']].copy()
            predictions['prediction'] = pred
            predictions.to_csv(dest / 'predictions.csv', index=False)
            np.savez(dest / 'scaler.npz', mean=scaler.mean, std=scaler.std)
            write(dest / 'metrics.json', metrics)
            results[f'{label}_seed{seed}'] = metrics
            print(f"DONE {label} seed={seed} | test r={metrics['test']['pearson_r']:.4f} RMSE={metrics['test']['rmse']:.4f}", flush=True)
    for label in LABELS:
        assert sha(SOURCE / label / 'embeddings.pt') == provenance[label]['embedding_sha256']
    write(out / 'comparison.json', dict(results=results, config=config, smoke=args.smoke, all_checks_passed=True, source_ridge=read(SOURCE / 'comparison.json')['results']))
    print(f'COMPLETE {out}', flush=True)
if __name__ == '__main__':
    main()
