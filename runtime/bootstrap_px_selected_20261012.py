"""Fixed-model paired PDB bootstrap; select head seed by validation MSE only."""
import hashlib
import json
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr, spearmanr, rankdata

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def corr(y, p):
    yc = y - y.mean(axis=-1, keepdims=True)
    pc = p - p.mean(axis=-1, keepdims=True)
    return (yc * pc).sum(axis=-1) / np.sqrt((yc * yc).sum(axis=-1) * (pc * pc).sum(axis=-1))

def metrics(y, p):
    d = p - y
    return np.stack([corr(y, p), corr(rankdata(y, axis=-1), rankdata(p, axis=-1)), np.sqrt(np.mean(d * d, axis=-1)), np.mean(np.abs(d), axis=-1)], axis=-1)

def main():
    assert True
    assert Path.cwd() == Path(__import__('os').environ['FMO_WORKDIR'])
    source = Path('results/frozen_px_full_7296')
    multi = Path('results/frozen_px_3seed_7298')
    out = Path('results') / ('bootstrap_px_selected_' + os.environ.get('SLURM_JOB_ID', 'standalone'))
    out.mkdir()
    shutil.copy2(__file__, out / Path(__file__).name)
    hashes = {str(p): sha(p) for p in [multi / 'metrics_per_seed.csv', source / 'cohort.csv', Path(__file__)]}
    allmetrics = pd.read_csv(multi / 'metrics_per_seed.csv')
    valid = allmetrics[allmetrics.split == 'valid'].copy()
    assert len(valid) == 18
    selected = valid.sort_values(['loss', 'seed']).groupby(['condition', 'representation'], sort=True).head(1)
    assert len(selected) == 6
    selected.to_csv(out / 'selected_heads.csv', index=False)
    cohort = pd.read_csv(source / 'cohort.csv')
    ref = cohort[cohort.split_pdbid == 'test'].sort_values('pdbid').reset_index(drop=True)
    assert len(ref) == 341 and ref.pdbid.is_unique
    y = ref.target_px.to_numpy(dtype=np.float64)
    ref[['pdbid', 'target_px', 'affinity_type']].to_csv(out / 'test_cohort.csv', index=False)
    rng = np.random.default_rng(20261012)
    idx = rng.integers(0, len(y), size=(10000, len(y)), dtype=np.int32)
    np.savez_compressed(out / 'resample_indices.npz', indices=idx)
    names = ['pearson_r', 'spearman_r', 'rmse', 'mae']
    rows, selection = ([], [])
    for kind in ['node', 'pair', 'concat']:
        points, draws = ({}, {})
        for cond in ['R3', 'R1']:
            sel = selected[(selected.condition == cond) & (selected.representation == kind)].iloc[0]
            seed = int(sel.seed)
            root = source if seed == 42 else multi
            path = root / cond / f'{kind}_mlp_seed{seed}' / 'predictions.csv'
            hashes[str(path)] = sha(path)
            df = pd.read_csv(path)
            df = df[df.split_pdbid == 'test'].sort_values('pdbid').reset_index(drop=True)
            assert df.pdbid.is_unique and df.pdbid.tolist() == ref.pdbid.tolist()
            np.testing.assert_array_equal(df.target_px.to_numpy(), y)
            assert df.affinity_type.tolist() == ref.affinity_type.tolist()
            p = df.prediction.to_numpy(dtype=np.float64)
            assert np.isfinite(p).all()
            point = metrics(y, p)
            np.testing.assert_allclose(point[:2], [pearsonr(y, p)[0], spearmanr(y, p)[0]], atol=1e-12)
            expected = allmetrics[(allmetrics.condition == cond) & (allmetrics.representation == kind) & (allmetrics.seed == seed) & (allmetrics.split == 'test')].iloc[0]
            np.testing.assert_allclose(point, expected[names].to_numpy(dtype=float), atol=1e-05)
            boot = np.concatenate([metrics(y[j], p[j]) for j in np.array_split(idx, 40)])
            assert boot.shape == (10000, 4) and np.isfinite(boot).all()
            for k in [0, 5000, 9999]:
                j = idx[k]
                np.testing.assert_allclose(boot[k, 0], pearsonr(y[j], p[j])[0], atol=1e-12)
                np.testing.assert_allclose(boot[k, 1], spearmanr(y[j], p[j])[0], atol=1e-12)
            np.testing.assert_allclose(metrics(y, y), [1, 1, 0, 0], atol=1e-12)
            points[cond], draws[cond] = (point, boot)
            selection.append(dict(condition=cond, representation=kind, seed=seed, validation_mse=float(sel.loss), predictions=str(path)))
            np.savez_compressed(out / f'{cond}_{kind}_bootstrap.npz', metrics=boot, metric_names=np.array(names))
            print(f'{kind} {cond}: seed={seed}, valid MSE={sel.loss:.6f}, test r={point[0]:.6f}', flush=True)
        for label, point, boot in [('GEO', points['R3'], draws['R3']), ('GEO+PIEDA', points['R1'], draws['R1']), ('difference', points['R1'] - points['R3'], draws['R1'] - draws['R3'])]:
            lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
            for k, name in enumerate(names):
                rows.append(dict(representation=kind, condition=label, metric=name, estimate=float(point[k]), ci_low=float(lo[k]), ci_high=float(hi[k]), n_test=341, n_bootstrap=10000))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / 'confidence_intervals.csv', index=False)
    for p, h in hashes.items():
        assert sha(p) == h
    config = dict(complete=True, model='SG/ESM-FP, all-subsets pretraining', selection='Minimum validation MSE separately per condition and representation; tie: smaller seed', selected_heads=selection, bootstrap='paired ordinary PDB bootstrap with replacement', n_test=341, n_resamples=10000, bootstrap_seed=20261012, confidence_level=0.95, interval='percentile 2.5th to 97.5th; numpy linear quantiles', difference='GEO+PIEDA minus GEO', same_indices_across_all_comparisons=True, retraining=False, protein_cluster_bootstrap=False, multiple_comparison_correction=False, caveat='Conditional on selected fitted models and fixed split. Does not include training/selection uncertainty or dependence between different PDBs of same protein. Not a seed-average or ensemble CI.', versions=dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__), source_hashes=hashes)
    (out / 'COMPLETE.json').write_text(json.dumps(config, indent=2))
    print(summary[summary.metric == 'pearson_r'].to_string(index=False), flush=True)
    print('COMPLETE: 10000 paired resamples; checks passed; no retraining', flush=True)
if __name__ == '__main__':
    main()
