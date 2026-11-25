"""Fixed-prediction, paired UniProt bootstrap on the previously verified cohort."""
import os, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
assert True
root = Path(__import__('os').environ['FMO_WORKDIR'])
cohort = root / 'results/target_results_verified_20261107/20261107_64標的_中心化した193リガンド.csv'
d = pd.read_csv(cohort)
assert len(d) == 193 and d.evaluation_target_key.nunique() == 64
out = root / 'results/within_target_all_representations_20261108'
out.mkdir(exist_ok=True)
paths = {}
cols = []
for rep in ['node', 'pair', 'concat']:
    for label, folder, condition in [('GEO', root / 'results/fpv2_px_7582', 'geo'), ('C0', root / 'results/fpv2_px_7582', 'geo_pieda'), ('C1', root / 'results/pieda_context_v2_20261103/model1_v6/px', 'geo_pieda')]:
        path = folder / condition / rep / 'alltrain/predictions.csv'
        paths[rep + '_' + label] = path
        p = pd.read_csv(path)
        p = p[p.split_pdbid == 'test'].set_index('pdbid')
        assert len(p) == 309 and p.index.is_unique
        name = rep + '_' + label
        cols.append(name)
        vals = []
        for row in d.itertuples():
            ids = row.pdbids.split(';')
            q = p.loc[ids]
            np.testing.assert_allclose(q.target_px, float(row.target_px), rtol=0, atol=1e-10)
            vals.append(float(q.prediction.mean()))
        d[name] = vals
        if rep == 'concat':
            old = {'GEO': 'GEO', 'C0': 'GEO+PIEDA', 'C1': 'GEO+PIEDA_v2'}[label]
            np.testing.assert_allclose(d[name], d[old], rtol=0, atol=1e-10)
terms = []
for u, g in d.groupby('evaluation_target_key', sort=True):
    yc = g.target_px.to_numpy() - g.target_px.mean()
    term = []
    for col in cols:
        pc = g[col].to_numpy() - g[col].mean()
        d.loc[g.index, 'centered_' + col] = pc
        term.append([float(yc @ pc), float(yc @ yc), float(pc @ pc)])
    terms.append(term)
t = np.asarray(terms)

def corr(a):
    s = a.sum(axis=0)
    return s[:, 0] / np.sqrt(s[:, 1] * s[:, 2])
point = corr(t)
rng = np.random.default_rng(42)
boot = np.asarray([corr(t[rng.integers(0, 64, 64)]) for _ in range(10000)])
assert np.isfinite(boot).all()
rows = []
for j, col in enumerate(cols):
    rep, label = col.split('_')
    base = cols.index(rep + '_GEO')
    lo, hi = np.quantile(boot[:, j], [0.025, 0.975])
    dl, dh = np.quantile(boot[:, j] - boot[:, base], [0.025, 0.975])
    rows.append(dict(representation=rep, condition=label, n_targets=64, n_ligands=193, within_target_r=point[j], r_CI_low=lo, r_CI_high=hi, delta_r_vs_GEO=point[j] - point[base], delta_r_CI_low=dl, delta_r_CI_high=dh))
summary = pd.DataFrame(rows)
old = pd.read_csv(root / 'results/target_results_verified_20261107/20261107_64標的_Within_target_Pearsonと95CI.csv').set_index('condition')
for _, row in summary[summary.representation == 'concat'].iterrows():
    ref = old.loc[{'GEO': 'GEO', 'C0': 'GEO+PIEDA', 'C1': 'GEO+PIEDA_v2'}[row.condition]]
    for key in ['within_target_r', 'r_CI_low', 'r_CI_high', 'delta_r_vs_GEO', 'delta_r_CI_low', 'delta_r_CI_high']:
        np.testing.assert_allclose(row[key], ref[key], rtol=0, atol=1e-10)
summary.to_csv(out / 'within_target_all_representations.csv', index=False)
d.to_csv(out / 'centered_predictions_193_ligands.csv', index=False)
(out / 'audit.json').write_text(json.dumps({'partition': os.environ.get('SLURM_JOB_PARTITION', 'standalone'), 'n_targets': 64, 'n_ligands': 193, 'bootstrap_repeats': 10000, 'seed': 42, 'same_target_draws_for_all_nine_series': True, 'existing_concat_results_reproduced': True, 'aggregation': 'same recorded pdbids group, average predictions; fixed observed pX', 'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [cohort, *paths.values()]}, 'no_training': True, 'no_inference': True}, indent=2))
print(summary.to_string(index=False), flush=True)
