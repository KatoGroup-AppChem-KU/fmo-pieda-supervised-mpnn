"""Recover bootstrap distribution with the exact existing 64-target rule."""
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
assert True
root = Path(__import__('os').environ['FMO_WORKDIR'])
d = pd.read_csv(root / 'results/target_results_verified_20261107/20261107_64標的_中心化した193リガンド.csv')
terms = []
for _, g in d.groupby('evaluation_target_key', sort=True):
    y = g.centered_true_px.to_numpy()
    terms.append([[float(y @ g['centered_' + m].to_numpy()), float(y @ y), float(g['centered_' + m].to_numpy() @ g['centered_' + m].to_numpy())] for m in ['GEO', 'GEO+PIEDA']])
t = np.asarray(terms)
assert len(t) == 64 and len(d) == 193

def corr(a):
    s = a.sum(axis=0)
    return s[:, 0] / np.sqrt(s[:, 1] * s[:, 2])
r = corr(t)
rng = np.random.default_rng(42)
boot = []
for _ in range(10000):
    rr = corr(t[rng.integers(0, 64, 64)])
    boot.append(float(rr[1] - rr[0]))
b = np.asarray(boot)
q1, med, q3 = np.quantile(b, [0.25, 0.5, 0.75])
iqr = q3 - q1
lo = float(b[b >= q1 - 1.5 * iqr].min())
hi = float(b[b <= q3 + 1.5 * iqr].max())
ci = np.quantile(b, [0.025, 0.975])
assert np.allclose(ci, [0.022593, 0.189913], atol=1e-06)
out = root / 'results/within_bootstrap_box_20261109'
out.mkdir(exist_ok=True)
(out / 'box.json').write_text(json.dumps({'estimate': float(r[1] - r[0]), 'ci': ci.tolist(), 'q1': q1, 'median': med, 'q3': q3, 'whislo': lo, 'whishi': hi, 'fliers': b[(b < lo) | (b > hi)].tolist(), 'draws': b.tolist(), 'bootstrap': 10000, 'seed': 42, 'partition': os.environ.get('SLURM_JOB_PARTITION', 'standalone'), 'n_targets': 64, 'n_ligands': 193}, indent=2))
print('CI', ci, 'box', q1, med, q3, lo, hi, flush=True)
