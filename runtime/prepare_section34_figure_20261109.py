"""Prepare exact boxplot/histogram geometry for editable PowerPoint export."""
import os, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.cbook import boxplot_stats
from matplotlib.ticker import MaxNLocator
assert True
root = Path(__import__('os').environ['FMO_WORKDIR'])
sources = [root / 'results/target_results_verified_20261107/20261107_64標的_標的別予測性能.csv', root / 'results/ligand_delta_px_20261105/20261105_リガンド間ΔpX_UniProt別MAE.csv']
d, pairs = [pd.read_csv(p) for p in sources]
assert len(d) == len(pairs) == 64 and set(d.uniprot) == set(pairs.uniprot)
assert (d.set_index('uniprot').n_ligands == pairs.set_index('uniprot').loc[d.uniprot, 'n_ligands']).all()
boxes = []
for metric in ['MAE', 'RMSE', 'rho']:
    rng = np.random.default_rng(42)
    groups = []
    for n in [2, 3, 5]:
        q = d.loc[d.n_ligands >= n, ['uniprot', 'GEO+PIEDA_change_' + metric]].dropna()
        values = q.iloc[:, 1].to_numpy()
        stats = boxplot_stats(values, whis=1.5)[0]
        groups.append({'min_ligands': n, 'n_targets': len(values), **{k: float(stats[k]) for k in ['q1', 'q3', 'med', 'whislo', 'whishi']}, 'points': [{'uniprot': str(u), 'value': float(v), 'jitter': float(j)} for u, v, j in zip(q.uniprot, values, rng.uniform(-0.13, 0.13, len(values)))]})
    boxes.append({'metric': metric, 'groups': groups})
col = 'GEO+PIEDA_delta_MAE_vs_GEO'
extent = float(np.ceil(np.abs(pairs[col]).max() / 0.25) * 0.25)
bins = np.arange(-extent, extent + 0.125, 0.25)
hist = []
for n in [2, 3, 5]:
    values = pairs.loc[pairs.n_ligands >= n, col].to_numpy()
    counts, _ = np.histogram(values, bins=bins)
    assert counts.sum() == len(values) == {2: 64, 3: 33, 5: 9}[n]
    ymax = float(MaxNLocator(nbins=5, integer=True).tick_values(0, counts.max() * 1.05)[-1])
    ticks = MaxNLocator(nbins=5, integer=True).tick_values(0, ymax).tolist()
    hist.append({'min_ligands': n, 'n_targets': len(values), 'improved': int((values < 0).sum()), 'bins': bins.tolist(), 'counts': counts.tolist(), 'extent': extent, 'ymax': ymax, 'yticks': ticks})
out = root / 'results/section34_figure_20261109'
out.mkdir(exist_ok=True)
(out / 'section34_figure_data.json').write_text(json.dumps({'boxes': boxes, 'histograms': hist, 'partition': os.environ.get('SLURM_JOB_PARTITION', 'standalone'), 'job_id': os.environ.get('SLURM_JOB_ID'), 'comparison': 'C0 minus GEO', 'no_training': True, 'no_new_bootstrap': True, 'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}, indent=2))
print('Verified common 64 / 33 / 9 targets; histograms improved:', [h['improved'] for h in hist])
print(out)
