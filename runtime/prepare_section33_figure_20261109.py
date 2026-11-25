"""Validate existing predictions and prepare editable Figure data on aurum-g01."""
import os, json, hashlib
from pathlib import Path
import pandas as pd
import numpy as np
assert True
root = Path(__import__('os').environ['FMO_WORKDIR'])
paths = [root / 'results/fpv2_concat_uniprot_20261020/test_predictions_with_uniprot.csv', root / 'results/target_results_verified_20261107/20261107_64標的_中心化した193リガンド.csv', root / 'results/target_results_verified_20261107/20261107_64標的_Within_target_Pearsonと95CI.csv']
g, w, ci = [pd.read_csv(p) for p in paths]
assert len(g) == 309 and len(w) == 193 and (w.evaluation_target_key.nunique() == 64)
panels = []
for col, title in [('Baseline', 'UniProt-mean baseline'), ('GEO', 'GEO'), ('GEO+PIEDA', 'GEO+PIEDA-C0')]:
    y, p = (g.target_px.to_numpy(), g[col].to_numpy())
    r = float(np.corrcoef(y, p)[0, 1])
    assert np.isfinite(y).all() and np.isfinite(p).all()
    assert ((y >= 1.5) & (y <= 12.5) & (p >= 1.5) & (p <= 12.5)).all()
    panels.append(dict(title=title, r=r, mae=float(abs(y - p).mean()), rmse=float(np.sqrt(((y - p) ** 2).mean())), points=[dict(x=float(a), y=float(b), group=str(c), id=str(d)) for a, b, c, d in zip(y, p, g.color_group, g.pdbid)]))
for col, title in [('GEO', 'GEO'), ('GEO+PIEDA', 'GEO+PIEDA-C0')]:
    y, p = (w.centered_true_px.to_numpy(), w['centered_' + col].to_numpy())
    row = ci.set_index('condition').loc[col]
    np.testing.assert_allclose(np.corrcoef(y, p)[0, 1], row.within_target_r, atol=1e-12)
    assert ((abs(y) <= 4.5) & (abs(p) <= 4.5)).all()
    panels.append(dict(title=title, r=float(row.within_target_r), mae=float(abs(y - p).mean()), rmse=float(np.sqrt(((y - p) ** 2).mean())), delta_r=float(row.delta_r_vs_GEO), ci=[float(row.delta_r_CI_low), float(row.delta_r_CI_high)], points=[dict(x=float(a), y=float(b), group=col, id=str(d)) for a, b, d in zip(y, p, w.pdbids)]))
out = root / 'results/section33_figure_20261109'
out.mkdir(exist_ok=True)
payload = dict(panels=panels, counts=g.color_group.value_counts().to_dict(), partition=os.environ.get('SLURM_JOB_PARTITION', 'standalone'), slurm_job=os.environ.get('SLURM_JOB_ID'), source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, new_training=False, new_bootstrap=False)
(out / 'section33_figure_data.json').write_text(json.dumps(payload, indent=2))
print('Validated global 309 structures and within-target 193 ligands / 64 targets. No clipped points.')
for panel in panels[3:]:
    print(panel['title'], 'centered RMSE=', panel['rmse'], 'MAE=', panel['mae'])
print(out)
