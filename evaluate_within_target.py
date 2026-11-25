"""Evaluate observed/predicted pX after separate within-target centering.

Derived from Hirokazu Okamoto's within_target_affinity_20261105.py.
The sufficient-statistic cluster bootstrap follows that adopted implementation.
Inputs are predictions, not raw FMO data. Exact UniProt-ID sets form the primary
nonduplicating groups. This does not estimate generalization to unseen targets.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def apply_paper_cohort(frame, cohort):
    """Use fixed target/ligand memberships; average predictions, never labels.

    This follows within_target_all_representations_20261108.py. The grouping
    metadata contain no measured activities or model predictions.
    """
    assert frame.pdbid.is_unique and cohort.cohort_row_id.is_unique
    source = frame.set_index('pdbid')
    rows = []
    for row in cohort.itertuples():
        ids = row.pdbids.split(';')
        values = source.loc[ids]
        np.testing.assert_allclose(values.target_px, values.target_px.iloc[0], rtol=0, atol=1e-10)
        rows.append(dict(pdbid=row.cohort_row_id, target_group=row.evaluation_target_key,
                         target_px=float(values.target_px.iloc[0]),
                         GEO=float(values.GEO.mean()), **{'GEO+PIEDA': float(values['GEO+PIEDA'].mean())},
                         pdbids=row.pdbids))
    return pd.DataFrame(rows)


def evaluate(frame, out, models=('GEO', 'GEO+PIEDA'), reps=10000, seed=42):
    out.mkdir(parents=True, exist_ok=False)
    columns = ['target_px', *models]
    assert frame.pdbid.is_unique
    assert np.isfinite(frame[columns].to_numpy(float)).all()
    frame = frame.fillna({'target_group': ''})
    frame = frame[frame.target_group.ne('')].copy()
    counts = frame.groupby('target_group').pdbid.transform('size')
    a = frame[counts.ge(2)].copy()
    if a.empty:
        raise ValueError('No target group has at least two structures.')
    centered = [v+'_centered' for v in columns]
    for value in columns:
        a[value+'_centered'] = a[value]-a.groupby('target_group')[value].transform('mean')
    for value in centered:
        np.testing.assert_allclose(a.groupby('target_group')[value].mean(), 0, atol=1e-12)
    units = [x for _, x in a.groupby('target_group', sort=True)]
    def moments(x):
        v = x[centered].to_numpy(float)
        return np.r_[len(v), v.sum(0), (v*v).sum(0), (v[:, 0, None]*v[:, 1:]).sum(0)]
    stats = np.stack([moments(x) for x in units])
    def rs(m):
        n, sums, squares, cross = m[..., 0], m[..., 1:4], m[..., 4:7], m[..., 7:9]
        variance = squares-sums*sums/n[..., None]
        numerator = cross-sums[..., :1]*sums[..., 1:]/n[..., None]
        denominator = np.sqrt(np.maximum(variance[..., :1]*variance[..., 1:], 0))
        with np.errstate(divide='ignore', invalid='ignore'):
            return numerator/denominator
    point = rs(stats.sum(0))
    values = a[centered].to_numpy(float)
    np.testing.assert_allclose(point, [np.corrcoef(values[:, 0], values[:, j])[0, 1] for j in (1, 2)], atol=1e-12)
    indices = np.random.default_rng(seed).integers(len(units), size=(reps, len(units)))
    draws = rs(stats[indices].sum(1))
    direct = pd.concat([units[j] for j in indices[0]], ignore_index=True)[centered].to_numpy(float)
    np.testing.assert_allclose(draws[0], [np.corrcoef(direct[:, 0], direct[:, j])[0, 1] for j in (1, 2)], atol=1e-12)
    finite = np.isfinite(draws).all(1)
    delta = draws[finite, 1]-draws[finite, 0]
    ci = np.quantile(delta, [.025, .975])
    result = dict(n_observations=len(a), n_unique_observation_ids=a.pdbid.nunique(), n_target_groups=len(units),
                  pearson_geo=float(point[0]), pearson_pieda=float(point[1]),
                  delta_r=float(point[1]-point[0]), ci95_low=float(ci[0]), ci95_high=float(ci[1]),
                  positive_bootstrap_fraction=float(np.mean(delta > 0)), bootstrap_replicates=reps,
                  bootstrap_valid=int(finite.sum()), seed=seed, grouping='exact UniProt ID set')
    a.to_csv(out/'centered_predictions.csv', index=False)
    pd.DataFrame({'r_geo': draws[:, 0], 'r_pieda': draws[:, 1],
                  'delta_r': draws[:, 1]-draws[:, 0]}).to_csv(out/'bootstrap.csv', index=False)
    (out/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True,
                        help='CSV columns: pdbid,target_group,target_px,GEO,GEO+PIEDA')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cohort', type=Path,
                        help='Optional fixed paper target/ligand membership CSV; no activity values required.')
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    if args.cohort:
        frame = apply_paper_cohort(frame, pd.read_csv(args.cohort))
    print(json.dumps(evaluate(frame, args.out), indent=2))
