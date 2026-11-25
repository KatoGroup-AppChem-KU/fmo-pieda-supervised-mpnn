"""Read-only SG/ESM-FP final concat diagnostics. Run on Slurm CPU only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
assert True
ROOT = Path(__import__('os').environ['FMO_WORKDIR'])
SOURCE = ROOT / 'results/fpv2_px_7582'
PLOT = ROOT / 'results/fpv2_concat_uniprot_20261020'
parser = argparse.ArgumentParser()
parser.add_argument('--variant', choices=['legacy', 'v2'], default='legacy')
args = parser.parse_args()
IS_V2 = args.variant == 'v2'
V2 = ROOT / 'results/pieda_context_v2_20261103/model1_v6/px'
LABEL = 'GEO+PIEDA_v2' if IS_V2 else 'GEO+PIEDA'
OUT = ROOT / ('results/within_target_affinity_v2_20261105' if IS_V2 else 'results/within_target_affinity_20261105')
OUT.mkdir(exist_ok=False)
SEED, REPS = (42, 10000)
MODELS = ['GEO', LABEL]
NAMES = {'P24941': 'CDK2', 'P00742': 'Factor X', 'P10275': 'AR', 'O60674': 'JAK2', 'O75530': 'EED'}
sources = [PLOT / 'test_predictions_with_uniprot.csv', PLOT / 'audit.json', SOURCE / 'COMPLETE.json', SOURCE / 'confidence_intervals.csv', SOURCE / 'geo/cohort_and_folds.csv', ROOT / 'paper/supporting_information/protein_classification_20261004/bdb_classified.csv']
sources += [SOURCE / c / 'concat/alltrain/predictions.csv' for c in ['geo', 'geo_pieda']]
if IS_V2:
    sources += [V2 / 'COMPLETE.json', V2 / 'confidence_intervals.csv', V2 / 'geo_pieda/concat/alltrain/predictions.csv']
    assert json.loads((V2 / 'COMPLETE.json').read_text())['complete']
sources.append(Path(__file__))
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
assert json.loads((SOURCE / 'COMPLETE.json').read_text())['complete']
saved_audit = json.loads((PLOT / 'audit.json').read_text())
for p, digest in saved_audit['source_sha256'].items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest, p
d = pd.read_csv(sources[0]).fillna({'uniprot': ''}).sort_values('pdbid').reset_index(drop=True)
assert len(d) == 309 and d.pdbid.is_unique and (set(d.split_pdbid) == {'test'})
for label, condition in zip(MODELS, ['geo', 'geo_pieda']):
    prediction_root = V2 if IS_V2 and condition == 'geo_pieda' else SOURCE
    original = pd.read_csv(prediction_root / condition / 'concat/alltrain/predictions.csv')
    original = original[original.split_pdbid == 'test'].sort_values('pdbid').reset_index(drop=True)
    assert d.pdbid.tolist() == original.pdbid.tolist()
    np.testing.assert_allclose(d.target_px, original.target_px, rtol=0, atol=1e-12)
    if label == 'GEO+PIEDA_v2':
        d[label] = original.prediction.to_numpy()
    else:
        np.testing.assert_allclose(d[label], original.prediction, rtol=0, atol=1e-12)
assert np.isfinite(d[['target_px'] + MODELS + ['Baseline']]).all().all()
if IS_V2:
    previous = pd.read_csv(ROOT / 'results/within_target_affinity_20261105/test_predictions_annotated.csv')
    assert d.pdbid.tolist() == previous.pdbid.tolist()
    np.testing.assert_allclose(d.target_px, previous.target_px, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(d.GEO, previous.GEO)
cohort = pd.read_csv(SOURCE / 'geo/cohort_and_folds.csv')
mapping = pd.read_csv(sources[5]).fillna({'uniprot': ''})
id_map = {r.pdb: sorted(set((t.strip() for t in r.uniprot.split(';') if t.strip()))) for r in mapping.itertuples()}
d['ids'] = d.pdbid.map(id_map)
d['n_uniprot_ids'] = d.ids.map(len)
d['target_group'] = d.ids.map(';'.join)
assert (d.target_group == d.uniprot).all()
train_ids = {u: set() for ids in id_map.values() for u in ids}
for p in cohort.loc[cohort.split_pdbid != 'test', 'pdbid']:
    for u in id_map[p]:
        train_ids[u].add(p)

def corr(y, p):
    y, p = (np.asarray(y, float), np.asarray(p, float))
    if len(y) < 2 or np.ptp(y) == 0 or np.ptp(p) == 0:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])

def errors(y, p):
    e = np.asarray(p) - np.asarray(y)
    return (float(np.mean(abs(e))), float(np.sqrt(np.mean(e * e))))

def group_table(frame, key):
    rows = []
    for label, group in frame.groupby(key, sort=True):
        if len(group) < 2:
            continue
        y = group.target_px.to_numpy(float)
        row = dict(uniprot=label, name=NAMES.get(label, ''), n_test=len(group), n_train=len(train_ids.get(label, set())), n_multi_id_structures=int((group.n_uniprot_ids > 1).sum()), n_unique_observed_px=len(np.unique(y)), observed_min=float(y.min()), observed_max=float(y.max()), n_unique_ligand_smiles=group.ligand_smiles.nunique(), affinity_types=';'.join(sorted(group.affinity_type.unique())), pdbids=';'.join(group.pdbid), correlation_caution='n=2: rank correlation is only +/-1 (or undefined)' if len(group) == 2 else 'n<5: exploratory' if len(group) < 5 else 'small group: descriptive', observed_constant=bool(np.ptp(y) == 0))
        for short, model in [('geo', MODELS[0]), ('pieda', MODELS[1])]:
            p = group[model].to_numpy(float)
            row['mae_' + short], row['rmse_' + short] = errors(y, p)
            row['pearson_' + short] = corr(y, p)
            row['spearman_' + short] = float(spearmanr(y, p).statistic) if np.ptp(y) > 0 and np.ptp(p) > 0 else np.nan
        row['mae_improvement'] = row['mae_geo'] - row['mae_pieda']
        row['rmse_improvement'] = row['rmse_geo'] - row['rmse_pieda']
        row['spearman_improvement'] = row['spearman_pieda'] - row['spearman_geo']
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['n_test', 'uniprot'], ascending=[False, True])
expanded = d[d.n_uniprot_ids > 0].explode('ids').rename(columns={'ids': 'individual_uniprot'})
per_id = group_table(expanded, 'individual_uniprot')
per_id.to_csv(OUT / 'per_uniprot_metrics.csv', index=False)
per_id.sort_values('mae_improvement', ascending=False).to_csv(OUT / 'per_uniprot_by_mae_improvement.csv', index=False)
per_id.sort_values('mae_improvement').to_csv(OUT / 'per_uniprot_by_mae_worsening.csv', index=False)
group_table(d[d.n_uniprot_ids > 0], 'target_group').to_csv(OUT / 'per_exact_id_set_metrics.csv', index=False)

def within(frame, label, key, cluster_key=None):
    counts = frame.groupby(key)[key].transform('size')
    a = frame[counts >= 2].copy()
    for value in ['target_px'] + MODELS:
        a[value + '_centered'] = a[value] - a.groupby(key)[value].transform('mean')
    centered_cols = [v + '_centered' for v in ['target_px'] + MODELS]
    centered = a[centered_cols].to_numpy(float)
    groupkey = cluster_key or key
    units = [x for _, x in a.groupby(groupkey, sort=True)]

    def moments(x):
        v = x[centered_cols].to_numpy(float)
        return np.r_[len(v), v.sum(0), (v * v).sum(0), (v[:, 0, None] * v[:, 1:]).sum(0)]
    stats = np.stack([moments(x) for x in units])

    def rs(m):
        n, sums, squares, cross = (m[..., 0], m[..., 1:4], m[..., 4:7], m[..., 7:9])
        variance = squares - sums * sums / n[..., None]
        numerator = cross - sums[..., :1] * sums[..., 1:] / n[..., None]
        denominator = np.sqrt(np.maximum(variance[..., :1] * variance[..., 1:], 0))
        with np.errstate(divide='ignore', invalid='ignore'):
            return numerator / denominator
    point = rs(stats.sum(0))
    np.testing.assert_allclose(point, [corr(centered[:, 0], centered[:, j]) for j in [1, 2]], atol=1e-12)
    rng = np.random.default_rng(SEED)
    indices = rng.integers(len(units), size=(REPS, len(units)))
    draws = rs(stats[indices].sum(1))
    direct = pd.concat([units[j] for j in indices[0]], ignore_index=True)
    np.testing.assert_allclose(draws[0], [corr(direct.target_px_centered, direct[m + '_centered']) for m in MODELS], atol=1e-12)
    finite = np.isfinite(draws).all(1)
    delta = draws[finite, 1] - draws[finite, 0]
    ci = np.quantile(delta, [0.025, 0.975])
    row = dict(analysis=label, centering=key, bootstrap_unit=groupkey, n_memberships=len(a), n_unique_pdb=a.pdbid.nunique(), n_target_groups=a[key].nunique(), n_bootstrap_clusters=len(units), n_multi_id_pdb=a.loc[a.n_uniprot_ids > 1, 'pdbid'].nunique(), pearson_geo=float(point[0]), pearson_pieda=float(point[1]), delta_r=float(point[1] - point[0]), ci95_low=float(ci[0]), ci95_high=float(ci[1]), positive_bootstrap_fraction=float(np.mean(delta > 0)), bootstrap_replicates=REPS, bootstrap_valid=int(finite.sum()), seed=SEED)
    a.drop(columns=['ids'], errors='ignore').to_csv(OUT / (label + '_centered_predictions.csv'), index=False)
    pd.DataFrame({'r_geo': draws[:, 0], 'r_pieda': draws[:, 1], 'delta_r': draws[:, 1] - draws[:, 0]}).to_csv(OUT / (label + '_bootstrap.csv'), index=False)
    return row
parent = {u: u for ids in d.ids for u in ids}

def find(u):
    while parent[u] != u:
        parent[u] = parent[parent[u]]
        u = parent[u]
    return u
for ids in d.ids:
    for u in ids[1:]:
        parent[find(u)] = find(ids[0])
d['shared_id_cluster'] = d.ids.map(lambda ids: find(ids[0]) if ids else '')
analyses = [within(d[d.n_uniprot_ids > 0], 'exact_id_sets', 'target_group'), within(d[d.n_uniprot_ids > 0], 'exact_id_sets_shared_id_bootstrap', 'target_group', 'shared_id_cluster'), within(d[d.n_uniprot_ids == 1], 'single_id_only', 'target_group'), within(expanded, 'individual_ids_overlap_exploratory', 'individual_uniprot')]
pd.DataFrame(analyses).to_csv(OUT / 'within_target_results.csv', index=False)
overall = []
for model in ['Baseline'] + MODELS:
    mae, rmse = errors(d.target_px, d[model])
    overall.append(dict(condition=model, n=309, pearson_r=corr(d.target_px, d[model]), mae=mae, rmse=rmse))
assert abs(overall[1]['pearson_r'] - 0.75906) < 1e-06
assert abs(overall[2]['pearson_r'] - (0.763041 if IS_V2 else 0.789211)) < 1e-06
pd.DataFrame(overall).to_csv(OUT / 'overall_metrics.csv', index=False)
d.drop(columns=['ids']).to_csv(OUT / 'test_predictions_annotated.csv', index=False)
inventory = dict(n_test=309, n_single_id=int((d.n_uniprot_ids == 1).sum()), n_multi_id=int((d.n_uniprot_ids > 1).sum()), n_no_id=int((d.n_uniprot_ids == 0).sum()), n_distinct_uniprot=expanded.individual_uniprot.nunique(), n_repeated_uniprot=len(per_id), n_pdb_in_repeated_uniprot=expanded[expanded.individual_uniprot.isin(per_id.uniprot)].pdbid.nunique(), n_mae_improved=int((per_id.mae_improvement > 0).sum()), n_mae_worsened=int((per_id.mae_improvement < 0).sum()), n_mae_tied=int((per_id.mae_improvement == 0).sum()), n_groups_n2=int((per_id.n_test == 2).sum()), n_groups_n3_4=int(per_id.n_test.between(3, 4).sum()), n_groups_n5plus=int((per_id.n_test >= 5).sum()), n_constant_observed_groups=int(per_id.observed_constant.sum()), n_defined_spearman=int(per_id.spearman_improvement.notna().sum()), n_spearman_improved=int((per_id.spearman_improvement > 1e-12).sum()), n_spearman_worsened=int((per_id.spearman_improvement < -1e-12).sum()), n_spearman_tied=int((per_id.spearman_improvement.abs() <= 1e-12).sum()))
audit = dict(complete=True, series='FPv2 C SG/ESM-FP final concat; ' + LABEL, no_retraining=True, input_hashes=hashes, inventory=inventory, improvement_sign='MAE/RMSE: GEO minus GEO+PIEDA; r/rho: GEO+PIEDA minus GEO', ci_method='paired whole-cluster percentile bootstrap, 10000 draws, seed42; no retraining', primary_grouping='Exact sorted UniProt ID sets; no PDB duplication. Shared-ID cluster sensitivity and single-ID sensitivity included.', individual_id_caveat='All individual IDs are listed. Multi-ID PDBs occur in more than one group; overlapping-ID pooled estimate is exploratory.', limitations=['Assay/indicator differences remain; same-target analysis is not same-assay analysis.', 'PDB count need not equal distinct ligand count; ligand SMILES counts reported.', 'n=2 Spearman is +/-1 or undefined; small groups descriptive, no per-target significance claims.', 'Positive bootstrap fraction is descriptive, not posterior probability.'])
(OUT / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))

def f(x):
    return 'NA' if pd.isna(x) else f'{x:.4f}'
lines = ['# UniProt内の活性差予測：SG／ESM-FP 最終・Node＋Pair連結', '', '- 固定Test309構造の保存済み予測を使用。従来GEO＋PIEDAを対象とし、v2ではない。再学習なし。', '- MAE・RMSEの改善量＝GEO−GEO＋PIEDA。相関の改善量＝GEO＋PIEDA−GEO。', '- pXは無次元の対数尺度。アッセイ条件・指標の混在は今回も残る。', '', '## 全体性能', '', '| 条件 | Pearson r | MAE | RMSE |', '|---|---:|---:|---:|']
for x in overall:
    lines.append(f"| {x['condition']} | {f(x['pearson_r'])} | {f(x['mae'])} | {f(x['rmse'])} |")
lines += ['', '## Within-target Pearson r', '', '- 同じ対象群の実測値と各モデルの予測値から、それぞれの群内平均を引き、中心化後の全構造をまとめて相関を計算。2構造以上の群を対象とする。', '- 同じ群を両モデルで共通に復元抽出するbootstrapを10,000回行い、Δrの2.5・97.5百分位を95% CIとした。群に属する構造はまとめて抽出する。', '- 複数ID併記の構造を重複させない主集計では同一ID集合を同一対象とする。同じIDを共有する群をまとめたbootstrapと、単一ID構造のみの解析も示す。個別IDへの展開結果は重複所属を含むため参考値。', '', '| 集計 | PDB数 | 群数 | GEO r | GEO＋PIEDA r | Δr [95% CI] |', '|---|---:|---:|---:|---:|---|']
for x in analyses:
    lines.append(f"| {x['analysis']} | {x['n_unique_pdb']} | {x['n_target_groups']} | {f(x['pearson_geo'])} | {f(x['pearson_pieda'])} | {f(x['delta_r'])} [{f(x['ci95_low'])}, {f(x['ci95_high'])}] |")
lines += ['', '## UniProt別の誤差と順位', '', f"- Test内のIDは計{inventory['n_distinct_uniprot']}種類。複数構造を持つIDは{len(per_id)}種類で、MAE改善{inventory['n_mae_improved']}、悪化{inventory['n_mae_worsened']}、同値{inventory['n_mae_tied']}。併記IDによる重複所属を含む。", f"- n=2の群は{inventory['n_groups_n2']}、n=3–4は{inventory['n_groups_n3_4']}、n≥5は{inventory['n_groups_n5plus']}。", f"- Spearmanが定義できる{inventory['n_defined_spearman']}群で、上昇{inventory['n_spearman_improved']}、低下{inventory['n_spearman_worsened']}、同値{inventory['n_spearman_tied']}。n=2では±1になり、順位改善の強い証拠とはしない。", '', '| UniProt | 名称 | Train数 | Test数 | GEO MAE | ＋PIEDA MAE | MAE改善 | GEO RMSE | ＋PIEDA RMSE | RMSE改善 | GEO ρ | ＋PIEDA ρ |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in per_id.itertuples():
    lines.append(f'| {r.uniprot} | {r.name} | {r.n_train} | {r.n_test} | {f(r.mae_geo)} | {f(r.mae_pieda)} | {f(r.mae_improvement)} | {f(r.rmse_geo)} | {f(r.rmse_pieda)} | {f(r.rmse_improvement)} | {f(r.spearman_geo)} | {f(r.spearman_pieda)} |')
lines += ['', '## 解釈上の範囲', '', '- 群内中心化はタンパク質ごとの平均活性・平均予測の差を除くが、同一タンパク質内の異なるアッセイ・指標・同一リガンド重複は除かない。', '- within-target相関は構造数と群内活性幅による重みを持つプール相関であり、UniProt別相関の単純平均ではない。', '- 全体rとwithin-target rは別の問いに答える。MAEの改善だけでは群内の活性差・順位をよりよく予測したとは断定しない。', '- 群内平均には評価データを使うが、固定予測を診断する統計処理にのみ使用し、モデル入力・学習・選択には用いない。', '- per_uniprot_metrics.csvに全対象、SMILES種類数、指標内訳、併記IDの有無、定数群フラグを収録。', '- source hash・解析条件はaudit.json、bootstrap全試行は各*_bootstrap.csvに保存。']
report = '\n'.join(lines) + '\n'
if IS_V2:
    report = report.replace('従来GEO＋PIEDAを対象とし、v2ではない。', '周辺PIEDAを入力するGEO＋PIEDA_v2を対象とする。')
    report = report.replace('GEO＋PIEDA', 'GEO＋PIEDA_v2').replace('GEO＋PIEDA_v2_v2', 'GEO＋PIEDA_v2')
    report = report.replace('＋PIEDA MAE', '＋PIEDA_v2 MAE').replace('＋PIEDA RMSE', '＋PIEDA_v2 RMSE').replace('＋PIEDA ρ', '＋PIEDA_v2 ρ')
(OUT / 'report.md').write_text(report)
print(json.dumps(inventory, ensure_ascii=False, indent=2), flush=True)
print(pd.DataFrame(overall).to_string(index=False), flush=True)
print(pd.DataFrame(analyses).to_string(index=False), flush=True)
print(per_id[['uniprot', 'name', 'n_train', 'n_test', 'mae_geo', 'mae_pieda', 'mae_improvement', 'spearman_geo', 'spearman_pieda']].to_string(index=False), flush=True)
