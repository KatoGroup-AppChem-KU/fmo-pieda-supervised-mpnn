"""Exploratory ligand similarity versus fixed-prediction error changes."""
import os
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
assert True
root = Path(__import__('os').environ['FMO_WORKDIR'])
sources = {'pairs': root / 'results/ligand_delta_px_20261105/20261105_リガンド間ΔpX_全ペア.csv', 'test': root / 'results/uniprot_delta_deduplicated_20261105/20261105_重複整理後_評価レコード301件.csv', 'train': root / 'results/target_training_counts_20261105/20261105_pX訓練2826構造_UniProtとリガンド.csv'}
out = root / 'results/ligand_similarity_20261106'
out.mkdir(exist_ok=False)
pairs, test, train = [pd.read_csv(sources[k]) for k in ['pairs', 'test', 'train']]
assert len(pairs) == 245 and pairs.uniprot.nunique() == 64
assert len(test) == 301 and len(train) == 2826
assert set(train.split) == {'train'} and train.pdbid.is_unique
test_pdbs = {p for s in test.pdbids for p in s.split(';')}
assert not test_pdbs.intersection(train.pdbid)
models = [('v1', 'GEO+PIEDA'), ('v2', 'GEO+PIEDA_v2')]
generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512, includeChirality=True)
cache = {}

def fp(smiles):
    if smiles not in cache:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError('Cannot parse SMILES: ' + smiles)
        cache[smiles] = generator.GetFingerprint(mol)
    return cache[smiles]
pairs['test_pair_tanimoto'] = [DataStructs.TanimotoSimilarity(fp(a), fp(b)) for a, b in zip(pairs.ligand_A_SMILES, pairs.ligand_B_SMILES)]
assert not pairs.duplicated(['uniprot', 'ligand_A_SMILES', 'ligand_B_SMILES']).any()
for _, m in models:
    np.testing.assert_allclose(pairs[f'{m}_delta_abs_error_vs_GEO'], pairs[f'{m}_abs_delta_px_error'] - pairs.GEO_abs_delta_px_error, atol=1e-12)
special = {'2ito', '2itz', '6rul', '7nae'}
eligible = test[(test.evaluation_target_kind == 'single_UniProt') & ~test.pdbid.isin(special)].copy()
assert len(eligible) == 296
assert not eligible.duplicated(['evaluation_target_key', 'canonical_smiles']).any()
single_train = train[(train.n_assigned_ids == 1) & train.evaluation_target_key.notna()].copy()
train_groups = {key: g for key, g in single_train.groupby('evaluation_target_key')}
rows = []
for _, r in eligible.iterrows():
    key, smiles = (r.evaluation_target_key, r.canonical_smiles)
    g = train_groups.get(key)
    row = {'uniprot': key, 'pdbids': r.pdbids, 'test_SMILES': smiles, 'observed_px': r.target_px, 'affinity_type': r.affinity_type, 'n_original_structures': r.n_original_structures, 'in_64_pair_targets': key in set(pairs.uniprot)}
    if g is None or len(g) == 0:
        row.update(status='no_same_UniProt_Train', n_train_structures=0, n_train_ligands=0)
    else:
        assert set(g.split) == {'train'}
        smiles_list = sorted(g.canonical_smiles.unique())
        similarities = np.array(DataStructs.BulkTanimotoSimilarity(fp(smiles), [fp(s) for s in smiles_list]))
        best = similarities.max()
        matched = [s for s, v in zip(smiles_list, similarities) if abs(v - best) < 1e-12]
        row.update(status='included', n_train_structures=len(g), n_train_ligands=len(smiles_list), max_train_tanimoto=float(best), nearest_train_SMILES=';'.join(matched), nearest_train_PDBs=';'.join(sorted(g[g.canonical_smiles.isin(matched)].pdbid)), exact_same_SMILES_in_Train=smiles in smiles_list)
    for _, m in [('GEO', 'GEO')] + models:
        row[f'{m}_predicted_px'] = r[m]
        row[f'{m}_abs_px_error'] = abs(r[m] - r.target_px)
    for _, m in models:
        row[f'{m}_delta_abs_px_error_vs_GEO'] = row[f'{m}_abs_px_error'] - row['GEO_abs_px_error']
    rows.append(row)
test_sim = pd.DataFrame(rows)
included = test_sim[test_sim.status == 'included'].copy()
assert np.isfinite(included.max_train_tanimoto).all()
assert pairs.test_pair_tanimoto.between(0, 1).all() and included.max_train_tanimoto.between(0, 1).all()
results = []
scopes = [('Test_pair_similarity', pairs, 'test_pair_tanimoto', 'delta_abs_error_vs_GEO'), ('Test_Train_similarity_all_eligible', included, 'max_train_tanimoto', 'delta_abs_px_error_vs_GEO'), ('Test_Train_similarity_64_pair_targets', included[included.in_64_pair_targets], 'max_train_tanimoto', 'delta_abs_px_error_vs_GEO')]
for scope, frame, xkey, suffix in scopes:
    for display, m in models:
        ykey = f'{m}_{suffix}'
        rho = spearmanr(frame[xkey], frame[ykey]).statistic
        results.append({'analysis': scope, 'condition': display, 'n_points': len(frame), 'n_UniProt': frame.uniprot.nunique(), 'spearman_rho': float(rho), 'x': xkey, 'y': ykey, 'mean_error_change': frame[ykey].mean(), 'median_error_change': frame[ykey].median()})
correlations = pd.DataFrame(results)
font_manager.fontManager.addfont(str(root / 'outputs/20261105_arial.ttf'))
plt.rcParams.update({'font.family': 'Arial', 'font.size': 11, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
for scope, frame, xkey, suffix in scopes[:2]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (display, m), color in zip(axes, models, ['#0072B2', '#D55E00']):
        ykey = f'{m}_{suffix}'
        ax.scatter(frame[xkey], frame[ykey], s=28, alpha=0.65, color=color, edgecolors='none')
        ax.axhline(0, color='black', ls='--', lw=1)
        rho = correlations[(correlations.analysis == scope) & (correlations.condition == display)].iloc[0].spearman_rho
        ax.set(title=f'{display}: Spearman rho = {rho:+.3f} (N={len(frame)})', xlabel='Test ligand-pair Tanimoto similarity' if scope == 'Test_pair_similarity' else 'Maximum Tanimoto to same-UniProt pX Train ligands', ylabel='Change in absolute delta-pX error\n(model - GEO)' if scope == 'Test_pair_similarity' else 'Change in absolute pX error\n(model - GEO)', xlim=(-0.02, 1.02))
    fig.suptitle('Morgan radius 2, 512 bits, chirality | Negative y: improvement')
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(out / f'{scope}_vs_error_change.{ext}', dpi=250, bbox_inches='tight')
    plt.close(fig)
for suffix, frame in [('Testリガンドペア.csv', pairs), ('TestとTrain.csv', test_sim), ('Spearman相関.csv', correlations)]:
    frame.to_csv(out / ('20261106_リガンド類似性_' + suffix), index=False, encoding='utf-8-sig')
audit = {'complete': True, 'partition': os.environ.get('SLURM_JOB_PARTITION', 'standalone'), 'fingerprint': {'radius': 2, 'bits': 512, 'includeChirality': True, 'RDKit': rdBase.rdkitVersion, 'standardization': 'existing canonical isomeric SMILES; no additional salt, tautomer, or protonation normalization'}, 'n_pairs': len(pairs), 'n_pair_targets': pairs.uniprot.nunique(), 'n_test_eligible': len(test_sim), 'n_test_with_same_UniProt_Train': len(included), 'n_test_targets': included.uniprot.nunique(), 'n_train_single_UniProt': len(single_train), 'n_test_exact_SMILES_in_Train': int(included.exact_same_SMILES_in_Train.sum()), 'no_retraining': True, 'no_inference': True, 'delta': 'model minus GEO; negative is improvement', 'limitations': ['Descriptive pooled-point Spearman; no independent-pair assumption, p-values, or confidence intervals', 'Targets with more ligands contribute more points; pairs share ligands', 'Train/test target assignment and reviewed-construct exclusions inherited from prior analyses', 'Train variants not exhaustively matched to Test constructs; single-UniProt mapping is the matching rule', 'Mixed affinity endpoints and assays, no assay adjustment', 'Morgan hash collisions and stereochemistry/molecular-preparation conventions affect similarity', 'v2 representations retain surrounding PIEDA context and are not FMO-free'], 'source_sha256': {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in sources.items()}, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(out / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
print(correlations.to_string(index=False), flush=True)
