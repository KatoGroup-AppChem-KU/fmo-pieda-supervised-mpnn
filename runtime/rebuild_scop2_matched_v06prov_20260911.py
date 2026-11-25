"""Rebuild matched scop2 ESM+Morgan / Morgan-only graphs on aurum-g01."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
import build_fmo_fragment_pyg_graph_v7_morganunified_esm as esm_builder
import build_fmo_fragment_pyg_graph_v7_morganunified as mfp_builder
import audit_fmo_pieda_v06_provisional_exclusion as qc
import pretrain_masked_edge_multigraph_v17r_rawdist as pretrain

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    require(True, 'Computation must run in a Slurm allocation on aurum-g01')
    torch.set_num_threads(1)
    RDLogger.DisableLog('rdApp.warning')
    root = args.out_root
    require(not root.exists(), f'Use a fresh output root: {root}')
    paths = sorted(Path('dataset/edges_v4_fmofrag_scop2_apomonomer').glob('*_edges.csv'))
    require(len(paths) == 5277, f'Expected 5277 source CSVs, found {len(paths)}')
    if args.limit:
        paths = paths[:args.limit]
    for p in paths:
        key = p.name.removesuffix('_edges.csv')
        for f in [Path('dataset/nodes_v4_fmofrag_scop2_apomonomer') / f'{key}_nodes.csv', Path('dataset/embeddings') / f'{key}_esm.npy']:
            require(f.is_file(), f'Missing input: {f}')
    root.mkdir(parents=True)
    for d in ['edges_pruned', 'edges_qc', 'graph_esm', 'graph_noesm', 'audits']:
        (root / d).mkdir()
    manifest = dict(partition=os.environ.get('SLURM_JOB_PARTITION', 'standalone'), job_id=os.environ.get('SLURM_JOB_ID'), expected_inputs=len(paths), dist_cutoff=None, edge_rules=['DIMER_ES != T', 'has_pieda == 1', 'abs(frag_i-frag_j) > 1'], qc='v06prov via existing audit_one_pdb; applied before graph normalization', max_n_edges_for_future_combination=50000, max_edges_applied_here=False, ligand_strict_applied=False, feature_variants={'graph_esm': 1026, 'graph_noesm': 546}, script_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in [__file__, 'build_fmo_fragment_pyg_graph_v7_morganunified_esm.py', 'build_fmo_fragment_pyg_graph_v7_morganunified.py', 'audit_fmo_pieda_v06_provisional_exclusion.py', 'pretrain_masked_edge_multigraph_v17r_rawdist.py']})
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    records = []
    for i, p in enumerate(paths, 1):
        key = p.name.removesuffix('_edges.csv')
        nodes_path = Path('dataset/nodes_v4_fmofrag_scop2_apomonomer') / f'{key}_nodes.csv'
        emb_path = Path('dataset/embeddings') / f'{key}_esm.npy'
        nodes = pd.read_csv(nodes_path)
        require(nodes.frag_id.is_unique and nodes.frag_id.is_monotonic_increasing, f'{key}: node ordering must match row-aligned ESM')
        require(not nodes.is_ligand.fillna(0).astype(bool).any(), f'{key}: unexpected ligand')
        emb = np.load(emb_path)
        require(emb.shape == (len(nodes), 480) and np.isfinite(emb).all(), f'{key}: invalid embedding')
        expected = np.zeros(len(nodes), dtype=bool)
        for _, group in nodes.groupby('rep_chain', sort=False):
            standard = group.is_standard_aa.fillna(0).astype(int) == 1
            aa = group.rep_aa1.fillna('').astype(str).str.strip().str.upper()
            idx = group.index[standard & aa.isin(set('ACDEFGHIKLMNPQRSTVWY'))][:1022]
            expected[idx] = True
        require((np.linalg.norm(emb[expected], axis=1) > 0).all(), f'{key}: unexpected zero ESM')
        e = pd.read_csv(p)
        mask = (e.DIMER_ES.astype(str) != 'T') & (e.has_pieda == 1) & ((e.frag_i - e.frag_j).abs() > 1)
        kept = e.loc[mask].copy().reset_index(drop=True)
        columns = ['DIST', 'centroid_distance', 'ES', 'EX', 'CTmix', 'DI', 'q_I_to_J']
        require(np.isfinite(kept[columns].to_numpy(float)).all(), f'{key}: invalid numeric edge data')
        require(not kept[['frag_i', 'frag_j']].duplicated().any(), f'{key}: duplicate pair')
        kept['PIEDA_sum'] = kept[['ES', 'EX', 'CTmix', 'DI']].sum(axis=1)
        pruned = root / 'edges_pruned' / p.name
        kept.to_csv(pruned, index=False)
        flags, excluded = qc.audit_one_pdb(key, pruned)
        if not excluded.empty:
            excluded.to_csv(root / 'audits' / f'{key}_excluded_pairs.csv', index=False)
        excluded_pairs = set(zip(excluded.frag_i, excluded.frag_j)) if not excluded.empty else set()
        final = kept.loc[[pair not in excluded_pairs for pair in zip(kept.frag_i, kept.frag_j)]].copy()
        final_path = root / 'edges_qc' / p.name
        final.to_csv(final_path, index=False)
        record = dict(key=key, n_nodes=len(nodes), original_pairs=len(e), pruned_pairs=len(kept), qc_removed_pairs=len(excluded), final_pairs=len(final), directed_edges=2 * len(final), eligible_max_edges_50000=bool(len(final) and 2 * len(final) <= 50000), max_dist=float(final.DIST.max()) if len(final) else None, qc_flags=flags, status='empty_after_filter' if final.empty else 'ok')
        if final.empty:
            records.append(record)
            print(json.dumps(record), flush=True)
            continue
        build_args = SimpleNamespace(pdbid=key, smiles='', ligand_table=None, pdbid_col='pdbid', smiles_col='het_smiles', hetid_col='hetid', ccd_smiles_json='ccd/ccd_other_hetero_smiles.json', morgan_bits=512, morgan_radius=2, no_chirality=False, no_water_smiles=False, keep_small_mol_hs=False, water_bridge_dist_cutoff=4.0, rbf_k=32, dist_rbf_max=10.0, centroid_rbf_max=30.0, directed=False)
        esm_out = root / 'graph_esm' / f'{key}.pt'
        noesm_out = root / 'graph_noesm' / f'{key}.pt'
        with (root / 'audits' / f'{key}_build.log').open('w') as log, contextlib.redirect_stdout(log):
            esm_builder.build_one(nodes_path, final_path, emb_path, esm_out, root / 'audits' / f'{key}_esm_graph_audit.json', build_args)
            mfp_builder.build_one(nodes_path, final_path, noesm_out, root / 'audits' / f'{key}_noesm_graph_audit.json', build_args)
        a = torch.load(esm_out, map_location='cpu', weights_only=False)
        b = torch.load(noesm_out, map_location='cpu', weights_only=False)
        require(tuple(a.x.shape) == (len(nodes), 1026) and tuple(b.x.shape) == (len(nodes), 546), f'{key}: feature dimensions')
        require(torch.equal(a.x[:, 480:], b.x), f'{key}: MFP/AA/flag mismatch')
        require(np.array_equal(a.x[:, :480].numpy(), emb), f'{key}: ESM alignment')
        require(torch.equal(a.edge_index, b.edge_index) and torch.equal(a.edge_attr, b.edge_attr), f'{key}: graph topology/features mismatch')
        require(a.edge_index.shape[1] == 2 * len(final), f'{key}: edge count mismatch')
        require(torch.isfinite(a.x).all() and torch.isfinite(a.edge_attr).all() and torch.isfinite(a.pos).all(), f'{key}: non-finite graph')
        raw_cols = ['PIEDA_sum', 'ES', 'EX', 'CTmix', 'DI', 'q_I_to_J']
        raw = final[raw_cols].to_numpy(float)
        recovered = a.edge_attr[:len(final), 66:72].numpy().astype(float) * np.array(list(a.fmo_std.values())) + np.array(list(a.fmo_mean.values()))
        require(np.allclose(raw, recovered, rtol=1e-05, atol=2e-05), f'{key}: raw PIEDA reconstruction')
        for g, out in [(a, esm_out), (b, noesm_out)]:
            g = pretrain.add_pair_cache(g)
            g.rebuild_provenance = {'manifest': str(root / 'manifest.json'), 'source_key': key}
            torch.save(g, out)
            loaded = torch.load(out, map_location='cpu', weights_only=False)
            require(loaded.edge_index.shape[1] == 2 * len(final), f'{key}: saved file check')
        records.append(record)
        (root / 'audits' / f'{key}_verified.json').write_text(json.dumps(record, indent=2) + '\n')
        if i <= 3 or i % 100 == 0 or i == len(paths):
            print(f'[{i}/{len(paths)}] {key}: pairs {len(e)} -> {len(kept)} -> {len(final)}', flush=True)
    pd.DataFrame(records).to_csv(root / 'graph_inventory.csv', index=False)
    summary = dict(inputs=len(records), graphs=sum((r['status'] == 'ok' for r in records)), empty_graph_keys=[r['key'] for r in records if r['status'] != 'ok'], total_original_pairs=sum((r['original_pairs'] for r in records)), total_pruned_pairs=sum((r['pruned_pairs'] for r in records)), total_final_pairs=sum((r['final_pairs'] for r in records)), max_directed_edges=max((r['directed_edges'] for r in records)), eligible_max_edges_50000=sum((r['eligible_max_edges_50000'] for r in records)), max_dist=max((r['max_dist'] for r in records if r['max_dist'] is not None), default=None))
    for d in ['graph_esm', 'graph_noesm']:
        require(len(list((root / d).glob('*.pt'))) == summary['graphs'], f'{d}: final file count')
    (root / 'COMPLETE.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
if __name__ == '__main__':
    main()
