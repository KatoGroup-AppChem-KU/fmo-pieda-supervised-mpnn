"""Build one graph from existing FMO fragment tables using the reviewed code.

Wrapper by Hirokazu Okamoto; the builders/QC retain their recorded authorship.
No upstream FMO dataset reconstruction or new FMO calculation is performed.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def apply_ligand_water_fp(graph, smiles, builder):
    """Reuse the FP assignment in rebuild_bdb_fp_v2_20261016.py.

    SMILES must already have been adopted/validated for the input structure.
    This function does not reproduce ligand-identity adjudication.
    """
    import torch
    names = list(graph.node_feature_names)
    cols = torch.tensor([names.index('small_mol_morgan_'+str(k)) for k in range(512)])
    water = graph.x[:, names.index('is_water')] > .5
    assignments = [(water, 'O')]
    if smiles:
        ligand = graph.x[:, names.index('is_ligand')] > .5
        assert int(ligand.sum()) == 1 and not torch.any(ligand & water)
        assignments.append((ligand, smiles))
    for mask, identity in assignments:
        value = torch.tensor(builder.morgan_fingerprint(identity))
        rows = mask.nonzero().flatten()
        graph.x[rows[:, None], cols[None, :]] = value.to(graph.x.dtype)
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes', type=Path, required=True)
    parser.add_argument('--edges', type=Path, required=True)
    parser.add_argument('--key', required=True)
    parser.add_argument('--esm', type=Path)
    parser.add_argument('--smiles', default='')
    parser.add_argument('--paper-fpv2', action='store_true',
                        help='Apply adopted ligand/water Morgan assignment; supplied ligand SMILES must already be validated.')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent/'runtime'))
    import pandas as pd
    import torch
    import audit_fmo_pieda_v06_provisional_exclusion as qc
    name = 'build_fmo_fragment_pyg_graph_v6' if args.esm else 'build_fmo_fragment_pyg_graph_v7_morganunified'
    builder = importlib.import_module(name)
    edges = pd.read_csv(args.edges)
    selected = edges[edges.DIMER_ES.astype(str).ne('T') & edges.has_pieda.eq(1)
                     & (edges.frag_i-edges.frag_j).abs().gt(1)].copy()
    if selected.empty:
        raise ValueError('No fragment pairs pass the adopted selection rules.')
    args.out.mkdir(parents=True, exist_ok=False)
    selected['PIEDA_sum'] = selected[['ES', 'EX', 'CTmix', 'DI']].sum(axis=1)
    selected.to_csv(args.out/'selected_edges.csv', index=False)
    flags, excluded = qc.audit_one_pdb(args.key, args.out/'selected_edges.csv')
    bad = set(zip(excluded.frag_i, excluded.frag_j))
    selected = selected[[pair not in bad for pair in zip(selected.frag_i, selected.frag_j)]].copy()
    if selected.empty:
        raise ValueError('No pairs remain after v06prov QC.')
    selected.to_csv(args.out/'edges_qc.csv', index=False)
    settings = SimpleNamespace(pdbid=args.key, smiles=args.smiles, ligand_table=None,
        pdbid_col='pdbid', smiles_col='het_smiles', hetid_col='hetid', ccd_smiles_json='',
        morgan_bits=512, morgan_radius=2, no_chirality=False, no_water_smiles=False,
        keep_small_mol_hs=False, water_bridge_dist_cutoff=4.0, rbf_k=32,
        dist_rbf_max=10.0, centroid_rbf_max=30.0, directed=False)
    if args.esm:
        builder.build_one(args.nodes, args.out/'edges_qc.csv', args.esm,
                          args.out/'graph.pt', args.out/'graph_audit.json', settings)
    else:
        builder.build_one(args.nodes, args.out/'edges_qc.csv', args.out/'graph.pt',
                          args.out/'graph_audit.json', settings)
    graph = torch.load(args.out/'graph.pt', map_location='cpu', weights_only=False)
    if args.paper_fpv2:
        graph = apply_ligand_water_fp(graph, args.smiles, builder)
        torch.save(graph, args.out/'graph.pt')
    assert graph.edge_index.shape[1] == 2*len(selected)
    assert torch.isfinite(graph.x).all() and torch.isfinite(graph.edge_attr).all()
    report = dict(key=args.key, nodes=len(graph.x), node_features=graph.x.shape[1],
                  directed_edges=graph.edge_index.shape[1], qc_flags=flags,
                  builder=name, upstream_fmo_recomputed=False, paper_ligand_water_fp=args.paper_fpv2,
                  note='Builder/QC output; not a certification of full paper FPv2 adoption.')
    (args.out/'construction.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
