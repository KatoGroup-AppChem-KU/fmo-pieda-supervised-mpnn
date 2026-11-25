"""Rebuild ligand/water features in copies; keep all FMO graph tensors unchanged."""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
import networkx as nx
import torch
from rdkit import Chem, RDLogger
assert True
ROOT = Path(__import__('os').environ['FMO_WORKDIR'])
sys.path.insert(0, str(ROOT))
from build_fmo_fragment_pyg_graph_v6 import morgan_fingerprint, rdkit_smiles_atom_graph
from prepare_geo_pieda_matched_20260911 import fingerprint, load
RDLogger.DisableLog('rdApp.warning')
torch.set_num_threads(1)
ap = argparse.ArgumentParser()
ap.add_argument('--mode', choices=['smoke', 'full'], required=True)
args = ap.parse_args()
out = ROOT / ('dataset/bdb_fp_v2_20261016_r2_' + args.mode)
out.mkdir(exist_ok=False)
excluded = set('1kyn 1qbv 2ity 1nde 2xno 4eil 4mdn 5xvg'.split())
affinity_excluded = set('1pxo 3sap 3sbh 4qsa 4qsb 4ww6 5fcw'.split())
sources = {}
auditpath = ROOT / 'results/input_pdb_public_ccd_20261015/per_structure.csv'
for r in csv.DictReader(auditpath.open()):
    if r['status'] == 'heavy_connectivity_match':
        sources[r['pdb_id'].lower()] = {'smiles': r['ccd_smiles'], 'source': 'public_CCD', 'ligand_id': r['ligand_id'], 'input_pdb': r['input_pdb']}
assert len(sources) == 3675
bdb = defaultdict(set)
for r in csv.DictReader((ROOT / 'results/input_mismatch_bdb_20261016/matching_records.csv').open()):
    bdb[r['pdb_id'].lower()].add(r['smile'])
details = {r['pdb_id']: r for r in json.loads((ROOT / 'results/public_mismatch_review_20261015/details.json').read_text())}
for pid, smiles in bdb.items():
    if pid == '3nxr':
        continue
    assert len(smiles) == 1, (pid, smiles)
    sources[pid] = {'smiles': next(iter(smiles)), 'source': 'input_matching_BindingDB', 'ligand_id': details[pid]['ligand_id'], 'input_pdb': details[pid]['input_pdb']}
for r in json.loads((ROOT / 'results/smiles19_candidates_20261016_v2/details.json').read_text()):
    assert r['connectivity_roundtrip_match']
    sources[r['pdb_id']] = {'smiles': r['candidate_smiles'], 'source': 'primary_paper_double_bond_input_stereo' if r['pdb_id'] == '3nxr' else 'CCD_input_substructure', 'ligand_id': r['ligand_id'], 'input_pdb': r['input_pdb']}
assert len(sources) == 3706 and (not set(sources) & excluded)

def input_match(s):
    g = nx.Graph()
    con = []
    for line in Path(s['input_pdb']).read_text().splitlines():
        if line.startswith(('ATOM  ', 'HETATM')) and line[17:20].strip() == s['ligand_id']:
            e = line[76:78].strip().title()
            if e not in ('H', 'D'):
                g.add_node(int(line[6:11]), element=e)
        elif line.startswith('CONECT'):
            con.append(line)
    for line in con:
        ns = [int(line[j:j + 5]) for j in range(6, len(line), 5) if line[j:j + 5].strip()]
        for b in ns[1:]:
            if ns[0] in g and b in g:
                g.add_edge(ns[0], b)
    m = Chem.MolFromSmiles(s['smiles'])
    assert m is not None
    mg = nx.Graph()
    for a in m.GetAtoms():
        if a.GetAtomicNum() > 1:
            mg.add_node(a.GetIdx(), element=a.GetSymbol())
    for b in m.GetBonds():
        if b.GetBeginAtomIdx() in mg and b.GetEndAtomIdx() in mg:
            mg.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    assert nx.is_isomorphic(g, mg, node_match=lambda a, b: a['element'] == b['element'])
    return len(g)

def eq(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return set(a) == set(b) and all((eq(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all((eq(x, y) for x, y in zip(a, b)))
    return a == b
manifest_path = ROOT / 'dataset/final_train_20261008/manifest.json'
old_manifest = json.loads(manifest_path.read_text())
records = [r for r in old_manifest['records'] if r['source'] == 'bdb' and r['pdb'] not in excluded]
assert len(records) == 3706 and {r['pdb'] for r in records} == set(sources)
if args.mode == 'smoke':
    picks = set('1a27 1c5o 1hut 3nxr 1pxo 3sap 2src 4d0x 3iit'.split())
    records = [r for r in records if r['pdb'] in picks]
    assert len(records) == len(picks)
water = torch.tensor(morgan_fingerprint('O'))
assert water.sum() > 0
allowed = {'x', 'ligand_smiles', 'bound_ligand_smiles', 'reference_ligand_smiles', 'bound_ligand_atom_x', 'bound_ligand_edge_index', 'bound_ligand_edge_attr', 'bound_ligand_atom_feature_names', 'bound_ligand_bond_feature_names', 'has_bound_ligand_atom_graph', 'small_mol_smiles', 'small_mol_graph_sources', 'small_mol_smiles_parse_statuses', 'small_mol_atom_x_list', 'small_mol_edge_index_list', 'small_mol_edge_attr_list', 'small_mol_pos_list', 'fp_rebuild_provenance'}
newrecords = []
audits = []
for j, r in enumerate(records, 1):
    pid = r['pdb']
    s = sources[pid]
    nh = input_match(s)
    fp = torch.tensor(morgan_fingerprint(s['smiles']))
    assert fp.sum() > 0
    ax, ei, ea, an, bn, status = rdkit_smiles_atom_graph(s['smiles'])
    assert an[0] == 'atomic_num' or 'atomic' in an[0].lower(), an
    atomic_numbers = (ax[:, 0] * 100).round().astype(int)
    assert int((atomic_numbers > 1).sum()) == nh, (pid, nh, atomic_numbers.tolist())
    tensors = [torch.from_numpy(x) for x in (ax, ei, ea)]
    nr = json.loads(json.dumps(r))
    gs = {}
    for v in ('v6', 'v7'):
        old = load(ROOT / r['paths'][v])
        assert fingerprint(old) == r['fingerprints'][v]
        g = old.clone()
        names = list(g.node_feature_names)
        cols = torch.tensor([names.index('small_mol_morgan_' + str(k)) for k in range(512)])
        lig = g.x[:, names.index('is_ligand')] > 0.5
        wat = g.x[:, names.index('is_water')] > 0.5
        assert int(lig.sum()) == 1 and (not torch.any(lig & wat))
        if hasattr(g, 'is_ligand'):
            assert torch.equal(lig, g.is_ligand.bool().reshape(-1))
        for mask, value in ((lig, fp), (wat, water)):
            rows = mask.nonzero().flatten()
            g.x[rows[:, None], cols[None, :]] = value.to(g.x.dtype)
        restored = g.x.clone()
        rows = (lig | wat).nonzero().flatten()
        restored[rows[:, None], cols[None, :]] = old.x[rows[:, None], cols[None, :]]
        assert torch.equal(restored, old.x)
        for k in ('ligand_smiles', 'bound_ligand_smiles', 'reference_ligand_smiles'):
            setattr(g, k, s['smiles'])
        for k, t in zip(('bound_ligand_atom_x', 'bound_ligand_edge_index', 'bound_ligand_edge_attr'), tensors):
            setattr(g, k, t)
        g.bound_ligand_atom_feature_names = an
        g.bound_ligand_bond_feature_names = bn
        g.has_bound_ligand_atom_graph = True
        frag = int(g.frag_id.reshape(-1)[lig][0])
        if hasattr(g, 'small_mol_frag_ids'):
            ids = list(g.small_mol_frag_ids)
            assert ids.count(frag) == 1
            i = ids.index(frag)
            values = {'small_mol_smiles': s['smiles'], 'small_mol_graph_sources': s['source'], 'small_mol_smiles_parse_statuses': status, 'small_mol_atom_x_list': tensors[0], 'small_mol_edge_index_list': tensors[1], 'small_mol_edge_attr_list': tensors[2], 'small_mol_pos_list': torch.zeros((ax.shape[0], 3))}
            for k, val in values.items():
                if hasattr(g, k):
                    arr = list(getattr(g, k))
                    arr[i] = val
                    setattr(g, k, arr)
        g.fp_rebuild_provenance = dict(s, water_smiles='O', radius=2, bits=512, chirality=True, hydrogens_not_identity_criterion=True)
        for k in old.keys():
            if k not in allowed:
                assert eq(old[k], g[k]), (pid, v, k)
        dest = out / v / (pid + '.pt')
        dest.parent.mkdir(exist_ok=True)
        torch.save(g, dest)
        again = load(dest)
        assert set(again.keys()) == set(g.keys()) and all((eq(g[k], again[k]) for k in g.keys()))
        nr['paths'][v] = str(dest)
        nr['fingerprints'][v] = fingerprint(g)
        gs[v] = g
        audits.append(dict(pdb_id=pid, version=v, source=s['source'], heavy_atoms=nh, atom_graph_nodes=int(ax.shape[0]), retained_hydrogens=int((atomic_numbers == 1).sum()), water_nodes=int(wat.sum()), ligand_fp_changed=not torch.equal(old.x[lig][:, cols], g.x[lig][:, cols]), other_attributes_unchanged=True))
    assert torch.equal(gs['v6'].x[:, 480:514], gs['v7'].x[:, :34])
    nr['affinity_excluded_by_structure'] = pid in affinity_excluded
    nr['fp_source'] = s
    newrecords.append(nr)
    if j % 100 == 0 or args.mode == 'smoke':
        print(f'Verified {j}/{len(records)}: {pid}', flush=True)
    (out / 'progress.json').write_text(json.dumps({'completed_structures': j, 'total': len(records)}))
(out / 'smiles_manifest.json').write_text(json.dumps(sources, indent=2))
(out / 'graph_manifest.json').write_text(json.dumps({'records': newrecords, 'source_manifest': str(manifest_path), 'split_assignments_preserved': True, 'training_normalization_recompute_required': True, 'not_training_ready': True, 'exclusions': sorted(excluded)}, indent=2))
(out / 'audit.json').write_text(json.dumps(audits, indent=2))
(out / 'COMPLETE.json').write_text(json.dumps({'complete': True, 'structures': len(records), 'graphs': len(audits), 'mode': args.mode, 'all_verified': True, 'old_graphs_modified': False, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'smiles_sources': dict(Counter((sources[r['pdb']]['source'] for r in records)))}, indent=2))
print('Rebuild complete; training manifests/statistics not yet regenerated.', flush=True)
