"""Build PyTorch Geometric graphs from v4 FMO-fragment nodes and edges. Morgan-FP-unified v7.

Derived from script_by_kkato/build_fmo_fragment_pyg_graph_v6.py (read-only original,
not modified). Differences from v6:

  - ESM embeddings are retained as the leading 480 node features. Embeddings
    produced by batch_extract_esm_all.py are aligned to node CSV rows; compact legacy
    arrays remain supported through the esm_index column.
  - Standard amino-acid residue fragments (is_standard_aa) now also get a Morgan
    fingerprint, computed from a fixed free-amino-acid SMILES dictionary keyed by
    rep_aa1 (the same key already used for the AA one-hot block). Ambiguous codes
    (B, Z, X) have no defined structure and are left as all-zero, consistent with
    morgan_fingerprint("")'s existing behavior for missing SMILES.
  - Water fragments (is_water) also get a Morgan fingerprint from SMILES "O" by
    default (disable with --no-water-smiles).
  - AA one-hot (25) and flags (9) blocks are kept unchanged, as identifiers.
  - Ligand / other_hetero SMILES resolution and atom-graph construction
    (collect_small_molecule_graphs) is unchanged from v6.

Output graph_version is "graph_v7_fmofrag_morganunified_esm" so it cannot be
confused with no-ESM v7 caches. This script and its outputs are intentionally fully
separate from existing graph_v4_fmofrag / graph_v6_* datasets and checkpoints;
it does not read, write, or delete anything under those paths unless pointed at
them explicitly via --out-dir / --out-pt (callers should use new output dirs).

Node feature layout (x): concat([ESM(480), AA one-hot(25), flags(9),
Morgan FP(default 512)]) -> default total dim = 1026.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
import numpy as np
import pandas as pd
import torch
try:
    from torch_geometric.data import Data
except Exception as e:
    raise RuntimeError('torch_geometric is required for this script') from e
AA_ORDER = list('ACDEFGHIKLMNPQRSTVWYUOBZX')
AA_TO_IDX = {a: i for i, a in enumerate(AA_ORDER)}
AA_FREE_SMILES: Dict[str, str] = {'A': 'C[C@@H](C(=O)O)N', 'R': 'C(C[C@@H](C(=O)O)N)CN=C(N)N', 'N': 'C([C@@H](C(=O)O)N)C(=O)N', 'D': 'C([C@@H](C(=O)O)N)C(=O)O', 'C': 'C([C@@H](C(=O)O)N)S', 'E': 'C(CC(=O)O)[C@@H](C(=O)O)N', 'Q': 'C(CC(=O)N)[C@@H](C(=O)O)N', 'G': 'C(C(=O)O)N', 'H': 'C1=C(NC=N1)C[C@@H](C(=O)O)N', 'I': 'CC[C@H](C)[C@@H](C(=O)O)N', 'L': 'CC(C)C[C@@H](C(=O)O)N', 'K': 'C(CCN)C[C@@H](C(=O)O)N', 'M': 'CSCC[C@@H](C(=O)O)N', 'F': 'C1=CC=C(C=C1)C[C@@H](C(=O)O)N', 'P': 'C1C[C@H](NC1)C(=O)O', 'S': 'C([C@@H](C(=O)O)N)O', 'T': 'C[C@H]([C@@H](C(=O)O)N)O', 'W': 'C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O)O)N', 'Y': 'C1=CC(=CC=C1C[C@@H](C(=O)O)N)O', 'V': 'CC(C)[C@@H](C(=O)O)N', 'U': 'C([C@@H](C(=O)O)N)[SeH]', 'O': 'C[C@@H]1CC=N[C@H]1C(=O)NCCCC[C@@H](C(=O)O)N'}
WATER_SMILES = 'O'
FMO_RAW_COLS = ['PIEDA_sum', 'ES', 'EX', 'CTmix', 'DI', 'q_I_to_J']
FMO_Z_COLS = ['PIEDA_sum_z', 'ES_z', 'EX_z', 'CTmix_z', 'DI_z', 'q_I_to_J_z']

def mol_from_smiles_robust(smiles: str, Chem=None):
    """Return (mol, parse_status) using a no-kekulize fallback.

    Same logic as v6 (rescues "Can't kekulize mol" failures for unusual SMILES).
    """
    if smiles is None:
        return (None, 'missing')
    smi = str(smiles).strip()
    if not smi or smi.lower() in {'nan', 'none', 'null'}:
        return (None, 'missing')
    if Chem is None:
        try:
            from rdkit import Chem as _Chem
            Chem = _Chem
        except Exception:
            return (None, 'rdkit_unavailable')
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return (mol, 'normal_ok')
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return (None, 'sanitize_false_failed')
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        return (mol, 'no_kekulize_ok')
    except Exception:
        try:
            mol.UpdatePropertyCache(strict=False)
            return (mol, 'unsanitized_only')
        except Exception:
            return (None, 'sanitize_failed')

def rbf_expand(x: np.ndarray, k: int, xmax: float) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    centers = np.linspace(0.0, xmax, k, dtype=float).reshape(1, -1)
    gamma = 1.0 / ((xmax / max(k - 1, 1)) ** 2 + 1e-12)
    return np.exp(-gamma * (x - centers) ** 2)

def morgan_fingerprint(smiles: str, n_bits: int=512, radius: int=2, use_chirality: bool=True) -> np.ndarray:
    if not smiles:
        return np.zeros(n_bits, dtype=np.float32)
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.DataStructs import ConvertToNumpyArray
    except Exception:
        return np.zeros(n_bits, dtype=np.float32)
    mol, _parse_status = mol_from_smiles_robust(smiles, Chem=Chem)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useChirality=use_chirality)
    arr = np.zeros((n_bits,), dtype=np.int8)
    ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)

def load_ccd_smiles_json(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    if not path.exists():
        raise RuntimeError(f'CCD smiles JSON not found: {path}')
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        key = str(k).strip()
        if not key:
            continue
        if isinstance(v, dict):
            out[key] = v
        else:
            out[key] = {'canonical_smiles': str(v)}
    return out

def pick_ccd_smiles(ccd_entry: Optional[Dict[str, Any]]) -> str:
    if not ccd_entry:
        return ''
    for key in ['canonical_smiles', 'smiles']:
        val = ccd_entry.get(key)
        if val is not None and str(val).strip() and (str(val).lower() != 'nan'):
            return str(val).strip()
    return ''

def refuse_unsafe_output_path(path: Path, kind: str) -> None:
    p = Path(path)
    parts = set(p.parts)
    if p.suffix.lower() in {'.pdb', '.log', '.csv'} and kind == 'graph PT':
        raise RuntimeError(f'Refusing unsafe {kind} output path: {p}')
    if p.suffix.lower() in {'.pdb', '.log'}:
        raise RuntimeError(f'Refusing unsafe {kind} output path with raw-data suffix: {p}')
    if 'fmo_results' in parts:
        raise RuntimeError(f'Refusing unsafe {kind} output path inside fmo_results: {p}')

def rdkit_smiles_atom_graph(smiles: str, remove_hs: bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], str]:
    """Return a 2D molecular graph from SMILES for downstream D-MPNN/GNN use. Unchanged from v6."""
    atom_feature_names = ['atomic_num', 'degree', 'formal_charge', 'num_hs', 'is_aromatic', 'is_in_ring', 'hyb_sp', 'hyb_sp2', 'hyb_sp3', 'hyb_sp3d', 'hyb_sp3d2', 'hyb_other']
    bond_feature_names = ['bond_single', 'bond_double', 'bond_triple', 'bond_aromatic', 'bond_other', 'is_conjugated', 'is_in_ring', 'stereo_none_or_any', 'stereo_specific']

    def empty_with_status(status: str):
        return (np.zeros((0, len(atom_feature_names)), dtype=np.float32), np.zeros((2, 0), dtype=np.int64), np.zeros((0, len(bond_feature_names)), dtype=np.float32), atom_feature_names, bond_feature_names, status)
    if not smiles:
        return empty_with_status('missing')
    try:
        from rdkit import Chem
    except Exception:
        return empty_with_status('rdkit_unavailable')
    mol, parse_status = mol_from_smiles_robust(smiles, Chem=Chem)
    if mol is None:
        return empty_with_status(parse_status)
    if remove_hs:
        try:
            mol = Chem.RemoveHs(mol, sanitize=True)
        except Exception:
            pass
    atom_rows = []
    for atom in mol.GetAtoms():
        try:
            hyb = str(atom.GetHybridization()).upper()
        except Exception:
            hyb = 'OTHER'
        try:
            total_degree = atom.GetTotalDegree()
        except Exception:
            total_degree = atom.GetDegree()
        try:
            formal_charge = atom.GetFormalCharge()
        except Exception:
            formal_charge = 0
        try:
            total_hs = atom.GetTotalNumHs()
        except Exception:
            total_hs = 0
        try:
            is_arom = atom.GetIsAromatic()
        except Exception:
            is_arom = False
        try:
            is_ring = atom.IsInRing()
        except Exception:
            is_ring = False
        atom_rows.append([atom.GetAtomicNum() / 100.0, total_degree / 8.0, formal_charge / 8.0, total_hs / 8.0, float(is_arom), float(is_ring), float(hyb == 'SP'), float(hyb == 'SP2'), float(hyb == 'SP3'), float(hyb == 'SP3D'), float(hyb == 'SP3D2'), float(hyb not in {'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2'})])
    edge_pairs = []
    bond_rows = []
    for bond in mol.GetBonds():
        btype = str(bond.GetBondType()).upper()
        stereo = str(bond.GetStereo()).upper()
        feat = [float(btype == 'SINGLE'), float(btype == 'DOUBLE'), float(btype == 'TRIPLE'), float(btype == 'AROMATIC'), float(btype not in {'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'}), float(bond.GetIsConjugated()), float(bond.IsInRing()), float(stereo in {'STEREONONE', 'STEREOANY'}), float(stereo not in {'STEREONONE', 'STEREOANY'})]
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_pairs.extend([(i, j), (j, i)])
        bond_rows.extend([feat, feat])
    atom_x = np.asarray(atom_rows, dtype=np.float32)
    if edge_pairs:
        edge_index = np.asarray(edge_pairs, dtype=np.int64).T
        edge_attr = np.asarray(bond_rows, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, len(bond_feature_names)), dtype=np.float32)
    return (atom_x, edge_index, edge_attr, atom_feature_names, bond_feature_names, parse_status)

def safe_str_list(series) -> List[str]:
    try:
        return [str(x) for x in series.fillna('').astype(str).tolist()]
    except Exception:
        return []

def load_esm(path: Optional[Path]) -> np.ndarray:
    if path is None:
        raise RuntimeError('ESM embedding is required for an ESM graph')
    if not path.exists():
        raise RuntimeError(f'ESM embedding not found: {path}')
    arr = np.load(path)
    if arr.ndim != 2:
        raise RuntimeError(f'ESM embedding must be 2D: {path} got {arr.shape}')
    if arr.shape[1] != 480:
        raise RuntimeError(f'ESM embedding dimension must be 480: {path} got {arr.shape}')
    if not np.isfinite(arr).all():
        raise RuntimeError(f'ESM embedding contains non-finite values: {path}')
    return arr.astype(np.float32)

def find_esm(pdbid: str, embedding_dir: Optional[Path]) -> Optional[Path]:
    if embedding_dir is None:
        return None
    for pat in [f'{pdbid}_esm.npy', f'{pdbid}*.npy']:
        xs = sorted(embedding_dir.glob(pat))
        if xs:
            return xs[0]
    return None

def make_node_features(nodes: pd.DataFrame, esm: Optional[np.ndarray], small_mol_fp_by_frag_id: Dict[int, np.ndarray], fp_dim: int) -> Tuple[np.ndarray, List[str]]:
    """v7+ESM: x = concat([ESM(esm_dim), AA one-hot(25), flags(9), Morgan FP(fp_dim)])."""
    if esm is None:
        raise RuntimeError('ESM embedding is required for an ESM graph')
    esm_dim = int(esm.shape[1])
    esm_feats = np.zeros((len(nodes), esm_dim), dtype=np.float32)
    if len(esm) == len(nodes):
        esm_feats[:] = esm
    else:
        esm_indices = nodes['esm_index'].fillna(-1).astype(int).to_numpy()
        required_rows = int(esm_indices.max()) + 1 if len(esm_indices) else 0
        if len(esm) < required_rows:
            raise RuntimeError(f'Compact ESM embedding has {len(esm)} rows but esm_index requires {required_rows}')
        for row_idx, esm_idx in enumerate(esm_indices):
            if 0 <= esm_idx < len(esm):
                esm_feats[row_idx] = esm[esm_idx]
    aa_onehot = np.zeros((len(nodes), len(AA_ORDER)), dtype=np.float32)
    for i, aa in enumerate(nodes.get('rep_aa1', pd.Series(['X'] * len(nodes))).fillna('X').astype(str)):
        aa_onehot[i, AA_TO_IDX.get(aa[:1], AA_TO_IDX['X'])] = 1.0
    flags = np.stack([nodes.get('is_standard_aa', pd.Series([0] * len(nodes))).fillna(0).astype(float).to_numpy(), nodes.get('is_hetero', pd.Series([0] * len(nodes))).fillna(0).astype(float).to_numpy(), nodes.get('is_ligand', pd.Series([0] * len(nodes))).fillna(0).astype(float).to_numpy(), nodes.get('is_water', pd.Series([0] * len(nodes))).fillna(0).astype(float).to_numpy(), nodes.get('is_other_hetero', pd.Series([0] * len(nodes))).fillna(0).astype(float).to_numpy(), nodes.get('n_residues_in_fragment', pd.Series([1] * len(nodes))).fillna(1).astype(float).to_numpy(), nodes.get('n_atoms_all', pd.Series([0] * len(nodes))).fillna(0).astype(float).to_numpy(), nodes.get('n_atoms_heavy', pd.Series([0] * len(nodes))).fillna(0).astype(float).to_numpy(), nodes.get('heavy_atom_radius', pd.Series([0] * len(nodes))).fillna(0).astype(float).to_numpy()], axis=1).astype(np.float32)
    flags[:, 5] = np.log1p(flags[:, 5])
    flags[:, 6] = np.log1p(flags[:, 6])
    flags[:, 7] = np.log1p(flags[:, 7])
    flags[:, 8] = flags[:, 8] / 10.0
    small_mol_mat = np.zeros((len(nodes), fp_dim), dtype=np.float32)
    if fp_dim:
        for row_idx, fid in enumerate(nodes['frag_id'].astype(int).to_numpy()):
            fp = small_mol_fp_by_frag_id.get(int(fid))
            if fp is not None and len(fp) == fp_dim:
                small_mol_mat[row_idx] = fp
    x = np.concatenate([esm_feats, aa_onehot, flags, small_mol_mat], axis=1).astype(np.float32)
    names = [f'esm_{i}' for i in range(esm_dim)]
    names += [f'aa_{aa}' for aa in AA_ORDER]
    names += ['is_standard_aa', 'is_hetero', 'is_ligand', 'is_water', 'is_other_hetero', 'log1p_n_residues_in_fragment', 'log1p_n_atoms_all', 'log1p_n_atoms_heavy', 'heavy_atom_radius_scaled']
    names += [f'small_mol_morgan_{i}' for i in range(fp_dim)]
    return (x, names)

def make_edge_features(edges: pd.DataFrame, rbf_k: int, dist_rbf_max: float, centroid_rbf_max: float, directed: bool) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, float], Dict[str, float]]:
    """Unchanged from v6."""
    e = edges.copy()
    for c in FMO_RAW_COLS:
        if c not in e.columns:
            e[c] = 0.0
        e[c] = e[c].fillna(0.0).astype(float)
    fmo_raw = e[FMO_RAW_COLS].to_numpy(float)
    mean = fmo_raw.mean(axis=0)
    std = fmo_raw.std(axis=0)
    std = np.where(std < 1e-08, 1.0, std)
    fmo_z = (fmo_raw - mean.reshape(1, -1)) / std.reshape(1, -1)
    dist = e['DIST'].astype(float).to_numpy()
    cent = e['centroid_distance'].astype(float).to_numpy()
    dist_rbf = rbf_expand(dist, rbf_k, dist_rbf_max)
    cent_rbf = rbf_expand(cent, rbf_k, centroid_rbf_max)
    edge_attr = np.concatenate([dist_rbf, dist.reshape(-1, 1), cent_rbf, cent.reshape(-1, 1), fmo_z], axis=1).astype(np.float32)
    names = [f'DIST_rbf_{i}' for i in range(rbf_k)] + ['DIST_raw']
    names += [f'centroid_distance_rbf_{i}' for i in range(rbf_k)] + ['centroid_distance_raw']
    names += FMO_Z_COLS
    edge_index = e[['src_idx', 'dst_idx']].to_numpy(dtype=np.int64).T
    if not directed:
        edge_index_rev = edge_index[[1, 0], :]
        edge_index = np.concatenate([edge_index, edge_index_rev], axis=1)
        edge_attr = np.concatenate([edge_attr, edge_attr], axis=0)
    mean_d = {k: float(v) for k, v in zip(FMO_Z_COLS, mean)}
    std_d = {k: float(v) for k, v in zip(FMO_Z_COLS, std)}
    return (edge_index, edge_attr, names, mean_d, std_d)

def collect_small_molecule_graphs(nodes: pd.DataFrame, ligand_smiles_by_resname: Dict[str, Tuple[str, str]], provided_smiles: str, ccd_map: Dict[str, Dict[str, Any]], n_bits: int, radius: int, use_chirality: bool, remove_hs: bool) -> Tuple[List[Dict[str, Any]], Dict[int, np.ndarray], List[str], List[str]]:
    """Collect per-fragment small molecule atom graphs for ligand and other_hetero nodes.

    Unchanged from v6: still only ligand/other_hetero get full atom/bond-level graphs
    (used for data.bound_ligand_* / data.small_mol_* D-MPNN-style records). Standard-aa
    and water Morgan FP are computed separately by compute_aa_water_fp_by_frag_id below,
    without building full atom graphs (no consumer needs a per-residue atom graph; only
    the 512-dim Morgan FP feeds into x).
    """
    records: List[Dict[str, Any]] = []
    fp_by_frag_id: Dict[int, np.ndarray] = {}
    atom_feature_names: List[str] = []
    bond_feature_names: List[str] = []
    for _, row in nodes.iterrows():
        kind = str(row.get('fragment_kind', '')).strip()
        if kind not in {'ligand', 'other_hetero'}:
            continue
        frag_id = int(row['frag_id'])
        resname = str(row.get('rep_resname', '')).strip()
        smiles = ''
        source = ''
        if kind == 'ligand':
            if provided_smiles:
                smiles = provided_smiles
                source = 'provided_smiles'
            elif resname in ligand_smiles_by_resname and ligand_smiles_by_resname[resname][0]:
                smiles, source = ligand_smiles_by_resname[resname]
            else:
                entry = ccd_map.get(resname)
                smiles = pick_ccd_smiles(entry)
                source = 'ccd_smiles_for_ligand' if smiles else 'missing_ligand_smiles'
        elif kind == 'other_hetero':
            entry = ccd_map.get(resname)
            smiles = pick_ccd_smiles(entry)
            source = 'ccd_smiles' if smiles else 'missing_ccd_smiles'
        atom_x, edge_index, edge_attr, atom_names, bond_names, smiles_parse_status = rdkit_smiles_atom_graph(smiles, remove_hs=remove_hs)
        if not atom_feature_names:
            atom_feature_names = atom_names
        if not bond_feature_names:
            bond_feature_names = bond_names
        if smiles:
            fp_by_frag_id[frag_id] = morgan_fingerprint(smiles, n_bits=n_bits, radius=radius, use_chirality=use_chirality)
        records.append({'frag_id': frag_id, 'resname': resname, 'fragment_kind': kind, 'smiles': smiles, 'graph_source': source, 'smiles_parse_status': smiles_parse_status, 'atom_x': torch.tensor(atom_x, dtype=torch.float32), 'edge_index': torch.tensor(edge_index, dtype=torch.long), 'edge_attr': torch.tensor(edge_attr, dtype=torch.float32), 'n_atoms': int(atom_x.shape[0]), 'n_edges': int(edge_index.shape[1])})
    return (records, fp_by_frag_id, atom_feature_names, bond_feature_names)

def compute_aa_water_fp_by_frag_id(nodes: pd.DataFrame, n_bits: int, radius: int, use_chirality: bool, include_water: bool) -> Tuple[Dict[int, np.ndarray], Dict[int, str]]:
    """Morgan FP for is_standard_aa fragments (from AA_FREE_SMILES dict, keyed by
    rep_aa1) and optionally is_water fragments (fixed SMILES "O"). No atom/bond graph
    is built, only the FP, since nothing downstream consumes per-residue atom graphs.
    """
    fp_by_frag_id: Dict[int, np.ndarray] = {}
    source_by_frag_id: Dict[int, str] = {}
    is_standard = nodes.get('is_standard_aa', pd.Series([0] * len(nodes))).fillna(0).astype(int).to_numpy().astype(bool)
    is_water = nodes.get('is_water', pd.Series([0] * len(nodes))).fillna(0).astype(int).to_numpy().astype(bool)
    rep_aa1 = nodes.get('rep_aa1', pd.Series(['X'] * len(nodes))).fillna('X').astype(str).to_numpy()
    frag_ids = nodes['frag_id'].astype(int).to_numpy()
    for i, fid in enumerate(frag_ids):
        if is_standard[i]:
            aa = rep_aa1[i][:1] or 'X'
            smiles = AA_FREE_SMILES.get(aa, '')
            source = f'aa_dict_{aa}' if smiles else f'aa_dict_missing_{aa}'
        elif include_water and is_water[i]:
            smiles = WATER_SMILES
            source = 'water_dict_smiles'
        else:
            continue
        fp_by_frag_id[int(fid)] = morgan_fingerprint(smiles, n_bits=n_bits, radius=radius, use_chirality=use_chirality)
        source_by_frag_id[int(fid)] = source
    return (fp_by_frag_id, source_by_frag_id)

def load_ligand_table_records(path: Optional[Path], pdbid_col: str, hetid_col: str, smiles_col: str) -> Tuple[Dict[Tuple[str, str], str], Dict[str, List[Tuple[str, str]]]]:
    """Unchanged from v6."""
    by_pair: Dict[Tuple[str, str], str] = {}
    by_pdbid: Dict[str, List[Tuple[str, str]]] = {}
    if path is None:
        return (by_pair, by_pdbid)
    df = pd.read_csv(path)
    need = [pdbid_col, hetid_col, smiles_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f'Ligand table missing columns {missing}. columns={list(df.columns)}')
    for _, r in df.iterrows():
        pdbid = str(r[pdbid_col]).strip().lower()
        hetid = '' if pd.isna(r[hetid_col]) else str(r[hetid_col]).strip()
        smi = '' if pd.isna(r[smiles_col]) else str(r[smiles_col]).strip()
        if not pdbid or not hetid or (not smi):
            continue
        by_pair.setdefault((pdbid, hetid), smi)
        by_pdbid.setdefault(pdbid, [])
        if (hetid, smi) not in by_pdbid[pdbid]:
            by_pdbid[pdbid].append((hetid, smi))
    return (by_pair, by_pdbid)

def resolve_ligand_smiles_by_resname(nodes: pd.DataFrame, pdbid: str, args) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, Any]]:
    """Unchanged from v6."""
    pdbid = str(pdbid or '').lower()
    lig_nodes = nodes[nodes.get('fragment_kind', pd.Series([''] * len(nodes))).astype(str) == 'ligand'] if 'fragment_kind' in nodes.columns else nodes.iloc[[]]
    ligand_resnames = sorted(set((str(x).strip() for x in lig_nodes.get('rep_resname', pd.Series([], dtype=str)).fillna('').astype(str).tolist() if str(x).strip())))
    by_resname: Dict[str, Tuple[str, str]] = {}
    audit: Dict[str, Any] = {'ligand_resnames_for_smiles_lookup': ligand_resnames, 'ligand_smiles_lookup': {}}
    if getattr(args, 'smiles', ''):
        for r in ligand_resnames:
            by_resname[r] = (args.smiles, 'provided_smiles')
            audit['ligand_smiles_lookup'][r] = 'provided_smiles'
        return (by_resname, audit)
    by_pair, by_pdbid = load_ligand_table_records(Path(args.ligand_table) if getattr(args, 'ligand_table', None) else None, args.pdbid_col, getattr(args, 'hetid_col', 'hetid'), args.smiles_col)
    pdb_entries = by_pdbid.get(pdbid, [])
    for r in ligand_resnames:
        smi = by_pair.get((pdbid, r), '')
        if smi:
            by_resname[r] = (smi, 'ligand_table_smiles')
            audit['ligand_smiles_lookup'][r] = 'ligand_table_smiles'
        elif len(pdb_entries) == 1 and pdb_entries[0][1]:
            by_resname[r] = (pdb_entries[0][1], 'ligand_table_singleton_smiles')
            audit['ligand_smiles_lookup'][r] = 'ligand_table_singleton_smiles'
        else:
            audit['ligand_smiles_lookup'][r] = 'not_found_in_ligand_table'
    return (by_resname, audit)

def compute_water_bridge_metadata(nodes: pd.DataFrame, edges: pd.DataFrame, dist_cutoff: float) -> Dict[str, Any]:
    """Unchanged from v6."""
    n = len(nodes)
    is_water = nodes.get('is_water', pd.Series([0] * n)).fillna(0).astype(int).to_numpy().astype(bool)
    is_ligand = nodes.get('is_ligand', pd.Series([0] * n)).fillna(0).astype(int).to_numpy().astype(bool)
    is_standard = nodes.get('is_standard_aa', pd.Series([0] * n)).fillna(0).astype(int).to_numpy().astype(bool)
    water_indices = np.where(is_water)[0].astype(np.int64)
    deg_lig = np.zeros(len(water_indices), dtype=np.int64)
    deg_std = np.zeros(len(water_indices), dtype=np.int64)
    min_lig = np.full(len(water_indices), np.inf, dtype=np.float32)
    min_std = np.full(len(water_indices), np.inf, dtype=np.float32)
    water_pos = {int(idx): k for k, idx in enumerate(water_indices.tolist())}
    ligand_water_pairs = set()
    water_standard_pairs = set()
    if len(water_indices) and {'src_idx', 'dst_idx', 'DIST'}.issubset(edges.columns):
        e = edges[['src_idx', 'dst_idx', 'DIST']].copy()
        e['DIST'] = e['DIST'].astype(float)
        e = e[e['DIST'] <= float(dist_cutoff)]
        for _, row in e.iterrows():
            a = int(row['src_idx'])
            b = int(row['dst_idx'])
            d = float(row['DIST'])
            for w, nb in ((a, b), (b, a)):
                if w not in water_pos:
                    continue
                wi = water_pos[w]
                if is_ligand[nb]:
                    deg_lig[wi] += 1
                    min_lig[wi] = min(min_lig[wi], d)
                    ligand_water_pairs.add((int(nb), int(w)))
                if is_standard[nb]:
                    deg_std[wi] += 1
                    min_std[wi] = min(min_std[wi], d)
                    water_standard_pairs.add((int(w), int(nb)))
    min_lig = np.where(np.isfinite(min_lig), min_lig, -1.0).astype(np.float32)
    min_std = np.where(np.isfinite(min_std), min_std, -1.0).astype(np.float32)
    bridge_mask = (deg_lig > 0) & (deg_std > 0)
    bridge_indices = water_indices[bridge_mask]
    return {'water_node_indices': torch.tensor(water_indices, dtype=torch.long), 'water_bridge_node_indices': torch.tensor(bridge_indices, dtype=torch.long), 'water_degree_to_ligand': torch.tensor(deg_lig, dtype=torch.long), 'water_degree_to_standard_aa': torch.tensor(deg_std, dtype=torch.long), 'water_min_dist_to_ligand': torch.tensor(min_lig, dtype=torch.float32), 'water_min_dist_to_standard_aa': torch.tensor(min_std, dtype=torch.float32), 'n_ligand_water_edges': len(ligand_water_pairs), 'n_water_standard_aa_edges': len(water_standard_pairs), 'n_ligand_water_protein_bridges': int(bridge_mask.sum()), 'water_bridge_dist_cutoff': float(dist_cutoff)}

def build_one(nodes_csv: Path, edges_csv: Path, esm_path: Optional[Path], out_pt: Path, audit_json: Optional[Path], args) -> None:
    refuse_unsafe_output_path(out_pt, 'graph PT')
    if audit_json is not None:
        refuse_unsafe_output_path(audit_json, 'audit JSON')
    nodes = pd.read_csv(nodes_csv).sort_values('frag_id').reset_index(drop=True)
    edges = pd.read_csv(edges_csv)
    frag_to_idx = {int(fid): i for i, fid in enumerate(nodes['frag_id'].astype(int))}
    missing = sorted((set(edges['frag_i'].astype(int)) | set(edges['frag_j'].astype(int))) - set(frag_to_idx))
    if missing:
        raise RuntimeError(f'Edges reference missing node frag_ids: n={len(missing)} first={missing[:10]}')
    edges = edges.copy()
    edges['src_idx'] = edges['frag_i'].astype(int).map(frag_to_idx)
    edges['dst_idx'] = edges['frag_j'].astype(int).map(frag_to_idx)
    pdbid_for_lookup = args.pdbid or nodes_csv.stem.replace('_nodes', '')
    ccd_map = load_ccd_smiles_json(Path(args.ccd_smiles_json) if getattr(args, 'ccd_smiles_json', None) else None)
    ligand_smiles_by_resname, ligand_lookup_audit = resolve_ligand_smiles_by_resname(nodes, pdbid_for_lookup, args)
    small_mol_records, small_mol_fp_by_frag_id, small_atom_feature_names, small_bond_feature_names = collect_small_molecule_graphs(nodes=nodes, ligand_smiles_by_resname=ligand_smiles_by_resname, provided_smiles=args.smiles or '', ccd_map=ccd_map, n_bits=args.morgan_bits, radius=args.morgan_radius, use_chirality=not args.no_chirality, remove_hs=not args.keep_small_mol_hs)
    aa_water_fp_by_frag_id, aa_water_source_by_frag_id = compute_aa_water_fp_by_frag_id(nodes=nodes, n_bits=args.morgan_bits, radius=args.morgan_radius, use_chirality=not args.no_chirality, include_water=not args.no_water_smiles)
    combined_fp_by_frag_id = dict(small_mol_fp_by_frag_id)
    combined_fp_by_frag_id.update(aa_water_fp_by_frag_id)
    lig_records = [r for r in small_mol_records if r['fragment_kind'] == 'ligand']
    lig_record = lig_records[0] if lig_records else None
    esm = load_esm(esm_path)
    x_np, node_feature_names = make_node_features(nodes, esm, combined_fp_by_frag_id, args.morgan_bits)
    edge_index_np, edge_attr_np, edge_attr_names, fmo_mean, fmo_std = make_edge_features(edges, args.rbf_k, args.dist_rbf_max, args.centroid_rbf_max, args.directed)
    pos_np = nodes[['centroid_x', 'centroid_y', 'centroid_z']].to_numpy(np.float32)
    is_ligand_np = nodes.get('is_ligand', pd.Series([0] * len(nodes))).fillna(0).astype(int).to_numpy().astype(bool)
    ligand_indices = np.where(is_ligand_np)[0].astype(np.int64)
    ligand_nodes_df = nodes.iloc[ligand_indices.tolist()] if len(ligand_indices) else nodes.iloc[[]]
    bound_ligand_resnames = sorted(set(safe_str_list(ligand_nodes_df.get('rep_resname', pd.Series([], dtype=str)))))
    bound_ligand_frag_ids = [int(x) for x in ligand_nodes_df.get('frag_id', pd.Series([], dtype=int)).tolist()]
    water_bridge = compute_water_bridge_metadata(nodes, edges, args.water_bridge_dist_cutoff)
    data = Data(x=torch.tensor(x_np, dtype=torch.float32), edge_index=torch.tensor(edge_index_np, dtype=torch.long), edge_attr=torch.tensor(edge_attr_np, dtype=torch.float32), pos=torch.tensor(pos_np, dtype=torch.float32))
    data.frag_id = torch.tensor(nodes['frag_id'].astype(int).to_numpy(), dtype=torch.long)
    data.is_ligand = torch.tensor(is_ligand_np, dtype=torch.bool)
    data.ligand_node_indices = torch.tensor(ligand_indices, dtype=torch.long)
    data.is_standard_aa = torch.tensor(nodes.get('is_standard_aa', pd.Series([0] * len(nodes))).fillna(0).astype(int).to_numpy().astype(bool), dtype=torch.bool)
    data.is_hetero = torch.tensor(nodes.get('is_hetero', pd.Series([0] * len(nodes))).fillna(0).astype(int).to_numpy().astype(bool), dtype=torch.bool)
    data.is_water = torch.tensor(nodes.get('is_water', pd.Series([0] * len(nodes))).fillna(0).astype(int).to_numpy().astype(bool), dtype=torch.bool)
    data.is_other_hetero = torch.tensor(nodes.get('is_other_hetero', pd.Series([0] * len(nodes))).fillna(0).astype(int).to_numpy().astype(bool), dtype=torch.bool)
    if 'fragment_kind' in nodes.columns:
        data.fragment_kind = list(nodes['fragment_kind'].astype(str))
    data.node_feature_names = node_feature_names
    data.edge_attr_names = edge_attr_names
    data.fmo_cols = FMO_Z_COLS
    data.fmo_mean = fmo_mean
    data.fmo_std = fmo_std
    data.graph_version = 'graph_v7_fmofrag_morganunified_esm'
    data.is_undirected_by_construction = bool(not args.directed)
    first_lig_smiles = lig_record['smiles'] if lig_record else args.smiles or ''
    data.ligand_smiles = first_lig_smiles
    data.bound_ligand_smiles = first_lig_smiles
    data.reference_ligand_smiles = first_lig_smiles
    data.bound_ligand_resnames = bound_ligand_resnames
    data.bound_ligand_frag_ids = bound_ligand_frag_ids
    data.ligand_feature_dim = int(args.morgan_bits)
    data.ligand_feature_names = [f'small_mol_morgan_{i}' for i in range(args.morgan_bits)]
    data.bound_ligand_atom_x = lig_record['atom_x'] if lig_record else torch.zeros((0, len(small_atom_feature_names) if small_atom_feature_names else 12), dtype=torch.float32)
    data.bound_ligand_edge_index = lig_record['edge_index'] if lig_record else torch.zeros((2, 0), dtype=torch.long)
    data.bound_ligand_edge_attr = lig_record['edge_attr'] if lig_record else torch.zeros((0, len(small_bond_feature_names) if small_bond_feature_names else 9), dtype=torch.float32)
    data.bound_ligand_atom_feature_names = small_atom_feature_names
    data.bound_ligand_bond_feature_names = small_bond_feature_names
    data.has_bound_ligand_atom_graph = bool(lig_record is not None and lig_record['n_atoms'] > 0)
    data.small_mol_resnames = [r['resname'] for r in small_mol_records]
    data.small_mol_fragment_kinds = [r['fragment_kind'] for r in small_mol_records]
    data.small_mol_frag_ids = [int(r['frag_id']) for r in small_mol_records]
    data.small_mol_smiles = [r['smiles'] for r in small_mol_records]
    data.small_mol_graph_sources = [r['graph_source'] for r in small_mol_records]
    data.small_mol_smiles_parse_statuses = [r.get('smiles_parse_status', '') for r in small_mol_records]
    data.small_mol_atom_x_list = [r['atom_x'] for r in small_mol_records]
    data.small_mol_edge_index_list = [r['edge_index'] for r in small_mol_records]
    data.small_mol_edge_attr_list = [r['edge_attr'] for r in small_mol_records]
    data.small_mol_pos_list = [torch.zeros((int(r['n_atoms']), 3), dtype=torch.float32) for r in small_mol_records]
    data.small_mol_atom_feature_names = small_atom_feature_names
    data.small_mol_bond_feature_names = small_bond_feature_names
    data.has_small_mol_graphs = bool(any((r['n_atoms'] > 0 for r in small_mol_records)))
    data.aa_water_fp_frag_ids = list(aa_water_source_by_frag_id.keys())
    data.aa_water_fp_sources = list(aa_water_source_by_frag_id.values())
    data.water_node_indices = water_bridge['water_node_indices']
    data.water_bridge_node_indices = water_bridge['water_bridge_node_indices']
    data.water_degree_to_ligand = water_bridge['water_degree_to_ligand']
    data.water_degree_to_standard_aa = water_bridge['water_degree_to_standard_aa']
    data.water_min_dist_to_ligand = water_bridge['water_min_dist_to_ligand']
    data.water_min_dist_to_standard_aa = water_bridge['water_min_dist_to_standard_aa']
    data.n_ligand_water_edges = int(water_bridge['n_ligand_water_edges'])
    data.n_water_standard_aa_edges = int(water_bridge['n_water_standard_aa_edges'])
    data.n_ligand_water_protein_bridges = int(water_bridge['n_ligand_water_protein_bridges'])
    data.water_bridge_dist_cutoff = float(water_bridge['water_bridge_dist_cutoff'])
    data.gnn_ready = True
    data.pdbid = args.pdbid or nodes_csv.stem.replace('_nodes', '')
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_pt)
    audit = {'pdbid': data.pdbid, 'nodes_csv': str(nodes_csv), 'edges_csv': str(edges_csv), 'out_pt': str(out_pt), 'n_nodes': int(data.x.shape[0]), 'n_edges': int(data.edge_index.shape[1]), 'node_dim': int(data.x.shape[1]), 'edge_dim': int(data.edge_attr.shape[1]), 'n_ligand_nodes': int(data.is_ligand.sum()), 'bound_ligand_resnames': bound_ligand_resnames, 'bound_ligand_frag_ids': bound_ligand_frag_ids, 'has_bound_ligand_atom_graph': bool(data.has_bound_ligand_atom_graph), 'bound_ligand_atom_graph_n_atoms': int(data.bound_ligand_atom_x.shape[0]), 'bound_ligand_atom_graph_n_edges': int(data.bound_ligand_edge_index.shape[1]), 'n_small_mol_graphs': int(len(small_mol_records)), 'n_small_mol_graphs_with_atoms': int(sum((1 for r in small_mol_records if r['n_atoms'] > 0))), 'small_mol_graph_source_counts': pd.Series([r['graph_source'] for r in small_mol_records]).value_counts().to_dict() if small_mol_records else {}, 'small_mol_smiles_parse_status_counts': dict(Counter([r.get('smiles_parse_status', '') for r in small_mol_records])) if small_mol_records else {}, 'small_mol_fragment_kind_counts': pd.Series([r['fragment_kind'] for r in small_mol_records]).value_counts().to_dict() if small_mol_records else {}, **ligand_lookup_audit, 'n_standard_aa_nodes': int(data.is_standard_aa.sum()), 'n_aa_water_fp_assigned': int(len(aa_water_fp_by_frag_id)), 'aa_water_fp_source_counts': dict(Counter(aa_water_source_by_frag_id.values())), 'n_water_nodes': int(data.is_water.sum()) if hasattr(data, 'is_water') else 0, 'n_water_bridge_nodes': int(data.water_bridge_node_indices.numel()), 'n_ligand_water_edges': int(data.n_ligand_water_edges), 'n_water_standard_aa_edges': int(data.n_water_standard_aa_edges), 'n_ligand_water_protein_bridges': int(data.n_ligand_water_protein_bridges), 'water_bridge_dist_cutoff': float(data.water_bridge_dist_cutoff), 'n_other_hetero_nodes': int(data.is_other_hetero.sum()) if hasattr(data, 'is_other_hetero') else 0, 'fragment_kind_counts': nodes['fragment_kind'].value_counts().to_dict() if 'fragment_kind' in nodes.columns else {}, 'graph_version': data.graph_version}
    if audit_json:
        audit_json.parent.mkdir(parents=True, exist_ok=True)
        with open(audit_json, 'w') as f:
            json.dump(audit, f, indent=2)
    print(f'Graph saved: {out_pt}')
    print(json.dumps(audit, indent=2))

def infer_pdbid_from_nodes(p: Path) -> str:
    return p.stem.replace('_nodes', '').lower()

def main() -> None:
    ap = argparse.ArgumentParser(description='Build PyG graph_v7_fmofrag_morganunified_esm from v4 nodes/edges (with ESM; AA/water/ligand/other_hetero all use Morgan FP).')
    one = ap.add_argument_group('single file')
    one.add_argument('--nodes')
    one.add_argument('--edges')
    one.add_argument('--out-pt')
    one.add_argument('--audit-json')
    one.add_argument('--pdbid')
    one.add_argument('--smiles', default='')
    one.add_argument('--esm')
    batch = ap.add_argument_group('batch')
    batch.add_argument('--nodes-dir')
    batch.add_argument('--edges-dir')
    batch.add_argument('--embedding-dir')
    batch.add_argument('--out-dir')
    batch.add_argument('--audit-dir')
    batch.add_argument('--ligand-table')
    batch.add_argument('--pdbid-col', default='pdbid')
    batch.add_argument('--smiles-col', default='het_smiles')
    batch.add_argument('--hetid-col', default='hetid')
    ap.add_argument('--ccd-smiles-json', default='')
    ap.add_argument('--water-bridge-dist-cutoff', type=float, default=4.0, help='DIST cutoff for ligand-water-protein bridge metadata.')
    ap.add_argument('--morgan-bits', type=int, default=512)
    ap.add_argument('--morgan-radius', type=int, default=2)
    ap.add_argument('--no-chirality', action='store_true')
    ap.add_argument('--no-water-smiles', action='store_true', help="Disable Morgan FP for water nodes (default: enabled, SMILES 'O').")
    ap.add_argument('--keep-small-mol-hs', action='store_true', help='Keep explicit hydrogens in stored ligand/other_hetero atom graphs. Default removes Hs.')
    ap.add_argument('--rbf-k', type=int, default=32)
    ap.add_argument('--dist-rbf-max', type=float, default=10.0)
    ap.add_argument('--centroid-rbf-max', type=float, default=30.0)
    ap.add_argument('--directed', action='store_true')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if args.nodes:
        if not (args.edges and args.out_pt):
            raise SystemExit('--edges and --out-pt are required with --nodes')
        build_one(nodes_csv=Path(args.nodes), edges_csv=Path(args.edges), esm_path=Path(args.esm) if args.esm else None, out_pt=Path(args.out_pt), audit_json=Path(args.audit_json) if args.audit_json else None, args=args)
        return
    if not (args.nodes_dir and args.edges_dir and args.out_dir):
        raise SystemExit('Use either --nodes/--edges/--out-pt or --nodes-dir/--edges-dir/--out-dir')
    nodes_paths = sorted(Path(args.nodes_dir).glob('*_nodes.csv'))
    print(f'Node files to process: {len(nodes_paths)}')
    for i, nodes_csv in enumerate(nodes_paths, 1):
        pdbid = infer_pdbid_from_nodes(nodes_csv)
        edges_csv = Path(args.edges_dir) / f'{pdbid}_edges.csv'
        if not edges_csv.exists():
            print(f'[{i}/{len(nodes_paths)}] {pdbid} missing edges')
            continue
        esm_path = find_esm(pdbid, Path(args.embedding_dir) if args.embedding_dir else None)
        out_pt = Path(args.out_dir) / f'{pdbid}.pt'
        audit_json = Path(args.audit_dir) / f'{pdbid}_graph_audit.json' if args.audit_dir else None
        if args.resume and out_pt.exists():
            print(f'[{i}/{len(nodes_paths)}] {pdbid} skipped')
            continue
        args.pdbid = pdbid
        args.smiles = ''
        print(f'[{i}/{len(nodes_paths)}] {pdbid}')
        build_one(nodes_csv, edges_csv, esm_path, out_pt, audit_json, args)
if __name__ == '__main__':
    main()
