"""
pretrain_masked_edge_multigraph_v17r_rawdist.py

v17R variant ("rawdist"): forked from
pretrain_masked_edge_multigraph_cached_timed_v17_joint_geom_fmo_blackout.py to
fix an information-amount asymmetry between the distance task and the FMO/PIEDA
task in the original v17 design.

Why this variant exists:
  The original v17 distance-reconstruction target is the 64-dim RBF expansion
  (DIST_rbf_* + centroid_distance_rbf_*), which is a smooth, bounded, highly
  redundant encoding of only 2 underlying scalars (distance, centroid distance).
  It is solved almost immediately (training loss ~1e-5) and is not on a
  comparable information/difficulty footing with the 5-dim z-scored PIEDA
  target. This makes "geom_only" vs "geom+PIEDA" downstream comparisons unfair.

What changed here:
  - New target: per-graph z-scored RAW distance values (DIST_raw_z,
    centroid_distance_raw_z), 2 dims total, standardized the same way PIEDA
    is standardized (per-graph mean/std, computed on the fly at graph-load
    time; see add_raw_distance_zscore()).
  - infer_distance_rawz_target_indices() replaces infer_distance_rbf_target_indices()
    as the source of distance_target_idx.
  - infer_input_ablation_indices() now also zeros the new DIST_raw_z /
    centroid_distance_raw_z columns under --ablation no_distance /
    no_distance_no_esm, so "erase all edge info" ablations are consistent.
  - The distance loss (both in train() and evaluate_loader()) now uses
    weighted_smooth_l1_loss(..., edge_weights, ...) instead of plain
    F.smooth_l1_loss(...). In the base v17 script, --ligand-edge-loss-weight
    only reweighted the FMO loss, never the distance loss -- an asymmetry.
    Here the SAME edge_weights (from --ligand-edge-loss-weight) apply to
    both loss terms, so distance and PIEDA stay on equal footing whether or
    not ligand-edge upweighting is used.
  - Everything else (masking, model architecture sizing) is unchanged and
    auto-adapts to the new (smaller) distance_target_idx.

Intended comparison (run with --ablation no_distance, i.e. all edge input
features zeroed for every edge -- no FMO, no RBF, no raw distance anywhere).
For a fair distance-vs-PIEDA comparison, keep --huber-beta and
--dist-huber-beta equal, --fmo-loss-weight and --dist-loss-weight equal, and
apply --ligand-edge-loss-weight (if used at all) identically to both, since it
now affects both loss terms symmetrically:
  A) --fmo-loss-weight 0 --dist-loss-weight 1.0
       -> predict only the 2 z-scored raw distance values per masked edge.
  B) --fmo-loss-weight 1.0 --dist-loss-weight 1.0 --fmo-target-mode real
       -> predict the 2 z-scored raw distance values + 5 PIEDA values jointly.
  (B) vs (A) isolates the marginal value of PIEDA as a supervisory target on
  an information-comparable, non-redundant geometry baseline.

Everything below this point is unchanged from the v17 base script unless
explicitly noted.

---

Multi-graph self-supervised pretraining for FMO/PIEDA residue interaction graphs.

Input:
  --graph-dir dataset/graph_v4_fmofrag
    containing many *.pt PyG Data objects created by build_fmo_fragment_pyg_graph_v5/v6.py

  Optional graph filtering:
    --strict-filter-csv audits/strict_graph_filter_after_v11.all.csv
    --require-single-ligand --require-ligand-atom-graph

Split:
  Graph-level train/valid/test split.
  A graph is assigned to only one split.

Modes:
  b_context:
    Setting B local context masking.
    For sampled target edges, mask their FMO/PIEDA components.
    Surrounding edges keep FMO/PIEDA.

  geometry_only:
    Remove FMO/PIEDA components from ALL input edges.
    Model uses node features + graph topology + DIST/centroid geometry only.

Targets:
  Optional edge FMO/PIEDA mask: PIEDA_sum_z/MP2_IFIE_z, ES_z, EX_z, CTmix_z, DI_z
  Optional joint edge task: masked distance/RBF + FMO/PIEDA reconstruction
  Optional node task: masked ligand sparse node feature reconstruction

Masked/removed:
  MP2_IFIE_z, ES_z, EX_z, CTmix_z, DI_z, q_I_to_J_z

Model:
  Edge-MLP MPNN compatible with PyG Batch objects:
    m_ij = MLP([h_i, h_j, e_ij])
    h_i <- GRU(sum_j m_ij, h_i)
    yhat_ij = MLP([h_i, h_j, e_geom_ij])

Notes:
  - This script samples a fixed fraction of undirected edge pairs per graph.
  - During valid/test, the mask is deterministic per graph and seed.
  - Best checkpoint is selected by graph-level validation loss.
"""
import argparse
import csv
import math
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_geometric.loader import DataLoader
except Exception as e:
    raise ImportError('torch_geometric is required. Install with: pip install torch-geometric') from e

def load_pt(path, device='cpu'):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)

def infer_pdbid_from_graph_path(path):
    """Infer lower-case PDBID from a graph .pt path."""
    stem = Path(path).stem
    return str(stem[:4]).lower()

def read_pdbid_list(path):
    """Read a PDBID list from txt/csv.

    Accepted formats:
      - one PDBID per line
      - CSV with a 'pdbid' column
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'PDBID list not found: {p}')
    if p.suffix.lower() == '.csv':
        df = pd.read_csv(p)
        if 'pdbid' not in df.columns:
            raise ValueError(f'CSV must contain a pdbid column: {p}')
        values = df['pdbid'].dropna().astype(str).tolist()
    else:
        values = []
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                s = s.replace(',', ' ').split()[0]
                values.append(s)
    return {str(x).lower().strip() for x in values if str(x).strip()}

def load_strict_filter_keep_set(path):
    """Read strict graph filter CSV produced by audit_extract_strict_exclusions.py."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Strict filter CSV not found: {p}')
    df = pd.read_csv(p)
    missing = [c for c in ['pdbid', 'keep_strict'] if c not in df.columns]
    if missing:
        raise ValueError(f'Strict filter CSV missing columns {missing}: {p}')
    keep = df[df['keep_strict'].astype(str).str.lower().isin(['true', '1', 'yes'])]
    return {str(x).lower().strip() for x in keep['pdbid'].dropna().astype(str).tolist()}

def graph_passes_runtime_filter(data, args):
    """Optional runtime graph-quality checks against the loaded PyG Data object."""
    reasons = []
    if args.require_single_ligand:
        n_lig = 0
        if hasattr(data, 'is_ligand'):
            try:
                n_lig = int(data.is_ligand.sum().item())
            except Exception:
                n_lig = int(np.asarray(data.is_ligand).sum())
        else:
            mask, _ = infer_ligand_node_mask(data)
            n_lig = int(mask.sum())
        if n_lig != 1:
            reasons.append(f'runtime_n_ligand_nodes_{n_lig}')
    if args.require_ligand_atom_graph:
        has_graph = bool(getattr(data, 'has_bound_ligand_atom_graph', False))
        n_atoms = 0
        if hasattr(data, 'bound_ligand_atom_x'):
            try:
                n_atoms = int(data.bound_ligand_atom_x.shape[0])
            except Exception:
                n_atoms = 0
        if not has_graph or n_atoms <= 0:
            reasons.append('runtime_no_bound_ligand_atom_graph')
    if args.allowed_graph_versions:
        gv = str(getattr(data, 'graph_version', ''))
        allowed = {x.strip() for x in args.allowed_graph_versions.split(',') if x.strip()}
        if allowed and gv not in allowed:
            reasons.append(f'runtime_graph_version_not_allowed:{gv}')
    return (len(reasons) == 0, reasons)

def filter_graph_paths(paths, args, out_dir):
    """Filter graph paths using strict keep/exclude lists and optional runtime checks."""
    include_set = read_pdbid_list(args.include_pdbids) if args.include_pdbids else None
    exclude_set = read_pdbid_list(args.exclude_pdbids) if args.exclude_pdbids else set()
    strict_keep_set = load_strict_filter_keep_set(args.strict_filter_csv) if args.strict_filter_csv else None
    rows = []
    kept = []
    for p in paths:
        pdbid = infer_pdbid_from_graph_path(p)
        reasons = []
        if include_set is not None and pdbid not in include_set:
            reasons.append('not_in_include_pdbids')
        if strict_keep_set is not None and pdbid not in strict_keep_set:
            reasons.append('not_in_strict_filter_keep')
        if pdbid in exclude_set:
            reasons.append('in_exclude_pdbids')
        if not reasons and (args.require_single_ligand or args.require_ligand_atom_graph or args.allowed_graph_versions):
            data = load_pt(p, 'cpu')
            ok, runtime_reasons = graph_passes_runtime_filter(data, args)
            if not ok:
                reasons.extend(runtime_reasons)
        keep = len(reasons) == 0
        if keep:
            kept.append(p)
        rows.append({'pdbid': pdbid, 'graph_path': str(p), 'keep_pretrain': keep, 'filter_reasons': ';'.join(reasons)})
    out_dir.mkdir(parents=True, exist_ok=True)
    filter_csv = out_dir / 'pretrain_graph_filter_audit.csv'
    pd.DataFrame(rows).to_csv(filter_csv, index=False)
    print(f'Graph filter audit saved: {filter_csv}')
    print(f'Graphs before filter: {len(paths)}')
    print(f'Graphs after filter: {len(kept)}')
    if len(kept) == 0:
        raise ValueError('No graphs remain after filtering. Check include/exclude/strict filter settings.')
    return (kept, filter_csv)

def format_seconds(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f'{seconds:.1f}s'
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f'{int(minutes)}m{sec:04.1f}s'
    hours, minutes = divmod(minutes, 60)
    return f'{int(hours)}h{int(minutes):02d}m{sec:04.1f}s'

def infer_ligand_node_mask(data):
    """Infer ligand-node boolean mask from common graph encodings.

    Priority:
      1. data.is_ligand attribute
      2. node_feature_names column containing ligand flags
      3. known alternate attributes such as ligand_mask / is_ligand_node

    This is needed because some graph builders store ligand flags in x with
    node_feature_names rather than as a standalone data.is_ligand tensor.
    """
    n_nodes = int(data.x.shape[0])
    for attr in ['is_ligand', 'ligand_mask', 'is_ligand_node', 'ligand_node_mask']:
        if hasattr(data, attr):
            val = getattr(data, attr)
            try:
                arr = val.detach().cpu().numpy()
            except Exception:
                arr = np.asarray(val)
            arr = arr.reshape(-1)
            if arr.shape[0] == n_nodes:
                return (arr.astype(bool), attr)
    node_names = list(getattr(data, 'node_feature_names', []))
    if node_names:
        lowered = [str(n).lower() for n in node_names]
        preferred = ['is_ligand', 'ligand', 'is_ligand_node', 'node_is_ligand', 'is_het', 'hetatm', 'is_hetatm']
        candidates = []
        for key in preferred:
            candidates.extend([i for i, n in enumerate(lowered) if n == key])
        if not candidates:
            candidates = [i for i, n in enumerate(lowered) if 'ligand' in n or 'hetatm' in n or n.endswith('_het') or n.endswith('_lig')]
        for i in candidates:
            col = data.x[:, i].detach().cpu().numpy()
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                continue
            unique = np.unique(finite)
            if len(unique) <= 3 and finite.max() > 0.5:
                return (col > 0.5, f'x[:, {i}] node_feature_names={node_names[i]}')
    return (np.zeros(n_nodes, dtype=bool), 'not_found')

def audit_ligand_node_sources(paths, max_graphs=20):
    rows = []
    for p in paths[:max_graphs]:
        data = load_pt(p, 'cpu')
        mask, source = infer_ligand_node_mask(data)
        rows.append({'path': str(p), 'source': source, 'n_nodes': int(data.x.shape[0]), 'n_ligand_nodes': int(mask.sum())})
    return rows

def add_pair_cache(data):
    """
    Precompute undirected pair -> directed edge indices for one graph.

    Stored edge indices are LOCAL to the graph before PyG batching.
    During batch training they are shifted by graph-specific edge offsets.

    Added attributes:
      pair_primary_edges:       [num_pairs]
      pair_directed_edges_flat: [num_directed_pair_edges]
      pair_directed_offsets:    [num_pairs + 1]
      pair_is_ligand:           [num_pairs] bool, True if either endpoint is ligand
      num_pairs_cached:         [1]

    Important:
      Older cached Data objects may already have pair_primary_edges but not pair_is_ligand.
      In that case we recompute the pair cache so that cached ligand_biased/ligand_only
      sampling works correctly.
    """
    has_complete_cache = hasattr(data, 'pair_primary_edges') and hasattr(data, 'pair_directed_edges_flat') and hasattr(data, 'pair_directed_offsets') and hasattr(data, 'num_pairs_cached') and hasattr(data, 'pair_is_ligand')
    if has_complete_cache:
        return data
    pair_map = {}
    ei = data.edge_index.detach().cpu()
    for e in range(ei.shape[1]):
        u = int(ei[0, e])
        v = int(ei[1, e])
        a, b = (u, v) if u <= v else (v, u)
        pair_map.setdefault((a, b), []).append(e)
    groups = list(pair_map.values())
    is_ligand_cpu, ligand_source = infer_ligand_node_mask(data)
    data.ligand_mask_source = ligand_source
    primary = []
    flat = []
    offsets = [0]
    pair_is_ligand = []
    for group in groups:
        primary_edge = group[0]
        primary.append(primary_edge)
        flat.extend(group)
        offsets.append(len(flat))
        u = int(ei[0, primary_edge])
        v = int(ei[1, primary_edge])
        pair_is_ligand.append(bool(is_ligand_cpu[u] or is_ligand_cpu[v]))
    data.pair_primary_edges = torch.tensor(primary, dtype=torch.long)
    data.pair_directed_edges_flat = torch.tensor(flat, dtype=torch.long)
    data.pair_directed_offsets = torch.tensor(offsets, dtype=torch.long)
    data.pair_is_ligand = torch.tensor(pair_is_ligand, dtype=torch.bool)
    data.num_pairs_cached = torch.tensor([len(groups)], dtype=torch.long)
    return data

def get_edge_offsets_for_batch(batch, device):
    batch_vec = batch.batch
    num_graphs = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 1
    edge_graph = batch_vec[batch.edge_index[0]]
    edge_counts = torch.bincount(edge_graph, minlength=num_graphs)
    edge_offsets = torch.zeros(num_graphs, dtype=torch.long, device=device)
    if num_graphs > 1:
        edge_offsets[1:] = torch.cumsum(edge_counts[:-1], dim=0)
    return edge_offsets

def get_dim_indices_from_names(names, prefixes=None, exact=None):
    prefixes = prefixes or []
    exact = exact or []
    idx = []
    for i, name in enumerate(names):
        if name in exact or any((name.startswith(p) for p in prefixes)):
            idx.append(i)
    return idx

def infer_feature_groups(data):
    names = list(getattr(data, 'edge_attr_names', []))
    if not names:
        raise ValueError('Data object has no edge_attr_names. Build graph.pt with build_pyg_graph.py.')
    geom_idx = get_dim_indices_from_names(names, prefixes=['DIST_rbf_', 'centroid_distance_rbf_'], exact=['DIST_raw', 'centroid_distance_raw'])
    target_names_v2 = ['PIEDA_sum_z', 'ES_z', 'EX_z', 'CTmix_z', 'DI_z']
    target_names_v1 = ['MP2_IFIE_z', 'ES_z', 'EX_z', 'CTmix_z', 'DI_z']
    if all((n in names for n in target_names_v2)):
        target_names = target_names_v2
    elif all((n in names for n in target_names_v1)):
        target_names = target_names_v1
    else:
        raise ValueError(f'Could not infer target FMO dims. Expected either {target_names_v2} or {target_names_v1}. Available: {names}')
    target_idx = get_dim_indices_from_names(names, exact=target_names)
    mask_extra_idx = get_dim_indices_from_names(names, exact=['q_I_to_J_z'])
    if len(target_idx) != 5:
        raise ValueError(f'Could not find all target dims {target_names}. Available: {names}')
    if len(geom_idx) == 0:
        raise ValueError('Could not find geometry dims in edge_attr_names.')
    fmo_mask_idx = sorted(set(target_idx + mask_extra_idx))
    return (geom_idx, target_idx, fmo_mask_idx, names)

def infer_distance_rbf_target_indices(data):
    """Distance/geometry reconstruction target dims.

    Prefer bounded RBF features over raw distances so the auxiliary loss scale
    is stable across graphs. Raw distances are deliberately excluded.
    """
    names = list(getattr(data, 'edge_attr_names', []))
    idx = [i for i, n in enumerate(names) if str(n).startswith('DIST_rbf_') or str(n).startswith('centroid_distance_rbf_')]
    if not idx:
        idx = get_dim_indices_from_names(names, exact=['DIST_raw', 'centroid_distance_raw'])
    if not idx:
        raise ValueError('Could not infer distance target dims from edge_attr_names.')
    return idx

def add_raw_distance_zscore(data):
    """
    Append per-graph z-scored raw distance columns to edge_attr.

    Standardized the same way PIEDA targets are standardized at graph-build
    time (per-graph mean/std over all directed edges in this single graph;
    std<1e-8 falls back to 1.0, matching the FMO z-scoring convention).
    Adds two new columns: DIST_raw_z, centroid_distance_raw_z.

    Must be called BEFORE sanitize_data_for_pretrain_batch() (which drops the
    non-tensor edge_attr_names attribute), and must be applied identically to
    every Data object used for index inference (`first`, loaded directly) and
    to every graph flowing through GraphFileDataset (actual train/eval
    batches), so the appended-column position is identical everywhere.
    """
    names = list(getattr(data, 'edge_attr_names', []))
    if 'DIST_raw_z' in names and 'centroid_distance_raw_z' in names:
        return data
    raw_cols = ['DIST_raw', 'centroid_distance_raw']
    idx = [names.index(c) for c in raw_cols if c in names]
    if len(idx) != 2:
        raise ValueError(f'Expected DIST_raw and centroid_distance_raw in edge_attr_names, got: {names}')
    raw_vals = data.edge_attr[:, idx].float()
    mean = raw_vals.mean(dim=0, keepdim=True)
    std = raw_vals.std(dim=0, unbiased=False, keepdim=True)
    std = torch.where(std < 1e-08, torch.ones_like(std), std)
    z_vals = (raw_vals - mean) / std
    data.edge_attr = torch.cat([data.edge_attr, z_vals], dim=1)
    data.edge_attr_names = names + ['DIST_raw_z', 'centroid_distance_raw_z']
    data.dist_raw_mean = mean.squeeze(0)
    data.dist_raw_std = std.squeeze(0)
    return data

def infer_distance_rawz_target_indices(data):
    """Distance reconstruction target dims: per-graph z-scored raw distance.

    Requires add_raw_distance_zscore() to have been called on this Data
    object first, so DIST_raw_z / centroid_distance_raw_z columns exist.
    """
    names = list(getattr(data, 'edge_attr_names', []))
    idx = get_dim_indices_from_names(names, exact=['DIST_raw_z', 'centroid_distance_raw_z'])
    if len(idx) != 2:
        raise ValueError('Could not find DIST_raw_z/centroid_distance_raw_z in edge_attr_names. Did you call add_raw_distance_zscore() on this Data object first?')
    return idx

def infer_ligand_sparse_node_block(data, default_start=514, default_end=1026):
    """Infer ligand sparse node feature block used for node-mask learning.

    Current graph format stores a 512-dim ligand sparse molecular block at
    x[:,514:1026]. We keep this explicit and validate against node dim.
    """
    n_dim = int(data.x.shape[1])
    start = int(default_start)
    end = int(min(default_end, n_dim))
    if end <= start:
        raise ValueError(f'Invalid ligand sparse node block [{start}:{end}] for node dim {n_dim}')
    return list(range(start, end))

def mask_ligand_sparse_node_features(batch, x_input, node_idx, rng, mask_ratio=1.0, device='cpu'):
    """Mask ligand-node sparse feature block and return BCE targets.

    By default, all ligand nodes in the batch are masked. For this dataset there
    is typically one ligand node per graph.
    """
    if not hasattr(batch, 'is_ligand') or not node_idx:
        return (x_input, None, None, 0)
    lig_ids = torch.where(batch.is_ligand.to(device).bool())[0]
    if lig_ids.numel() == 0:
        return (x_input, None, None, 0)
    if mask_ratio < 1.0:
        ids_cpu = lig_ids.detach().cpu().tolist()
        k = max(1, int(round(len(ids_cpu) * float(mask_ratio))))
        ids_cpu = rng.sample(ids_cpu, min(k, len(ids_cpu)))
        lig_ids = torch.tensor(ids_cpu, dtype=torch.long, device=device)
    idx_t = torch.tensor(node_idx, dtype=torch.long, device=device)
    target = batch.x[lig_ids[:, None], idx_t].detach().clone()
    x_new = x_input.clone()
    x_new[lig_ids[:, None], idx_t] = 0.0
    return (x_new, lig_ids, target, int(lig_ids.numel()))

def infer_input_ablation_indices(data, fmo_mask_idx):
    """
    Identify node/edge input dimensions that can be zeroed for ablation studies.

    Edge:
      distance dims are DIST and centroid_distance features.
      FMO dims are fmo_mask_idx.

    Node:
      ESM dims are inferred from node_feature_names starting with 'esm_'.
      If node_feature_names are absent, assume ESM dims are the leading block
      before AA one-hot/flags if names are unavailable.
    """
    edge_names = list(getattr(data, 'edge_attr_names', []))
    node_names = list(getattr(data, 'node_feature_names', []))
    distance_idx = get_dim_indices_from_names(edge_names, prefixes=['DIST_rbf_', 'centroid_distance_rbf_'], exact=['DIST_raw', 'centroid_distance_raw', 'DIST_raw_z', 'centroid_distance_raw_z'])
    esm_idx = []
    if node_names:
        esm_idx = [i for i, n in enumerate(node_names) if str(n).startswith('esm_')]
    else:
        n_node_dim = int(data.x.shape[1])
        fallback_dim = min(480, n_node_dim)
        esm_idx = list(range(fallback_dim))
    return {'distance_idx': sorted(set(distance_idx)), 'fmo_idx': sorted(set(fmo_mask_idx)), 'esm_idx': sorted(set(esm_idx))}

def resolve_mode_and_ablation(mode, ablation):
    """
    Backward-compatible interpretation.

    Recommended new usage:
      --mode b_context --ablation none
      --mode b_context --ablation no_fmo_context
      --mode b_context --ablation no_distance
      --mode b_context --ablation no_esm
      --mode b_context --ablation no_distance_no_esm

    Legacy:
      --mode geometry_only
    is treated as:
      --mode b_context --ablation no_fmo_context
    """
    if mode == 'geometry_only':
        if ablation != 'none':
            print('Warning: --mode geometry_only is a legacy alias for --ablation no_fmo_context. The explicit --ablation value is ignored.')
        return ('b_context', 'no_fmo_context')
    return (mode, ablation)

def apply_input_ablation(batch, edge_attr_input, ablation, ablation_indices):
    """
    Apply input ablations.

    Semantics:
      none:
        no extra ablation.

      no_fmo_context:
        remove FMO/PIEDA/q_I_to_J from all input edges.
        Equivalent to legacy geometry-only.

      no_distance:
        remove FMO context and distance features.
        Keep ESM/node features and topology.

      no_esm:
        remove FMO context and ESM node features.
        Keep distance features, AA one-hot, flags, topology.

      no_distance_no_esm:
        remove FMO context, distance features, and ESM.
        Keep AA one-hot, flags, topology.

    Note:
      no_distance/no_esm baselines remove FMO context by design, so that
      the ablation isolates non-FMO cues.
    """
    x_input = batch.x
    if ablation == 'none':
        return (x_input, edge_attr_input)
    edge_zero = set()
    node_zero = set()
    if ablation in {'no_fmo_context', 'no_distance', 'no_esm', 'no_distance_no_esm'}:
        edge_zero.update(ablation_indices['fmo_idx'])
    if ablation in {'no_distance', 'no_distance_no_esm'}:
        edge_zero.update(ablation_indices['distance_idx'])
    if ablation in {'no_esm', 'no_distance_no_esm'}:
        node_zero.update(ablation_indices['esm_idx'])
    if edge_zero:
        idx = torch.tensor(sorted(edge_zero), dtype=torch.long, device=edge_attr_input.device)
        edge_attr_input = edge_attr_input.clone()
        all_edges = torch.arange(edge_attr_input.shape[0], device=edge_attr_input.device)
        edge_attr_input[all_edges[:, None], idx] = 0.0
    if node_zero:
        idx = torch.tensor(sorted(node_zero), dtype=torch.long, device=batch.x.device)
        x_input = batch.x.clone()
        all_nodes = torch.arange(x_input.shape[0], device=x_input.device)
        x_input[all_nodes[:, None], idx] = 0.0
    return (x_input, edge_attr_input)

def sample_pair_ids(num_pairs, ratio, rng, min_pairs=1):
    k = max(min_pairs, int(round(num_pairs * ratio)))
    k = min(k, num_pairs)
    return rng.sample(range(num_pairs), k)

def sample_pair_ids_with_cached_ligand_flags(pair_is_ligand, ratio, rng, mask_sampling='uniform', ligand_mask_frac=0.7, min_pairs=1):
    """Sample pair ids using cached per-pair ligand flags.

    pair_is_ligand:
      list/array of bool, length = num_pairs for one graph.

    This mirrors sample_pair_ids_with_ligand_bias() but avoids reconstructing
    pair_groups from edge_index when pair cache is enabled.
    """
    flags = [bool(x) for x in pair_is_ligand]
    num_pairs = len(flags)
    if num_pairs == 0:
        return []
    k = max(min_pairs, int(round(num_pairs * ratio)))
    k = min(k, num_pairs)
    if mask_sampling == 'uniform':
        return rng.sample(range(num_pairs), k)
    ligand_pair_ids = [i for i, flag in enumerate(flags) if flag]
    other_pair_ids = [i for i, flag in enumerate(flags) if not flag]
    if mask_sampling == 'ligand_only':
        if len(ligand_pair_ids) == 0:
            return []
        return rng.sample(ligand_pair_ids, min(k, len(ligand_pair_ids)))
    if mask_sampling == 'ligand_biased':
        if len(ligand_pair_ids) == 0:
            return rng.sample(range(num_pairs), k)
        k_lig = max(1, int(round(k * ligand_mask_frac)))
        k_lig = min(k_lig, len(ligand_pair_ids))
        k_other = min(k - k_lig, len(other_pair_ids))
        selected = []
        if k_lig > 0:
            selected.extend(rng.sample(ligand_pair_ids, k_lig))
        if k_other > 0:
            selected.extend(rng.sample(other_pair_ids, k_other))
        if len(selected) < k:
            selected_set = set(selected)
            remaining_pool = [pid for pid in range(num_pairs) if pid not in selected_set]
            if remaining_pool:
                selected.extend(rng.sample(remaining_pool, min(k - len(selected), len(remaining_pool))))
        rng.shuffle(selected)
        return selected
    raise ValueError(f'Unknown mask_sampling: {mask_sampling}')

def pair_ids_to_edges(pair_groups, pair_ids):
    directed_edges = []
    primary_edges = []
    for pid in pair_ids:
        group = pair_groups[int(pid)]
        directed_edges.extend(group)
        primary_edges.append(group[0])
    return (directed_edges, primary_edges)

def make_graph_split(paths, args, out_dir):
    split_path = Path(args.split_path) if args.split_path else out_dir / f'graph_split_seed{args.split_seed}.pt'
    if split_path.exists() and (not args.overwrite_split):
        split = load_pt(split_path, 'cpu')
        print(f'Loaded graph split: {split_path}')
        return (split, split_path)
    n = len(paths)
    indices = np.arange(n)
    rng = np.random.default_rng(args.split_seed)
    rng.shuffle(indices)
    test_n = int(round(n * args.test_ratio))
    valid_n = int(round(n * args.valid_ratio))
    test_idx = indices[:test_n]
    valid_idx = indices[test_n:test_n + valid_n]
    train_idx = indices[test_n + valid_n:]
    split = {'train_graph_indices': torch.tensor(train_idx, dtype=torch.long), 'valid_graph_indices': torch.tensor(valid_idx, dtype=torch.long), 'test_graph_indices': torch.tensor(test_idx, dtype=torch.long), 'graph_paths': [str(p) for p in paths], 'split_seed': args.split_seed, 'valid_ratio': args.valid_ratio, 'test_ratio': args.test_ratio}
    split_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(split, split_path)
    csv_path = split_path.with_suffix('.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'graph_index', 'graph_path'])
        for split_name, key in [('train', 'train_graph_indices'), ('valid', 'valid_graph_indices'), ('test', 'test_graph_indices')]:
            for i in split[key].tolist():
                writer.writerow([split_name, i, str(paths[i])])
    print(f'Saved graph split: {split_path}')
    print(f'Saved graph split CSV: {csv_path}')
    return (split, split_path)
PRETRAIN_EXCLUDE_KEYS = ['node_feature_names', 'edge_attr_names', 'bound_ligand_atom_feature_names', 'bound_ligand_bond_feature_names', 'bound_ligand_atom_x', 'bound_ligand_edge_index', 'bound_ligand_edge_attr', 'bound_ligand_atom_names', 'bound_ligand_atom_elements', 'bound_ligand_atom_resnames', 'bound_ligand_atom_resids', 'bound_ligand_atom_chains', 'small_mol_atom_x_list', 'small_mol_edge_index_list', 'small_mol_edge_attr_list', 'small_mol_resnames', 'small_mol_resids', 'small_mol_chains', 'small_mol_smiles', 'small_mol_graph_sources', 'small_mol_smiles_parse_statuses', 'water_node_indices', 'water_bridge_node_indices', 'water_degree_to_ligand', 'water_degree_to_standard_aa', 'water_min_dist_to_ligand', 'water_min_dist_to_standard_aa', 'n_ligand_water_edges', 'n_water_standard_aa_edges', 'n_ligand_water_protein_bridges', 'graph_version', 'pdbid', 'source_nodes_csv', 'source_edges_csv', 'source_esm_npy', 'pair_cache_source_graph_path', 'pair_cache_version', 'has_pair_cache', 'ligand_mask_source']

def sanitize_data_for_pretrain_batch(data):
    """Remove variable-length metadata/downstream attributes before PyG batching.

    PyG Batch.from_data_list recursively collates Python lists. Graphs can have
    different numbers of small molecules, atom graphs, water-bridge records, etc.,
    which may cause list index errors during batching. Current masked-edge
    pretraining only needs tensor attributes such as x, edge_index, edge_attr,
    is_ligand, and pair-cache tensors, so non-tensor metadata is dropped here.
    """
    removed = []
    try:
        keys = list(data.keys())
    except Exception:
        keys = []
    for key in keys:
        try:
            value = data[key]
        except Exception:
            continue
        if torch.is_tensor(value):
            continue
        try:
            del data[key]
            removed.append(key)
        except Exception:
            pass
    data.pretrain_sanitized = torch.tensor([1], dtype=torch.long)
    data.n_pretrain_removed_attrs = torch.tensor([len(removed)], dtype=torch.long)
    return data

class GraphFileDataset(torch.utils.data.Dataset):

    def __init__(self, paths, preload=False, cache_pairs=True, add_dist_z=True):
        self.paths = [Path(p) for p in paths]
        self.preload = preload
        self.cache_pairs = cache_pairs
        self.add_dist_z = add_dist_z
        self.data_list = None
        if self.preload:
            self.data_list = []
            for p in self.paths:
                data = load_pt(p, 'cpu')
                data.graph_path = str(p)
                if self.cache_pairs:
                    data = add_pair_cache(data)
                if self.add_dist_z:
                    data = add_raw_distance_zscore(data)
                data = sanitize_data_for_pretrain_batch(data)
                self.data_list.append(data)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if self.preload:
            return self.data_list[idx]
        data = load_pt(self.paths[idx], 'cpu')
        data.graph_path = str(self.paths[idx])
        if self.cache_pairs:
            data = add_pair_cache(data)
        if self.add_dist_z:
            data = add_raw_distance_zscore(data)
        data = sanitize_data_for_pretrain_batch(data)
        return data

class EdgeMLPMPNN(nn.Module):

    def __init__(self, node_in_dim, edge_in_dim, geom_dim, target_dim, hidden_dim=256, num_layers=4, dropout=0.1, decoder_head='shared', component_head_hidden=0, distance_target_dim=0, node_target_dim=0):
        super().__init__()
        if decoder_head not in {'shared', 'hybrid', 'linear', 'linear_nogeom', 'hybrid_nogeom'}:
            raise ValueError(f'Unknown decoder_head: {decoder_head}')
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.decoder_head = decoder_head
        self.component_head_hidden = int(component_head_hidden)
        self.target_dim = int(target_dim)
        self.node_encoder = nn.Sequential(nn.Linear(node_in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.edge_encoder = nn.Sequential(nn.Linear(edge_in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.message_mlps = nn.ModuleList()
        self.grus = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.message_mlps.append(nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
            self.grus.append(nn.GRUCell(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        decoder_in_dim = hidden_dim * 2 + geom_dim
        decoder_latent_dim = hidden_dim // 2
        if decoder_head == 'shared':
            self.edge_decoder = nn.Sequential(nn.Linear(decoder_in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, decoder_latent_dim), nn.SiLU(), nn.Linear(decoder_latent_dim, target_dim))
            self.edge_decoder_trunk = None
            self.component_heads = None
        elif decoder_head in ('hybrid', 'hybrid_nogeom'):
            trunk_in_dim = hidden_dim * 2 if decoder_head == 'hybrid_nogeom' else decoder_in_dim
            self.edge_decoder = None
            self.edge_decoder_trunk = nn.Sequential(nn.Linear(trunk_in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, decoder_latent_dim), nn.SiLU())
            heads = []
            for _ in range(target_dim):
                if self.component_head_hidden > 0:
                    heads.append(nn.Sequential(nn.Linear(decoder_latent_dim, self.component_head_hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(self.component_head_hidden, 1)))
                else:
                    heads.append(nn.Linear(decoder_latent_dim, 1))
            self.component_heads = nn.ModuleList(heads)
        else:
            linear_in_dim = hidden_dim * 2 if decoder_head == 'linear_nogeom' else decoder_in_dim
            self.edge_decoder = nn.Linear(linear_in_dim, target_dim)
            self.edge_decoder_trunk = None
            self.component_heads = None
        self.distance_target_dim = int(distance_target_dim)
        self.node_target_dim = int(node_target_dim)
        self.distance_decoder = None
        if self.distance_target_dim > 0:
            self.distance_decoder = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, self.distance_target_dim))
        self.node_decoder = None
        if self.node_target_dim > 0:
            self.node_decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, self.node_target_dim))

    def decode_distance(self, h, edge_index, decode_edge_indices):
        if self.distance_decoder is None:
            raise RuntimeError('distance_decoder is disabled')
        dec_e = decode_edge_indices
        u = edge_index[0, dec_e]
        v = edge_index[1, dec_e]
        return self.distance_decoder(torch.cat([h[u], h[v]], dim=-1))

    def decode_nodes(self, h, node_indices):
        if self.node_decoder is None:
            raise RuntimeError('node_decoder is disabled')
        return self.node_decoder(h[node_indices])

    def encode_nodes(self, x, edge_index, edge_attr_input):
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr_input)
        src = edge_index[0]
        dst = edge_index[1]
        for layer in range(self.num_layers):
            msg_input = torch.cat([h[src], h[dst], e], dim=-1)
            msg = self.message_mlps[layer](msg_input)
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, msg)
            h_new = self.grus[layer](agg, h)
            h = self.norms[layer](h + F.dropout(h_new, p=self.dropout, training=self.training))
        return h

    def decode_edges(self, h, edge_index, edge_geom, decode_edge_indices, return_latent=False):
        dec_e = decode_edge_indices
        u = edge_index[0, dec_e]
        v = edge_index[1, dec_e]
        if self.decoder_head in ('linear_nogeom', 'hybrid_nogeom'):
            dec_input = torch.cat([h[u], h[v]], dim=-1)
        else:
            dec_input = torch.cat([h[u], h[v], edge_geom[dec_e]], dim=-1)
        if self.decoder_head in ('shared', 'linear', 'linear_nogeom'):
            pred = self.edge_decoder(dec_input)
            if return_latent:
                return (pred, None)
            return pred
        z_edge = self.edge_decoder_trunk(dec_input)
        pred = torch.cat([head(z_edge) for head in self.component_heads], dim=-1)
        if return_latent:
            return (pred, z_edge)
        return pred

    def forward(self, x, edge_index, edge_attr_input, edge_geom, decode_edge_indices, return_latent=False):
        h = self.encode_nodes(x, edge_index, edge_attr_input)
        return self.decode_edges(h=h, edge_index=edge_index, edge_geom=edge_geom, decode_edge_indices=decode_edge_indices, return_latent=return_latent)

def edge_is_ligand_pair(edge_index_cpu, is_ligand_cpu, edge_id):
    u = int(edge_index_cpu[0, edge_id])
    v = int(edge_index_cpu[1, edge_id])
    return bool(is_ligand_cpu[u] or is_ligand_cpu[v])

def split_pair_groups_by_ligand(pair_groups, edge_index_cpu, is_ligand_cpu):
    ligand_pair_ids = []
    other_pair_ids = []
    for pid, group in enumerate(pair_groups):
        if edge_is_ligand_pair(edge_index_cpu, is_ligand_cpu, group[0]):
            ligand_pair_ids.append(pid)
        else:
            other_pair_ids.append(pid)
    return (ligand_pair_ids, other_pair_ids)

def sample_pair_ids_with_ligand_bias(pair_groups, edge_index_cpu, is_ligand_cpu, ratio, rng, mask_sampling='uniform', ligand_mask_frac=0.7, min_pairs=1):
    num_pairs = len(pair_groups)
    if num_pairs == 0:
        return []
    k = max(min_pairs, int(round(num_pairs * ratio)))
    k = min(k, num_pairs)
    if mask_sampling == 'uniform':
        return rng.sample(range(num_pairs), k)
    ligand_pair_ids, other_pair_ids = split_pair_groups_by_ligand(pair_groups=pair_groups, edge_index_cpu=edge_index_cpu, is_ligand_cpu=is_ligand_cpu)
    if mask_sampling == 'ligand_only':
        if len(ligand_pair_ids) == 0:
            return rng.sample(range(num_pairs), k)
        return rng.sample(ligand_pair_ids, min(k, len(ligand_pair_ids)))
    if mask_sampling == 'ligand_biased':
        if len(ligand_pair_ids) == 0:
            return rng.sample(range(num_pairs), k)
        k_lig = max(1, int(round(k * ligand_mask_frac)))
        k_lig = min(k_lig, len(ligand_pair_ids))
        k_other = min(k - k_lig, len(other_pair_ids))
        selected = []
        if k_lig > 0:
            selected.extend(rng.sample(ligand_pair_ids, k_lig))
        if k_other > 0:
            selected.extend(rng.sample(other_pair_ids, k_other))
        if len(selected) < k:
            selected_set = set(selected)
            remaining_pool = [pid for pid in range(num_pairs) if pid not in selected_set]
            if remaining_pool:
                selected.extend(rng.sample(remaining_pool, min(k - len(selected), len(remaining_pool))))
        rng.shuffle(selected)
        return selected
    raise ValueError(f'Unknown mask_sampling: {mask_sampling}')

def compute_decode_edge_weights(batch, decode_edges, ligand_edge_loss_weight, device):
    weights = torch.ones(decode_edges.shape[0], dtype=torch.float32, device=device)
    if ligand_edge_loss_weight == 1.0 or not hasattr(batch, 'is_ligand'):
        return weights
    is_ligand = batch.is_ligand.to(device)
    u = batch.edge_index[0, decode_edges]
    v = batch.edge_index[1, decode_edges]
    lig_mask = is_ligand[u] | is_ligand[v]
    weights[lig_mask] = float(ligand_edge_loss_weight)
    return weights

def weighted_smooth_l1_loss(pred, target, weights, beta=1.0):
    loss_elem = F.smooth_l1_loss(pred, target, beta=beta, reduction='none')
    w = weights[:, None]
    return (loss_elem * w).sum() / (w.sum() * pred.shape[1] + 1e-12)

def weighted_abs_err_sum(pred, target, weights):
    return ((pred.detach() - target.detach()).abs() * weights[:, None]).sum(dim=0).cpu()

def make_fmo_target_control(batch, decode_edges, target_idx, mode, rng, device, distance_source_idx=None, distance_bins=(3.5, 5.0, 7.0, 10.0)):
    """Return real or shuffled FMO/PIEDA targets for decoded primary edges.

    mode="real": true target.
    mode="graph_shuffle": shuffle target vectors among decoded edges within each graph.
    mode="distbin_shuffle": shuffle target vectors within graph and distance bin.
    Vector shuffle keeps component correlations intact.
    """
    target_idx_t = torch.tensor(target_idx, dtype=torch.long, device=device)
    y = batch.edge_attr[decode_edges[:, None], target_idx_t].detach().clone()
    if mode == 'real':
        return y
    if mode not in {'graph_shuffle', 'distbin_shuffle'}:
        raise ValueError(f'Unknown fmo_target_mode: {mode}')
    if decode_edges.numel() <= 1:
        return y
    edge_graph = batch.batch[batch.edge_index[0, decode_edges]].detach().cpu().numpy()
    if mode == 'distbin_shuffle' and distance_source_idx is not None:
        dist = batch.edge_attr[decode_edges, int(distance_source_idx)].detach().cpu().numpy()
        bins = np.digitize(dist, list(distance_bins), right=False)
    else:
        bins = np.zeros(len(decode_edges), dtype=np.int64)
    y_out = y.clone()
    n = int(decode_edges.numel())
    for g in np.unique(edge_graph):
        if mode == 'graph_shuffle':
            ids = np.where(edge_graph == g)[0]
            groups = [ids]
        else:
            groups = [np.where((edge_graph == g) & (bins == b))[0] for b in np.unique(bins[edge_graph == g])]
        for ids in groups:
            if len(ids) <= 1:
                continue
            perm = list(ids)
            rng.shuffle(perm)
            ids_t = torch.tensor(ids, dtype=torch.long, device=device)
            perm_t = torch.tensor(perm, dtype=torch.long, device=device)
            y_out[ids_t] = y[perm_t]
    return y_out

def apply_local_fmo_blackout(edge_attr_input, batch, decode_edges, fmo_mask_idx, device, include_target_edges=True, zero_all_dims=True):
    """Zero features on edges incident to endpoints of decoded edges.

    Prevents the model from using information from edges directly adjacent to
    the masked edge endpoints.
    zero_all_dims=True (default, full mode): zeros all edge feature dims.
    zero_all_dims=False (pieda_only mode): zeros only fmo_mask_idx dims.
    """
    if decode_edges.numel() == 0:
        return edge_attr_input
    ei = batch.edge_index
    endpoints = torch.unique(ei[:, decode_edges].reshape(-1))
    src = ei[0]
    dst = ei[1]
    incident = torch.isin(src, endpoints) | torch.isin(dst, endpoints)
    if include_target_edges:
        incident[decode_edges] = True
    eids = torch.nonzero(incident, as_tuple=False).view(-1)
    if eids.numel() == 0:
        return edge_attr_input
    out = edge_attr_input.clone()
    if zero_all_dims:
        out[eids] = 0.0
    else:
        idx_t = torch.tensor(fmo_mask_idx, dtype=torch.long, device=device)
        out[eids[:, None], idx_t] = 0.0
    return out

def build_batch_mask_and_targets_cached(batch, target_idx, fmo_mask_idx, mode, mask_ratio, rng, device, mask_sampling='uniform', ligand_mask_frac=0.7):
    """
    Faster mask builder using per-graph cached pair information.

    Fixed behavior:
      In older versions, cached mode ignored mask_sampling and always sampled uniformly.
      This version uses cached pair_is_ligand, so ligand_biased and ligand_only
      work even when pair cache is enabled.
    """
    edge_attr_input = batch.edge_attr.clone()
    fmo_mask_idx_t = torch.tensor(fmo_mask_idx, dtype=torch.long, device=device)
    target_idx_t = torch.tensor(target_idx, dtype=torch.long, device=device)
    if not hasattr(batch, 'num_pairs_cached'):
        raise RuntimeError('Pair cache attributes not found. Use GraphFileDataset(..., cache_pairs=True).')
    if not hasattr(batch, 'pair_is_ligand'):
        raise RuntimeError('Pair cache is missing pair_is_ligand. Rebuild pair cache with the patched add_pair_cache(), or run once with --no-pair-cache.')
    num_pairs_per_graph = batch.num_pairs_cached.view(-1).detach().cpu().tolist()
    num_graphs = len(num_pairs_per_graph)
    edge_offsets = get_edge_offsets_for_batch(batch, device)
    pair_primary = batch.pair_primary_edges.to(device)
    pair_flat = batch.pair_directed_edges_flat.to(device)
    pair_offsets = batch.pair_directed_offsets.to(device)
    pair_is_ligand_all = batch.pair_is_ligand.detach().cpu().numpy().astype(bool)
    pair_start = 0
    offset_start = 0
    flat_start = 0
    decode_edges_all = []
    directed_edges_all = []
    target_pairs_total = 0
    for g in range(num_graphs):
        n_pairs = int(num_pairs_per_graph[g])
        if n_pairs <= 0:
            offset_start += 1
            continue
        flags = pair_is_ligand_all[pair_start:pair_start + n_pairs]
        pair_ids = sample_pair_ids_with_cached_ligand_flags(pair_is_ligand=flags, ratio=mask_ratio, rng=rng, mask_sampling=mask_sampling, ligand_mask_frac=ligand_mask_frac)
        if len(pair_ids) == 0:
            pair_start += n_pairs
            off_chunk_tmp = pair_offsets[offset_start:offset_start + n_pairs + 1]
            flat_start += int(off_chunk_tmp[-1].item())
            offset_start += n_pairs + 1
            continue
        pair_ids_t = torch.tensor(pair_ids, dtype=torch.long, device=device)
        primary_local = pair_primary[pair_start + pair_ids_t]
        primary_global = primary_local + edge_offsets[g]
        decode_edges_all.append(primary_global)
        off_chunk = pair_offsets[offset_start:offset_start + n_pairs + 1]
        lengths = off_chunk[1:] - off_chunk[:-1]
        if bool(torch.all(lengths[pair_ids_t] == 2).item()):
            starts = off_chunk[pair_ids_t]
            gather_idx = torch.stack([starts, starts + 1], dim=1).reshape(-1)
            directed_local = pair_flat[flat_start + gather_idx]
        else:
            chunks = []
            for pid in pair_ids:
                s = int(off_chunk[pid].item())
                e = int(off_chunk[pid + 1].item())
                if e > s:
                    chunks.append(pair_flat[flat_start + s:flat_start + e])
            directed_local = torch.cat(chunks, dim=0) if chunks else torch.empty(0, dtype=torch.long, device=device)
        if directed_local.numel() > 0:
            directed_global = directed_local + edge_offsets[g]
            directed_edges_all.append(directed_global)
        target_pairs_total += len(pair_ids)
        pair_start += n_pairs
        n_flat = int(off_chunk[-1].item())
        flat_start += n_flat
        offset_start += n_pairs + 1
    if len(decode_edges_all) == 0:
        raise RuntimeError('No decode edges sampled from batch.')
    decode_edges = torch.cat(decode_edges_all, dim=0)
    directed_edges = torch.cat(directed_edges_all, dim=0) if directed_edges_all else decode_edges
    if mode == 'b_context' and directed_edges.numel() > 0:
        edge_attr_input[directed_edges[:, None], fmo_mask_idx_t] = 0.0
    target = batch.edge_attr[decode_edges[:, None], target_idx_t]
    return (edge_attr_input, decode_edges, directed_edges, target, target_pairs_total)

def mask_distance_features_for_edges(edge_attr_input, directed_edges_list, distance_mask_idx, device):
    """Zero distance/RBF features for selected directed edges."""
    if not distance_mask_idx or not directed_edges_list:
        return edge_attr_input
    idx_t = torch.tensor(distance_mask_idx, dtype=torch.long, device=device)
    directed_edges = torch.cat(directed_edges_list, dim=0) if isinstance(directed_edges_list[0], torch.Tensor) else torch.tensor(directed_edges_list, dtype=torch.long, device=device)
    if directed_edges.numel() > 0:
        edge_attr_input = edge_attr_input.clone()
        edge_attr_input[directed_edges[:, None], idx_t] = 0.0
    return edge_attr_input

def build_batch_mask_and_targets(batch, target_idx, fmo_mask_idx, mode, mask_ratio, rng, device, mask_sampling='uniform', ligand_mask_frac=0.7):
    """
    Build edge_attr_input and prediction targets for a PyG Batch.

    Returns:
      edge_attr_input
      decode_edges tensor
      target tensor
      num_target_pairs
    """
    edge_attr_input = batch.edge_attr.clone()
    edge_index = batch.edge_index
    edge_geom = None
    fmo_mask_idx_t = torch.tensor(fmo_mask_idx, dtype=torch.long, device=device)
    target_idx_t = torch.tensor(target_idx, dtype=torch.long, device=device)
    decode_edges_all = []
    directed_edges_all = []
    target_pairs_total = 0
    batch_vec = batch.batch
    num_graphs = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 1
    edge_graph = batch_vec[edge_index[0]].detach().cpu().numpy()
    edge_index_cpu = edge_index.detach().cpu()
    if hasattr(batch, 'is_ligand'):
        is_ligand_cpu = batch.is_ligand.detach().cpu().numpy().astype(bool)
    else:
        is_ligand_cpu = np.zeros(batch.x.shape[0], dtype=bool)
    for g in range(num_graphs):
        edge_ids = np.where(edge_graph == g)[0].tolist()
        if len(edge_ids) == 0:
            continue
        local_pair_map = {}
        for e in edge_ids:
            u = int(edge_index_cpu[0, e])
            v = int(edge_index_cpu[1, e])
            a, b = (u, v) if u <= v else (v, u)
            local_pair_map.setdefault((a, b), []).append(e)
        pair_groups = list(local_pair_map.values())
        if len(pair_groups) == 0:
            continue
        pair_ids = sample_pair_ids_with_ligand_bias(pair_groups=pair_groups, edge_index_cpu=edge_index_cpu, is_ligand_cpu=is_ligand_cpu, ratio=mask_ratio, rng=rng, mask_sampling=mask_sampling, ligand_mask_frac=ligand_mask_frac)
        directed_edges, primary_edges = pair_ids_to_edges(pair_groups, pair_ids)
        if len(directed_edges) > 0:
            directed_t = torch.tensor(directed_edges, dtype=torch.long, device=device)
            directed_edges_all.append(directed_t)
            if mode == 'b_context':
                edge_attr_input[directed_t[:, None], fmo_mask_idx_t] = 0.0
        decode_edges_all.extend(primary_edges)
        target_pairs_total += len(primary_edges)
    if len(decode_edges_all) == 0:
        raise RuntimeError('No decode edges sampled from batch.')
    decode_edges = torch.tensor(decode_edges_all, dtype=torch.long, device=device)
    directed_edges = torch.cat(directed_edges_all, dim=0) if directed_edges_all else decode_edges
    target = batch.edge_attr[decode_edges[:, None], target_idx_t]
    return (edge_attr_input, decode_edges, directed_edges, target, target_pairs_total)

@torch.no_grad()
def evaluate_loader(model, loader, target_idx, fmo_mask_idx, geom_idx_t, mode, ablation, ablation_indices, mask_ratio, loss_fn, device, seed, mask_sampling='uniform', ligand_mask_frac=0.7, ligand_edge_loss_weight=1.0, fmo_ligand_edge_loss_weight=1.0):
    model.eval()
    total_loss_weighted = 0.0
    total_pairs = 0
    component_abs_sum = None
    rng = random.Random(seed)
    for batch in loader:
        batch = batch.to(device)
        builder = build_batch_mask_and_targets_cached if getattr(batch, 'num_pairs_cached', None) is not None else build_batch_mask_and_targets
        edge_attr_input, decode_edges, directed_edges, target, n_pairs = builder(batch=batch, target_idx=target_idx, fmo_mask_idx=fmo_mask_idx, mode=mode, mask_ratio=mask_ratio, rng=rng, device=device, mask_sampling=mask_sampling, ligand_mask_frac=ligand_mask_frac)
        target = make_fmo_target_control(batch=batch, decode_edges=decode_edges, target_idx=target_idx, mode=getattr(evaluate_loader, 'fmo_target_mode', 'real'), rng=rng, device=device, distance_source_idx=getattr(evaluate_loader, 'distance_source_idx', None))
        if getattr(evaluate_loader, 'local_fmo_blackout', False):
            edge_attr_input = apply_local_fmo_blackout(edge_attr_input=edge_attr_input, batch=batch, decode_edges=decode_edges, fmo_mask_idx=fmo_mask_idx, device=device, zero_all_dims=getattr(evaluate_loader, 'zero_all_dims_blackout', True))
        x_input, edge_attr_input = apply_input_ablation(batch=batch, edge_attr_input=edge_attr_input, ablation=ablation, ablation_indices=ablation_indices)
        node_target = None
        node_ids = None
        if getattr(model, 'node_decoder', None) is not None and getattr(evaluate_loader, 'node_mask_task', 'none') == 'ligand_block':
            x_input, node_ids, node_target, _ = mask_ligand_sparse_node_features(batch=batch, x_input=x_input, node_idx=evaluate_loader.ligand_node_idx, rng=rng, mask_ratio=evaluate_loader.node_mask_ratio, device=device)
        dist_target = None
        if getattr(model, 'distance_decoder', None) is not None and getattr(evaluate_loader, 'dist_loss_weight', 0.0) > 0:
            dist_idx_t = torch.tensor(evaluate_loader.distance_target_idx, dtype=torch.long, device=device)
            dist_target = batch.edge_attr[decode_edges[:, None], dist_idx_t].detach()
            dist_mask_edges = directed_edges if getattr(evaluate_loader, 'joint_mask_distance_directions', True) else decode_edges
            edge_attr_input = edge_attr_input.clone()
            edge_attr_input[dist_mask_edges[:, None], dist_idx_t] = 0.0
            edge_attr_input[dist_mask_edges[:, None], geom_idx_t] = 0.0
        edge_geom = edge_attr_input[:, geom_idx_t]
        h = model.encode_nodes(x_input, batch.edge_index, edge_attr_input)
        pred = model.decode_edges(h, batch.edge_index, edge_geom, decode_edges)
        dist_edge_weights = compute_decode_edge_weights(batch=batch, decode_edges=decode_edges, ligand_edge_loss_weight=ligand_edge_loss_weight, device=device)
        fmo_edge_weights = compute_decode_edge_weights(batch=batch, decode_edges=decode_edges, ligand_edge_loss_weight=fmo_ligand_edge_loss_weight, device=device)
        fmo_loss = weighted_smooth_l1_loss(pred, target, fmo_edge_weights, beta=loss_fn.beta)
        loss = fmo_loss * float(getattr(evaluate_loader, 'fmo_loss_weight', 1.0))
        if dist_target is not None:
            dist_pred = model.decode_distance(h, batch.edge_index, decode_edges)
            loss = loss + float(evaluate_loader.dist_loss_weight) * weighted_smooth_l1_loss(dist_pred, dist_target, dist_edge_weights, beta=evaluate_loader.dist_huber_beta)
        if node_target is not None and node_ids is not None and (node_ids.numel() > 0):
            node_pred = model.decode_nodes(h, node_ids)
            loss = loss + float(evaluate_loader.node_loss_weight) * F.binary_cross_entropy_with_logits(node_pred, node_target)
        abs_err = weighted_abs_err_sum(pred, target, fmo_edge_weights)
        weight_sum = float(fmo_edge_weights.sum().item())
        total_loss_weighted += loss.item() * weight_sum
        total_pairs += weight_sum
        if component_abs_sum is None:
            component_abs_sum = abs_err
        else:
            component_abs_sum += abs_err
    avg_loss = total_loss_weighted / max(total_pairs, 1)
    component_mae = component_abs_sum / max(total_pairs, 1)
    return (avg_loss, component_mae)

def audit_pair_cache_ligand_fraction(paths, max_graphs=50):
    """Quick CPU audit of cached pair ligand flags on a subset of graphs."""
    n_graphs = 0
    total_pairs = 0
    ligand_pairs = 0
    total_ligand_nodes = 0
    sources = {}
    examples = []
    for p in paths[:max_graphs]:
        data = load_pt(p, 'cpu')
        data = add_pair_cache(data)
        flags = data.pair_is_ligand.detach().cpu().numpy().astype(bool)
        total_pairs += int(flags.shape[0])
        ligand_pairs += int(flags.sum())
        n_graphs += 1
        mask, source = infer_ligand_node_mask(data)
        total_ligand_nodes += int(mask.sum())
        sources[source] = sources.get(source, 0) + 1
        if len(examples) < 5:
            examples.append({'path': str(p), 'source': source, 'n_nodes': int(data.x.shape[0]), 'n_ligand_nodes': int(mask.sum()), 'n_pairs': int(flags.shape[0]), 'n_ligand_pairs': int(flags.sum())})
    frac = ligand_pairs / max(total_pairs, 1)
    return {'audit_graphs': n_graphs, 'audit_total_pairs': total_pairs, 'audit_ligand_pairs': ligand_pairs, 'audit_ligand_pair_fraction': frac, 'audit_total_ligand_nodes': total_ligand_nodes, 'ligand_mask_sources': sources, 'examples': examples}

def train(args):
    device = torch.device(args.device if args.device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    effective_mode, effective_ablation = resolve_mode_and_ablation(args.mode, args.ablation)
    graph_dir = Path(args.graph_dir)
    paths_all = sorted(graph_dir.glob(args.pattern))
    if len(paths_all) == 0:
        raise FileNotFoundError(f'No graph files found in {graph_dir} with pattern {args.pattern}')
    paths, graph_filter_audit_csv = filter_graph_paths(paths_all, args, out_dir)
    pieda_only = args.masking_mode == 'pieda_only'
    if pieda_only and args.dist_loss_weight > 0:
        raise ValueError('--masking-mode pieda_only requires --dist-loss-weight 0.0 (distance dims are not added in this mode).')
    first = load_pt(paths[0], 'cpu')
    if not pieda_only:
        first = add_raw_distance_zscore(first)
    geom_idx, target_idx, fmo_mask_idx, edge_attr_names = infer_feature_groups(first)
    ablation_indices = infer_input_ablation_indices(first, fmo_mask_idx)
    distance_target_idx = infer_distance_rawz_target_indices(first) if not pieda_only else []
    distance_source_idx = None
    for _name in ['DIST_raw', 'distance', 'dist', 'r', 'r_ij']:
        if _name in edge_attr_names:
            distance_source_idx = edge_attr_names.index(_name)
            break
    ligand_node_idx = infer_ligand_sparse_node_block(first, args.ligand_node_start, args.ligand_node_end)
    node_in_dim = first.x.shape[1]
    edge_in_dim = first.edge_attr.shape[1]
    geom_dim = len(geom_idx)
    target_dim = len(target_idx)
    split, split_path = make_graph_split(paths, args, out_dir)
    train_paths = [paths[i] for i in split['train_graph_indices'].tolist()]
    valid_paths = [paths[i] for i in split['valid_graph_indices'].tolist()]
    test_paths = [paths[i] for i in split['test_graph_indices'].tolist()]
    pair_ligand_audit = audit_pair_cache_ligand_fraction(train_paths, max_graphs=min(50, len(train_paths)))
    if len(train_paths) == 0:
        raise ValueError('Train split is empty. Reduce valid/test ratios or add more graphs.')
    if len(valid_paths) == 0:
        raise ValueError('Valid split is empty. Reduce valid ratio or add more graphs.')
    if len(test_paths) == 0:
        raise ValueError('Test split is empty. Reduce test ratio or add more graphs.')
    persistent_workers = args.persistent_workers and args.num_workers > 0
    train_loader = DataLoader(GraphFileDataset(train_paths, preload=args.preload, cache_pairs=not args.no_pair_cache, add_dist_z=not pieda_only), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory, persistent_workers=persistent_workers, exclude_keys=PRETRAIN_EXCLUDE_KEYS)
    valid_loader = DataLoader(GraphFileDataset(valid_paths, preload=args.preload, cache_pairs=not args.no_pair_cache, add_dist_z=not pieda_only), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, persistent_workers=persistent_workers, exclude_keys=PRETRAIN_EXCLUDE_KEYS)
    test_loader = DataLoader(GraphFileDataset(test_paths, preload=args.preload, cache_pairs=not args.no_pair_cache, add_dist_z=not pieda_only), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, persistent_workers=persistent_workers, exclude_keys=PRETRAIN_EXCLUDE_KEYS)
    print(f'Device: {device}')
    print(f'Graph dir: {graph_dir}')
    print(f'Graphs total after filter: {len(paths)}')
    print(f'Graph filter audit CSV: {graph_filter_audit_csv}')
    print(f'Train graphs: {len(train_paths)}')
    print(f'Valid graphs: {len(valid_paths)}')
    print(f'Test graphs: {len(test_paths)}')
    print(f'Split path: {split_path}')
    print(f'Mode: {args.mode}')
    print(f'Effective mode: {effective_mode}')
    print(f'Ablation: {effective_ablation}')
    print(f'Mask sampling: {args.mask_sampling}')
    print(f'Ligand mask fraction: {args.ligand_mask_frac}')
    print(f'Ligand edge loss weight (dist): {args.ligand_edge_loss_weight}')
    print(f'Ligand edge loss weight (FMO): {args.fmo_ligand_edge_loss_weight}')
    print(f'Preload graphs into RAM: {args.preload}')
    print(f'DataLoader num_workers: {args.num_workers}')
    print(f'DataLoader pin_memory: {args.pin_memory}')
    print(f'DataLoader persistent_workers: {args.persistent_workers and args.num_workers > 0}')
    print(f'Pair cache enabled: {not args.no_pair_cache}')
    print(f'PyG batch exclude_keys: {len(PRETRAIN_EXCLUDE_KEYS)} metadata/downstream attributes')
    print('Pretrain Data sanitize: drop all non-tensor Data attributes before PyG batching')
    print(f'Pair cache ligand audit: {pair_ligand_audit}')
    print(f'Node dim: {node_in_dim}')
    print(f'Edge dim: {edge_in_dim}')
    print(f'Geometry dims: {geom_dim}')
    print('Target dims:')
    for i in target_idx:
        print(f'  {i}: {edge_attr_names[i]}')
    print('Masked/removed FMO dims:')
    for i in fmo_mask_idx:
        print(f'  {i}: {edge_attr_names[i]}')
    print(f"Distance input dims: {len(ablation_indices['distance_idx'])}")
    print(f"ESM input dims: {len(ablation_indices['esm_idx'])}")
    print('Decoder geometry source: ablated edge_attr_input')
    print(f'Decoder head: {args.decoder_head}')
    print(f'Component head hidden dim: {args.component_head_hidden}')
    print(f'Aux fmo loss weight: {args.fmo_loss_weight}')
    print(f'Aux distance loss weight: {args.dist_loss_weight}')
    print(f'Distance target dims: {len(distance_target_idx)}')
    print(f'Distance source idx for distbin shuffle: {distance_source_idx}')
    print(f'FMO target mode: {args.fmo_target_mode}')
    print(f'Local FMO blackout: {args.local_fmo_blackout}')
    print(f'Masking mode: {args.masking_mode}')
    print(f'Joint mask distance directions: {args.joint_mask_distance_directions}')
    print(f'Node mask task: {args.node_mask_task}')
    print(f'Node loss weight: {args.node_loss_weight}')
    print(f'Ligand sparse node block dims: {len(ligand_node_idx)} [{args.ligand_node_start}:{args.ligand_node_end}]')
    model = EdgeMLPMPNN(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim, geom_dim=geom_dim, target_dim=target_dim, hidden_dim=args.hidden_dim, num_layers=args.layers, dropout=args.dropout, decoder_head=args.decoder_head, component_head_hidden=args.component_head_hidden, distance_target_dim=len(distance_target_idx) if args.dist_loss_weight > 0 else 0, node_target_dim=len(ligand_node_idx) if args.node_mask_task == 'ligand_block' and args.node_loss_weight > 0 else 0).to(device)
    geom_idx_t = torch.tensor(geom_idx, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    loss_fn = nn.SmoothL1Loss(beta=args.huber_beta)
    evaluate_loader.fmo_loss_weight = float(args.fmo_loss_weight)
    evaluate_loader.dist_loss_weight = float(args.dist_loss_weight)
    evaluate_loader.dist_huber_beta = float(args.dist_huber_beta)
    evaluate_loader.fmo_ligand_edge_loss_weight = float(args.fmo_ligand_edge_loss_weight)
    evaluate_loader.node_loss_weight = float(args.node_loss_weight)
    evaluate_loader.node_mask_task = str(args.node_mask_task)
    evaluate_loader.node_mask_ratio = float(args.node_mask_ratio)
    evaluate_loader.distance_target_idx = list(distance_target_idx)
    evaluate_loader.distance_source_idx = distance_source_idx
    evaluate_loader.fmo_target_mode = str(args.fmo_target_mode)
    evaluate_loader.local_fmo_blackout = bool(args.local_fmo_blackout)
    evaluate_loader.zero_all_dims_blackout = not pieda_only
    evaluate_loader.joint_mask_distance_directions = bool(args.joint_mask_distance_directions)
    evaluate_loader.ligand_node_idx = list(ligand_node_idx)
    best_valid = math.inf
    best_epoch = -1
    total_start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.perf_counter()
        model.train()
        total_loss_weighted = 0.0
        total_pairs = 0
        component_abs_sum = None
        rng = random.Random(args.seed + epoch)
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            builder = build_batch_mask_and_targets_cached if getattr(batch, 'num_pairs_cached', None) is not None else build_batch_mask_and_targets
            edge_attr_input, decode_edges, directed_edges, target, n_pairs = builder(batch=batch, target_idx=target_idx, fmo_mask_idx=fmo_mask_idx, mode=effective_mode, mask_ratio=args.mask_ratio, rng=rng, device=device, mask_sampling=args.mask_sampling, ligand_mask_frac=args.ligand_mask_frac)
            target = make_fmo_target_control(batch=batch, decode_edges=decode_edges, target_idx=target_idx, mode=args.fmo_target_mode, rng=rng, device=device, distance_source_idx=distance_source_idx)
            if args.local_fmo_blackout:
                edge_attr_input = apply_local_fmo_blackout(edge_attr_input=edge_attr_input, batch=batch, decode_edges=decode_edges, fmo_mask_idx=fmo_mask_idx, device=device, zero_all_dims=not pieda_only)
            x_input, edge_attr_input = apply_input_ablation(batch=batch, edge_attr_input=edge_attr_input, ablation=effective_ablation, ablation_indices=ablation_indices)
            node_target = None
            node_ids = None
            if args.node_mask_task == 'ligand_block' and args.node_loss_weight > 0:
                x_input, node_ids, node_target, _ = mask_ligand_sparse_node_features(batch=batch, x_input=x_input, node_idx=ligand_node_idx, rng=rng, mask_ratio=args.node_mask_ratio, device=device)
            dist_target = None
            if args.dist_loss_weight > 0:
                dist_idx_t = torch.tensor(distance_target_idx, dtype=torch.long, device=device)
                dist_target = batch.edge_attr[decode_edges[:, None], dist_idx_t].detach()
                dist_mask_edges = directed_edges if args.joint_mask_distance_directions else decode_edges
                edge_attr_input = edge_attr_input.clone()
                edge_attr_input[dist_mask_edges[:, None], dist_idx_t] = 0.0
                edge_attr_input[dist_mask_edges[:, None], geom_idx_t] = 0.0
            edge_geom = edge_attr_input[:, geom_idx_t]
            h = model.encode_nodes(x_input, batch.edge_index, edge_attr_input)
            pred = model.decode_edges(h=h, edge_index=batch.edge_index, edge_geom=edge_geom, decode_edge_indices=decode_edges)
            dist_edge_weights = compute_decode_edge_weights(batch=batch, decode_edges=decode_edges, ligand_edge_loss_weight=args.ligand_edge_loss_weight, device=device)
            fmo_edge_weights = compute_decode_edge_weights(batch=batch, decode_edges=decode_edges, ligand_edge_loss_weight=args.fmo_ligand_edge_loss_weight, device=device)
            fmo_loss = weighted_smooth_l1_loss(pred, target, fmo_edge_weights, beta=args.huber_beta)
            loss = fmo_loss * float(args.fmo_loss_weight)
            if dist_target is not None:
                dist_pred = model.decode_distance(h, batch.edge_index, decode_edges)
                loss = loss + float(args.dist_loss_weight) * weighted_smooth_l1_loss(dist_pred, dist_target, dist_edge_weights, beta=args.dist_huber_beta)
            if node_target is not None and node_ids is not None and (node_ids.numel() > 0):
                node_pred = model.decode_nodes(h, node_ids)
                loss = loss + float(args.node_loss_weight) * F.binary_cross_entropy_with_logits(node_pred, node_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            abs_err = weighted_abs_err_sum(pred, target, fmo_edge_weights)
            weight_sum = float(fmo_edge_weights.sum().item())
            total_loss_weighted += loss.item() * weight_sum
            total_pairs += weight_sum
            if component_abs_sum is None:
                component_abs_sum = abs_err
            else:
                component_abs_sum += abs_err
        scheduler.step()
        epoch_elapsed = time.perf_counter() - epoch_start_time
        total_elapsed = time.perf_counter() - total_start_time
        train_loss = total_loss_weighted / max(total_pairs, 1)
        train_component_mae = component_abs_sum / max(total_pairs, 1)
        do_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        if do_eval:
            valid_loss, valid_component_mae = evaluate_loader(model=model, loader=valid_loader, target_idx=target_idx, fmo_mask_idx=fmo_mask_idx, geom_idx_t=geom_idx_t, mode=effective_mode, ablation=effective_ablation, ablation_indices=ablation_indices, mask_ratio=args.eval_mask_ratio, loss_fn=loss_fn, device=device, seed=args.eval_seed, mask_sampling=args.mask_sampling, ligand_mask_frac=args.ligand_mask_frac, ligand_edge_loss_weight=args.ligand_edge_loss_weight, fmo_ligand_edge_loss_weight=args.fmo_ligand_edge_loss_weight)
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_epoch = epoch
                ckpt = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch, 'train_loss': train_loss, 'valid_loss': valid_loss, 'args': {**vars(args), 'effective_mode': effective_mode, 'effective_ablation': effective_ablation}, 'edge_attr_names': edge_attr_names, 'geom_idx': geom_idx, 'target_idx': target_idx, 'fmo_mask_idx': fmo_mask_idx, 'masking_mode': args.masking_mode, 'split_path': str(split_path)}
                torch.save(ckpt, out_dir / 'best_model.pt')
            if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                train_mae_str = ', '.join((f"{edge_attr_names[target_idx[i]].replace('_z', '')}={train_component_mae[i].item():.4f}" for i in range(len(target_idx))))
                valid_mae_str = ', '.join((f"{edge_attr_names[target_idx[i]].replace('_z', '')}={valid_component_mae[i].item():.4f}" for i in range(len(target_idx))))
                print(f'epoch {epoch:05d} | train {train_loss:.6f} | valid {valid_loss:.6f} | best_valid {best_valid:.6f}@{best_epoch} | train_pairs {total_pairs} | epoch_time {format_seconds(epoch_elapsed)} | total_time {format_seconds(total_elapsed)}')
                print(f'  train MAE(z): {train_mae_str}')
                print(f'  valid MAE(z): {valid_mae_str}')
    best_path = out_dir / 'best_model.pt'
    if not best_path.exists():
        raise RuntimeError('No best checkpoint was saved.')
    ckpt = load_pt(best_path, device)
    model.load_state_dict(ckpt['model_state_dict'])
    test_loss, test_component_mae = evaluate_loader(model=model, loader=test_loader, target_idx=target_idx, fmo_mask_idx=fmo_mask_idx, geom_idx_t=geom_idx_t, mode=effective_mode, ablation=effective_ablation, ablation_indices=ablation_indices, mask_ratio=args.eval_mask_ratio, loss_fn=loss_fn, device=device, seed=args.eval_seed + 10000, mask_sampling=args.mask_sampling, ligand_mask_frac=args.ligand_mask_frac, ligand_edge_loss_weight=args.ligand_edge_loss_weight)
    test_mae_str = ', '.join((f"{edge_attr_names[target_idx[i]].replace('_z', '')}={test_component_mae[i].item():.4f}" for i in range(len(target_idx))))
    total_elapsed = time.perf_counter() - total_start_time
    summary = {'best_epoch': best_epoch, 'best_valid_loss': float(best_valid), 'test_loss': float(test_loss), 'total_training_time_sec': float(total_elapsed), 'test_component_mae': {edge_attr_names[target_idx[i]].replace('_z', ''): float(test_component_mae[i].item()) for i in range(len(target_idx))}, 'split_path': str(split_path), 'mode': args.mode, 'effective_mode': effective_mode, 'ablation': effective_ablation, 'decoder_geometry_source': 'ablated_edge_attr_input', 'decoder_head': args.decoder_head, 'component_head_hidden': args.component_head_hidden, 'mask_sampling': args.mask_sampling, 'ligand_mask_frac': args.ligand_mask_frac, 'ligand_edge_loss_weight': args.ligand_edge_loss_weight, 'fmo_ligand_edge_loss_weight': args.fmo_ligand_edge_loss_weight, 'fmo_loss_weight': args.fmo_loss_weight, 'dist_loss_weight': args.dist_loss_weight, 'masking_mode': args.masking_mode, 'node_mask_task': args.node_mask_task, 'node_loss_weight': args.node_loss_weight, 'distance_target_idx': distance_target_idx, 'ligand_node_idx': ligand_node_idx, 'num_graphs_total_after_filter': len(paths), 'graph_filter_audit_csv': str(graph_filter_audit_csv), 'num_graphs_train': len(train_paths), 'num_graphs_valid': len(valid_paths), 'num_graphs_test': len(test_paths), 'pair_ligand_audit': pair_ligand_audit}
    torch.save(summary, out_dir / 'test_summary.pt')
    print('Final evaluation from best checkpoint')
    print(f'Best epoch: {best_epoch}')
    print(f'Best valid loss: {best_valid:.6f}')
    print(f'Test loss: {test_loss:.6f}')
    print(f'Test MAE(z): {test_mae_str}')
    print(f'Total training/eval time: {format_seconds(total_elapsed)}')
    print(f'Saved best checkpoint: {best_path}')
    print(f"Saved test summary: {out_dir / 'test_summary.pt'}")

def main():
    ap = argparse.ArgumentParser(description='Multi-graph masked edge pretraining for FMO/PIEDA graphs.')
    ap.add_argument('--graph-dir', required=True, help='Directory containing graph .pt files')
    ap.add_argument('--pattern', default='*.pt', help='Glob pattern inside graph-dir')
    ap.add_argument('--out-dir', default='pretrain_multigraph_out', help='Output directory')
    ap.add_argument('--include-pdbids', default=None, help='Optional txt/csv PDBID list. If supplied, only these PDBIDs are used.')
    ap.add_argument('--exclude-pdbids', default=None, help='Optional txt/csv PDBID list to exclude.')
    ap.add_argument('--strict-filter-csv', default=None, help='Optional strict graph filter CSV from audit_extract_strict_exclusions.py. Only rows with keep_strict=True are used.')
    ap.add_argument('--require-single-ligand', action='store_true', help='Runtime safety check: keep only graphs with exactly one ligand node.')
    ap.add_argument('--require-ligand-atom-graph', action='store_true', help='Runtime safety check: keep only graphs with a non-empty bound ligand atom graph.')
    ap.add_argument('--allowed-graph-versions', default=None, help='Optional comma-separated graph_version allowlist. Usually leave unset to allow mixed v5/v6 graphs after 4d0x robust-SMILES repair.')
    ap.add_argument('--mode', choices=['b_context', 'geometry_only'], default='b_context', help='Masked-edge setting. Use b_context for the recommended setting. geometry_only is kept as a legacy alias for --ablation no_fmo_context.')
    ap.add_argument('--ablation', choices=['none', 'no_fmo_context', 'no_distance', 'no_esm', 'no_distance_no_esm'], default='none', help='Input ablation. none keeps the default b_context behavior. no_fmo_context removes all FMO/PIEDA input edge features. no_distance removes FMO context and distance features. no_esm removes FMO context and ESM node features. no_distance_no_esm removes FMO context, distance features, and ESM features.')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--log-every', type=int, default=10)
    ap.add_argument('--eval-every', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--preload', action='store_true', help='Load all graphs in the selected split into CPU RAM at Dataset initialization to reduce torch.load I/O overhead.')
    ap.add_argument('--pin-memory', action='store_true', help='Use DataLoader pin_memory=True. Often useful when using CUDA.')
    ap.add_argument('--persistent-workers', action='store_true', help='Keep DataLoader workers alive across epochs. Only active when num_workers > 0.')
    ap.add_argument('--no-pair-cache', action='store_true', help='Disable cached undirected-pair information and use the original slower per-batch pair construction.')
    ap.add_argument('--hidden-dim', type=int, default=256)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--decoder-head', choices=['shared', 'hybrid', 'linear', 'linear_nogeom', 'hybrid_nogeom'], default='shared', help='Prediction head architecture. shared reproduces the legacy single multi-output decoder. hybrid uses a shared decoder trunk followed by one component-specific head per target. hybrid_nogeom is hybrid without edge_geom in the decoder input (decoder capacity without direct geometry access).')
    ap.add_argument('--component-head-hidden', type=int, default=0, help='Hidden dimension for each component-specific head when --decoder-head hybrid. 0 uses a single Linear(latent, 1) head per component.')
    ap.add_argument('--mask-ratio', type=float, default=0.15)
    ap.add_argument('--eval-mask-ratio', type=float, default=0.15)
    ap.add_argument('--mask-sampling', choices=['uniform', 'ligand_biased', 'ligand_only'], default='uniform', help='How to sample masked edge pairs. uniform samples all pairs uniformly. ligand_biased samples a fraction from ligand-protein edges. ligand_only samples only ligand-protein edges when available.')
    ap.add_argument('--ligand-mask-frac', type=float, default=0.7, help='For ligand_biased sampling, target fraction of masked pairs from ligand-protein edges.')
    ap.add_argument('--ligand-edge-loss-weight', type=float, default=1.0, help='Loss weight for decoded ligand-protein edges in the DISTANCE loss. 1.0 disables weighting.')
    ap.add_argument('--fmo-ligand-edge-loss-weight', type=float, default=1.0, help='Loss weight for decoded ligand-protein edges in the FMO/PIEDA loss. 1.0 disables weighting. Use e.g. 20.0 to upweight PIEDA on ligand edges without affecting the distance loss.')
    ap.add_argument('--fmo-loss-weight', type=float, default=1.0, help='Weight for masked FMO/PIEDA edge prediction loss. Set 0 for no-FMO geometry/node pretraining.')
    ap.add_argument('--dist-loss-weight', type=float, default=0.0, help='Weight for masked distance/RBF edge reconstruction loss.')
    ap.add_argument('--dist-huber-beta', type=float, default=0.2, help='SmoothL1 beta for distance/RBF reconstruction.')
    ap.add_argument('--fmo-target-mode', choices=['real', 'graph_shuffle', 'distbin_shuffle'], default='real', help='FMO/PIEDA target control for decoded edges.')
    ap.add_argument('--local-fmo-blackout', action='store_true', help='Zero FMO/PIEDA input on all edges incident to decoded edge endpoints.')
    ap.add_argument('--masking-mode', choices=['full', 'pieda_only'], default='full', help='full (default): v17ctx behavior — mask+adjacent edges zeroed on all dims, dist_z added (edge_dim=74). pieda_only: zero only PIEDA+q dims (66-71) for mask+adjacent edges, dist_z not added (edge_dim=72). Requires --dist-loss-weight 0.0 (default). Use with --local-fmo-blackout.')
    ap.add_argument('--joint-mask-distance-directions', action='store_true', default=True, help='Mask distance/RBF features on both directed edges of each sampled pair. Disable with --no-joint-mask-distance-directions.')
    ap.add_argument('--no-joint-mask-distance-directions', dest='joint_mask_distance_directions', action='store_false')
    ap.add_argument('--node-mask-task', choices=['none', 'ligand_block'], default='none', help='Optional node mask task. ligand_block masks x[:, ligand_node_start:ligand_node_end] on ligand nodes.')
    ap.add_argument('--node-loss-weight', type=float, default=0.0, help='Weight for ligand node sparse block BCE reconstruction.')
    ap.add_argument('--node-mask-ratio', type=float, default=1.0, help='Fraction of ligand nodes to mask for the node task.')
    ap.add_argument('--ligand-node-start', type=int, default=514, help='Start index of ligand sparse node block.')
    ap.add_argument('--ligand-node-end', type=int, default=1026, help='End index of ligand sparse node block.')
    ap.add_argument('--valid-ratio', type=float, default=0.1)
    ap.add_argument('--test-ratio', type=float, default=0.1)
    ap.add_argument('--split-seed', type=int, default=42)
    ap.add_argument('--split-path', default=None, help='Optional graph split .pt path to reuse')
    ap.add_argument('--overwrite-split', action='store_true')
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--weight-decay', type=float, default=0.0001)
    ap.add_argument('--huber-beta', type=float, default=1.0)
    ap.add_argument('--grad-clip', type=float, default=5.0)
    ap.add_argument('--device', default='auto', help='auto, cpu, cuda, cuda:0, etc.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--eval-seed', type=int, default=12345)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed % (2 ** 32 - 1))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train(args)
if __name__ == '__main__':
    main()
