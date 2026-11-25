"""Create FMO-fragment-level edge CSVs directly from ABINIT-MP logs.

This script does not reuse old v3 edge CSVs. It parses IFIE and PIEDA tables
from the ABINIT-MP log, then joins them with v4 FMO-fragment nodes created by
`create_fmo_fragment_nodes.py`.

Edge definition
---------------
- edge = FMO fragment pair `(frag_i, frag_j)` from IFIE/PIEDA
- DIST = ABINIT-MP IFIE distance
- centroid_distance = distance between CHECK-FRAGMENT-derived fragment centroids
- residue_i/residue_j = representative residue IDs from the node table
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
DATA_LINE = re.compile('^\\s*\\d+\\s+\\d+')

def parse_ifie_section(lines):
    records = []
    inside = False
    for line in lines:
        if re.search('^\\s*IJ-PAIR', line):
            inside = True
            continue
        if inside:
            if line.strip() == '':
                continue
            if line.startswith(' ##') and records:
                break
            if not DATA_LINE.match(line):
                continue
            cols = line.split()
            if len(cols) < 8:
                continue
            try:
                records.append({'frag_i': int(cols[0]), 'frag_j': int(cols[1]), 'DIST': float(cols[2]), 'DIMER_ES': cols[3], 'HF_IFIE': float(cols[4]), 'MP2_IFIE': float(cols[5])})
            except Exception:
                continue
    return pd.DataFrame(records)

def parse_pieda_section(lines):
    records = []
    in_pieda = False
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped == '## PIEDA':
            in_pieda = True
            in_table = False
            continue
        if in_pieda and 'IJ-PAIR' in line:
            in_table = True
            continue
        if in_pieda and in_table:
            if stripped == '':
                continue
            if line.startswith(' ##') and records:
                break
            if not DATA_LINE.match(line):
                continue
            cols = line.split()
            if len(cols) < 7:
                continue
            try:
                records.append({'frag_i': int(cols[0]), 'frag_j': int(cols[1]), 'ES': float(cols[2]), 'EX': float(cols[3]), 'CTmix': float(cols[4]), 'DI': float(cols[5]), 'q_I_to_J': float(cols[6]), 'has_pieda': 1})
            except Exception:
                continue
    return pd.DataFrame(records)

def add_centroid_distance(df: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    nodes = nodes.copy()
    nodes['frag_id'] = nodes['frag_id'].astype(int)
    lookup = nodes.set_index('frag_id')
    missing = sorted((set(df['frag_i'].astype(int)) | set(df['frag_j'].astype(int))) - set(lookup.index.astype(int)))
    if missing:
        raise RuntimeError(f'Edge references missing node frag_ids: n={len(missing)} first={missing[:10]}')
    ci = lookup.loc[df['frag_i'].astype(int), ['centroid_x', 'centroid_y', 'centroid_z']].to_numpy(float)
    cj = lookup.loc[df['frag_j'].astype(int), ['centroid_x', 'centroid_y', 'centroid_z']].to_numpy(float)
    df['centroid_distance'] = np.linalg.norm(ci - cj, axis=1)
    df['residue_i'] = lookup.loc[df['frag_i'].astype(int), 'rep_residue_id'].to_numpy()
    df['residue_j'] = lookup.loc[df['frag_j'].astype(int), 'rep_residue_id'].to_numpy()
    return df

def make_audit(df: pd.DataFrame, nodes: pd.DataFrame, log_file: Path, out_csv: Path) -> dict:
    c = df['centroid_distance'].to_numpy(float) if len(df) else np.array([])
    dist = df['DIST'].to_numpy(float) if len(df) else np.array([])
    audit = {'log_file': str(log_file), 'output': str(out_csv), 'n_nodes': int(len(nodes)), 'n_edges': int(len(df)), 'n_has_pieda': int(df.get('has_pieda', pd.Series(dtype=int)).fillna(0).sum()) if len(df) else 0, 'n_dimer_es_T': int((df['DIMER_ES'].astype(str) == 'T').sum()) if len(df) else 0, 'max_frag_id_edges': int(max(df['frag_i'].max(), df['frag_j'].max())) if len(df) else None, 'n_DIST_le6_centroid_gt20': int(((df['DIST'] <= 6.0) & (df['centroid_distance'] > 20.0)).sum()) if len(df) else 0, 'n_DIST_le6_centroid_gt30': int(((df['DIST'] <= 6.0) & (df['centroid_distance'] > 30.0)).sum()) if len(df) else 0, 'n_DIST_le8_centroid_gt30': int(((df['DIST'] <= 8.0) & (df['centroid_distance'] > 30.0)).sum()) if len(df) else 0}
    if len(df):
        audit['dist_centroid_corr'] = float(np.corrcoef(dist, c)[0, 1]) if len(df) > 2 and np.std(dist) > 0 and (np.std(c) > 0) else None
        audit['DIST_quantiles'] = {str(q): float(np.quantile(dist, q)) for q in [0, 0.5, 0.9, 0.95, 0.99, 1]}
        audit['centroid_distance_quantiles'] = {str(q): float(np.quantile(c, q)) for q in [0, 0.5, 0.9, 0.95, 0.99, 1]}
        bad = df[(df['DIST'] <= 6.0) & (df['centroid_distance'] > 30.0)].copy()
        audit['bad_examples'] = bad.head(10)[['frag_i', 'frag_j', 'residue_i', 'residue_j', 'DIST', 'centroid_distance']].to_dict('records')
    return audit

def build_one(log_file: Path, nodes_csv: Path, out_csv: Path, audit_json: Optional[Path], args) -> None:
    with open(log_file, 'r', errors='replace') as f:
        lines = f.readlines()
    ifie = parse_ifie_section(lines)
    if ifie.empty:
        raise RuntimeError(f'No IFIE pairs parsed from {log_file}')
    pieda = parse_pieda_section(lines)
    if pieda.empty:
        pieda = pd.DataFrame(columns=['frag_i', 'frag_j', 'ES', 'EX', 'CTmix', 'DI', 'q_I_to_J', 'has_pieda'])
    df = ifie.merge(pieda, on=['frag_i', 'frag_j'], how='left')
    for col in ['ES', 'EX', 'CTmix', 'DI', 'q_I_to_J']:
        if col not in df.columns:
            df[col] = np.nan
    df['has_pieda'] = df['has_pieda'].fillna(0).astype(int)
    if args.exclude_dimer_es:
        df = df[df['DIMER_ES'].astype(str) != 'T'].copy()
    if args.require_pieda:
        df = df[df['has_pieda'] == 1].copy()
    if args.exclude_neighbor_fragments:
        df = df[(df['frag_j'] - df['frag_i']).abs() > 1].copy()
    if args.dist_cutoff is not None:
        df = df[df['DIST'] <= args.dist_cutoff].copy()
    nodes = pd.read_csv(nodes_csv)
    df = add_centroid_distance(df, nodes)
    df['PIEDA_sum'] = df['ES'] + df['EX'] + df['CTmix'] + df['DI']
    cols = ['frag_i', 'frag_j', 'residue_i', 'residue_j', 'DIST', 'centroid_distance', 'DIMER_ES', 'HF_IFIE', 'MP2_IFIE', 'ES', 'EX', 'CTmix', 'DI', 'PIEDA_sum', 'q_I_to_J', 'has_pieda']
    df = df[cols].sort_values(['frag_i', 'frag_j']).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    audit = make_audit(df, nodes, log_file, out_csv)
    if audit_json:
        audit_json.parent.mkdir(parents=True, exist_ok=True)
        with open(audit_json, 'w') as f:
            json.dump(audit, f, indent=2)
    print(f'Edges written: {len(df)} -> {out_csv}')
    print(json.dumps(audit, indent=2))

def infer_pdbid_from_log(p: Path) -> str:
    name = p.name.lower()
    m = re.match('([0-9][a-z0-9]{3})', name)
    if m:
        return m.group(1)
    return p.stem.split('_')[0].lower()

def main() -> None:
    ap = argparse.ArgumentParser(description='Create FMO-fragment-level edge CSVs directly from ABINIT-MP logs.')
    one = ap.add_argument_group('single file')
    one.add_argument('--log-file')
    one.add_argument('--nodes')
    one.add_argument('--out-csv')
    one.add_argument('--audit-json')
    batch = ap.add_argument_group('batch')
    batch.add_argument('--log-dir')
    batch.add_argument('--nodes-dir')
    batch.add_argument('--out-dir')
    batch.add_argument('--audit-dir')
    ap.add_argument('--pattern', default='*.log')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--exclude-dimer-es', action='store_true')
    ap.add_argument('--require-pieda', action='store_true')
    ap.add_argument('--exclude-neighbor-fragments', action='store_true')
    ap.add_argument('--dist-cutoff', type=float, default=None)
    args = ap.parse_args()
    if args.log_file:
        if not (args.nodes and args.out_csv):
            raise SystemExit('--nodes and --out-csv are required with --log-file')
        build_one(Path(args.log_file), Path(args.nodes), Path(args.out_csv), Path(args.audit_json) if args.audit_json else None, args)
        return
    if not (args.log_dir and args.nodes_dir and args.out_dir):
        raise SystemExit('Use either --log-file/--nodes/--out-csv or --log-dir/--nodes-dir/--out-dir')
    logs = sorted(Path(args.log_dir).glob(args.pattern))
    print(f'Logs to process: {len(logs)}')
    for i, log_file in enumerate(logs, 1):
        pdbid = infer_pdbid_from_log(log_file)
        nodes_csv = Path(args.nodes_dir) / f'{pdbid}_nodes.csv'
        out_csv = Path(args.out_dir) / f'{pdbid}_edges.csv'
        audit_json = Path(args.audit_dir) / f'{pdbid}_edges_audit.json' if args.audit_dir else None
        if args.resume and out_csv.exists():
            print(f'[{i}/{len(logs)}] {pdbid} skipped')
            continue
        if not nodes_csv.exists():
            print(f'[{i}/{len(logs)}] {pdbid} missing nodes: {nodes_csv}')
            continue
        print(f'[{i}/{len(logs)}] {pdbid}')
        build_one(log_file, nodes_csv, out_csv, audit_json, args)
if __name__ == '__main__':
    main()
