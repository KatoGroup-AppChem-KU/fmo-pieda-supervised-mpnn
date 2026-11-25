from __future__ import annotations
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
STANDARD_AA3 = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'SEC', 'PYL'}
AA3_TO_1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V', 'SEC': 'U', 'PYL': 'O'}
DEFAULT_WATER = {'HOH', 'WAT', 'H2O', 'DOD'}
HEAVY_ELEMENTS = {'B', 'C', 'N', 'O', 'F', 'P', 'S', 'CL', 'BR', 'I', 'NA', 'MG', 'K', 'CA', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN', 'SE', 'SI', 'AL', 'CD', 'HG'}

def norm_pdbid(x: Any) -> str:
    if x is None:
        return ''
    s = str(x).strip().lower()
    return '' if s in {'', 'nan', 'none', 'null'} else s

def norm_resname(x: Any) -> str:
    if x is None:
        return ''
    s = str(x).strip().upper()
    return '' if s in {'', 'NAN', 'NONE', 'NULL'} else s

def norm_chain(x: Any) -> str:
    if x is None:
        return '_'
    s = str(x).strip()
    return s if s else '_'

def is_heavy_element(elem: str) -> bool:
    e = str(elem).strip().upper()
    return e != '' and e != 'H'

def residue_id(chain: str, resname: str, resseq: Any, icode: str='') -> str:
    chain = norm_chain(chain)
    resname = norm_resname(resname)
    resseq_s = str(resseq).strip()
    icode_s = str(icode).strip()
    return f'{chain}:{resname}:{resseq_s}{icode_s}'

def read_text_lines(path: Path) -> list[str]:
    with open(path, 'r', errors='replace') as f:
        return f.read().splitlines()

def find_line(lines: list[str], pattern: str, start: int=0) -> int | None:
    for i in range(start, len(lines)):
        if pattern in lines[i]:
            return i
    return None

def parse_ligand_charge_resnames(lines: list[str]) -> set[str]:
    ligs: set[str] = set()
    for line in lines:
        if 'LigandCharge' not in line or '=' not in line:
            continue
        rhs = line.split('=', 1)[1].strip()
        if not rhs:
            continue
        for part in re.split('[,;\\s]+', rhs):
            part = part.strip()
            if not part:
                continue
            token = re.split('[:=]', part)[0].strip()
            token = norm_resname(token)
            if token:
                ligs.add(token)
    return ligs

def read_ligand_resnames(ligand_table: str | None, pdbid: str | None, pdbid_col: str, ligand_resname_col: str, explicit_ligand_resname: str | None, water_resnames: set[str], log_ligands: set[str], include_log_ligand_charge: bool=False) -> set[str]:
    ligs: set[str] = set(log_ligands) if include_log_ligand_charge else set()
    if explicit_ligand_resname:
        for x in str(explicit_ligand_resname).replace(';', ',').split(','):
            r = norm_resname(x)
            if r:
                ligs.add(r)
    if ligand_table:
        lt = Path(ligand_table)
        if lt.exists():
            df = pd.read_csv(lt)
            if pdbid_col not in df.columns:
                raise ValueError(f'ligand table missing pdbid column {pdbid_col}: {list(df.columns)}')
            if ligand_resname_col not in df.columns:
                raise ValueError(f'ligand table missing ligand resname column {ligand_resname_col}: {list(df.columns)}')
            tmp = df.copy()
            tmp['_pdbid_norm'] = tmp[pdbid_col].map(norm_pdbid)
            tmp['_hetid_norm'] = tmp[ligand_resname_col].map(norm_resname)
            if pdbid:
                tmp = tmp[tmp['_pdbid_norm'] == norm_pdbid(pdbid)]
            for r in tmp['_hetid_norm'].dropna().astype(str):
                r = norm_resname(r)
                if r:
                    ligs.add(r)
    return {x for x in ligs if x and x not in water_resnames}

def parse_read_molecular_structure(lines: list[str]) -> tuple[dict[int, dict[str, Any]], tuple[int | None, int | None]]:
    """Parse '## READ MOLECULAR STRUCTURE FROM PDB-FILE' atom table.

    Expected line format:
        3367   N     1      -24.665000   23.197000   50.604000           1

    We use the first integer as FMO internal atom index, the second field as element/atom symbol,
    and the three floats as coordinates.
    """
    start = find_line(lines, '## READ MOLECULAR STRUCTURE FROM PDB-FILE')
    if start is None:
        raise RuntimeError("Could not find '## READ MOLECULAR STRUCTURE FROM PDB-FILE' section in log")
    atoms: dict[int, dict[str, Any]] = {}
    first_line = None
    last_line = None
    in_table = False
    atom_re = re.compile('^\\s*(\\d+)\\s+([A-Za-z][A-Za-z0-9+-]*)\\s+(-?\\d+)\\s+([-+]?\\d+(?:\\.\\d*)?(?:[Ee][-+]?\\d+)?)\\s+([-+]?\\d+(?:\\.\\d*)?(?:[Ee][-+]?\\d+)?)\\s+([-+]?\\d+(?:\\.\\d*)?(?:[Ee][-+]?\\d+)?)')
    for i in range(start + 1, len(lines)):
        line = lines[i]
        m = atom_re.match(line)
        if m:
            idx = int(m.group(1))
            sym_raw = m.group(2).strip()
            elem = re.sub('[^A-Za-z]', '', sym_raw).upper()
            if len(elem) > 1 and elem not in {'CL', 'BR', 'NA', 'MG', 'CA', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN', 'SE', 'SI', 'AL', 'CD', 'HG'}:
                elem = elem[0]
            x, y, z = (float(m.group(4)), float(m.group(5)), float(m.group(6)))
            atoms[idx] = {'atom_index': idx, 'symbol': sym_raw, 'element': elem, 'x': x, 'y': y, 'z': z, 'is_heavy': is_heavy_element(elem), 'source_line': i + 1}
            in_table = True
            first_line = first_line or i + 1
            last_line = i + 1
            continue
        if in_table:
            if not line.strip():
                break
            if not atom_re.match(line):
                break
    if not atoms:
        raise RuntimeError('No atoms parsed from READ MOLECULAR STRUCTURE section')
    return (atoms, (first_line, last_line))

def parse_automatic_fragmentation(lines: list[str]) -> tuple[dict[int, dict[str, Any]], tuple[int | None, int | None]]:
    """Parse '## AUTOMATIC FRAGMENTATION' summary table.

    Example:
      A 105    209    SAM           T        T      1
      B 104    208    TYR           F        T     -1 Uncharged Polar
    """
    start = find_line(lines, '## AUTOMATIC FRAGMENTATION')
    if start is None:
        raise RuntimeError("Could not find '## AUTOMATIC FRAGMENTATION' section in log")
    frags: dict[int, dict[str, Any]] = {}
    first_line = None
    last_line = None
    frag_re_spaced = re.compile('^\\s*(\\S+)\\s+(-?\\d+[A-Za-z]?)\\s+(\\d+)\\s+([A-Za-z0-9]+)(?:\\s+(\\d+))?\\s+(\\S+)\\s+(\\S+)\\s+(-?\\d+)')
    frag_re_concat = re.compile('^\\s*([A-Za-z])(-?\\d+[A-Za-z]?)\\s+(\\d+)\\s+([A-Za-z0-9]+)(?:\\s+(\\d+))?\\s+(\\S+)\\s+(\\S+)\\s+(-?\\d+)')
    in_table = False
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if 'The system has no ion' in line or 'CHARGE OF MOLECULE' in line or '## CHECK FRAGMENT' in line:
            if in_table:
                break
        m = frag_re_spaced.match(line)
        if m:
            chain, resseq, frag_id_s, resname, ss_index, flag1, flag2, charge_s = m.groups()
        else:
            m = frag_re_concat.match(line)
            if m:
                chain, resseq, frag_id_s, resname, ss_index, flag1, flag2, charge_s = m.groups()
            else:
                if in_table and (not line.strip()):
                    break
                continue
        frag_id = int(frag_id_s)
        resname_n = norm_resname(resname)
        frags[frag_id] = {'frag_id': frag_id, 'chain': norm_chain(chain), 'resseq': str(resseq).strip(), 'icode': '', 'resname': resname_n, 'residue_id': residue_id(chain, resname_n, resseq), 'auto_frag_ss_index': ss_index or '', 'auto_frag_flag1': flag1, 'auto_frag_flag2': flag2, 'charge': int(charge_s), 'source_line': i + 1, 'raw_line': line.rstrip()}
        in_table = True
        first_line = first_line or i + 1
        last_line = i + 1
    if not frags:
        raise RuntimeError('No fragments parsed from AUTOMATIC FRAGMENTATION section')
    return (frags, (first_line, last_line))

def parse_number_of_fragments_atom_lists(lines: list[str]) -> tuple[dict[int, dict[str, Any]], int, tuple[int | None, int | None]]:
    """Parse 'NUMBER OF FRAGMENTS' atom membership table.

    Example:
       Frag.   Elec.   ATOM
           1      24       1     2 ...
                          23    24 ...
    """
    start = find_line(lines, 'NUMBER OF FRAGMENTS')
    if start is None:
        raise RuntimeError("Could not find 'NUMBER OF FRAGMENTS' section in log")
    m_nf = re.search('NUMBER OF FRAGMENTS\\s*=\\s*(\\d+)', lines[start])
    n_frag = int(m_nf.group(1)) if m_nf else None
    table: dict[int, dict[str, Any]] = {}
    current_frag: int | None = None
    first_line = None
    last_line = None
    new_re = re.compile('^\\s*(\\d+)\\s+(\\d+)\\s*(.*)$')
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if not line.strip():
            if current_frag is not None and len(table) >= (n_frag or 10 ** 9):
                break
            continue
        if current_frag is not None and len(table) >= (n_frag or 10 ** 9):
            break
        indent = len(line) - len(line.lstrip(' '))
        m = new_re.match(line)
        if m and indent < 20:
            frag_id = int(m.group(1))
            elec = int(m.group(2))
            rest = m.group(3)
            atoms = [int(x) for x in re.findall('\\d+', rest)]
            current_frag = frag_id
            table.setdefault(frag_id, {'frag_id': frag_id, 'elec': elec, 'atom_indices': [], 'source_line': i + 1})
            table[frag_id]['atom_indices'].extend(atoms)
            first_line = first_line or i + 1
            last_line = i + 1
            continue
        if current_frag is not None and indent >= 20:
            nums = [int(x) for x in re.findall('\\d+', line)]
            if nums:
                table[current_frag]['atom_indices'].extend(nums)
                last_line = i + 1
                continue
        if table and ('##' in line or '===' in line):
            break
    if n_frag is None:
        n_frag = len(table)
    if not table:
        raise RuntimeError('No fragment atom lists parsed after NUMBER OF FRAGMENTS')
    return (table, n_frag, (first_line, last_line))

def classify_fragment(resname: str, ligand_resnames: set[str], water_resnames: set[str]) -> str:
    """Classify an FMO fragment.

    v11 policy:
      1. Water is always water.
      2. Standard amino-acid residue names are always standard_aa.
      3. Only non-water, non-standard-AA names listed in the ligand table are ligand.
      4. Everything else is other_hetero.

    This prevents ligand-table hetids such as GLU from turning ordinary protein GLU
    residues into ligand nodes.
    """
    r = norm_resname(resname)
    if r in water_resnames:
        return 'water'
    if r in STANDARD_AA3:
        return 'standard_aa'
    if r in ligand_resnames:
        return 'ligand'
    return 'other_hetero'

def compute_geometry(atom_indices: list[int], atoms: dict[int, dict[str, Any]]) -> dict[str, Any]:
    present = [idx for idx in atom_indices if idx in atoms]
    missing = [idx for idx in atom_indices if idx not in atoms]
    all_coords = np.array([[atoms[i]['x'], atoms[i]['y'], atoms[i]['z']] for i in present], dtype=float) if present else np.zeros((0, 3), dtype=float)
    heavy = [i for i in present if atoms[i]['is_heavy']]
    heavy_coords = np.array([[atoms[i]['x'], atoms[i]['y'], atoms[i]['z']] for i in heavy], dtype=float) if heavy else all_coords
    if len(heavy_coords):
        centroid = heavy_coords.mean(axis=0)
        radius = float(np.linalg.norm(heavy_coords - centroid[None, :], axis=1).max()) if len(heavy_coords) else 0.0
    elif len(all_coords):
        centroid = all_coords.mean(axis=0)
        radius = float(np.linalg.norm(all_coords - centroid[None, :], axis=1).max()) if len(all_coords) else 0.0
    else:
        centroid = np.array([np.nan, np.nan, np.nan], dtype=float)
        radius = float('nan')
    elements = sorted(set((atoms[i]['element'] for i in present)))
    symbols = sorted(set((atoms[i]['symbol'] for i in present)))
    return {'present_atom_indices': present, 'missing_atom_indices': missing, 'n_atoms_all': int(len(present)), 'n_atoms_heavy': int(len(heavy)), 'centroid_x': float(centroid[0]), 'centroid_y': float(centroid[1]), 'centroid_z': float(centroid[2]), 'heavy_atom_radius': radius, 'elements': elements, 'symbols': symbols}

def build_nodes(log_file: Path, out_csv: Path, audit_json: Path | None, ligand_table: str | None, pdbid: str | None, pdbid_col: str, ligand_resname_col: str, ligand_resname: str | None, water_resnames: set[str], include_log_ligand_charge: bool=False) -> dict[str, Any]:
    lines = read_text_lines(log_file)
    log_ligands = parse_ligand_charge_resnames(lines)
    ligand_resnames = read_ligand_resnames(ligand_table=ligand_table, pdbid=pdbid, pdbid_col=pdbid_col, ligand_resname_col=ligand_resname_col, explicit_ligand_resname=ligand_resname, water_resnames=water_resnames, log_ligands=log_ligands, include_log_ligand_charge=include_log_ligand_charge)
    atoms, readgeom_range = parse_read_molecular_structure(lines)
    frag_summary, autofrag_range = parse_automatic_fragmentation(lines)
    frag_atoms, n_frag_reported, fraglist_range = parse_number_of_fragments_atom_lists(lines)
    max_frag_id = max(set(frag_summary) | set(frag_atoms))
    rows = []
    missing_frag_summary = []
    missing_frag_atoms = []
    all_missing_atom_indices = []
    esm_counter = 0
    for frag_id in range(1, max_frag_id + 1):
        fs = frag_summary.get(frag_id)
        fa = frag_atoms.get(frag_id)
        if fs is None:
            missing_frag_summary.append(frag_id)
            fs = {'frag_id': frag_id, 'chain': '_', 'resseq': str(frag_id), 'icode': '', 'resname': 'UNK', 'residue_id': residue_id('_', 'UNK', frag_id), 'charge': 0, 'auto_frag_flag1': '', 'auto_frag_flag2': '', 'source_line': None, 'raw_line': ''}
        if fa is None:
            missing_frag_atoms.append(frag_id)
            fa = {'atom_indices': [], 'elec': None, 'source_line': None}
        resname = norm_resname(fs['resname'])
        kind = classify_fragment(resname, ligand_resnames, water_resnames)
        is_standard = int(kind == 'standard_aa')
        is_ligand = int(kind == 'ligand')
        is_water = int(kind == 'water')
        is_other = int(kind == 'other_hetero')
        is_hetero = int(kind in {'ligand', 'water', 'other_hetero'})
        geom = compute_geometry(fa.get('atom_indices', []), atoms)
        all_missing_atom_indices.extend(geom['missing_atom_indices'])
        if kind == 'standard_aa':
            esm_index = esm_counter
            esm_counter += 1
        else:
            esm_index = -1
        rid = fs['residue_id']
        residue_atom_counts = {rid: geom['n_atoms_all']}
        ligand_resnames_in_fragment = resname if is_ligand else ''
        water_resnames_in_fragment = resname if is_water else ''
        other_hetero_resnames_in_fragment = resname if is_other else ''
        row = {'frag_id': frag_id, 'rep_chain': fs['chain'], 'rep_resseq': fs['resseq'], 'rep_icode': fs.get('icode', ''), 'rep_resname': resname, 'rep_aa1': AA3_TO_1.get(resname, ''), 'rep_residue_id': rid, 'esm_index': esm_index, 'is_standard_aa': is_standard, 'is_hetero': is_hetero, 'is_ligand': is_ligand, 'is_water': is_water, 'is_other_hetero': is_other, 'fragment_kind': kind, 'n_atoms_all': geom['n_atoms_all'], 'n_atoms_heavy': geom['n_atoms_heavy'], 'centroid_x': geom['centroid_x'], 'centroid_y': geom['centroid_y'], 'centroid_z': geom['centroid_z'], 'heavy_atom_radius': geom['heavy_atom_radius'], 'atom_serials_json': json.dumps(fa.get('atom_indices', []), ensure_ascii=False), 'atom_names_in_fragment': ';'.join(geom['symbols']), 'elements_in_fragment': ';'.join(geom['elements']), 'resnames_in_fragment': resname, 'residue_ids_in_fragment': rid, 'ligand_resnames_in_fragment': ligand_resnames_in_fragment, 'water_resnames_in_fragment': water_resnames_in_fragment, 'other_hetero_resnames_in_fragment': other_hetero_resnames_in_fragment, 'gnn_node_role': kind, 'n_residues_in_fragment': 1, 'residue_atom_counts_json': json.dumps(residue_atom_counts, ensure_ascii=False, sort_keys=True), 'rep_heavy_atoms': geom['n_atoms_heavy'], 'rep_has_ca': int(False), 'log_only_node_source': True, 'auto_frag_charge': fs.get('charge'), 'auto_frag_ss_index': fs.get('auto_frag_ss_index', ''), 'auto_frag_flag1': fs.get('auto_frag_flag1'), 'auto_frag_flag2': fs.get('auto_frag_flag2'), 'auto_frag_source_line': fs.get('source_line'), 'frag_atom_list_source_line': fa.get('source_line'), 'frag_electron_count': fa.get('elec'), 'n_missing_atom_indices': len(geom['missing_atom_indices']), 'missing_atom_indices_json': json.dumps(geom['missing_atom_indices'], ensure_ascii=False)}
        rows.append(row)
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    kind_counts = {str(k): int(v) for k, v in df['fragment_kind'].value_counts().to_dict().items()}
    audit = {'log_file': str(log_file), 'coord_source': str(log_file), 'node_source': 'fmo_log_only_v11', 'output': str(out_csv), 'n_atoms_read_molecular_structure': int(len(atoms)), 'read_molecular_structure_line_range': list(readgeom_range), 'automatic_fragmentation_line_range': list(autofrag_range), 'fragment_atom_list_line_range': list(fraglist_range), 'n_fragments_reported': int(n_frag_reported), 'n_fragments': int(len(df)), 'max_frag_id': int(max_frag_id), 'n_ligand_fragments': int(df['is_ligand'].sum()), 'n_water_fragments': int(df['is_water'].sum()), 'n_other_hetero_fragments': int(df['is_other_hetero'].sum()), 'n_standard_aa_fragments': int((df['fragment_kind'] == 'standard_aa').sum()), 'n_fragments_with_esm_index': int((df['esm_index'] >= 0).sum()), 'fragment_kind_counts': kind_counts, 'classification_rule': 'fmo_log_only_v11; automatic_fragmentation_resname_first; supports_spaced_and_concatenated_chain_resseq; supports_optional_disulfide_ss_index; water_and_standard_aa_prioritized_over_ligand_table; ligand_table_hetid_only_by_default; LigandCharge_saved_for_audit_only; water_resnames_excluded_from_ligand', 'include_log_ligand_charge_as_ligand': bool(include_log_ligand_charge), 'ligand_resnames': sorted(ligand_resnames), 'log_ligand_charge_resnames': sorted(log_ligands), 'water_resnames': sorted(water_resnames), 'n_missing_fragment_summary': int(len(missing_frag_summary)), 'missing_fragment_summary_first20': missing_frag_summary[:20], 'n_missing_fragment_atom_lists': int(len(missing_frag_atoms)), 'missing_fragment_atom_lists_first20': missing_frag_atoms[:20], 'n_missing_atom_indices_total': int(len(all_missing_atom_indices)), 'missing_atom_indices_first20': all_missing_atom_indices[:20], 'n_fragments_with_missing_atom_indices': int((df['n_missing_atom_indices'] > 0).sum()), 'fragments_with_missing_atom_indices_first20': {str(int(r.frag_id)): json.loads(r.missing_atom_indices_json) for r in df[df['n_missing_atom_indices'] > 0].head(20).itertuples(index=False)}, 'gnn_ready': True}
    if audit_json is not None:
        audit_json.parent.mkdir(parents=True, exist_ok=True)
        with open(audit_json, 'w', encoding='utf-8') as f:
            json.dump(audit, f, indent=2, ensure_ascii=False)
    return audit

def main():
    ap = argparse.ArgumentParser(description='Create FMO fragment node CSV using only ABINIT-MP/FMO log information. No PDB file is required. Uses READ MOLECULAR STRUCTURE for internal atom coordinates, AUTOMATIC FRAGMENTATION for fragment residue labels, and NUMBER OF FRAGMENTS for atom lists.')
    ap.add_argument('--log-file', required=True)
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--audit-json', default=None)
    ap.add_argument('--ligand-table', default=None)
    ap.add_argument('--pdbid', default=None)
    ap.add_argument('--pdbid-col', default='pdbid')
    ap.add_argument('--ligand-resname-col', '--hetid-col', dest='ligand_resname_col', default='hetid')
    ap.add_argument('--ligand-resname', default=None)
    ap.add_argument('--include-log-ligand-charge', action='store_true', help='Also treat resnames in the FMO log LigandCharge field as ligand. Default is OFF because LigandCharge can include cofactors/other hetero such as NAP.')
    ap.add_argument('--water-resnames', default='HOH,WAT,H2O,DOD')
    args = ap.parse_args()
    water_resnames = {norm_resname(x) for x in args.water_resnames.split(',') if norm_resname(x)}
    audit = build_nodes(log_file=Path(args.log_file), out_csv=Path(args.out_csv), audit_json=Path(args.audit_json) if args.audit_json else None, ligand_table=args.ligand_table, pdbid=args.pdbid, pdbid_col=args.pdbid_col, ligand_resname_col=args.ligand_resname_col, ligand_resname=args.ligand_resname, water_resnames=water_resnames, include_log_ligand_charge=args.include_log_ligand_charge)
    print(f"Nodes written: {audit['n_fragments']} -> {args.out_csv}")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
if __name__ == '__main__':
    main()
