"""
audit_fmo_pieda_v06_provisional_exclusion.py

2026-07-17のディスカッションで決まった暫定閾値による異常値フィルタリングの再検証。
既存のv05ルール（charge-aware ES・distance-contact・same-residue-split等の複合条件、
audit_fmo_pieda_v05_exclusion.py）とは別に、よりシンプルな単一閾値ベースのルールセット
で除外候補edgeを洗い出し、除外規模・v05との差分を比較する。

適用ルール（2026-07-17暫定案。policy/20260717_方針.md 参照）:
  ES:    |ES| >= 200
  EX:    EX <= -1  または EX >= +50
  CTmix: CTmix >= +1  または CTmix <= -50
  DI:    DI >= +1  または DI <= -50

v05との主な違い:
  - ESはcharge-aware判定をやめ、単一閾値(|ES|>=200)に単純化
  - EXに下限(EX<=-1)を新設。EX(交換反発)は本来正の値のはずで、負の値は物理的異常の疑い
  - CTmix/DIは正方向の閾値を大幅に厳格化(v05: +50/-100 -> 今回: +1/-50)

edges_v4_fmofrag CSV(frag_i, frag_j, residue_i, residue_j, DIST, ES, EX, CTmix, DI,
q_I_to_J等)のみから判定可能。nodes CSVのcharge情報等は使わない
(charge-aware判定自体を今回は行わないため)。

出力:
  {out_prefix}.exclusion_edges.csv     フラグの立ったedgeのみ(pdbid, frag_i, frag_j, ...)
  {out_prefix}.excluded_pdbids.txt     除外edgeを1件以上含むPDBID一覧
  {out_prefix}.summary.json            件数サマリ(フラグ種別ごとの合算ヒット数を含む)

Usage:
  python audit_fmo_pieda_v06_provisional_exclusion.py \\
      --edges-dir dataset/edges_v4_fmofrag \\
      --pdbid-list dataset/rebuild_graphs_v7_fmofrag_morganunified_logs/keep_pdbids.txt \\
      --out-prefix audits/fmo_pieda_v06_provisional_exclusion
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

def audit_one_pdb(pdbid: str, edges_csv: Path):
    edges = pd.read_csv(edges_csv)
    if edges.empty:
        return ({}, edges)
    ES, EX, CTmix, DI = (edges['ES'], edges['EX'], edges['CTmix'], edges['DI'])
    flag_masks = {'es_large': ES.abs() >= 200, 'ex_negative': EX <= -1, 'ex_positive_large': EX >= 50, 'ctmix_positive': CTmix >= 1, 'ctmix_negative_large': CTmix <= -50, 'di_positive': DI >= 1, 'di_negative_large': DI <= -50}
    for name, mask in flag_masks.items():
        edges[name] = mask.fillna(False)
    flag_cols = list(flag_masks.keys())
    any_flag = edges[flag_cols].any(axis=1)
    edges['pdbid'] = pdbid
    reasons = []
    for _, row in edges[flag_cols].iterrows():
        hits = [c for c in flag_cols if row[c]]
        reasons.append(';'.join(hits))
    edges['qc_reasons'] = reasons
    flag_counts = {name: int(edges[name].sum()) for name in flag_cols}
    excluded = edges[any_flag].copy()
    return (flag_counts, excluded)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges-dir', required=True)
    ap.add_argument('--pdbid-list', default=None, help='1行1PDBIDのテキストファイル。省略時はedges-dir内の全PDBを対象にする。')
    ap.add_argument('--out-prefix', required=True)
    args = ap.parse_args()
    edges_dir = Path(args.edges_dir)
    if args.pdbid_list:
        pdbids = [line.strip() for line in Path(args.pdbid_list).read_text(encoding='utf-8').splitlines() if line.strip()]
    else:
        pdbids = sorted((p.stem.replace('_edges', '') for p in edges_dir.glob('*_edges.csv')))
    print(f'Target PDBs: {len(pdbids)}')
    all_excluded = []
    n_total_edges = 0
    n_missing = 0
    flag_totals: dict[str, int] = {}
    for i, pdbid in enumerate(pdbids, start=1):
        edges_csv = edges_dir / f'{pdbid}_edges.csv'
        if not edges_csv.exists():
            n_missing += 1
            continue
        edges_raw = pd.read_csv(edges_csv, usecols=['frag_i'])
        n_total_edges += len(edges_raw)
        flag_counts, excluded = audit_one_pdb(pdbid, edges_csv)
        for name, cnt in flag_counts.items():
            flag_totals[name] = flag_totals.get(name, 0) + cnt
        if not excluded.empty:
            all_excluded.append(excluded)
        if i % 200 == 0 or i == len(pdbids):
            print(f'  {i}/{len(pdbids)} PDBs processed, excluded so far: {sum((len(x) for x in all_excluded))}')
    result = pd.concat(all_excluded, axis=0, ignore_index=True) if all_excluded else pd.DataFrame()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = ['pdbid', 'frag_i', 'frag_j', 'residue_i', 'residue_j', 'DIST', 'ES', 'EX', 'CTmix', 'DI', 'q_I_to_J', 'qc_reasons']
    if not result.empty:
        keep_cols = [c for c in keep_cols if c in result.columns]
        result = result[keep_cols].sort_values(['pdbid', 'frag_i', 'frag_j'])
    result.to_csv(f'{out_prefix}.exclusion_edges.csv', index=False)
    excluded_pdbids = sorted(result['pdbid'].unique().tolist()) if not result.empty else []
    Path(f'{out_prefix}.excluded_pdbids.txt').write_text('\n'.join(excluded_pdbids) + ('\n' if excluded_pdbids else ''), encoding='utf-8')
    summary = {'n_target_pdbs': len(pdbids), 'n_missing_pdbs': n_missing, 'n_total_edges_scanned': int(n_total_edges), 'n_excluded_edges': int(len(result)), 'n_excluded_pdbs': len(excluded_pdbids), 'excluded_edge_fraction': len(result) / n_total_edges if n_total_edges else 0.0, 'flag_hit_counts_total': flag_totals}
    with open(f'{out_prefix}.summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print('=' * 80)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Saved: {out_prefix}.exclusion_edges.csv')
    print(f'Saved: {out_prefix}.excluded_pdbids.txt')
    print(f'Saved: {out_prefix}.summary.json')
if __name__ == '__main__':
    main()
