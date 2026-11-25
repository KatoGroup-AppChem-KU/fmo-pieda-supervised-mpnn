"""Apply agreed structural target keys; inventory ligand repeats without merging."""
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
assert True
root = Path(__import__('os').environ['FMO_WORKDIR'])
source = root / 'results/affinity_duplicate_inventory_20261105/20261105_全309構造_SMILESとID一覧.csv'
assignments = root / 'outputs/20261105_複数UniProt42構造_評価用割当案.csv'
out = root / 'results/ligand_duplicates_after_assignment_20261105'
out.mkdir(exist_ok=False)

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write(name, rows):
    with (out / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
data = read(source)
assignment_rows = read(assignments)
assignment = {r['PDB ID']: r for r in assignment_rows}
assert len(data) == len({r['pdbid'] for r in data}) == 309
assert len(assignment) == 42
groups = defaultdict(list)
for row in data:
    pdb = row['pdbid']
    ids = row['uniprot'].split(';')
    if pdb in assignment:
        key = assignment[pdb]['提案する構造側評価キー']
        assert assignment[pdb]['既存UniProt IDs'] == row['uniprot']
    else:
        assert len(ids) == 1
        key = ids[0]
    row['evaluation_target_key'] = key
    row['evaluation_target_kind'] = 'complex' if ';' in key else 'single_UniProt'
    assert row['canonical_smiles']
    groups[key, row['canonical_smiles']].append(row)
repeated = []
for (key, smiles), members in sorted(groups.items()):
    if len(members) == 1:
        continue
    members.sort(key=lambda r: r['pdbid'])
    ys = [float(r['target_px']) for r in members]
    different = max(ys) - min(ys) > 1e-10
    repeated.append({'評価標的キー': key, 'canonicalSMILES': smiles, '構造数': len(members), 'PDB ID一覧': ';'.join((r['pdbid'] for r in members)), '実測pX一覧（PDB順）': ';'.join((r['target_px'] for r in members)), '実測pX範囲': max(ys) - min(ys), '実測pXが異なる': different, '活性指標': ';'.join(sorted({r['affinity_type'] for r in members})), 'GEO予測一覧': ';'.join((r['GEO'] for r in members)), 'v1予測一覧': ';'.join((r['GEO+PIEDA'] for r in members)), 'v2予測一覧': ';'.join((r['GEO+PIEDA_v2'] for r in members)), '扱い': '測定対象・条件の確認まで統合保留' if different else '同一リガンドの反復構造。予測平均の候補・未実施'})
write('20261105_UniProt整理後_全309構造_評価キー.csv', data)
write('20261105_UniProt整理後_同一リガンド複数構造一覧.csv', repeated)
summary = {'n_pdb': len(data), 'n_target_ligand_keys': len(groups), 'n_repeated_groups': len(repeated), 'n_pdb_in_repeated_groups': sum((r['構造数'] for r in repeated)), 'n_same_px_groups': sum((not r['実測pXが異なる'] for r in repeated)), 'n_different_px_groups': sum((r['実測pXが異なる'] for r in repeated)), 'n_unique_target_keys': len({r['evaluation_target_key'] for r in data}), 'n_complex_pdb': sum((r['evaluation_target_kind'] == 'complex' for r in data))}
(out / 'audit.json').write_text(json.dumps({'summary': summary, 'assignment_policy': 'User approved 20261105; structural grouping, not assay verification', 'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, assignments]}, 'no_averaging': True, 'no_exclusions': True, 'no_predictions_changed': True, 'no_delta_px_pairs': True}, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2))
for row in repeated:
    print(row['評価標的キー'], row['PDB ID一覧'], row['実測pXが異なる'])
