#!/usr/bin/env python3
"""Materialize fixed manifests using user-supplied graph paths (stdlib only).

No tensor is loaded, transformed or certified by this metadata operation.
Graph fingerprints are retained for the scientific loader to check later.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def prepare(package, graph_root, output, check_files):
    if output.exists():
        raise FileExistsError('Use a new output directory: ' + str(output))
    with (package/'data/structures.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len({r['key'] for r in rows}) == 8983
    schema = json.loads((package/'data/schema.json').read_text())
    stats = json.loads((package/'data/normalization.json').read_text())
    payloads = {}
    for name, normalization in stats.items():
        field = 'final_split' if name.startswith('final/') else Path(name).stem
        records, pdb_splits = [], {}
        for row in rows:
            split = row[field]
            assert pdb_splits.setdefault(row['pdb'], split) == split
            paths = {v: str((graph_root/row[v+'_relative_path']).resolve()) for v in ('v6', 'v7')}
            if check_files:
                for path in paths.values():
                    if not Path(path).is_file():
                        raise FileNotFoundError(path)
            records.append({'key': row['key'], 'source': row['source'], 'pdb': row['pdb'],
                            'subset': row['subset'], 'split': split, 'edges': int(row['edges']),
                            'paths': paths, 'fingerprints': {v: row[v+'_fingerprint'] for v in ('v6', 'v7')}})
        train_keys = '\n'.join(r['key'] for r in records if r['split'] == 'train').encode()
        assert hashlib.sha256(train_keys).hexdigest() == normalization['normalization_train_keys_sha256']
        payloads[name] = {**schema, **normalization, 'records': records,
                          'metadata_only': True, 'graphs_validated': False}
    for name, payload in payloads.items():
        write_json(output/name, payload)
    for directory in ('final', 'cv'):
        folder = output/directory
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.json')}
        write_json(folder/'PREPARED.json', {'metadata_prepared': True, 'files_sha256': hashes,
                                          'graphs_validated': False})
    return {'manifests': len(payloads), 'structures': len(rows), 'graph_presence_checked': check_files}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph-root', type=Path, required=True, help='Parent of graphs/bdb and graphs/scop2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-files', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parent, args.graph_root, args.output, args.check_files)))
