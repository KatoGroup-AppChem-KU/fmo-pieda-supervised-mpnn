"""Create isolated Random-FP inputs; run only in Slurm aurum-g01.

Change standard-AA rows only: zero the AA one-hot block, replace the
512-bit Morgan block by a fixed AA-type code. Preserve all other data.
"""
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from prepare_geo_pieda_matched_20260911 import fingerprint, load
ROOT = Path(__import__('os').environ['FMO_WORKDIR'])
SOURCE = ROOT / 'dataset/fpv2_prepared_20261016'
DEST = ROOT / 'dataset/random_fp_20261030'
AA_ORDER = list('ACDEFGHIKLMNPQRSTVWYUOBZX')
STANDARD20 = set('ACDEFGHIKLMNPQRSTVWY')
CODE_SEED = 42
BITS = 512

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text() == payload, f'existing output differs: {path}'
    else:
        path.write_text(payload)

def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and (a.shape == b.shape) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b, equal_nan=True)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all((equal(a[k], b[k]) for k in a))
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all((equal(x, y) for x, y in zip(a, b)))
    return a == b

def transform(original, codes):
    names = list(original.node_feature_names)
    aa = [names.index('aa_' + symbol) for symbol in AA_ORDER]
    fp = [names.index('small_mol_morgan_' + str(i)) for i in range(BITS)]
    assert len(names) == 546 and len(set(aa + fp)) == 537
    mask = original.x[:, names.index('is_standard_aa')] > 0.5
    if hasattr(original, 'is_standard_aa'):
        assert torch.equal(mask, original.is_standard_aa.bool().reshape(-1))
    onehot = original.x[mask][:, aa]
    assert bool(((onehot == 0) | (onehot == 1)).all()) and bool((onehot.sum(1) == 1).all())
    identities = onehot.argmax(1)
    derived = original.clone()
    derived.x = original.x.clone()
    rows = torch.where(mask)[0]
    derived.x[rows[:, None], torch.tensor(aa)] = 0
    derived.x[rows[:, None], torch.tensor(fp)] = codes[identities].to(derived.x.dtype)
    validate(original, derived, codes)
    return (derived, Counter((AA_ORDER[i] for i in identities.tolist())))

def validate(original, derived, codes):
    names = list(original.node_feature_names)
    aa = [names.index('aa_' + symbol) for symbol in AA_ORDER]
    fp = [names.index('small_mol_morgan_' + str(i)) for i in range(BITS)]
    mask = original.x[:, names.index('is_standard_aa')] > 0.5
    identities = original.x[mask][:, aa].argmax(1)
    other = [i for i in range(len(names)) if i not in set(aa + fp)]
    assert original.x.shape == derived.x.shape and original.x.dtype == derived.x.dtype
    assert torch.isfinite(derived.x).all()
    assert torch.equal(original.x[~mask], derived.x[~mask]), 'hetero/ligand/water features changed'
    assert torch.equal(original.x[:, other], derived.x[:, other]), 'flags or size features changed'
    assert torch.count_nonzero(derived.x[mask][:, aa]) == 0
    assert torch.equal(derived.x[mask][:, fp], codes[identities].to(derived.x.dtype))
    before, after = (original.to_dict(), derived.to_dict())
    assert before.keys() == after.keys()
    assert all((equal(before[k], after[k]) for k in before if k != 'x')), 'non-x graph attribute changed'

def main():
    assert True
    assert Path.cwd() == ROOT
    torch.set_num_threads(4)
    inputs = [SOURCE / 'final/manifest.json'] + [SOURCE / f'cv/fold{k}.json' for k in range(1, 6)]
    manifests = {}
    for path in inputs:
        parent_gate = json.loads((path.parent / 'PREPARED.json').read_text())
        assert sha(path) == parent_gate['files_sha256'][path.name]
        manifests[str(path.relative_to(SOURCE))] = json.loads(path.read_text())
    final = manifests['final/manifest.json']
    records = final['records']
    assert len(records) == 8983 and sum((r['split'] == 'train' for r in records)) == 8085
    codes_np = np.random.default_rng(CODE_SEED).integers(0, 2, size=(len(AA_ORDER), BITS), dtype=np.uint8)
    assert len(np.unique(codes_np, axis=0)) == len(AA_ORDER)
    codes = torch.from_numpy(codes_np)
    DEST.mkdir(parents=True, exist_ok=True)
    recipe = dict(name='Random-FP', base='Unified-FP', standard_aa_onehot='zero; standard-AA rows only', nonstandard_nodes='all features unchanged', bits=BITS, code_seed=CODE_SEED, generator='numpy.default_rng / PCG64 / integers(0,2,dtype=uint8)', probability_one=0.5, aa_order=AA_ORDER, standard20=sorted(STANDARD20), extra_codes='U,O,B,Z,X have fixed fallback codes if present in standard-AA rows', shared_across='GEO/GEO+PIEDA, all folds, final, training/validation/test and downstream', codebook={a: row.tolist() for a, row in zip(AA_ORDER, codes_np)}, source_manifests={str(p.relative_to(SOURCE)): sha(p) for p in inputs})
    write(DEST / 'codebook.json', recipe)
    replacements = {}
    counts = Counter()
    source_counts = Counter()
    bit_sums = Counter()
    for i, record in enumerate(records, 1):
        original = load(record['paths']['v7'])
        assert fingerprint(original) == record['fingerprints']['v7'], record['key']
        derived, observed = transform(original, codes)
        counts.update(observed)
        for a, n in observed.items():
            source_counts[record['source'], a] += n
            bit_sums[a] += n * int(codes_np[AA_ORDER.index(a)].sum())
        target = DEST / 'graphs' / record['source'] / f'{i:05d}.pt'
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            torch.save(derived, target)
        reread = load(target)
        validate(original, reread, codes)
        assert fingerprint(derived) == fingerprint(reread)
        identity = (record['source'], record['key'])
        assert identity not in replacements
        replacements[identity] = (str(target), fingerprint(reread))
        if i % 500 == 0 or i == len(records):
            print(f'Random-FP prepared and reloaded {i}/{len(records)}; all non-AA features unchanged', flush=True)
    source_hashes = {}
    for relative, old in manifests.items():
        new = copy.deepcopy(old)
        for r in new['records']:
            r['paths']['v7'], r['fingerprints']['v7'] = replacements[r['source'], r['key']]
            r['random_fp_codebook_sha256'] = sha(DEST / 'codebook.json')
        recovered = copy.deepcopy(new)
        old_by_key = {(r['source'], r['key']): r for r in old['records']}
        for r in recovered['records']:
            ref = old_by_key[r['source'], r['key']]
            r['paths']['v7'] = ref['paths']['v7']
            r['fingerprints']['v7'] = ref['fingerprints']['v7']
            del r['random_fp_codebook_sha256']
        assert recovered == old, 'split, targets or training normalization changed'
        path = DEST / relative
        write(path, new)
        source_hashes[relative] = sha(SOURCE / relative)
    for sub in ('cv', 'final'):
        files = sorted((DEST / sub).glob('*.json'))
        write(DEST / sub / 'PREPARED.json', dict(prepared=True, training_authorized=False, files_sha256={p.name: sha(p) for p in files if p.name != 'PREPARED.json'}, representation='Random-FP', codebook_sha256=sha(DEST / 'codebook.json')))
    write(DEST / 'COMPLETE.json', dict(complete=True, graphs=8983, all_graphs_reloaded=True, input_dim=546, aa_counts=dict(counts), noncanonical_standard_aa={a: n for a, n in counts.items() if a not in STANDARD20}, counts_by_source={src: {a: source_counts[src, a] for a in AA_ORDER if source_counts[src, a]} for src in ('bdb', 'scop2')}, code_ones={a: int(row.sum()) for a, row in zip(AA_ORDER, codes_np)}, code_density=float(sum(bit_sums.values()) / (sum(counts.values()) * BITS)), nonstandard_nodes_unchanged=True, all_non_x_attributes_unchanged=True, all_split_and_target_statistics_unchanged=True, source_hashes=source_hashes, codebook_sha256=sha(DEST / 'codebook.json'), preparation_script_sha256=sha(__file__)))
    print('Random-FP PREPARATION COMPLETE; production training not submitted.', flush=True)
if __name__ == '__main__':
    main()
