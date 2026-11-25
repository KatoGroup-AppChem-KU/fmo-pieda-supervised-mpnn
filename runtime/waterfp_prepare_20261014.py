"""Change ONLY water Morgan bits; preserve original graphs and all split/statistics."""
import os, json, hashlib, shutil
from pathlib import Path
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
from prepare_geo_pieda_matched_20260911 import load, fingerprint
assert True
torch.set_num_threads(4)
ROOT = Path('dataset/waterfp_20261014')
ROOT.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, o):
    p.write_text(json.dumps(o, indent=2))
original = Path('dataset/final_train_20261008/manifest.json')
manifest = json.loads(original.read_text())
bits = np.zeros(512, dtype=np.int8)
DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles('O'), 2, nBits=512, useChirality=True), bits)
assert bits.sum() > 0
mapping = {}
audit = []
for i, r in enumerate(manifest['records'], 1):
    g = load(r['paths']['v6'])
    assert fingerprint(g) == r['fingerprints']['v6']
    names = list(g.node_feature_names)
    cols = [names.index('small_mol_morgan_' + str(k)) for k in range(512)]
    water = g.is_water.bool().reshape(-1)
    assert torch.equal(water, g.x[:, names.index('is_water')] > 0.5)
    old = g.x.clone()
    assert torch.count_nonzero(old[water][:, cols]) == 0
    rows = water.nonzero().flatten()
    g.x[rows[:, None], torch.tensor(cols)[None, :]] = torch.tensor(bits, dtype=g.x.dtype)[None, :]
    restored = g.x.clone()
    restored[rows[:, None], torch.tensor(cols)[None, :]] = 0
    assert torch.equal(restored, old)
    canonical = load(r['paths']['v7'])
    assert fingerprint(canonical) == r['fingerprints']['v7']
    assert torch.equal(canonical.is_water.bool().reshape(-1), water)
    ccols = [list(canonical.node_feature_names).index('small_mol_morgan_' + str(k)) for k in range(512)]
    assert torch.equal(g.x[water][:, cols], canonical.x[water][:, ccols])
    dest = ROOT / 'graphs' / r['source'] / (r['key'] + '.pt')
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(g, dest)
    newhash = fingerprint(g)
    assert fingerprint(load(dest)) == newhash
    mapping[r['source'], r['key']] = (str(dest), newhash)
    audit.append(dict(source=r['source'], key=r['key'], water_nodes=int(water.sum()), old_fingerprint=r['fingerprints']['v6'], new_fingerprint=newhash))
    if i % 250 == 0:
        print('Verified water-only change', i, '/', len(manifest['records']), flush=True)
assert len(mapping) == 8991
for source, target, files in [(Path('dataset/fixedtest_cv5_20260925_prepared'), ROOT / 'cv', [f'fold{k}.json' for k in range(1, 6)]), (original.parent, ROOT / 'final', ['manifest.json'])]:
    target.mkdir()
    hashes = {}
    for name in files:
        mf = json.loads((source / name).read_text())
        for r in mf['records']:
            r['paths']['v6'], r['fingerprints']['v6'] = mapping[r['source'], r['key']]
        mf['water_fp_update'] = dict(smiles='O', radius=2, bits=512, use_chirality=True, source_manifest=str(source / name), source_sha256=sha(source / name), only_water_morgan_changed=True)
        write(target / name, mf)
        hashes[name] = sha(target / name)
    write(target / 'COMPLETE.json', dict(complete=True, files_sha256=hashes, manifest_sha256=hashes.get('manifest.json')))
write(ROOT / 'graph_audit.json', audit)
write(ROOT / 'COMPLETE.json', dict(complete=True, graphs=8991, water_nodes=sum((a['water_nodes'] for a in audit)), water_fp_on_bits=np.flatnonzero(bits).tolist(), only_water_morgan_changed=True, split_and_normalization_unchanged=True, source_sha256=sha(original)))
print('PREPARATION COMPLETE', flush=True)
