"""Final SG/ESM-FP frozen probes with paired PDB-cluster percentile intervals."""
import os, json, shutil, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import run_cv5_20260925 as cv
import cv5_loops_20260925 as loops
import geopair_cfg_20261006 as gp

def write(p, obj):
    p.write_text(json.dumps(obj, indent=2))

def pooled(s):
    n, sy, sp, syy, spp, syp, se, sa = np.moveaxis(s, -1, 0)
    vy = syy - sy * sy / n
    vp = spp - sp * sp / n
    return np.stack([(syp - sy * sp / n) / np.sqrt(vy * vp), np.sqrt(se / n), sa / n, 1 - se / vy], axis=-1)

def bootstrap(out, records):
    pdbs = sorted({r['pdb'] for r in records})
    lookup = {p: i for i, p in enumerate(pdbs)}
    group = np.array([lookup[r['pdb']] for r in records])
    rng = np.random.default_rng(20261012)
    sampled = rng.integers(len(pdbs), size=(10000, len(pdbs)), dtype=np.int32)
    np.savez_compressed(out / 'pdb_bootstrap_indices.npz', indices=sampled, pdbids=np.array(pdbs))
    names = ['PIEDA_sum', 'ES', 'EX', 'CTmix', 'DI']
    met = ['pearson_r', 'rmse', 'mae', 'r2']
    rows = []
    for kind in ['node', 'pair']:
        points = {}
        draws = {}
        reference = None
        for cond in ['R3', 'R1']:
            obj = np.load(out / cond / (kind + '_raw_probe') / 'test_predictions.npz')
            y = obj['y'].astype(float)
            p = obj['prediction'].astype(float)
            ids = obj['graph_ids']
            if reference is None:
                reference = (y, ids)
            else:
                np.testing.assert_array_equal(y, reference[0])
                np.testing.assert_array_equal(ids, reference[1])
            g = group[ids]
            stats = np.zeros((len(pdbs), 5, 8))
            for j in range(5):
                a, b = (y[:, j], p[:, j])
                for k, v in enumerate([np.ones(len(y)), a, b, a * a, b * b, a * b, (a - b) ** 2, np.abs(a - b)]):
                    stats[:, j, k] = np.bincount(g, weights=v, minlength=len(pdbs))
            point = pooled(stats.sum(0))
            draw = []
            for chunk in np.array_split(sampled, 100):
                counts = np.stack([np.bincount(i, minlength=len(pdbs)) for i in chunk])
                draw.append(pooled(np.einsum('bg,gck->bck', counts, stats)))
            draw = np.concatenate(draw)
            assert np.isfinite(draw).all()
            for j in range(5):
                np.testing.assert_allclose(point[j, 0], np.corrcoef(y[:, j], p[:, j])[0, 1], atol=1e-10)
                np.testing.assert_allclose(point[j, 1], np.sqrt(np.mean((y[:, j] - p[:, j]) ** 2)), atol=1e-10)
            mult = np.bincount(sampled[0], minlength=len(pdbs))
            ii = np.repeat(np.arange(len(y)), mult[g])
            for j in range(5):
                np.testing.assert_allclose(draw[0, j, 0], np.corrcoef(y[ii, j], p[ii, j])[0, 1], atol=1e-10)
            np.savez_compressed(out / f'{cond}_{kind}_bootstrap.npz', metrics=draw, statistics=stats)
            points[cond] = point
            draws[cond] = draw
        for label, point, draw in [('GEO', points['R3'], draws['R3']), ('GEO+PIEDA', points['R1'], draws['R1']), ('difference', points['R1'] - points['R3'], draws['R1'] - draws['R3'])]:
            lo, hi = np.quantile(draw, [0.025, 0.975], axis=0)
            for j, target in enumerate(names):
                for k, metric in enumerate(met):
                    rows.append(dict(representation=kind, condition=label, target=target, metric=metric, estimate=point[j, k], ci_low=lo[j, k], ci_high=hi[j, k], n_pdb=len(pdbs), n_graphs=len(records)))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'confidence_intervals.csv', index=False)
    print(frame[frame.metric == 'pearson_r'].to_string(index=False), flush=True)

def main():
    assert True and True
    assert Path.cwd() == Path(__import__('os').environ['FMO_WORKDIR']) and torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    number = int(os.environ.get('EVAL_MODEL_NUMBER', '1'))
    version = os.environ.get('EVAL_GRAPH_VERSION', 'v6')
    assert number in [1, 2] and version in ['v6', 'v7']
    assert cv.sha(loops.__file__) == '1ee097f8b7dd552b0ae73bd5315039aea1f690a5540f16a99f74c532f00b156e'
    out = Path(os.environ.get('PIEDA_CI_OUT', 'results/final_pieda_ci_' + os.environ.get('SLURM_JOB_ID', 'standalone')))
    out.mkdir()
    shutil.copy2(__file__, out / Path(__file__).name)
    manifest_path = cv.PREP / 'fold1.json'
    manifest = json.loads(manifest_path.read_text())
    prep = json.loads((cv.PREP / 'COMPLETE.json').read_text())
    assert cv.sha(manifest_path) == prep['files_sha256']['fold1.json']
    records = manifest['records']
    test = [r for r in records if r['split'] == 'test']
    assert len(test) == 900
    write(out / 'fold_manifest.json', manifest)
    sources = cv.source_hashes()
    graphs = {s: [] for s in ['train', 'valid', 'test']}
    for i, r in enumerate(records, 1):
        graphs[r['split']].append(cv.load(r, manifest, version))
        if i % 500 == 0:
            print(f'preload {i}/{len(records)}', flush=True)
    config = dict(model=os.environ.get('EVAL_MODEL_LABEL', 'SG/ESM-FP'), encoder='all-subsets fixed epoch300', probe='existing fold1 train/validation; not fivefold head CV or all-development head refit', seed=42, max_epochs=80, patience=10, batch_size=4096, optimizer='Adam', lr=0.001, train_edge_cap=800000, valid_edge_cap=150000, edge_selection_seed=123, inputs='geometry retained; all FMO/q zero', bootstrap='paired PDB cluster; pooled selected-edge metrics', n_resamples=10000, interval='percentile 2.5-97.5%', caveat='Encoder saw probe train/validation structures and PIEDA teachers, not fixed Test. CI conditional on fitted model; does not cover training uncertainty or shared proteins across different PDBs.')
    write(out / 'config.json', config)
    hashes = {str(manifest_path): cv.sha(manifest_path)}
    for cond in ['R3', 'R1']:
        run = Path('results/final_train_20261008/full') / f'model{number}_{version}_{cond}_alltrain'
        ckpath = run / 'epoch_300.pt'
        ckhash = cv.sha(ckpath)
        done = json.loads((run / 'COMPLETE.json').read_text())
        assert done['complete'] and done['checkpoint_sha256'] == ckhash
        ck = cv.base.load(ckpath)
        assert ck['epoch'] == 300 and ck['config']['train_graphs'] == 8091
        pre = json.loads((run / 'train_manifest.json').read_text())
        assert not {r['pdb'] for r in test} & {r['pdb'] for r in pre['records'] if r['split'] == 'train'}
        model = gp.make_model(number, version) if cond == 'R3' else cv.make_model(number, version, 'R1')
        model.load_state_dict(ck['model_state'], strict=True)
        del ck
        model.eval().requires_grad_(False)
        for kind in ['node', 'pair']:
            dest = out / cond / (kind + '_raw_probe')
            dest.mkdir(parents=True)
            loops.extract = cv.prior.ORIGINAL_EXTRACT if kind == 'node' else cv.extract_pair
            loops.Probe = cv.prior.ORIGINAL_PROBE if kind == 'node' or number == 2 else cv.prior.PairProbe
            start = time.monotonic()
            print(f'BEGIN {cond} {kind}', flush=True)
            loops.run_probe(model, graphs, SimpleNamespace(mode='full'), dest)
            print(f'DONE {cond} {kind}: {time.monotonic() - start:.1f}s', flush=True)
            for sp in graphs:
                a = json.loads((dest / f'{sp}_probe_audit.json').read_text())
                ref = json.loads((out / 'R3/node_raw_probe' / f'{sp}_probe_audit.json').read_text())
                assert a == ref
        hashes[str(ckpath)] = ckhash
        del model
        torch.cuda.empty_cache()
    bootstrap(out, test)
    for path, h in hashes.items():
        assert cv.sha(path) == h
    assert sources == cv.source_hashes()
    write(out / 'COMPLETE.json', dict(complete=True, config=config, source_hashes=sources, input_hashes=hashes))
    print('COMPLETE: frozen final encoders, 4 probes, paired PDB bootstrap', flush=True)
if __name__ == '__main__':
    main()
