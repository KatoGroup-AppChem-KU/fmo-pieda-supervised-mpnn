"""Random-FP SG controlled experiment, reusing the complete FPv2 C runner.

Only --stage smoke is authorized now. run/aggregate are prepared entry points
for later explicit production submission. Inputs are immutable derived graphs.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import fpv2_full_20261017 as full
import random_fp_prepare_20261030 as prep
import random_fp_pieda_20261030 as pieda
import random_fp_px_20261030 as affinity
ROOT = Path(__import__('os').environ['FMO_WORKDIR'])
SERIES = ROOT / 'results/random_fp_20261030'
TASKS = [(k, c) for k in range(1, 6) for c in ('geo', 'geo_pieda')]
TASKS += [(0, c) for c in ('geo', 'geo_pieda')]

def configure():
    assert True and Path.cwd() == ROOT
    full.torch.set_num_threads(4)
    full.torch.backends.cudnn.benchmark = False
    gate = json.loads((prep.DEST / 'COMPLETE.json').read_text())
    assert gate['complete'] and gate['graphs'] == 8983 and gate['all_non_x_attributes_unchanged']
    assert gate['preparation_script_sha256'] == prep.sha(prep.__file__)
    assert gate['codebook_sha256'] == prep.sha(prep.DEST / 'codebook.json')
    full.ph, full.px = (pieda, affinity)
    paths = {ROOT / n if (ROOT / n).is_file() else ROOT / 'outputs' / n for n in full.BASE_HASH()}
    paths |= {Path(full.__file__).resolve(), Path(full.ph.__file__).resolve(), Path(full.px.__file__).resolve(), ROOT / 'outputs/aggregate_fpv2_full_pieda_20261017.py', Path(full.px.fit.__code__.co_filename).resolve(), ROOT / 'outputs/final_pieda_ci_20261012.py', ROOT / 'outputs/bootstrap_px_selected_20261012.py', Path(__file__).resolve(), Path(prep.__file__).resolve()}
    assert all((p.is_file() for p in paths))
    full.SOURCE_PATHS = sorted(paths)
    full.SERIES = SERIES
    full.NUMBER, full.VERSION = (1, 'v7')
    full.fp.PREP = prep.DEST
    full.fp.RUNS = SERIES / 'model1_v7'
    full.fp.make_model = full.make_model
    full.fp.source_hashes = full.sources
    os.environ['FPV2_MODEL'], os.environ['FPV2_VERSION'] = ('1', 'v7')
    os.environ['PIEDA_HEADCV_ROOT'] = str(full.fp.RUNS / 'pieda')
    return gate

def ready():
    sm = SERIES / 'smoke/model1_v7/COMPLETE.json'
    d = json.loads(sm.read_text())
    assert d['complete'] and d['source_hashes'] == full.sources()
    data_gate = prep.DEST / 'COMPLETE.json'
    plan = dict(ready=True, production_submitted=False, representation='Random-FP', model='SG', input_dim=546, node_dim=256, pair_dim=256, encoder_epochs=300, seed=42, head_seed=42, code_seed=42, probability_one=0.5, standard_aa_onehot='zero', nonstandard_features='unchanged', input_policy='all PIEDA/q zero; selected pair geometry masked in both directions', tasks=[dict(task=i, fold=k, condition=c, final_encoder=k == 0) for i, (k, c) in enumerate(TASKS)], encoder_tasks=12, maximum_concurrent_tasks=2, downstream='original heads; frozen node/pair PIEDA; node/pair/concat affinity; final head CV/refit and paired CI', prepared_manifest_root=str(prep.DEST), preparation_sha256=prep.sha(data_gate), codebook_sha256=prep.sha(prep.DEST / 'codebook.json'), smoke_sha256=prep.sha(sm), source_hashes=full.sources(), resource_check=d['checks'], production_command='sbatch --array=0-11%2 outputs/slurm_random_fp_20261030.sh run', aggregation_command='sbatch outputs/slurm_random_fp_20261030.sh aggregate')
    prep.write(SERIES / 'READY_TO_SUBMIT.json', plan)
    print('READY: 12 Random-FP tasks prepared; NO production job submitted.', flush=True)

def aggregate():
    assert all(((full.fp.RUNS / (f'cv/fold{k}/{c}' if k else f'final/{c}') / 'EVALUATION_COMPLETE.json').exists() for k, c in TASKS))
    sys.argv = ['aggregate', str(full.fp.RUNS / 'pieda')]
    runpy.run_path(str(ROOT / 'outputs/aggregate_fpv2_full_pieda_20261017.py'), run_name='__main__')
    full.px.aggregate(full.fp.RUNS / 'px')
    rows = []
    for k in range(1, 6):
        for c in ('geo', 'geo_pieda'):
            d = full.fp.RUNS / f'cv/fold{k}/{c}'
            for kind in ('node', 'pair'):
                for target, metrics in json.loads((d / f'pieda/{kind}_raw_probe/test_metrics.json').read_text()).items():
                    rows.append(dict(task='PIEDA', condition=c, fold=k, representation=kind, target=target, **metrics))
            for kind in full.px.KINDS:
                met = json.loads((d / f'px/{kind}/metrics.json').read_text())['test']
                rows.append(dict(task='affinity', condition=c, fold=k, representation=kind, target='pX', **met))
    frame = full.pd.DataFrame(rows)
    frame.to_csv(full.fp.RUNS / 'encoder_cv_metrics.csv', index=False)
    frame.groupby(['task', 'condition', 'representation', 'target'])[['pearson_r', 'rmse', 'mae']].agg(['mean', 'std']).to_csv(full.fp.RUNS / 'encoder_cv_mean_sd.csv')
    counts = []
    for key, g in frame.groupby(['task', 'representation', 'target']):
        p = g.pivot(index='fold', columns='condition', values='pearson_r')
        counts.append(dict(task=key[0], representation=key[1], target=key[2], improved_folds=int((p.geo_pieda > p.geo).sum()), folds=5))
    full.pd.DataFrame(counts).to_csv(full.fp.RUNS / 'encoder_cv_improvements.csv', index=False)
    prep.write(full.fp.RUNS / 'COMPLETE.json', dict(complete=True, representation='Random-FP', source_hashes=full.sources()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('smoke', 'run', 'aggregate'), required=True)
    ap.add_argument('--task', type=int, default=0)
    args = ap.parse_args()
    configure()
    if args.stage == 'smoke':
        full.smoke()
        ready()
    elif args.stage == 'run':
        assert 0 <= args.task < len(TASKS)
        d = json.loads((SERIES / 'READY_TO_SUBMIT.json').read_text())
        assert d['ready'] and d['source_hashes'] == full.sources()
        assert d['preparation_sha256'] == prep.sha(prep.DEST / 'COMPLETE.json')
        k, c = TASKS[args.task]
        full.run(k, c)
    else:
        aggregate()
if __name__ == '__main__':
    main()
