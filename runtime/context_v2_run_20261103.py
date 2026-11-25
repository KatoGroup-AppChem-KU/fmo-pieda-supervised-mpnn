"""SG/ESM-FP GEO+PIEDA_v2, six encoders; completed GEO is a frozen reference."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import context_v2_full_20261103 as full
import pieda_context_v2_20261103 as context
ROOT = Path(__import__('os').environ['FMO_WORKDIR'])
SERIES = ROOT / 'results/pieda_context_v2_20261103'
TASKS = [(k, 'geo_pieda') for k in (0, 1, 2, 3, 4, 5)]
GEO_CV = ROOT / 'results/fpv2_C_full_20261017/model1_v6/cv'
GEO_FINAL = ROOT / 'results/fpv2_sg_b_unitloss_20261016/final/geo'
GEO_PIEDA = ROOT / 'results/fpv2_pieda_headcv_7571/model1_v6'
GEO_PX = ROOT / 'results/fpv2_px_7582/geo'

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def configure():
    assert True and Path.cwd() == ROOT
    full.torch.set_num_threads(4)
    full.torch.backends.cudnn.benchmark = False
    paths = {ROOT / n if (ROOT / n).is_file() else ROOT / 'outputs' / n for n in full.BASE_HASH()}
    paths |= {Path(m.__file__).resolve() for m in (full, full.ph, full.px, context)}
    paths |= {Path(__file__).resolve(), ROOT / 'outputs/aggregate_fpv2_full_pieda_20261017.py', Path(full.px.fit.__code__.co_filename).resolve(), ROOT / 'outputs/final_pieda_ci_20261012.py', ROOT / 'outputs/bootstrap_px_selected_20261012.py'}
    assert all((p.is_file() for p in paths))
    full.SOURCE_PATHS = sorted(paths)
    full.SERIES = SERIES
    full.NUMBER = 1
    full.VERSION = 'v6'
    full.fp.RUNS = SERIES / 'model1_v6'
    full.fp.make_model = full.make_model
    full.fp.source_hashes = full.sources
    full.fp.cv.load = context.load
    full.fp.prior.masked = context.masked
    full.fp.cv.prior.ORIGINAL_EXTRACT = context.extract_node
    full.fp.cv.extract_pair = context.extract_pair
    full.fp.smoke = context.smoke
    os.environ['FPV2_MODEL'], os.environ['FPV2_VERSION'] = ('1', 'v6')
    os.environ['PIEDA_HEADCV_ROOT'] = str(full.fp.RUNS / 'pieda')

def link(source, dest):
    assert source.exists(), str(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        assert dest.resolve() == source.resolve()
    else:
        assert not dest.exists(), str(dest)
        dest.symlink_to(source, target_is_directory=source.is_dir())

def reference_setup():
    """Read-only links: never alter any existing GEO file or checkpoint."""
    audit = {}
    for k in range(1, 6):
        src = GEO_CV / f'fold{k}/geo'
        assert json.loads((src / 'EVALUATION_COMPLETE.json').read_text())['complete']
        link(src, full.fp.RUNS / f'cv/fold{k}/geo')
        audit[str(src / 'EVALUATION_COMPLETE.json')] = full.fp.cv.sha(src / 'EVALUATION_COMPLETE.json')
    link(GEO_FINAL, full.fp.RUNS / 'final/geo')
    assert json.loads((GEO_FINAL / 'COMPLETE.json').read_text())['complete']
    audit[str(GEO_FINAL / 'COMPLETE.json')] = full.fp.cv.sha(GEO_FINAL / 'COMPLETE.json')
    dest = full.fp.RUNS / 'pieda/model1_v6'
    for src in GEO_PIEDA.iterdir():
        if src.name.startswith('geo_') and (not src.name.startswith('geo_pieda_')):
            link(src, dest / src.name)
    link(GEO_PIEDA / 'R3', dest / 'R3')
    link(GEO_PIEDA / 'cv/R3', dest / 'cv/R3')
    link(GEO_PX, full.fp.RUNS / 'px/geo')
    for src in (GEO_PIEDA / 'geo_COMPLETE.json', GEO_PX / 'COMPLETE.json'):
        assert json.loads(src.read_text())['complete']
        audit[str(src)] = full.fp.cv.sha(src)
    return audit

def ready():
    sm = SERIES / 'smoke/model1_v6/COMPLETE.json'
    gate = json.loads(sm.read_text())
    assert gate['complete'] and gate['source_hashes'] == full.sources()
    refs = reference_setup()
    manifests = [full.manifest(k)[0] for k in (0, 1, 2, 3, 4, 5)]
    write(SERIES / 'READY_TO_SUBMIT.json', dict(ready=True, model='SG/ESM-FP', experiment='GEO+PIEDA_v2', source_hashes=full.sources(), smoke_sha256=full.fp.cv.sha(sm), reference_hashes=refs, manifests={str(p): full.fp.cv.sha(p) for p in manifests}, tasks=[dict(task=i, fold=k, condition=c) for i, (k, c) in enumerate(TASKS)], new_encoder_tasks=6, reused_geo_encoders=6, encoder_epochs=300, head_seed=42, mask_rate=0.15, input_normalization='canonical raw PIEDA / encoder TRAIN mean and std', pretrain='selected pair geometry and all 5 PIEDA masked both ways; surrounding geometry and PIEDA retained', pieda_probe='target PIEDA masked both ways, <=15% pair chunks; geometry retained', affinity='all geometry and PIEDA visible', q='zero in every stage', missing_fmo_imputation='not implemented or evaluated in this experiment', comparison='GEO uses no PIEDA input; v2 adds both PIEDA supervision and PIEDA context', checks=gate['checks']))
    print('READY: six GEO+PIEDA_v2 tasks; GEO references reused read-only', flush=True)

def aggregate():
    refs = reference_setup()
    gate = json.loads((SERIES / 'READY_TO_SUBMIT.json').read_text())
    assert refs == gate['reference_hashes']
    assert all(((full.fp.RUNS / (f'cv/fold{k}/{c}' if k else f'final/{c}') / 'EVALUATION_COMPLETE.json').exists() for k, c in TASKS))
    sys.argv = ['aggregate', str(full.fp.RUNS / 'pieda')]
    runpy.run_path(str(ROOT / 'outputs/aggregate_fpv2_full_pieda_20261017.py'), run_name='__main__')
    full.px.aggregate(full.fp.RUNS / 'px')
    rows = []
    for k in range(1, 6):
        for c in ('geo', 'geo_pieda'):
            d = full.fp.RUNS / f'cv/fold{k}/{c}'
            for kind in ('node', 'pair'):
                ms = json.loads((d / f'pieda/{kind}_raw_probe/test_metrics.json').read_text())
                for target, metrics in ms.items():
                    rows.append(dict(task='PIEDA', condition=c, fold=k, representation=kind, target=target, **metrics))
            for kind in full.px.KINDS:
                ms = json.loads((d / f'px/{kind}/metrics.json').read_text())['test']
                rows.append(dict(task='affinity', condition=c, fold=k, representation=kind, target='pX', **ms))
    df = full.pd.DataFrame(rows)
    df.to_csv(full.fp.RUNS / 'encoder_cv_metrics.csv', index=False)
    df.groupby(['task', 'condition', 'representation', 'target'])[['pearson_r', 'rmse', 'mae']].agg(['mean', 'std']).to_csv(full.fp.RUNS / 'encoder_cv_mean_sd.csv')
    counts = []
    for key, g in df.groupby(['task', 'representation', 'target']):
        p = g.pivot(index='fold', columns='condition', values='pearson_r')
        counts.append(dict(task=key[0], representation=key[1], target=key[2], improved_folds=int((p.geo_pieda > p.geo).sum()), folds=5))
    full.pd.DataFrame(counts).to_csv(full.fp.RUNS / 'encoder_cv_improvements.csv', index=False)
    write(full.fp.RUNS / 'COMPLETE.json', dict(complete=True, experiment='GEO+PIEDA_v2', source_hashes=full.sources(), reused_geo_references=refs, caveat='v2 has real surrounding PIEDA as input; missing-FMO imputation remains future work'))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['smoke', 'run', 'aggregate'], required=True)
    ap.add_argument('--task', type=int, default=0)
    a = ap.parse_args()
    configure()
    if a.stage == 'smoke':
        reference_setup()
        full.smoke()
        ready()
    else:
        gate = json.loads((SERIES / 'READY_TO_SUBMIT.json').read_text())
        assert gate['ready'] and gate['source_hashes'] == full.sources()
        assert gate['reference_hashes'] == reference_setup()
        assert all((full.fp.cv.sha(p) == h for p, h in gate['manifests'].items()))
        if a.stage == 'run':
            assert 0 <= a.task < len(TASKS)
            full.run(*TASKS[a.task])
        else:
            aggregate()
if __name__ == '__main__':
    main()
