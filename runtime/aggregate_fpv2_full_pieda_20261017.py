import os, json, sys
from pathlib import Path
import pandas as pd
from final_pieda_ci_20261012 import bootstrap, write
assert True
out = Path(sys.argv[1]) / f"model{os.environ['FPV2_MODEL']}_{os.environ['FPV2_VERSION']}"
done = [json.loads((out / (c + '_COMPLETE.json')).read_text()) for c in ['geo', 'geo_pieda']]
assert all((d['complete'] and (not d['config']['smoke']) for d in done))
records = [json.loads((out / (c + '_test_records.json')).read_text()) for c in ['geo', 'geo_pieda']]
assert records[0] == records[1] and len(records[0]) == 898
for kind in ['node', 'pair']:
    for fold in range(1, 6):
        for split in ['train', 'valid']:
            paths = [out / 'cv' / c / kind / f'fold{fold}' / f'{split}_probe_audit.json' for c in ['R3', 'R1']]
            assert json.loads(paths[0].read_text()) == json.loads(paths[1].read_text())
    for split in ['train', 'test']:
        paths = [out / c / (kind + '_raw_probe') / f'{split}_probe_audit.json' for c in ['R3', 'R1']]
        assert json.loads(paths[0].read_text()) == json.loads(paths[1].read_text())
cv = pd.concat([pd.read_csv(out / (c + '_cv_metrics.csv')) for c in ['geo', 'geo_pieda']], ignore_index=True)
assert len(cv) == 200
cv.to_csv(out / 'cv_metrics.csv', index=False)
cv.groupby(['model', 'condition', 'representation', 'split', 'target'])[['pearson_r', 'rmse', 'mae', 'r2']].agg(['mean', 'std']).to_csv(out / 'cv_mean_sd.csv')
bootstrap(out, records[0])
write(out / 'COMPLETE.json', dict(complete=True, series='FPv2 C full series', n_test_graphs=898, head_cv=True, encoder_cv=False, conditions=done))
