"""Reuse established frozen ligand-centred pIC50 extraction, Ridge and MLP."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import predict_pic50_mlp_probe as probe
from pic50_matched_mlp import read, write, sha

def evaluate(model, manifest, cohort_path, args, out, cfg, ckpath, b):
    out.mkdir()
    assert sha(probe.__file__) == 'bea288b0b088affda8c970ad66e40dcbdec68ae65c008c857e66dabbfaef4afe'
    checkpoint_sha = sha(ckpath)
    records = {r['pdb']: r for r in manifest['records'] if r['source'] == 'bdb'}
    df = pd.read_csv(cohort_path)
    assert df.pdbid.is_unique and len(df) == 1973
    assert all((records[p]['split'] == s for p, s in zip(df.pdbid, df.split_pdbid)))
    assert not set(df.loc[df.split_pdbid == 'test', 'pdbid']) & {r['pdb'] for r in manifest['records'] if r['split'] == 'train'}
    if args.smoke:
        df = df.groupby('split_pdbid', group_keys=False).head(8).sort_values('pdbid').reset_index(drop=True)
    df.to_csv(out / 'cohort.csv', index=False)
    pair_dim = 256 if args.model_number == 1 else 512
    model.eval().requires_grad_(False)
    before = b.digest_state(model.state_dict())
    vectors = {k: [] for k in ('node', 'pair', 'concat')}
    ih, nh = (hashlib.sha256(), hashlib.sha256())
    with torch.inference_mode():
        for i, row in enumerate(df.itertuples()):
            r = records[row.pdbid]
            assert r['split'] == row.split_pdbid
            g = b.load(r['paths'][args.version])
            assert b.fingerprint(g) == r['fingerprints'][args.version]
            assert g.x.shape[1] == (1026 if args.version == 'v6' else 546)
            geom, _, fmo, names = b.pm.infer_feature_groups(g)
            assert (len(geom), len(fmo), len(names)) == (66, 6, 72)
            ea = g.edge_attr.clone().float()
            ea[:, fmo] = 0
            ei = g.edge_index.long()
            lig = g.is_ligand.bool().reshape(-1)
            assert int(lig.sum()) == 1
            selected = lig.clone()
            selected[ei[:, lig[ei[0]] | lig[ei[1]]].reshape(-1)] = True
            edges = torch.where(lig[ei[0]] & ~lig[ei[1]])[0]
            neighbors = torch.where(selected & ~lig)[0]
            assert len(edges) > 0 and len(edges) == len(neighbors)
            assert torch.equal(torch.sort(ei[1, edges]).values, neighbors)
            ih.update(row.pdbid.encode())
            ih.update(ei.numpy().tobytes())
            ih.update(ea.numpy().tobytes())
            nh.update(row.pdbid.encode())
            nh.update(selected.numpy().tobytes())
            h = model.encode_nodes(g.x.float().cuda(), ei.cuda(), ea.cuda())
            node = h[selected.cuda()].mean(0).cpu()
            pair = model.pair_state[edges.cuda()].mean(0).cpu()
            assert node.shape == (256,) and pair.shape == (pair_dim,)
            for key, value in [('node', node), ('pair', pair), ('concat', torch.cat([node, pair]))]:
                assert torch.isfinite(value).all()
                vectors[key].append(value)
            if (i + 1) % 250 == 0:
                print(f'EXTRACT {args.condition} {i + 1}/{len(df)}', flush=True)
    assert before == b.digest_state(model.state_dict())
    config = dict(graph_mode='pretrained', target_col='reference_pIC50', split_col='split_pdbid', hidden_dim=256, mlp_layers=2, dropout=0.2, lr=0.001, weight_decay=0.001, batch_size=64, num_workers=2, epochs=2 if args.smoke else 500, patience=50, log_every=1 if args.smoke else 10)
    write(out / 'audit.json', dict(config=config, checkpoint_sha256=checkpoint_sha, encoder_frozen=True, model_number=args.model_number, version=args.version, pretrain_source='combined', checkpoint_path=str(ckpath), test_pretrain_train_overlap=0, cohort_sha256=sha(out / 'cohort.csv'), input_sha256=ih.hexdigest(), pooling_nodes_sha256=nh.hexdigest(), pooling='node: ligand+neighbors mean; pair: ligand-first saved edges mean; concat: both', counts=df.split_pdbid.value_counts().to_dict(), source_hashes=cfg['source_hashes']))
    results = {}
    for kind, values in vectors.items():
        cache = out / f'{kind}_embeddings.pt'
        torch.save(dict(embeddings=torch.stack(values), pdbids=df.pdbid.tolist()), cache)
        mapping, dim, _ = probe.load_graph_cache(str(cache))
        assert dim == {'node': 256, 'pair': pair_dim, 'concat': 256 + pair_dim}[kind] and set(mapping) == set(df.pdbid)
        x = np.stack([mapping[p] for p in df.pdbid]).astype(np.float32)
        y = df.reference_pIC50.to_numpy(dtype=np.float32)
        tr = df.split_pdbid.to_numpy() == 'train'
        scaler = StandardScaler().fit(x[tr])
        ridge = Ridge(alpha=1).fit(scaler.transform(x[tr]), y[tr])
        pred = ridge.predict(scaler.transform(x))
        dest = out / f'{kind}_ridge'
        dest.mkdir()
        met = {sp: probe.regression_metrics(y[df.split_pdbid == sp], pred[df.split_pdbid == sp]) for sp in ('train', 'valid', 'test')}
        write(dest / 'metrics.json', met)
        results[f'{kind}_ridge'] = met
        frame = df[['pdbid', 'split_pdbid', 'reference_pIC50']].copy()
        frame['prediction'] = pred
        frame.to_csv(dest / 'predictions.csv', index=False)
        np.savez(dest / 'model.npz', mean=scaler.mean_, scale=scaler.scale_, coef=ridge.coef_, intercept=ridge.intercept_)
        for seed in [42]:
            dest = out / f'{kind}_mlp_seed{seed}'
            dest.mkdir()
            probe.seed_all(seed)
            met = probe.run_mlp_probe(SimpleNamespace(**config, seed=seed), df, mapping, dim, dest)
            write(dest / 'metrics.json', met)
            results[f'{kind}_mlp_seed{seed}'] = met
            ids = sorted(df.loc[df.split_pdbid == 'train', 'pdbid'])
            std = probe.Standardizer(np.stack([mapping[p] for p in ids]))
            head = probe.GraphMLPProbe(dim, 256, 0.2, 2).eval()
            head.load_state_dict(b.load(dest / 'best_model.pt')['model'])
            with torch.no_grad():
                pred = head(torch.from_numpy(std.transform(x))).numpy()
            frame['prediction'] = pred
            frame.to_csv(dest / 'predictions.csv', index=False)
            np.savez(dest / 'scaler.npz', mean=std.mean, std=std.std)
            print(f"DONE {args.condition} {kind} seed={seed} test r={met['test']['pearson_r']:.4f}", flush=True)
    for met in results.values():
        for sp in ('train', 'valid', 'test'):
            assert met[sp]['n'] == int((df.split_pdbid == sp).sum())
            assert all((np.isfinite(met[sp][k]) for k in ('pearson_r', 'spearman_r', 'rmse', 'mae')))
    write(out / 'comparison.json', dict(results=results, condition=args.condition, model_number=args.model_number, version=args.version, fold=args.fold, head_seed=42, checkpoint_sha256=checkpoint_sha, all_checks_passed=True, smoke=args.smoke))
    print(f'PIC50_COMPLETE {out}', flush=True)
