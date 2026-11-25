"""
predict_pic50.py

案1 --task zero_shot
    GNN埋め込み [pdbid → 1280次元] を Ridge 回帰（線形プローブ）で
    reference_pIC50 に回帰する。追加の深層学習なし。
    pdbidレベルの回帰（訓練データで Ridge だけをフィット）。

案1b --task mlp_probe
    GNN埋め込み [pdbid → 1280次元] を MLP 回帰で reference_pIC50 に
    回帰する。GNNは固定。MLPのみ訓練セットでフィット。

案2 --task finetune
    GNN埋め込み + クエリリガンド Morgan 指紋 → MLP → query_pIC50
    MLP のみ訓練（GNN 重みはキャッシュ済みの固定値）。

共通オプション:
    --graph-mode pretrained / random / zero / none
    どちらのタスクでも 4条件アブレーションが有効。

使用例:
  # 案1: zero_shot × 4条件
  python predict_pic50.py --task zero_shot \\
      --data-csv policyB_delta_bound_exact_graphcovered_strictsplits.csv \\
      --split-col split_pdbid \\
      --graph-embedding-cache checkpoints/pretrain_v11s_4435/graph_embeddings_global_mean.pt \\
      --graph-mode pretrained --out-dir results/zs_pretrained

  # 案2: finetune × 4条件
  python predict_pic50.py --task finetune \\
      --data-csv policyB_delta_bound_exact_graphcovered_strictsplits.csv \\
      --split-col split_pdbid \\
      --graph-embedding-cache checkpoints/pretrain_v11s_4435/graph_embeddings_global_mean.pt \\
      --graph-mode pretrained --out-dir results/ft_pretrained
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
try:
    from scipy import stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False
try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def norm_pdb(x):
    return str(x).strip().lower()

def load_graph_cache(path: str):
    try:
        obj = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location='cpu')
    if not isinstance(obj, dict):
        raise ValueError('Cache must be a dict')
    emb = None
    for k in ['embeddings', 'embedding', 'graph_embeddings', 'x']:
        if k in obj:
            emb = obj[k]
            break
    ids = None
    for k in ['pdbids', 'pdb_ids', 'pdbid', 'ids', 'keys']:
        if k in obj:
            ids = obj[k]
            break
    if emb is None or ids is None:
        raise ValueError(f'embeddings/pdbids not found. keys={list(obj.keys())}')
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().float().numpy()
    else:
        emb = np.asarray(emb, dtype=np.float32)
    ids = [norm_pdb(x) for x in ids]
    graph_map = {p: emb[i].astype(np.float32) for i, p in enumerate(ids)}
    meta = {'n_cache': len(ids), 'n_unique': len(graph_map)}
    return (graph_map, int(emb.shape[1]), meta)

def make_rand_map(pdbids, graph_dim: int, seed: int):
    rng = np.random.default_rng(seed)
    return {p: rng.normal(0, 1, graph_dim).astype(np.float32) for p in pdbids}

def get_graph_vec(pdb: str, graph_map: dict, rand_map: dict, graph_mode: str, graph_dim: int):
    """graph_mode に応じた埋め込みベクトルを返す。none モードは None を返す。"""
    if graph_mode == 'none':
        return None
    if graph_mode == 'zero':
        return np.zeros(graph_dim, dtype=np.float32)
    if graph_mode == 'random':
        return rand_map[pdb]
    return graph_map[pdb]

def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    nan = float('nan')
    if n < 2:
        return {'n': n, 'pearson_r': nan, 'spearman_r': nan, 'mae': nan, 'rmse': nan}
    if HAS_SCIPY:
        pr, pp = stats.pearsonr(y_true, y_pred)
        sr, sp = stats.spearmanr(y_true, y_pred)
    else:
        yt_c = y_true - y_true.mean()
        yp_c = y_pred - y_pred.mean()
        d = np.sqrt((yt_c ** 2).sum()) * np.sqrt((yp_c ** 2).sum())
        pr = float((yt_c * yp_c).sum() / d) if d > 0 else 0.0
        sr = nan
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {'n': n, 'pearson_r': float(pr), 'spearman_r': float(sr), 'mae': mae, 'rmse': rmse}

def ligand_feature(smiles: str, fpgen, n_bits: int):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fp = fpgen.GetFingerprint(mol)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    desc = np.array([Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol), Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), Descriptors.NumRotatableBonds(mol), Descriptors.RingCount(mol), Descriptors.HeavyAtomCount(mol), Descriptors.FractionCSP3(mol), Descriptors.NumAromaticRings(mol), Descriptors.NumAliphaticRings(mol), Descriptors.BertzCT(mol), Descriptors.Chi0v(mol), Descriptors.Chi1v(mol), Descriptors.Kappa1(mol), Descriptors.Kappa2(mol)], dtype=np.float32)
    desc[~np.isfinite(desc)] = 0.0
    return np.concatenate([arr, desc]).astype(np.float32)

class Standardizer:

    def __init__(self, x):
        self.mean = x.mean(axis=0).astype(np.float32)
        self.std = x.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-06] = 1.0

    def transform(self, x):
        return ((x - self.mean) / self.std).astype(np.float32)

def run_zero_shot(args, df: pd.DataFrame, graph_map: dict, graph_dim: int) -> dict:
    """
    pdbidレベルで GNN 埋め込み → Ridge 回帰 → reference_pIC50。
    GNN は固定。Ridge のみ訓練セットでフィットする。
    none モードは定数予測（訓練平均）を使う。
    """
    if not HAS_SKLEARN:
        raise RuntimeError('scikit-learn が必要です (pip install scikit-learn)')
    target_col = args.target_col or 'reference_pIC50'
    if target_col not in df.columns:
        raise ValueError(f'--target-col {target_col!r} が CSV に存在しません。列: {df.columns.tolist()}')
    df = df.copy()
    df['pdbid'] = df['pdbid'].map(norm_pdb)
    df = df[df['pdbid'].isin(graph_map) & df[target_col].notna()].copy()
    ref = df.groupby('pdbid', as_index=False).agg({target_col: 'mean', args.split_col: 'first'})
    rand_map = make_rand_map(ref['pdbid'].tolist(), graph_dim, args.seed)
    by_split = {str(sp): g for sp, g in ref.groupby(args.split_col)}
    train_df = by_split.get('train', pd.DataFrame())
    print('=' * 100)
    print(f'Task        : zero_shot')
    print(f'Graph mode  : {args.graph_mode}')
    print(f'Target col  : {target_col}  |  Split col: {args.split_col}')
    for sp, g in sorted(by_split.items()):
        print(f'  {sp:8s}: {len(g)} PDBs  mean_pIC50={g[target_col].mean():.3f}')
    if args.graph_mode == 'none':
        train_mean = float(train_df[target_col].mean()) if len(train_df) > 0 else 0.0
        results = {}
        for sp, sub in sorted(by_split.items()):
            y = sub[target_col].values
            m = regression_metrics(y, np.full(len(y), train_mean))
            m['note'] = 'constant_mean_predictor'
            results[sp] = m
        return results

    def build_Xy(sub: pd.DataFrame):
        X, y = ([], [])
        for _, row in sub.iterrows():
            v = get_graph_vec(row['pdbid'], graph_map, rand_map, args.graph_mode, graph_dim)
            X.append(v)
            y.append(float(row[target_col]))
        return (np.array(X, dtype=np.float32), np.array(y, dtype=np.float32))
    X_tr, y_tr = build_Xy(train_df)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    ridge = Ridge(alpha=args.ridge_alpha)
    ridge.fit(X_tr_sc, y_tr)
    tr_r = regression_metrics(y_tr, ridge.predict(X_tr_sc))['pearson_r']
    print(f'Ridge alpha={args.ridge_alpha}  train_pearson_r={tr_r:.4f}')
    results = {}
    for sp, sub in sorted(by_split.items()):
        X, y = build_Xy(sub)
        pred = ridge.predict(scaler.transform(X))
        results[sp] = regression_metrics(y, pred)
    return results

class GraphProbeDataset(Dataset):

    def __init__(self, rows: list, graph_map: dict, graph_mode: str, graph_dim: int, rand_map: dict, graph_std=None):
        self.rows = rows
        self.graph_map = graph_map
        self.graph_mode = graph_mode
        self.graph_dim = graph_dim
        self.rand_map = rand_map
        self.graph_std = graph_std

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        g = get_graph_vec(r['pdbid'], self.graph_map, self.rand_map, self.graph_mode, self.graph_dim)
        if g is None:
            g = np.zeros(self.graph_dim, dtype=np.float32)
        if self.graph_std is not None:
            g = self.graph_std.transform(g)
        return (torch.from_numpy(g), torch.tensor(r['pic50'], dtype=torch.float32))

class GraphMLPProbe(nn.Module):

    def __init__(self, graph_dim: int, hidden: int, dropout: float, n_layers: int):
        super().__init__()
        layers = [nn.Linear(graph_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout)]
        d = hidden
        for _ in range(max(n_layers - 1, 0)):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, g):
        return self.net(g).view(-1)

def eval_graph_probe(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    loss_fn = nn.MSELoss()
    ys, preds, total, n = ([], [], 0.0, 0)
    with torch.no_grad():
        for g, y in loader:
            g, y = (g.to(device), y.to(device))
            p = model(g)
            total += float(loss_fn(p, y)) * len(y)
            n += len(y)
            ys.append(y.cpu().numpy())
            preds.append(p.cpu().numpy())
    m = regression_metrics(np.concatenate(ys), np.concatenate(preds))
    m['loss'] = total / max(n, 1)
    return m

def run_mlp_probe(args, df: pd.DataFrame, graph_map: dict, graph_dim: int, out_dir: Path) -> dict:
    """
    pdbidレベルで frozen graph embedding → MLP → reference_pIC50。
    GNNはキャッシュ済み固定値。MLPのみtrain splitで訓練する。
    Ridge線形プローブの非線形版として使う。
    """
    target_col = args.target_col or 'reference_pIC50'
    if target_col not in df.columns:
        raise ValueError(f'--target-col {target_col!r} が CSV に存在しません。列: {df.columns.tolist()}')
    df = df.copy()
    df['pdbid'] = df['pdbid'].map(norm_pdb)
    df = df[df['pdbid'].isin(graph_map) & df[target_col].notna()].copy()
    nunique = df.groupby('pdbid')[target_col].nunique(dropna=True)
    n_multi = int((nunique > 1).sum())
    if n_multi > 0:
        print(f'Warning: {target_col} has >1 unique value for {n_multi} PDBIDs. PDB-level MLP probe will use the mean per PDB. This is appropriate for reference_pIC50 but may be inappropriate for query_pIC50.')
    ref = df.groupby('pdbid', as_index=False).agg({target_col: 'mean', args.split_col: 'first'})
    by_split = {str(sp): g for sp, g in ref.groupby(args.split_col)}
    train_df = by_split.get('train', pd.DataFrame())
    if len(train_df) == 0:
        raise ValueError('Train split is empty for mlp_probe')
    print('=' * 100)
    print(f'Task        : mlp_probe')
    print(f'Graph mode  : {args.graph_mode}')
    print(f'Target col  : {target_col}  |  Split col: {args.split_col}')
    for sp, g in sorted(by_split.items()):
        print(f'  {sp:8s}: {len(g)} PDBs  mean_pIC50={g[target_col].mean():.3f}')
    if args.graph_mode == 'none':
        train_mean = float(train_df[target_col].mean())
        results = {}
        for sp, sub in sorted(by_split.items()):
            y = sub[target_col].values
            m = regression_metrics(y, np.full(len(y), train_mean))
            m['note'] = 'constant_mean_predictor'
            results[sp] = m
        return {'best_epoch': 0, 'best_valid_loss': float('nan'), **results}
    rand_map = make_rand_map(ref['pdbid'].tolist(), graph_dim, args.seed)
    rows = {}
    for sp, sub in by_split.items():
        rows[sp] = [{'pdbid': r['pdbid'], 'pic50': float(r[target_col])} for _, r in sub.iterrows()]
    train_X = []
    for r in rows.get('train', []):
        train_X.append(get_graph_vec(r['pdbid'], graph_map, rand_map, args.graph_mode, graph_dim))
    graph_std = Standardizer(np.stack(train_X, axis=0).astype(np.float32))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders = {}
    for sp in ['train', 'valid', 'test']:
        ds = GraphProbeDataset(rows.get(sp, []), graph_map, args.graph_mode, graph_dim, rand_map, graph_std)
        loaders[sp] = DataLoader(ds, batch_size=args.batch_size, shuffle=sp == 'train', num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    model = GraphMLPProbe(graph_dim=graph_dim, hidden=args.hidden_dim, dropout=args.dropout, n_layers=args.mlp_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    print(f'MLP hidden  : {args.hidden_dim}  layers={args.mlp_layers}  dropout={args.dropout}')
    print(f'Device      : {device}')
    best, best_epoch = (math.inf, -1)
    best_path = out_dir / 'best_model.pt'
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        ys, preds, total, n = ([], [], 0.0, 0)
        for g, y in loaders['train']:
            g, y = (g.to(device), y.to(device))
            p = model(g)
            loss = loss_fn(p, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(y)
            n += len(y)
            ys.append(y.detach().cpu().numpy())
            preds.append(p.detach().cpu().numpy())
        tr = regression_metrics(np.concatenate(ys), np.concatenate(preds))
        tr['loss'] = total / max(n, 1)
        va = eval_graph_probe(model, loaders['valid'], device)
        history.append({'epoch': epoch, 'train': tr, 'valid': va})
        if va['loss'] < best:
            best, best_epoch = (va['loss'], epoch)
            torch.save({'model': model.state_dict()}, best_path)
        if epoch % args.log_every == 0:
            print(f"epoch {epoch:04d} | train loss {tr['loss']:.4f} r {tr['pearson_r']:.4f} | valid loss {va['loss']:.4f} r {va['pearson_r']:.4f} | best {best:.4f}@{best_epoch}")
        if epoch - best_epoch >= args.patience:
            print(f'Early stopping at epoch {epoch}; best epoch {best_epoch}')
            break
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    final = {'best_epoch': best_epoch, 'best_valid_loss': float(best)}
    for sp in ['train', 'valid', 'test']:
        final[sp] = eval_graph_probe(model, loaders[sp], device)
    (out_dir / 'history.json').write_text(json.dumps(history, indent=2))
    return final

class AffinityDataset(Dataset):

    def __init__(self, rows: list, graph_map: dict, lig_map: dict, graph_mode: str, graph_dim: int, rand_map: dict, lig_std=None):
        self.rows = rows
        self.graph_map = graph_map
        self.lig_map = lig_map
        self.graph_mode = graph_mode
        self.graph_dim = graph_dim
        self.rand_map = rand_map
        self.lig_std = lig_std

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        g = get_graph_vec(r['pdbid'], self.graph_map, self.rand_map, self.graph_mode, self.graph_dim)
        if g is None:
            g = np.zeros(self.graph_dim, dtype=np.float32)
        lig = self.lig_map[r['smiles']]
        if self.lig_std is not None:
            lig = self.lig_std.transform(lig)
        return (torch.from_numpy(g), torch.from_numpy(lig), torch.tensor(r['pic50'], dtype=torch.float32))

class AffinityMLP(nn.Module):

    def __init__(self, graph_dim: int, lig_dim: int, graph_mode: str, g_hidden: int, l_hidden: int, hidden: int, dropout: float, n_layers: int):
        super().__init__()
        self.graph_mode = graph_mode

        def branch(in_d, out_d):
            return nn.Sequential(nn.Linear(in_d, out_d), nn.LayerNorm(out_d), nn.SiLU(), nn.Dropout(dropout), nn.Linear(out_d, out_d), nn.LayerNorm(out_d), nn.SiLU())
        merged = 0
        self.graph_branch = None
        if graph_mode != 'none':
            self.graph_branch = branch(graph_dim, g_hidden)
            merged += g_hidden
        self.lig_branch = branch(lig_dim, l_hidden)
        merged += l_hidden
        layers, d = ([], merged)
        for _ in range(max(n_layers - 1, 0)):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, g, lig):
        xs = []
        if self.graph_branch is not None:
            xs.append(self.graph_branch(g))
        xs.append(self.lig_branch(lig))
        return self.mlp(torch.cat(xs, dim=-1)).view(-1)

def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    loss_fn = nn.MSELoss()
    ys, preds, total, n = ([], [], 0.0, 0)
    with torch.no_grad():
        for g, lig, y in loader:
            g, lig, y = (g.to(device), lig.to(device), y.to(device))
            p = model(g, lig)
            total += float(loss_fn(p, y)) * len(y)
            n += len(y)
            ys.append(y.cpu().numpy())
            preds.append(p.cpu().numpy())
    m = regression_metrics(np.concatenate(ys), np.concatenate(preds))
    m['loss'] = total / max(n, 1)
    return m

def run_finetune(args, df: pd.DataFrame, graph_map: dict, graph_dim: int, out_dir: Path) -> dict:
    if not HAS_RDKIT:
        raise RuntimeError('RDKit が必要です')
    target_col = args.target_col or 'query_pIC50'
    smiles_col = args.smiles_col
    if target_col not in df.columns:
        raise ValueError(f'--target-col {target_col!r} が CSV に存在しません')
    if smiles_col not in df.columns:
        raise ValueError(f'--smiles-col {smiles_col!r} が CSV に存在しません')
    df = df.copy()
    df['pdbid'] = df['pdbid'].map(norm_pdb)
    df = df[df['pdbid'].isin(graph_map) & df[target_col].notna() & df[smiles_col].notna()].copy()
    rand_map = make_rand_map(df['pdbid'].unique().tolist(), graph_dim, args.seed)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=args.radius, fpSize=args.n_bits)
    smiles_list = sorted(df[smiles_col].unique())
    lig_map, bad = ({}, 0)
    for smi in smiles_list:
        feat = ligand_feature(smi, fpgen, args.n_bits)
        if feat is None:
            bad += 1
            feat = np.zeros(args.n_bits + 16, dtype=np.float32)
        lig_map[smi] = feat
    lig_dim = len(next(iter(lig_map.values())))
    splits: dict[str, list] = {'train': [], 'valid': [], 'test': []}
    for _, row in df.iterrows():
        sp = str(row[args.split_col])
        if sp in splits:
            splits[sp].append({'pdbid': row['pdbid'], 'smiles': row[smiles_col], 'pic50': float(row[target_col])})
    train_ligs = np.stack([lig_map[r['smiles']] for r in splits['train']], axis=0)
    lig_std = Standardizer(train_ligs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 100)
    print(f'Task        : finetune')
    print(f'Graph mode  : {args.graph_mode}')
    print(f'Target col  : {target_col}  |  SMILES col: {smiles_col}  |  Split: {args.split_col}')
    print(f"Examples    : train={len(splits['train'])}  valid={len(splits['valid'])}  test={len(splits['test'])}")
    print(f'Ligands     : {len(smiles_list)} unique  bad={bad}')
    print(f'Graph dim   : {graph_dim}  |  Lig dim: {lig_dim}')
    print(f'Device      : {device}')
    loaders = {}
    for sp in ['train', 'valid', 'test']:
        ds = AffinityDataset(splits[sp], graph_map, lig_map, args.graph_mode, graph_dim, rand_map, lig_std)
        loaders[sp] = DataLoader(ds, batch_size=args.batch_size, shuffle=sp == 'train', num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    model = AffinityMLP(graph_dim, lig_dim, args.graph_mode, args.graph_hidden_dim, args.ligand_hidden_dim, args.hidden_dim, args.dropout, args.mlp_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    best, best_epoch = (math.inf, -1)
    best_path = out_dir / 'best_model.pt'
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        ys, preds, total, n = ([], [], 0.0, 0)
        for g, lig, y in loaders['train']:
            g, lig, y = (g.to(device), lig.to(device), y.to(device))
            p = model(g, lig)
            loss = loss_fn(p, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(y)
            n += len(y)
            ys.append(y.detach().cpu().numpy())
            preds.append(p.detach().cpu().numpy())
        tr = regression_metrics(np.concatenate(ys), np.concatenate(preds))
        tr['loss'] = total / max(n, 1)
        va = eval_epoch(model, loaders['valid'], device)
        history.append({'epoch': epoch, 'train': tr, 'valid': va})
        if va['loss'] < best:
            best, best_epoch = (va['loss'], epoch)
            torch.save({'model': model.state_dict()}, best_path)
        if epoch % args.log_every == 0:
            print(f"epoch {epoch:04d} | train loss {tr['loss']:.4f} r {tr['pearson_r']:.4f} | valid loss {va['loss']:.4f} r {va['pearson_r']:.4f} | best {best:.4f}@{best_epoch}")
        if epoch - best_epoch >= args.patience:
            print(f'Early stopping at epoch {epoch}; best epoch {best_epoch}')
            break
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    final = {'best_epoch': best_epoch, 'best_valid_loss': float(best)}
    for sp in ['train', 'valid', 'test']:
        final[sp] = eval_epoch(model, loaders[sp], device)
    (out_dir / 'history.json').write_text(json.dumps(history, indent=2))
    return final

def main():
    ap = argparse.ArgumentParser(description='pIC50 prediction from GNN embeddings (zero_shot / finetune)')
    ap.add_argument('--task', choices=['zero_shot', 'mlp_probe', 'finetune'], required=True, help='zero_shot: Ridge線形プローブ  mlp_probe: PDB-level graph MLP probe  finetune: ligand+graph MLP訓練')
    ap.add_argument('--data-csv', required=True)
    ap.add_argument('--split-col', required=True, help='分割列名（例: split_pdbid, split_uniprot）')
    ap.add_argument('--graph-embedding-cache', required=True, help='export_graph_embeddings_from_pretrain_v8.py の出力 .pt ファイル')
    ap.add_argument('--graph-mode', choices=['pretrained', 'random', 'zero', 'none'], default='pretrained', help='pretrained: 事前学習埋め込み  random: ランダム初期化  zero: ゼロ  none: 使用しない')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--target-col', default=None, help='予測対象の pIC50 列名。デフォルト: zero_shot/mlp_probe→reference_pIC50, finetune→query_pIC50')
    ap.add_argument('--ridge-alpha', type=float, default=1.0, help='Ridge 回帰の正則化係数（zero_shot のみ）')
    ap.add_argument('--smiles-col', default='query_smiles_norm', help='クエリ SMILES 列名（finetune のみ）')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--weight-decay', type=float, default=0.001)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--hidden-dim', type=int, default=256)
    ap.add_argument('--graph-hidden-dim', type=int, default=256)
    ap.add_argument('--ligand-hidden-dim', type=int, default=256)
    ap.add_argument('--mlp-layers', type=int, default=2)
    ap.add_argument('--radius', type=int, default=2)
    ap.add_argument('--n-bits', type=int, default=2048)
    ap.add_argument('--log-every', type=int, default=1)
    args = ap.parse_args()
    seed_all(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data_csv)
    graph_map, graph_dim, cache_meta = load_graph_cache(args.graph_embedding_cache)
    print(f'Graph cache : {args.graph_embedding_cache}')
    print(f"  n_cache={cache_meta['n_cache']}  n_unique={cache_meta['n_unique']}  dim={graph_dim}")
    if args.task == 'zero_shot':
        results = run_zero_shot(args, df, graph_map, graph_dim)
        out = {'task': 'zero_shot', 'graph_mode': args.graph_mode, 'target_col': args.target_col or 'reference_pIC50', 'split_col': args.split_col, **results}
    elif args.task == 'mlp_probe':
        results = run_mlp_probe(args, df, graph_map, graph_dim, out_dir)
        out = {'task': 'mlp_probe', 'graph_mode': args.graph_mode, 'target_col': args.target_col or 'reference_pIC50', 'split_col': args.split_col, **results}
    else:
        results = run_finetune(args, df, graph_map, graph_dim, out_dir)
        out = {'task': 'finetune', 'graph_mode': args.graph_mode, 'target_col': args.target_col or 'query_pIC50', 'split_col': args.split_col, **results}
    (out_dir / 'metrics.json').write_text(json.dumps(out, indent=2))
    print('\nFinal evaluation')
    print(json.dumps(out, indent=2))
if __name__ == '__main__':
    main()
