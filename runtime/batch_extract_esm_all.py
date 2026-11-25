"""
batch_extract_esm_all.py

Loads the ESM model ONCE and processes all nodes.csv files in a directory.
Significantly faster than calling extract_esm_embedding.py once per protein.

Usage:
  python batch_extract_esm_all.py     --nodes-dir dataset/nodes_v4_fmofrag     --out-dir   dataset/embeddings     --model-name facebook/esm2_t12_35M_UR50D
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
STANDARD_AA = set(list('ACDEFGHIKLMNPQRSTVWY'))

def build_chain_sequences(nodes):
    chain_col = 'rep_chain' if 'rep_chain' in nodes.columns else 'chain'
    aa1_col = 'rep_aa1' if 'rep_aa1' in nodes.columns else 'aa1'
    chain_items = []
    for chain, g in nodes.groupby(chain_col, sort=False):
        row_indices, seq_chars = ([], [])
        for idx, row in g.iterrows():
            aa1 = str(row.get(aa1_col, 'X')).strip().upper()
            is_std = int(row.get('is_standard_aa', int(aa1 in STANDARD_AA)))
            if is_std == 1 and aa1 in STANDARD_AA:
                row_indices.append(idx)
                seq_chars.append(aa1)
        if seq_chars:
            chain_items.append({'chain': chain, 'row_indices': row_indices, 'sequence': ''.join(seq_chars)})
    return chain_items

@torch.no_grad()
def embed_sequence(sequence, tokenizer, model, device):
    encoded = tokenizer(sequence, return_tensors='pt', add_special_tokens=True)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    outputs = model(**encoded)
    token_embeddings = outputs.last_hidden_state
    residue_embeddings = token_embeddings[0, 1:len(sequence) + 1, :]
    return residue_embeddings.detach().cpu().numpy()

def process_one(nodes_path, out_npy, tokenizer, model, device, max_length):
    nodes = pd.read_csv(nodes_path).reset_index(drop=True)
    has_v9 = {'rep_chain', 'rep_aa1'}.issubset(nodes.columns)
    has_leg = {'chain', 'aa1'}.issubset(nodes.columns)
    if not (has_v9 or has_leg):
        raise ValueError(f'Missing columns: need (rep_chain+rep_aa1) or (chain+aa1). Got: {list(nodes.columns)}')
    chain_items = build_chain_sequences(nodes)
    if not chain_items:
        print('  SKIP: no standard AA residues')
        return False
    embeddings = None
    has_esm = np.zeros(len(nodes), dtype=np.int64)
    for item in chain_items:
        seq = item['sequence']
        row_indices = item['row_indices']
        if len(seq) > max_length:
            print(f"  WARN chain {item['chain']}: {len(seq)} > {max_length}, truncating")
            seq = seq[:max_length]
            row_indices = row_indices[:max_length]
        emb = embed_sequence(seq, tokenizer, model, device)
        if emb.shape[0] != len(row_indices):
            raise RuntimeError(f"Embedding length mismatch chain {item['chain']}: {emb.shape[0]} vs {len(row_indices)}")
        if embeddings is None:
            embeddings = np.zeros((len(nodes), emb.shape[1]), dtype=np.float32)
        embeddings[np.array(row_indices), :] = emb.astype(np.float32)
        has_esm[np.array(row_indices)] = 1
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, embeddings)
    print(f'  Saved {int(has_esm.sum())}/{len(nodes)} embedded nodes → {out_npy}')
    return True

def main():
    ap = argparse.ArgumentParser(description='Batch ESM extraction: loads model once, processes all nodes.csv files.')
    ap.add_argument('--nodes-dir', required=True, help='dataset/nodes_v4_fmofrag/')
    ap.add_argument('--out-dir', required=True, help='dataset/embeddings/')
    ap.add_argument('--model-name', default='facebook/esm2_t12_35M_UR50D', help='HuggingFace ESM model name')
    ap.add_argument('--device', default='auto', help='auto | cpu | cuda | cuda:0')
    ap.add_argument('--max-length', type=int, default=1022, help='Max residues per chain (ESM-2 limit). Longer chains are truncated.')
    ap.add_argument('--force', action='store_true', default=False, help='Recompute even if output .npy already exists')
    args = ap.parse_args()
    nodes_dir = Path(args.nodes_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_files = sorted(nodes_dir.glob('*_nodes.csv'))
    print(f'Found {len(nodes_files)} nodes.csv files in {nodes_dir}')
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Loading model: {args.model_name}')
    print(f'Device: {device}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.to(device)
    model.eval()
    print('Model loaded.')
    n_done = 0
    n_skip = 0
    n_error = 0
    for i, nodes_path in enumerate(nodes_files, 1):
        pdbid = nodes_path.stem.replace('_nodes', '')
        out_npy = out_dir / f'{pdbid}_esm.npy'
        if not args.force and out_npy.exists():
            n_skip += 1
            continue
        print(f'[{i}/{len(nodes_files)}] {pdbid}')
        try:
            ok = process_one(nodes_path, out_npy, tokenizer, model, device, args.max_length)
            if ok:
                n_done += 1
            else:
                n_skip += 1
        except Exception as e:
            print(f'  ERROR {pdbid}: {e}')
            n_error += 1
        if i % 100 == 0:
            import datetime
            print(f'Progress {i}/{len(nodes_files)} — {datetime.datetime.now()}')
    print(f'\nFinished: done={n_done}, skipped={n_skip}, errors={n_error}')
if __name__ == '__main__':
    main()
