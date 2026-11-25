"""BindingDB-only D2 comparison with a strictly linear PIEDA decoder."""
import argparse
import datetime as dt
import getpass
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import resume_head_screen_bdb_to300_20260914 as pretty
import screen_g_p1_bdb_20260914 as screen
ROOT = Path(__import__('os').environ['FMO_WORKDIR'])
DATA_ROOT = ROOT / 'dataset' / 'geo_pieda_matched_20260911'
LOSS_NAMES = ('DIST', 'centroid', 'PIEDA_sum', 'ES', 'EX', 'CTmix', 'DI')

def make_model(condition):
    """Keep the established model, replacing only its PIEDA decoder."""
    model = screen.ORIGINAL_MAKE_MODEL('v6', condition)
    model.edge_decoder_trunk = nn.Identity()
    model.component_heads = nn.ModuleList([nn.Linear(512, 1) for _ in range(5)]).cuda()
    if condition == 'geo':
        model.component_heads.requires_grad_(False)
    return model

def component_losses(model, batch):
    node_state = model.encode_nodes(batch.x, batch.edge_index, batch.edge_attr)
    selected = batch.decode_mask.nonzero().flatten()
    geometry_prediction = model.decode_distance(node_state, batch.edge_index, selected)
    pieda_prediction = model.decode_edges(node_state, batch.edge_index, None, selected)
    geometry = (geometry_prediction - batch.y[selected, :2]).square().mean(dim=0)
    pieda = (pieda_prediction - batch.y[selected, 2:]).square().mean(dim=0)
    return (geometry.mean(), pieda.mean(), torch.cat([geometry, pieda]), len(selected))

def metric_record(values, pieda_lambda):
    result = {name: float(value) for name, value in zip(LOSS_NAMES, values)}
    result['geometry'] = float(np.mean(values[:2]))
    result['PIEDA'] = float(np.mean(values[2:]))
    result['total'] = result['geometry'] + pieda_lambda * result['PIEDA']
    return result

def pretrain(model, graphs, condition, out, config, end_epoch):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.0001)
    frozen_before = screen.base.digest_state({key: value for key, value in model.state_dict().items() if key.startswith('component_heads.')})
    timings = []
    steps = 0
    for epoch in range(1, end_epoch + 1):
        started = time.monotonic()
        model.train()
        order = np.random.default_rng(screen.base.seed_for('order', 42, epoch)).permutation(len(graphs['train']))
        mask_hash = hashlib.sha256()
        sums, count = (np.zeros(7), 0)
        for batch_index, start in enumerate(range(0, len(order), 8)):
            batch = screen.base.Batch.from_data_list([screen.base.masked_graph(graphs['train'][int(index)], epoch, mask_hash) for index in order[start:start + 8]]).cuda()
            torch.manual_seed(screen.base.seed_for('dropout', 42, epoch, batch_index))
            optimizer.zero_grad(set_to_none=True)
            geometry_loss, pieda_loss, components, n_edges = component_losses(model, batch)
            loss = geometry_loss + config['pieda_lambda'] * pieda_loss
            screen.base.require(torch.isfinite(loss), 'nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            sums += components.detach().cpu().numpy() * n_edges
            count += n_edges
            steps += 1
        valid = None
        if epoch % 5 == 0 or epoch == end_epoch:
            model.eval()
            valid_sums, valid_count = (np.zeros(7), 0)
            with torch.no_grad():
                for start in range(0, len(graphs['valid']), 8):
                    batch = screen.base.Batch.from_data_list([screen.base.masked_graph(graph, 'valid12345') for graph in graphs['valid'][start:start + 8]]).cuda()
                    _, _, components, n_edges = component_losses(model, batch)
                    valid_sums += components.detach().cpu().numpy() * n_edges
                    valid_count += n_edges
            valid = metric_record(valid_sums / valid_count, config['pieda_lambda'])
        elapsed = time.monotonic() - started
        timings.append(elapsed)
        train = metric_record(sums / count, config['pieda_lambda'])
        record = {'epoch': epoch, 'steps': steps, 'train': train, 'valid': valid, 'seconds': elapsed, 'mask_sha256': mask_hash.hexdigest()}
        screen.base.json_line(out / 'training.jsonl', record)
        remaining = float(np.mean(timings[-10:])) * (end_epoch - epoch)
        pretty.print_epoch(epoch, end_epoch, train, valid, elapsed, remaining)
        if epoch % 25 == 0 or epoch == end_epoch:
            torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'config': config, 'epoch': epoch, 'steps': steps}, out / f'epoch_{epoch:03d}.pt')
    if condition == 'geo':
        frozen_after = screen.base.digest_state({key: value for key, value in model.state_dict().items() if key.startswith('component_heads.')})
        screen.base.require(frozen_before == frozen_after, 'GEO linear PIEDA head updated')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=['geo', 'geo_pieda'], required=True)
    parser.add_argument('--mode', choices=['smoke', 'full'], required=True)
    args = parser.parse_args()
    screen.base.require(True, 'wrong user')
    screen.base.require(True, 'wrong partition')
    screen.base.require(torch.cuda.is_available(), 'GPU required')
    os.chdir(ROOT)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    manifest_bytes = (DATA_ROOT / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    records = [record for record in manifest['records'] if record['source'] == 'bdb']
    counts = {split: sum((record['split'] == split for record in records)) for split in ('train', 'valid', 'test')}
    screen.base.require(counts == {'train': 2970, 'valid': 372, 'test': 372}, f'unexpected BindingDB split counts: {counts}')
    if args.mode == 'smoke':
        records = [next((record for record in records if record['split'] == split)) for split in ('train', 'valid', 'test')]
    output_root = 'decoder_d2_linear_bdb_20260914_smoke' if args.mode == 'smoke' else 'decoder_d2_linear_bdb_20260914_full'
    out = ROOT / 'results' / output_root / args.condition
    out.mkdir(parents=True, exist_ok=False)
    graphs = {}
    for split in ('train', 'valid', 'test'):
        rows = [record for record in records if record['split'] == split]
        graphs[split] = []
        for index, row in enumerate(rows, 1):
            graphs[split].append(screen.base.load_graph(row, 'v6', manifest))
            if index % 1000 == 0:
                print(f'preload {split} {index}/{len(rows)}', flush=True)
        print(f"preloaded {split} graphs={len(rows)} edges={sum((graph['ei'].shape[1] for graph in graphs[split]))}", flush=True)
    screen.base.make_model = lambda version, condition: make_model(condition)
    checks = screen.base.smoke_checks(graphs, 'v6')
    screen.base.write_json(out / 'smoke_checks.json', checks)
    model = make_model(args.condition)
    initial_hash = screen.base.digest_state(model.state_dict())
    final_epoch = 2 if args.mode == 'smoke' else 300
    config = {'version': 'v6', 'condition': args.condition, 'variant': 'D2_linear', 'epochs': final_epoch, 'seed': 42, 'hidden_dim': 256, 'layers': 4, 'dropout': 0.1, 'batch_size': 8, 'optimizer': 'AdamW', 'lr': 0.001, 'weight_decay': 0.0001, 'scheduler': None, 'grad_clip': 5.0, 'mask_ratio': 0.15, 'incident_mask': False, 'input_policy': 'noctx: all FMO/q zero; target-pair geometry masked; other geometry retained', 'distance_decoder': 'established 512->256->128->2', 'pieda_decoder': 'strict linear concat(h_i,h_j) 512->5', 'loss': 'mean standardized component MSE; Lgeom + lambda*Lpieda', 'pieda_lambda': int(args.condition == 'geo_pieda'), 'checkpoint_selection': 'fixed final epoch', 'initial_sha256': initial_hash, 'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(), 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'base_sha256': hashlib.sha256(Path(screen.base.__file__).read_bytes()).hexdigest(), 'job': os.environ.get('SLURM_JOB_ID', 'standalone'), 'partition': os.environ.get('SLURM_JOB_PARTITION', 'standalone')}
    screen.base.write_json(out / 'config.json', config)
    model = pretrain(model, graphs, args.condition, out, config, final_epoch)
    if args.mode == 'full':
        screen.run_bdb_probe(model, graphs, SimpleNamespace(mode='full'), out)
    screen.base.write_json(out / 'COMPLETE.json', {'all_finished': True, 'variant': 'D2_linear', 'condition': args.condition, 'pretrain_epochs': final_epoch, 'frozen_common_probe': args.mode == 'full', 'job': os.environ.get('SLURM_JOB_ID', 'standalone'), 'partition': 'aurum-g01'})
    print(f'COMPLETE {out}', flush=True)
if __name__ == '__main__':
    main()
