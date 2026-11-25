"""Resume the BindingDB-only G/P1 comparison from epoch 100 to epoch 300."""
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
import screen_g_p1_bdb_20260914 as screen
ROOT = Path(__import__('os').environ['FMO_WORKDIR'])
DATA_ROOT = ROOT / 'dataset' / 'geo_pieda_matched_20260911'
START_EPOCH = 100
FINAL_EPOCH = 300
LOSS_NAMES = ('DIST', 'centroid', 'PIEDA_sum', 'ES', 'EX', 'CTmix', 'DI')

def component_losses(model, batch):
    """Return the unchanged aggregate losses plus their per-target MSEs."""
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

def print_epoch(epoch, end_epoch, train, valid, seconds, remaining):
    eta = (dt.datetime.now().astimezone() + dt.timedelta(seconds=remaining)).strftime('%H:%M:%S')
    print(f'Epoch {epoch:03d}/{end_epoch:03d} | {seconds:5.1f}s | ETA {eta}', flush=True)
    print(f"  Train | total={train['total']:.6f} | GEO={train['geometry']:.6f} (DIST={train['DIST']:.6f}, centroid={train['centroid']:.6f})", flush=True)
    print(f"        | PIEDA={train['PIEDA']:.6f} (sum={train['PIEDA_sum']:.6f}, ES={train['ES']:.6f}, EX={train['EX']:.6f}, CTmix={train['CTmix']:.6f}, DI={train['DI']:.6f})", flush=True)
    if valid is None:
        print('  Valid | - (evaluated every 5 epochs)', flush=True)
    else:
        print(f"  Valid | total={valid['total']:.6f} | GEO={valid['geometry']:.6f} (DIST={valid['DIST']:.6f}, centroid={valid['centroid']:.6f})", flush=True)
        print(f"        | PIEDA={valid['PIEDA']:.6f} (sum={valid['PIEDA_sum']:.6f}, ES={valid['ES']:.6f}, EX={valid['EX']:.6f}, CTmix={valid['CTmix']:.6f}, DI={valid['DI']:.6f})", flush=True)

def resume_pretrain(model, graphs, checkpoint, condition, out, config, end_epoch):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.0001)
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    screen.base.require(len(optimizer.param_groups) == 1, 'unexpected optimizer groups')
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                screen.base.require(bool(torch.isfinite(value).all()), 'nonfinite optimizer state')
    steps = int(checkpoint['steps'])
    expected_steps = START_EPOCH * int(np.ceil(len(graphs['train']) / 8))
    if len(graphs['train']) == 2970:
        screen.base.require(steps == expected_steps, f'unexpected checkpoint steps: {steps} != {expected_steps}')
    frozen_before = screen.base.digest_state({key: value for key, value in model.state_dict().items() if key.startswith(('edge_decoder_trunk.', 'component_heads.'))})
    timings = []
    for epoch in range(START_EPOCH + 1, end_epoch + 1):
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
        screen.base.json_line(out / 'training_continuation.jsonl', record)
        remaining = float(np.mean(timings[-10:])) * (end_epoch - epoch)
        print_epoch(epoch, end_epoch, train, valid, elapsed, remaining)
        if epoch % 25 == 0 or epoch == end_epoch:
            torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'config': config, 'epoch': epoch, 'steps': steps, 'resumed_from_epoch': START_EPOCH}, out / f'epoch_{epoch:03d}.pt')
    if condition == 'geo':
        frozen_after = screen.base.digest_state({key: value for key, value in model.state_dict().items() if key.startswith(('edge_decoder_trunk.', 'component_heads.'))})
        screen.base.require(frozen_before == frozen_after, 'GEO PIEDA head updated')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['G', 'P1'], required=True)
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
    source = ROOT / 'results' / 'head_screen_bdb_20260914_full' / f'{args.variant}_{args.condition}'
    checkpoint_path = source / 'epoch_100.pt'
    screen.base.require(checkpoint_path.is_file(), f'missing checkpoint: {checkpoint_path}')
    output_root = 'head_screen_bdb_20260914_resume_smoke' if args.mode == 'smoke' else 'head_screen_bdb_20260914_resume_to300'
    out = ROOT / 'results' / output_root / f'{args.variant}_{args.condition}'
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
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    screen.base.require(checkpoint.get('epoch') == START_EPOCH, 'checkpoint epoch is not 100')
    old_config = checkpoint.get('config', {})
    expected = {'version': 'v6', 'condition': args.condition, 'seed': 42, 'hidden_dim': 256, 'layers': 4, 'batch_size': 8, 'optimizer': 'AdamW', 'lr': 0.001, 'weight_decay': 0.0001, 'mask_ratio': 0.15, 'incident_mask': False, 'pieda_lambda': int(args.condition == 'geo_pieda')}
    for key, value in expected.items():
        screen.base.require(old_config.get(key) == value, f'checkpoint config mismatch: {key}')
    model = screen.make_model('v6', args.condition, args.variant)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    config = dict(old_config)
    config.update({'epochs': FINAL_EPOCH, 'resume_epoch': START_EPOCH, 'resume_checkpoint': str(checkpoint_path), 'resume_checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(), 'resume_job': os.environ.get('SLURM_JOB_ID', 'standalone'), 'resume_partition': os.environ.get('SLURM_JOB_PARTITION', 'standalone'), 'resume_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    screen.base.write_json(out / 'resume_config.json', config)
    end_epoch = START_EPOCH + 1 if args.mode == 'smoke' else FINAL_EPOCH
    model = resume_pretrain(model, graphs, checkpoint, args.condition, out, config, end_epoch)
    if args.mode == 'full':
        screen.run_bdb_probe(model, graphs, SimpleNamespace(mode='full'), out)
    screen.base.write_json(out / 'COMPLETE.json', {'all_finished': True, 'variant': args.variant, 'condition': args.condition, 'start_epoch': START_EPOCH, 'end_epoch': end_epoch, 'optimizer_restored': True, 'frozen_common_probe': args.mode == 'full', 'partition': 'aurum-g01'})
    print(f'COMPLETE {out}', flush=True)
if __name__ == '__main__':
    main()
