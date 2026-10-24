"""Train original DLinear, then evaluate an independent raw-window kNN predictor."""
import argparse
import json
import hashlib
import inspect
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.knn_mts.data import DATASETS, GeneralData, file_hash
from src.knn_mts.panel import WEB_DATASETS, PanelData
from src.knn_mts.model import MODEL_PATH, make_model, forecast
from src.knn_mts.retrieval import METHODS, RawWindowBank, synchronize

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=DATASETS+WEB_DATASETS)
    p.add_argument('--panel-artifact', type=Path, help='verified frozen RQ1 200K export, required for web-scale')
    p.add_argument('--data-root', type=Path, default=ROOT.parent/'data')
    p.add_argument('--csv', type=Path, help='override CSV location; dataset split rules remain unchanged')
    p.add_argument('--output', type=Path, required=True, help='new directory; existing outputs are never overwritten')
    p.add_argument('--seq-len', type=int, default=720)
    p.add_argument('--pred-len', type=int, default=96)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-batch-size', type=int, default=32)
    p.add_argument('--learning-rate', type=float, default=.0001)
    p.add_argument('--weight-decay', type=float, default=0.)
    p.add_argument('--num-workers', type=int, default=10)
    p.add_argument('--lradj', choices=['type1','multistep'], default='type1')
    p.add_argument('--metric', choices=['pearson','squared_l2'], default='pearson')
    p.add_argument('--lr-milestones', nargs='*', type=int, default=[1, 50, 80])
    p.add_argument('--lr-gamma', type=float, default=.5)
    p.add_argument('--individual', action='store_true')
    p.add_argument('--method', choices=METHODS, default='KNN')
    p.add_argument('--value-transfer', choices=['none','query_last'], default='query_last',
                   help='query_last: candidate future - candidate past last + query past last')
    p.add_argument('--crest-pool', type=int, default=100)
    p.add_argument('--profile-sections', type=int, default=6)
    p.add_argument('--stp-weights', type=float, nargs=3, default=[.5,.25,.25])
    p.add_argument('--reliability-k', type=int, default=None, help='default: K')
    p.add_argument('--tau-v', type=float, default=1.0,
                   help='candidate-future distance temperature; no automatic calibration')
    p.add_argument('--sigma-min', type=float, default=.05)
    p.add_argument('--sigma-max', type=float, default=.30)
    p.add_argument('--mmr-chunk', type=int, default=1024)
    p.add_argument('--hnsw-m', type=int, default=32)
    p.add_argument('--hnsw-ef-construction', type=int, default=400)
    p.add_argument('--hnsw-ef-search', type=int, default=400)
    p.add_argument('--hnsw-build-threads', type=int, default=32)
    p.add_argument('--hnsw-search-threads', type=int, default=32)
    p.add_argument('--k', type=int, default=50)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--alpha', type=float, default=.2, help='lambda=alpha/(mean selected distance+alpha); Pearson distance=1-rho')
    p.add_argument('--retrieval-batch-size', type=int, default=1024, help='number of scalar-series queries, independent of training batch')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--retrieval-device', choices=['cpu', 'cuda'], help='default: same as model; no silent fallback')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--search-backend', choices=['auto','cpu_exact','faiss_cpu'], default='auto',
                   help='optional CPU Pearson search for KNN/KNN_CREST; auto preserves existing behavior')
    p.add_argument('--cpu-search-threads', type=int, default=32)
    p.add_argument('--faiss-nprobe', type=int, default=32)
    p.add_argument('--checkpoint', type=Path, help='evaluate an existing compatible checkpoint without training')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--artifact-mode', choices=['full','minimal'], default='minimal')
    p.add_argument('--shared-array-index', type=Path, help='content-hash index for immutable hard-linked arrays')
    return p


def source_hashes():
    paths = [Path(__file__).resolve(), MODEL_PATH, *sorted((ROOT).rglob('*.py'))]
    result = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    result['shared/candidate_reliability.py'] = file_hash(ROOT.parent/'utils/candidate_reliability.py')
    return result


def training_source_hashes():
    # Search/controller changes do not invalidate an independently trained model.
    paths = [MODEL_PATH, ROOT/'model.py', ROOT/'data.py', ROOT/'panel.py']
    result = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    result['training_functions'] = hashlib.sha256(
        (inspect.getsource(train)+inspect.getsource(model_validation)).encode()).hexdigest()
    return result


def score_arrays(prediction, target, mean, scale):
    error = prediction.astype(np.float64)-target.astype(np.float64)
    raw_error = error*(scale if scale.ndim == 3 else scale.reshape(1, 1, -1))
    return np.array([np.square(error).sum(), np.abs(error).sum(),
                     np.square(raw_error).sum(), np.abs(raw_error).sum(), error.size], dtype=np.float64)


def summarize(total):
    return dict(MSE=float(total[0]/total[4]), MAE=float(total[1]/total[4]),
                raw_MSE=float(total[2]/total[4]), raw_MAE=float(total[3]/total[4]), scalar_count=int(total[4]))


@torch.no_grad()
def model_validation(model, dataset, device, batch_size, num_workers=0):
    model.eval()
    squared, count = 0., 0
    for past, target, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers):
        pred = forecast(model, past.to(device))
        if not torch.isfinite(pred).all():
            raise RuntimeError('nonfinite validation prediction')
        error = pred.double()-target.to(device).double()
        squared += error.square().sum().item()
        count += error.numel()
    return squared/count


def train(model, data, args, out):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)
                 if args.lradj == 'multistep' else None)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(data.splits['train'], batch_size=args.batch_size, shuffle=True,
                        drop_last=False, generator=generator, num_workers=args.num_workers)
    history, best, stale = [], float('inf'), 0
    for epoch in range(1, args.epochs+1):
        model.train()
        started = time.perf_counter()
        total, count = 0., 0
        lr = optimizer.param_groups[0]['lr']
        for past, target, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = forecast(model, past.to(args.device))
            loss = torch.nn.functional.mse_loss(pred, target.to(args.device))
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            loss.backward()
            optimizer.step()
            total += loss.item()*target.numel()
            count += target.numel()
        validation = model_validation(model, data.splits['val'], args.device, args.eval_batch_size, args.num_workers)
        if validation <= best:
            best, stale = validation, 0
            torch.save(dict(model=model.state_dict(), epoch=epoch, best_val_MSE=best, data_audit=data.audit,
                            individual=args.individual, source_hashes=source_hashes(),
                            training_source_hashes=training_source_hashes(), seed=args.seed,
                            training_settings={k:getattr(args,k) for k in ('epochs','patience','batch_size','eval_batch_size',
                                'learning_rate','weight_decay','lradj','lr_milestones','lr_gamma','num_workers')}), out/'checkpoint.pt')
        else:
            stale += 1
        row = dict(epoch=epoch, train_MSE=total/count, val_MSE=validation, learning_rate=lr,
                   seconds=time.perf_counter()-started)
        history.append(row)
        write_json(out/'training_history.json', history)
        print('EPOCH', json.dumps(row), flush=True)
        if stale >= args.patience:
            break
        if scheduler is not None:
            scheduler.step()
        else:
            for group in optimizer.param_groups:
                group['lr'] = args.learning_rate * .5**(epoch-1)
    return history


@torch.no_grad()
def evaluate(model, bank, data, split, args, out):
    dataset = data.splits[split]
    directory = out/split
    directory.mkdir()
    shape = (len(dataset), args.pred_len, data.channels)
    arrays = {name: np.lib.format.open_memmap(directory/(name+'.npy'), mode='w+', dtype='float32', shape=shape)
              for name in ('backbone_pred', 'knn_pred', 'prediction', 'target')}
    ids = np.lib.format.open_memmap(directory/'neighbor_ids.npy', mode='w+', dtype='int64', shape=(len(dataset), data.channels, args.k))
    distances_out = np.lib.format.open_memmap(directory/('neighbor_pearson_distance.npy' if args.metric == 'pearson' else 'neighbor_squared_l2.npy'), mode='w+', dtype='float32', shape=ids.shape)
    gates = np.lib.format.open_memmap(directory/'mixing_lambda.npy', mode='w+', dtype='float32', shape=(len(dataset), data.channels))
    candidate_weights = np.lib.format.open_memmap(directory/'candidate_weights.npy', mode='w+', dtype='float32', shape=ids.shape)
    effective = np.lib.format.open_memmap(directory/'effective_candidates.npy', mode='w+', dtype='float32', shape=gates.shape)
    controller = (np.lib.format.open_memmap(directory/'crest_controller.npy', mode='w+', dtype='float32',
                   shape=(*gates.shape, 3)) if bank.crest else None)
    totals = {name: np.zeros(5) for name in ('backbone', 'knn', 'combined')}
    from .minimal_artifacts import Stats
    components = {name: Stats() for name in ('A_q', 'V_q', 'N_q_size')}
    model.eval()
    begin, model_seconds = 0, 0.
    old_timing = dict(bank.timings)
    started = time.perf_counter()
    for past, target, past_start in DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers):
        synchronize(args.device)
        forward_start = time.perf_counter()
        backbone = forecast(model, past.to(args.device)).cpu().numpy()
        synchronize(args.device)
        model_seconds += time.perf_counter()-forward_start
        queries = past.permute(0, 2, 1).contiguous().view(-1, args.seq_len)
        knn = np.empty((len(queries), args.pred_len), dtype=np.float32)
        lam = np.empty(len(queries), dtype=np.float32)
        found = np.empty((len(queries), args.k), dtype=np.int64)
        dists = np.empty((len(queries), args.k), dtype=np.float32)
        global_ids = (past_start[:, None]*data.channels+torch.arange(data.channels)[None, :]).reshape(-1)
        batch_weights = np.empty_like(dists)
        batch_effective = np.empty_like(lam)
        batch_controller = np.empty((len(queries), 3), dtype=np.float32) if bank.crest else None
        for first in range(0, len(queries), args.retrieval_batch_size):
            last = min(first+args.retrieval_batch_size, len(queries))
            value, mixing, labels, distances = bank.predict(queries[first:last], global_ids[first:last], split)
            knn[first:last], lam[first:last] = value.cpu().numpy(), mixing.cpu().numpy()
            found[first:last], dists[first:last] = labels.cpu().numpy(), distances.cpu().numpy()
            batch_weights[first:last] = bank.last['weights'].cpu().numpy()
            batch_effective[first:last] = bank.last['effective_candidates'].cpu().numpy()
            if bank.crest:
                batch_controller[first:last] = bank.last['controller'].cpu().numpy()
                for name, stats in components.items():
                    stats.add(bank.last['reliability_components'][name].cpu().numpy())
        B = len(past)
        knn = knn.reshape(B, data.channels, args.pred_len).transpose(0, 2, 1)
        lam = lam.reshape(B, data.channels)
        combined = (1-lam[:, None, :])*backbone + lam[:, None, :]*knn
        truth = target.numpy()
        if not all(np.isfinite(x).all() for x in (backbone, knn, combined, lam, dists)):
            raise RuntimeError('nonfinite evaluation output')
        end = begin+B
        for name, value in [('backbone_pred', backbone), ('knn_pred', knn), ('prediction', combined), ('target', truth)]:
            arrays[name][begin:end] = value
        ids[begin:end] = found.reshape(B, data.channels, args.k)
        distances_out[begin:end] = dists.reshape(B, data.channels, args.k)
        gates[begin:end] = lam
        candidate_weights[begin:end] = batch_weights.reshape(B, data.channels, args.k)
        effective[begin:end] = batch_effective.reshape(B, data.channels)
        if bank.crest:
            controller[begin:end] = batch_controller.reshape(B, data.channels, 3)
        for name, pred in [('backbone', backbone), ('knn', knn), ('combined', combined)]:
            scale = data.evaluation_scale(split, begin, end) if isinstance(data, PanelData) else data.scale
            totals[name] += score_arrays(pred, truth, data.mean, scale)
        begin = end
    for array in [*arrays.values(), ids, distances_out, gates, candidate_weights, effective]+([controller] if bank.crest else []):
        array.flush()
    if isinstance(data, PanelData):
        np.savez(directory/'query_identity.npz', **data.identities[split])
    else:
        np.savez(directory/'query_identity.npz', past_start=dataset.starts,
                 forecast_start=dataset.starts+args.seq_len, variable_names=np.asarray(data.columns))
    metrics = {name: summarize(total) for name, total in totals.items()}
    if isinstance(data, PanelData) and split == 'test':
        metrics.update(panel_test_metrics(data, arrays, directory))
        metrics.update(full_test_MSE=metrics['combined']['MSE'], full_test_MAE=metrics['combined']['MAE'])
    metrics['mixing_lambda'] = dict(mean=float(gates.mean(dtype=np.float64)), min=float(gates.min()), max=float(gates.max()))
    metrics['effective_candidates'] = dict(mean=float(effective.mean(dtype=np.float64)),
        min=float(effective.min()), max=float(effective.max()), fraction_below_1_01=float((effective < 1.01).mean()))
    if bank.crest:
        metrics['reliability_components'] = {name: stats.result() for name, stats in components.items()}
        metrics['crest_controller'] = {name: dict(mean=float(controller[..., i].mean(dtype=np.float64)),
            min=float(controller[..., i].min()), max=float(controller[..., i].max()))
            for i, name in enumerate(('cr_q','lambda_q','sigma_q'))}
    timing = dict(wall_seconds=time.perf_counter()-started, backbone_forward_seconds=model_seconds,
                  scalar_series_queries=len(dataset)*data.channels,
                  **{k: bank.timings[k]-old_timing[k] for k in bank.timings})
    write_json(directory/'metrics.json', metrics)
    write_json(directory/'timing.json', timing)
    print('EVALUATED', split, json.dumps(metrics), flush=True)
    return metrics, timing


def panel_test_metrics(data, arrays, directory):
    """Full and anchor metrics use identical saved predictions; targets never enter retrieval."""
    fixed = {name: np.zeros(5) for name in ('backbone', 'knn', 'combined')}
    for begin in range(0, len(data.fixed_mask), 1024):
        end = min(begin+1024, len(data.fixed_mask)); keep = data.fixed_mask[begin:end]
        if not keep.any():
            continue
        scale = data.evaluation_scale('test', begin, end)[keep]
        truth = arrays['target'][begin:end][keep]
        for name, key in [('backbone', 'backbone_pred'), ('knn', 'knn_pred'), ('combined', 'prediction')]:
            fixed[name] += score_arrays(arrays[key][begin:end][keep], truth, data.mean, scale)
    fixed = {k: summarize(v) for k, v in fixed.items()}
    np.save(directory/'fixed_test_mask.npy', data.fixed_mask)
    return dict(fixed=fixed, fixed_test_MSE=fixed['combined']['MSE'],
                fixed_test_MAE=fixed['combined']['MAE'], fixed_test_queries=int(data.fixed_mask.sum()),
                fixed_scalar_count=fixed['combined']['scalar_count'])


def run(args):
    if args.learning_rate != .0001 or list(args.stp_weights) != [.5,.25,.25]:
        raise ValueError("Public RQ1 fixes LR=1e-4 and CREST weights=(0.5,0.25,0.25)")
    for key in ('seq_len','pred_len','epochs','patience','batch_size','eval_batch_size','k','retrieval_batch_size','threads'):
        if getattr(args, key) < 1:
            raise ValueError(key+' must be positive')
    for key in ('learning_rate','temperature','lr_gamma','weight_decay','alpha'):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0 or (key in ('learning_rate','temperature','lr_gamma') and value == 0):
            raise ValueError('invalid '+key)
    if not 0 <= args.seed < 2**32 or any(v < 0 for v in args.lr_milestones):
        raise ValueError('invalid seed or scheduler milestones')
    if args.num_workers < 0:
        raise ValueError('num_workers must be nonnegative')
    args.retrieval_device = args.retrieval_device or args.device
    if not args.preflight_only and 'cuda' in (args.device, args.retrieval_device) and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: explicitly choose --device cpu; no automatic fallback')
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    path = args.csv or args.data_root/('ETT' if args.dataset.startswith('ETT') else 'experiment')/(args.dataset+('.csv' if args.dataset.startswith('ETT') else '_wide.csv'))
    panel = args.dataset in WEB_DATASETS
    if panel and args.panel_artifact is None:
        raise ValueError('web-scale requires --panel-artifact; use scripts/train_forecasting.py --framework knn_mts')
    data = (PanelData(args.panel_artifact, args.dataset, args.seq_len, args.pred_len) if panel
            else GeneralData(args.dataset, path, args.seq_len, args.pred_len))
    if args.k > data.audit['bank_candidates']:
        raise ValueError('K exceeds bank candidates')
    if 'CREST' in args.method:
        if not math.isfinite(args.tau_v) or args.tau_v <= 0:
            raise ValueError('tau_v must be finite and positive')
        if not args.k <= args.crest_pool <= data.audit['bank_candidates']:
            raise ValueError('CREST requires K <= pool <= candidate count')
        if (args.profile_sections < 2 or args.seq_len//args.profile_sections < 2 or args.mmr_chunk < 1
                or not 1 <= (args.reliability_k or args.k) <= args.crest_pool
                or not all(math.isfinite(v) and v >= 0 for v in args.stp_weights)
                or abs(sum(args.stp_weights)-1)>1e-6
                or not math.isfinite(args.sigma_min) or not math.isfinite(args.sigma_max)
                or not 0 < args.sigma_min <= args.sigma_max):
            raise ValueError('invalid CREST parameters')
    if 'SHARP' in args.method and min(args.hnsw_m,args.hnsw_ef_construction,args.hnsw_ef_search,
                                     args.hnsw_build_threads,args.hnsw_search_threads) < 1:
        raise ValueError('invalid HNSW parameters')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_json(out/'config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    write_json(out/'source_hashes.json', source_hashes())
    from threadpoolctl import threadpool_info
    import os
    write_json(out/'runtime.json', dict(torch=torch.__version__, numpy=np.__version__,
        torch_threads=torch.get_num_threads(), pools=threadpool_info(),
        environment={k:os.environ.get(k) for k in ('MKL_THREADING_LAYER','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}))
    write_json(out/'data_audit.json', data.audit)
    np.savez(out/'scaler.npz', mean=data.mean, scale=data.scale, columns=np.asarray(data.columns))
    if panel:
        np.savez(out/'candidate_identity.npz', **data.candidate_identity)
        np.save(out/'scaler_series_id.npy', data.series_ids)
    write_json(out/'status.json', dict(state='preflight_passed' if args.preflight_only else 'running'))
    if args.preflight_only:
        print('PREFLIGHT_PASS', json.dumps(data.audit), flush=True)
        return
    try:
        model = make_model(args.seq_len, args.pred_len, data.channels, args.individual).to(args.device)
        training_start = time.perf_counter()
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            history = train(model, data, args, out)
            checkpoint_path = out/'checkpoint.pt'
        training_seconds = time.perf_counter()-training_start if not args.checkpoint else 0.
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if checkpoint['data_audit'] != data.audit or checkpoint['individual'] != args.individual:
            raise ValueError('checkpoint data/scaler/architecture differs from requested run')
        source_ok = (checkpoint['training_source_hashes'] == training_source_hashes()
                     if 'training_source_hashes' in checkpoint else checkpoint['source_hashes'] == source_hashes())
        if panel:
            source_ok = source_ok and checkpoint['source_hashes'].get('panel.py') == file_hash(ROOT/'panel.py')
        if checkpoint.get('training_settings') != {k: getattr(args, k) for k in ('epochs','patience','batch_size','eval_batch_size','learning_rate','weight_decay','lradj','lr_milestones','lr_gamma','num_workers')}:
            raise ValueError('checkpoint training settings differ from requested run')
        if not source_ok or checkpoint['seed'] != args.seed:
            raise ValueError('checkpoint training source or seed differs from requested run')
        model.load_state_dict(checkpoint['model'])
        bank = RawWindowBank(data.bank_values if panel else data.values[:data.ends['train']], args.seq_len, args.pred_len,
                             args.retrieval_device, args.k, args.temperature, args.alpha,
                             candidate_starts=data.candidate_starts if panel else None,
                             candidate_series_ids=data.candidate_identity['source_id'] if panel else None,
                             candidate_local_starts=data.candidate_identity['past_start'] if panel else None,
                             **{key: getattr(args, key) for key in ('method','metric','value_transfer','seed','crest_pool','profile_sections',
                                'stp_weights','reliability_k','tau_v','sigma_min','sigma_max','mmr_chunk','hnsw_m',
                                'hnsw_ef_construction','hnsw_ef_search','hnsw_build_threads','hnsw_search_threads',
                                'search_backend','cpu_search_threads','faiss_nprobe')})
        write_json(out/'retrieval_runtime.json', dict(pools=threadpool_info(), search_backend=bank.search_backend,
            search_backend_info=bank.search_backend_info, controller_device=str(bank.device)))
        write_json(out/'retrieval_protocol.json', dict(method=args.method, metric=args.metric, value_transfer=args.value_transfer,
            search_device='cpu' if bank.sharp or bank.search_backend!='auto' else args.retrieval_device,
            search_backend=bank.search_backend,search_backend_info=bank.search_backend_info,
            controller_aggregation_device=args.retrieval_device,
            query_retrieval='validation/test only; no masked training retrieval',
            candidate_pool=bank.pool, candidate_unit='scalar-variable past window; global bank',
            bank_id_rule=data.audit['bank_id_rule'],
            mmr_relevance='rho' if args.metric == 'pearson' else '-squared_l2/temperature', mmr_redundancy='centered future cosine',
            gaussian='softmax(-(1-rho)**2/(2*sigma_q**2))' if args.metric == 'pearson' else 'softmax(-squared_l2/(2*sigma_q**2))',
            distance_definition='1-rho' if args.metric == 'pearson' else 'squared_l2',
            constant_window_policy='rho=0 if either centered norm <=1e-12 or window exactly constant',
            stp_input='original train-standardized windows, not Pearson-normalized keys',
            final_gate='alpha/(mean_final_selected_distance+alpha); alpha=0 -> zero',
            adaptive_two_stage='not invoked: no training queries or forbidden candidate intervals',
            controller_columns=['cr_q','lambda_q','sigma_q'],
            reliability_definition=bank.reliability_configuration,
            profile_preparation_batch=4096, checkpoint_training_source_hashes=checkpoint.get('training_source_hashes')))
        metrics, splits = {}, {}
        evaluator = evaluate
        if args.artifact_mode == 'minimal':
            from src.knn_mts.minimal_artifacts import evaluate_minimal
            evaluator = evaluate_minimal
        for split in ('val', 'test'):
            metrics[split], splits[split] = evaluator(model, bank, data, split, args, out)
        if panel:
            metrics['test'].update(full_test_MSE=metrics['test']['combined']['MSE'],
                                   full_test_MAE=metrics['test']['combined']['MAE'])
        write_json(out/'query_batch_protocol.json', dict(training_batch=args.batch_size, forecast_batch=args.eval_batch_size,
            query_batch=args.retrieval_batch_size, mmr_batch=args.mmr_chunk, calls=bank.query_batches))
        write_json(out/'metrics.json', metrics)
        write_json(out/'timing.json', dict(training_including_validation_seconds=training_seconds,
                   candidate_key_prepare_seconds=bank.prepare_seconds, preparation=bank.prepare_timing, splits=splits,
                   end_to_end_seconds=time.perf_counter()-started,
                   train_retrieval_seconds=None, train_retrieval_status='not_performed',
                   definition='end_to_end starts after CSV preprocessing; includes checkpoint, bank, val/test inference and artifact writes; nested split timers overlap'))
        write_json(out/'training_protocol.json', dict(loss='normalized unmasked MSE', best_epoch=checkpoint['epoch'],
                   best_val_MSE=checkpoint['best_val_MSE'], checkpoint=str(checkpoint_path.resolve()),
                   checkpoint_sha256=file_hash(checkpoint_path), checkpoint_reused=bool(args.checkpoint),
                   optimizer='Adam', use_amp=False, itr=1, training_settings=checkpoint.get('training_settings'),
                   epochs_completed=len(history) if not args.checkpoint else None,
                   retrieval_training=False, model_device=args.device, retrieval_device=args.retrieval_device,
                   torch=torch.__version__, numpy=np.__version__,
                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if 'cuda' in (args.device,args.retrieval_device) else None))
        write_json(out/'status.json', dict(state='complete'))
        if args.shared_array_index:
            from src.knn_mts.minimal_artifacts import deduplicate
            deduplicate(out, args.shared_array_index)
        write_json(out/'completion.json', dict(state='complete', hashes={str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'completion.json'}))
    except BaseException as error:
        write_json(out/'status.json', dict(state='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    # Must be set before the first CUDA context for deterministic GEMM.
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    run(parser().parse_args())
