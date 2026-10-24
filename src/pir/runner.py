"""Explicit, isolated PIR four-way entry point; one condition per invocation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from src.pir.options import build_parser as native_parser
from src.pir.utils.pir_results import write_json, file_hashes


WEB_INPUT_LENGTHS = {'quito_corpus_hour': 576, 'quito_hour': 576,
                     'quito_corpus_min': 576, 'quito_min': 576,
                     'extended_wikipedia': 576, 'tailedts_202401': 168}


def resolve_defaults(args):
    dataset = None
    if args.panel_manifest:
        dataset = json.loads(Path(args.panel_manifest).read_text())['dataset']
    if args.seq_len is None:
        if args.panel_manifest and dataset not in WEB_INPUT_LENGTHS:
            raise ValueError('unknown panel dataset requires explicit --seq_len')
        args.seq_len = WEB_INPUT_LENGTHS[dataset] if args.panel_manifest else 96
    if args.sharp_c is None:
        args.sharp_c = 5.0 if args.panel_manifest else 2.0
    return args


class ProtocolParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        return resolve_defaults(super().parse_args(args, namespace))


def build_parser():
    p = ProtocolParser(parents=[native_parser()], add_help=False)
    p.description = __doc__
    p.set_defaults(retrieval_num=50, seq_len=None, load_pretrained_backbone=0)
    p.add_argument('--retrieval_method', choices=['PIR', 'PIR_SHARP', 'PIR_CREST', 'PIR_SHARP_CREST'], default='PIR')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--torch_threads', type=int, default=8)
    p.add_argument('--hnsw_threads', type=int, default=32)
    p.add_argument('--hnsw_build_threads', type=int, default=32)
    p.add_argument('--retrieval_batch_size', type=int, default=1024,
                   help='query batch for one-time retrieval preparation, independent of training batch')
    p.add_argument('--retrieval_execution', choices=['prepared', 'inline'], default='prepared')
    p.add_argument('--hnsw_m', type=int, default=32)
    p.add_argument('--hnsw_ef_construction', type=int, default=400)
    p.add_argument('--hnsw_ef_search', type=int, default=400)
    p.add_argument('--sharp_c', type=float, default=None,
                   help='narrow search multiplier: default general=2, panel=5')
    p.add_argument('--crest_pool', type=int, default=100)
    p.add_argument('--tau_v', type=float, default=1.0,
                   help='candidate-future distance temperature; no automatic calibration')
    p.add_argument('--reliability_r', type=int, default=40)
    p.add_argument('--profile_sections', type=int, default=6)
    p.add_argument('--stp_weights', type=float, nargs=3, default=[.5, .25, .25])
    p.add_argument('--panel_root', default='')
    p.add_argument('--panel_manifest', default='')
    p.add_argument('--panel_anchor_manifest', default='')
    p.add_argument('--panel_channel', type=int, default=0)
    p.add_argument('--pretrained_backbone_path', default='')
    p.add_argument('--shared_pretraining_cache', default='',
                   help='share native pretraining model/RNG boundary across methods; empty disables sharing')
    p.add_argument('--result_dir', required=True)
    p.add_argument('--preflight_only', action='store_true')
    p.add_argument('--cpu', action='store_true')
    return p


def validate(args):
    resolve_defaults(args)
    if not math.isfinite(args.tau_v) or args.tau_v <= 0:
        raise ValueError("tau_v must be finite and positive")
    if args.learning_rate != .0001 or args.refine_lr != .0001 or list(args.stp_weights) != [.5,.25,.25]:
        raise ValueError("Public RQ1 fixes LR=1e-4 and CREST weights=(0.5,0.25,0.25)")
    if args.model != 'PIR' or args.retrieval_num != 50 or args.retrieval_stride != 1:
        raise ValueError('agreed PIR protocol requires model=PIR, K=50, stride=1')
    if args.seq_len < 16 or (not args.panel_manifest and args.seq_len != 96):
        raise ValueError('requires general L=96 or panel L>=16')
    if args.panel_manifest:
        dataset = json.loads(Path(args.panel_manifest).read_text())['dataset']
        if dataset in WEB_INPUT_LENGTHS and args.seq_len != WEB_INPUT_LENGTHS[dataset]:
            raise ValueError('panel input length differs from frozen dataset protocol')
    if args.use_multi_gpu or args.use_amp or args.itr != 1 or not args.is_training:
        raise ValueError('this entry point supports single-device FP32 training, one condition per invocation')
    if args.use_norm != 1 or args.label_len != 0 or not math.isfinite(args.sharp_c) or args.sharp_c < 1:
        raise ValueError('requires native normalization, label_len=0, finite c>=1')
    if args.percent != 100:
        raise ValueError('whole training split is required')
    if args.backbone not in ('PatchTST',):
        raise ValueError('unsupported PIR backbone')
    if args.pred_len < 1 or not 2 <= args.profile_sections <= args.seq_len//2:
        raise ValueError('invalid horizon or profile section count')
    if args.crest_pool < args.retrieval_num or not 1 <= args.reliability_r <= args.crest_pool:
        raise ValueError('requires M >= K and 1 <= R <= M')
    if any(not math.isfinite(w) or w < 0 for w in args.stp_weights) or not math.isclose(sum(args.stp_weights), 1, abs_tol=1e-6):
        raise ValueError('STP weights must be nonnegative, finite and sum to one')
    if args.seed < 0 or args.seed >= 2**32:
        raise ValueError('seed must be in [0,2**32)')
    for key in ('batch_size', 'retrieval_batch_size', 'torch_threads', 'hnsw_threads', 'hnsw_build_threads', 'train_epochs', 'refine_epochs'):
        if getattr(args, key) < 1:
            raise ValueError(key+' must be positive')
    if args.hnsw_m < 2 or args.hnsw_ef_search < 1 or args.hnsw_ef_construction < args.hnsw_m:
        raise ValueError('invalid HNSW construction/search settings')
    if args.load_pretrained_backbone not in (0, 2):
        raise ValueError('use native pretraining (0), or explicit backbone-only checkpoint (2)')
    if args.load_pretrained_backbone == 2 and not Path(args.pretrained_backbone_path).is_file():
        raise ValueError('load_pretrained_backbone=2 requires an existing --pretrained_backbone_path')
    if args.shared_pretraining_cache and args.load_pretrained_backbone != 0:
        raise ValueError('shared pretraining requires load_pretrained_backbone=0')
    if args.panel_manifest:
        if not args.panel_root or not args.panel_anchor_manifest:
            raise ValueError('panel requires canonical root, cohort and anchor manifests')
        if args.enc_in != 1 or args.dec_in != 1 or args.c_out != 1:
            raise ValueError('web-scale requires channel dimensions 1')
    elif args.data not in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'exchange_rate', 'weather', 'custom'):
        raise ValueError('unsupported general dataset')


def source_record():
    root = Path(__file__).resolve().parent
    paths = [*root.glob('*.py'), *root.glob('models/*.py'), *root.glob('layers/*.py'),
             *root.glob('data_provider/*.py'), *root.glob('exp/*.py'), *root.glob('utils/*.py'), *root.glob('retrieval/*.py')]
    record = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    shared = root.parent/'reproduction/model_data.py'
    record['shared/candidate_reliability.py'] = hashlib.sha256((Path(__file__).resolve().parents[1]/'utils/candidate_reliability.py').read_bytes()).hexdigest()
    record['shared/model_data.py'] = hashlib.sha256(shared.read_bytes()).hexdigest()
    return record


def run(args):
    validate(args)
    root = Path(args.result_dir).resolve()
    # Never overwrite an older experiment, checkpoint or partial run.
    root.mkdir(parents=True, exist_ok=False)
    args.result_dir = str(root)
    args.checkpoints = str(root/'checkpoints')
    args.bakcbone_checkpoints = str(root/'backbone')
    args.output_index = 1
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; explicitly choose --cpu')
    args.use_gpu = not args.cpu
    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    write_json(root/'config.json', vars(args))
    write_json(root/'source_hashes.json', source_record())
    write_json(root/'runtime.json', {'torch': torch.__version__, 'numpy': np.__version__,
               'device': torch.cuda.get_device_name(args.gpu) if args.use_gpu else 'cpu',
               'cuda_version': torch.version.cuda, 'torch_threads': torch.get_num_threads(),
               'hnsw_build_threads': args.hnsw_build_threads, 'hnsw_query_threads': args.hnsw_threads,
               'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
               'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
               'thread_environment': {name: os.environ.get(name) for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}})
    write_json(root/'status.json', {'state': 'running'})
    started = time.perf_counter()
    try:
        from src.pir.exp.exp_long_term_forecasting_pir import Exp_Long_Term_Forecast_PIR
        from src.pir.data_provider.pir_panel import candidate_metadata, array_hash
        exp = Exp_Long_Term_Forecast_PIR(args)
        # Preflight all splits before starting an optimizer. Preserve RNG so
        # metadata checks do not change native minibatch order.
        data_audit = {}
        for flag in ('train', 'retrieval', 'val', 'test'):
            ds, _ = exp._get_data(flag)
            if len(ds) < 1:
                raise ValueError('empty '+flag+' split')
            sample = ds[0]
            if sample[0].shape != (args.seq_len, args.enc_in) or sample[1].shape != (args.pred_len, args.enc_in):
                raise ValueError('dataset/model dimensions disagree')
            metadata = candidate_metadata(ds)
            data_audit[flag] = {'count': len(ds), 'series_id_hash': array_hash(metadata['series_id']), 'past_start_hash': array_hash(metadata['start'])}
            if flag == 'retrieval':
                _, counts = np.unique(metadata['series_id'], return_counts=True)
                worst_mask = int(np.minimum(counts, 2*args.seq_len+1).max())
                target = args.crest_pool if 'CREST' in args.retrieval_method else args.retrieval_num
                if len(ds)-worst_mask < target:
                    raise ValueError('not enough eligible candidates for every training query')
                data_audit[flag]['minimum_eligible_candidates'] = len(ds)-worst_mask
                data_audit[flag]['key_value_float32_bytes'] = len(ds)*(args.seq_len+args.pred_len)*args.enc_in*4
        if args.panel_manifest:
            write_json(root/'panel_manifest.json', args._pir_panel_bundle.audit())
        else:
            data_audit['source_csv_sha256'] = hashlib.sha256((Path(args.root_path)/args.data_path).read_bytes()).hexdigest()
        write_json(root/'data_audit.json', data_audit)
        if args.preflight_only:
            write_json(root/'status.json', {'state': 'preflight_passed', 'training_started': False})
            return
        train_start = time.perf_counter()
        exp.train('model')
        if args.use_gpu:
            torch.cuda.synchronize(exp.device)
        train_seconds = time.perf_counter()-train_start
        test_start = time.perf_counter()
        exp.test('model')
        if args.use_gpu:
            torch.cuda.synchronize(exp.device)
        write_json(root/'wall_timing.json', {'train_pipeline_seconds': train_seconds,
                   'final_test_seconds': time.perf_counter()-test_start,
                   'end_to_end_seconds': time.perf_counter()-started,
                   'note': 'train_pipeline includes pretraining, bank/index construction, refine training and per-epoch validation/test; not retrieval-only timing'})
        write_json(root/'completion.json', {'state': 'complete', 'hashes': file_hashes(root)})
        write_json(root/'status.json', {'state': 'complete'})
        print('PIR_FOURWAY_SUCCESS', args.retrieval_method, (root/'metrics.json').read_text(), flush=True)
    except Exception:
        write_json(root/'status.json', {'state': 'failed', 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    run(build_parser().parse_args())
