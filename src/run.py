import argparse
import hashlib
import random
import time

import numpy as np
import torch

from src.exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from src.data_provider.webscale_panel_adapter import (
    DEFAULT_PANEL_GROUPED_ROOT,
    PANEL_SPLIT_PRESETS,
    get_panel_split_preset,
    resolve_panel_root,
)
from src.utils.panel_crest import validate_panel_crest_options
from src.utils.value_mmr import validate_value_mmr_args, value_mmr_setting_suffix
from src.utils.profile_reliability import (
    profile_setting_suffix,
    resolved_reliability_k,
    validate_profile_args,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Modular exact/SHARP retrieval with optional CREST for RAFT'
    )
    parser.add_argument('--task_name', default='long_term_forecast',
                        choices=['long_term_forecast'])
    parser.add_argument('--is_training', type=int, default=1, choices=[0, 1])
    parser.add_argument('--retrieval_only', action='store_true')
    parser.add_argument('--stage', choices=['all', 'retrieval'], default='all')
    parser.add_argument('--cache_root', default='./retrieval-cache')
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--model_id', required=True)
    parser.add_argument('--model', default='RAFT', choices=['RAFT'])

    parser.add_argument('--data', required=True,
                        choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'custom', 'panel'])
    parser.add_argument('--root_path', default='')
    parser.add_argument('--data_path', default='')
    parser.add_argument('--features', default='M', choices=['M', 'S', 'MS'])
    parser.add_argument('--target', default='OT')
    parser.add_argument('--checkpoints', default='./checkpoints')
    parser.add_argument('--inverse', action='store_true')
    parser.add_argument('--main_protocol', action='store_true')
    parser.add_argument('--main_method', choices=['RAFT', 'RAFT_SHARP', 'RAFT_CREST', 'RAFT_SHARP_CREST'])
    parser.add_argument('--main_dataset')
    parser.add_argument('--canonical_root', default='./data/web/processed')
    parser.add_argument('--crest_weighting', choices=['gaussian', 'adaptive_softmax'], default='gaussian')
    parser.add_argument('--min_delta', type=float, default=0.0)

    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--enc_in', type=int, required=True)

    parser.add_argument('--n_period', type=int, default=3, choices=range(1, 6))
    parser.add_argument('--topm', type=int, default=20)
    parser.add_argument('--hnsw_m', type=int, default=16)
    parser.add_argument('--hnsw_ef_construction', type=int, default=200)
    parser.add_argument('--hnsw_ef_search', type=int, default=200)
    parser.add_argument('--hnsw_seed', type=int, default=0)
    parser.add_argument('--retrieval_batch_size', type=int, default=1024)
    parser.add_argument(
        '--retrieval_backend', choices=['exact', 'hnsw'], default='hnsw'
    )
    parser.add_argument('--hnsw_oversample_factor', type=int, default=5)
    parser.add_argument(
        '--candidate_search_strategy',
        default='two_stage',
        choices=['two_stage', 'wide'],
        help=(
            'training-mask candidate search: SHARP two-stage search or naive '
            'wide search for every query'
        ),
    )
    parser.add_argument('--runtime_breakdown', action='store_true')
    parser.add_argument('--retrieval_postprocess_device',
                        choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--mmr_device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument(
        '--mmr_redundancy_source', choices=['key', 'value'], default='value'
    )
    parser.add_argument('--mmr_value_repr', choices=['raw', 'paa', 'random_projection'],
                        default='raw')
    parser.add_argument('--mmr_paa_segments', type=int, default=8)
    parser.add_argument('--mmr_rp_dim', type=int, default=32)
    parser.add_argument('--mmr_rp_seed', type=int, default=0)
    parser.add_argument('--value_mmr_chunk_size', type=int, default=8)
    parser.add_argument('--retrieval_aggregation_chunk_size', type=int, default=32)
    parser.add_argument('--hnsw_build_batch_size', type=int, default=4096)
    parser.add_argument('--hnsw_build_num_threads', type=int, default=-1)
    parser.add_argument('--hnsw_search_num_threads', type=int, default=32)
    parser.add_argument('--memory_report', action='store_true')

    parser.add_argument(
        '--hnsw_recall_check', action='store_true', default=False,
        help='exact HNSW recall: full queries for general data, sampled for panel',
    )
    parser.add_argument('--no_hnsw_recall_check', action='store_false', dest='hnsw_recall_check')
    parser.add_argument('--hnsw_recall_queries', type=int, default=100,
                        help='panel query sample size; ignored for general recall')
    parser.add_argument('--hnsw_recall_seed', type=int, default=0)
    parser.add_argument('--hnsw_recall_splits', default='valid',
                        help='comma-separated train,valid,test (valid maps to provider val)')
    parser.add_argument(
        '--hnsw_recall_candidate_chunk_size', type=int, default=4096
    )
    parser.add_argument(
        '--hnsw_recall_query_batch_size', type=int, default=32
    )
    parser.add_argument(
        '--hnsw_recall_device', choices=['cpu', 'cuda'], default='cpu'
    )
    parser.add_argument('--hnsw_recall_cache_dir', default='',
                        help='panel exact cache directory; general recall is uncached')


    panel = parser.add_argument_group('web-scale panel')
    panel.add_argument('--panel_root', default='')
    panel.add_argument('--panel_exact_max_candidates', type=int, default=6000000,
                       help='panel exact candidate limit (1..6000000); HNSW is unaffected')
    panel.add_argument(
        '--panel_group', default='',
        help='existing group directory name; overrides --panel_root',
    )
    panel.add_argument(
        '--panel_grouped_root', default=DEFAULT_PANEL_GROUPED_ROOT,
        help='parent of <panel_dataset>/<panel_group> canonical directories',
    )
    panel.add_argument(
        '--panel_dataset', choices=sorted(PANEL_SPLIT_PRESETS)
    )
    subset = panel.add_mutually_exclusive_group()
    subset.add_argument('--series_fraction', type=float)
    subset.add_argument('--max_series', type=int)
    panel.add_argument('--series_subset_seed', type=int, default=0)
    panel.add_argument('--series_subset_path', default='')
    panel.add_argument('--candidate_stride', type=int, default=1)
    panel.add_argument('--query_stride', type=int, default=1)

    parser.add_argument('--retrieval_controller',
                        choices=['baseline', 'crest', 'mean_profile'],
                        default='baseline')
    parser.add_argument('--mmr_candidate_pool', type=int, default=100)
    parser.add_argument('--crest_profile', choices=['stp', 'mean'], default='stp')
    parser.add_argument('--stp_state_weight', type=float, default=0.5)
    parser.add_argument('--stp_dispersion_weight', type=float, default=0.25)
    parser.add_argument('--stp_transition_weight', type=float, default=0.25)
    parser.add_argument('--profile_sections', type=int, default=6)
    parser.add_argument('--profile_reliability_k', type=int, default=-1)
    parser.add_argument('--mean_profile_temperature', type=float, default=1.0)
    parser.add_argument('--tau_v', type=float, default=1.0,
                        help='candidate-future distance temperature (fixed; no automatic calibration)')
    parser.add_argument('--sigma_min', type=float, default=0.05)
    parser.add_argument('--sigma_max', type=float, default=0.30)
    parser.add_argument('--lambda_min', type=float, default=0.30)
    parser.add_argument('--lambda_max', type=float, default=0.90)
    parser.add_argument('--fusion_mode', choices=['concat_projection'],
                        default='concat_projection')

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lradj', default='type1', choices=['type1', 'constant'])
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--des', default='sharp')

    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu')
    parser.add_argument('--gpu', type=int, default=0)
    return parser


def _safe_setting_name(setting, max_length=220):
    """Keep one path component below common filesystem NAME_MAX limits."""
    if len(setting.encode('utf-8')) <= max_length:
        return setting

    digest = hashlib.sha1(setting.encode('utf-8')).hexdigest()[:12]
    suffix = f'__{digest}'
    prefix_length = max_length - len(suffix)
    shortened = setting[:prefix_length].rstrip('._-') + suffix
    print(
        'Experiment setting exceeded the safe filename length; '
        f'using shortened setting: {shortened}'
    )
    return shortened


def experiment_setting(args, iteration):
    data_identifier = args.data
    if args.data == 'panel':
        data_identifier = f'panel_{args.panel_dataset}'
    setting = (
        f'{args.model_id}_{args.model}_{data_identifier}_ft{args.features}_'
        f'sl{args.seq_len}_pl{args.pred_len}_np{args.n_period}_'
        f'topm{args.topm}_{args.des}_{iteration}'
    )
    setting += f'_backend{args.retrieval_backend}'
    if args.retrieval_controller != 'baseline':
        setting += profile_setting_suffix(args)
        setting += (
            f'_red{args.mmr_redundancy_source}'
            f'_vchunk{args.value_mmr_chunk_size}'
        )
        setting += value_mmr_setting_suffix(args)
        setting += f"_crAxV_spaceH_tauv{getattr(args, 'tau_v', 1.0)}"
    return _safe_setting_name(setting)


def print_args(args):
    print('SHARP configuration')
    print(f'  device: {args.device}')
    if args.data == 'panel':
        preset = get_panel_split_preset(args.panel_dataset)
        if getattr(args, 'panel_group', None):
            print(f'  Panel dataset: {args.panel_dataset}')
            print(f'  Panel group: {args.panel_group}')
            print(f'  Panel root: {args.panel_root}')
        else:
            print(f'  Panel dataset: {args.panel_dataset} ({args.panel_root})')
        print('  Split protocol: fixed absolute endpoints')
        print(f"  Protocol description: {preset['protocol']}")
        print(f"  Series length: {preset['expected_length']}")
        print(
            f"  Train: [0,{preset['train_end']}) = "
            f"{preset['train_end']}"
        )
        print(
            f"  Validation target: [{preset['train_end']},"
            f"{preset['valid_end']}) = "
            f"{preset['valid_end'] - preset['train_end']}"
        )
        print(
            f"  Test target: [{preset['valid_end']},"
            f"{preset['test_end']}) = "
            f"{preset['test_end'] - preset['valid_end']}"
        )
        print(
            '  Validation context starts at: '
            f"{preset['train_end'] - args.seq_len}"
        )
        print(
            '  Test context starts at: '
            f"{preset['valid_end'] - args.seq_len}"
        )
        selection = (
            f'fraction={args.series_fraction}' if args.series_fraction is not None
            else f'max_series={args.max_series}' if args.max_series is not None
            else 'all series'
        )
        print(
            f'  panel selection: {selection}, seed={args.series_subset_seed}, '
            f'subset_file={args.series_subset_path or "none"}'
        )
        print(
            f'  panel stride: candidate_stride={args.candidate_stride}, '
            f'query_stride={args.query_stride}'
        )
        print(
            f'  panel streaming: build_batch={args.hnsw_build_batch_size}, '
            f'memory_report={args.memory_report}'
        )
    else:
        print(f'  dataset: {args.data} ({args.root_path}/{args.data_path})')
    print(f'  features: {args.features}, channels: {args.enc_in}')
    print(
        f'  windows: seq={args.seq_len}, label={args.label_len}, '
        f'pred={args.pred_len}'
    )
    crest_enabled = args.retrieval_controller != 'baseline'
    target_count = (
        args.topm if args.retrieval_controller == 'baseline'
        else args.mmr_candidate_pool
    )
    max_forbidden = 2 * (args.seq_len + args.pred_len) - 1
    if args.data == 'panel':
        max_forbidden = (
            (2 * (args.seq_len + args.pred_len) - 2) // args.candidate_stride
            + 1
        )
    print(
        f"  Search backend: {'HNSW (SHARP)' if args.retrieval_backend == 'hnsw' else 'Exact'}"
    )
    print(f"  CREST: {'Enabled' if crest_enabled else 'Disabled'}")
    print(
        f'  retrieval: controller={args.retrieval_controller}, '
        f'periods={args.n_period}, candidate-pool={target_count}, top-m={args.topm}'
    )
    if args.retrieval_backend == 'hnsw' and args.candidate_search_strategy == 'two_stage':
        print(
            '  retrieval search: strategy=two_stage, '
            f'narrow_k=min(n_train, {args.hnsw_oversample_factor * target_count}), '
            f'wide_k=min(n_train, candidate-pool + '
            f'{max_forbidden}), '
            f'batch={args.retrieval_batch_size}'
        )
    elif args.retrieval_backend == 'hnsw':
        print(
            '  retrieval search: strategy=wide, '
            f'k=min(n_train, candidate-pool + '
            f'{max_forbidden}), '
            f'batch={args.retrieval_batch_size}'
        )
    print(f'  fusion: {args.fusion_mode}')
    if crest_enabled:
        reliability_k = resolved_reliability_k(
            args.profile_reliability_k, args.topm
        )
        print(
            '  CREST controller: '
            f'profile={args.crest_profile}, '
            f'M={args.mmr_candidate_pool}, R={reliability_k}, '
            f'K={args.topm}, redundancy={args.mmr_redundancy_source}, '
            f'value_chunk={args.value_mmr_chunk_size}'
            f'{value_mmr_setting_suffix(args)}'
        )
        print(
            '  retrieval postprocess: '
            f'device={args.retrieval_postprocess_device}, '
            f'mmr_device={args.mmr_device}, '
            f'aggregation_chunk={args.retrieval_aggregation_chunk_size}, '
            f'breakdown={args.runtime_breakdown}'
        )
    else:
        print(
            '  CREST disabled; profile, MMR, Gaussian weighting, and '
            'MMR redundancy options are inactive'
        )
    if (crest_enabled and args.retrieval_postprocess_device == 'cpu'
            and args.mmr_device == 'cuda'):
        print(
            '  warning: CPU profile + CUDA MMR requires extra CPU/GPU transfers; '
            'use --retrieval_postprocess_device cuda for the recommended GPU path'
        )
    if args.retrieval_backend == 'hnsw':
        print(
            f'  HNSW: M={args.hnsw_m}, '
            f'ef_construction={args.hnsw_ef_construction}, '
            f'ef_search={args.hnsw_ef_search}'
        )
    print(
        f'  training: epochs={args.train_epochs}, batch={args.batch_size}, '
        f'lr={args.learning_rate}, seed={args.seed}'
    )


def main():
    program_started_at = time.perf_counter()
    args = build_parser().parse_args()
    args._program_started_at = program_started_at
    if args.main_protocol:
        from src.utils.main_protocol import validate_main_args
        validate_main_args(args)
    if args.data != 'panel' and args.crest_weighting != 'gaussian':
        raise ValueError('adaptive_softmax is a panel weighting mode; General uses gaussian')
    args.retrieval_only = args.retrieval_only or args.stage == 'retrieval'
    if args.hnsw_recall_check and args.retrieval_backend != 'hnsw':
        raise ValueError('HNSW recall requires retrieval_backend=hnsw')
    if args.hnsw_build_batch_size < 1:
        raise ValueError('hnsw_build_batch_size must be >= 1')
    if args.candidate_stride < 1 or args.query_stride < 1:
        raise ValueError('panel candidate/query strides must be >= 1')
    if args.data == 'panel':
        if args.panel_dataset is None:
            raise ValueError('--panel_dataset is required for panel data')
        resolve_panel_root(args)
        if not args.panel_root:
            raise ValueError('--panel_root is required for panel data')
        validate_panel_crest_options(args)
    elif not args.root_path or not args.data_path:
        raise ValueError('--root_path and --data_path are required for legacy data')
    validate_profile_args(args)
    validate_value_mmr_args(args)
    if args.retrieval_aggregation_chunk_size < 1:
        raise ValueError('retrieval_aggregation_chunk_size must be >= 1')
    if args.value_mmr_chunk_size < 1:
        raise ValueError('value_mmr_chunk_size must be >= 1')
    if args.retrieval_controller == 'mean_profile':
        args.retrieval_controller = 'crest'
        args.crest_profile = 'mean'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    use_cuda = args.use_gpu and torch.cuda.is_available()
    args.device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
    if args.use_amp and not use_cuda:
        print('AMP requested without CUDA; disabling AMP')
        args.use_amp = False

    print_args(args)
    for iteration in range(args.itr if args.is_training else 1):
        setting = experiment_setting(args, iteration)
        if args.main_protocol:
            setting = _safe_setting_name('main4way_' + args.main_dataset + '_' + args.main_method + '_' + setting)
        experiment = Exp_Long_Term_Forecast(args)
        if args.retrieval_only:
            print('Retrieval-only benchmark complete; skipping training and evaluation.')
            return
        if args.is_training:
            print(f'>>>>>>> start training: {setting} >>>>>>>')
            experiment.train(setting)
            print(f'<<<<<<< testing: {setting} <<<<<<<')
            experiment.test(setting)
        else:
            print(f'<<<<<<< testing: {setting} <<<<<<<')
            experiment.test(setting, test=1)
        if use_cuda:
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
