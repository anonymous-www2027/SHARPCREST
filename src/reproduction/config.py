"""Paper defaults with explicit overrides; no seed/horizon whitelist."""
import argparse
import json
import sys
from pathlib import Path
import torch
from src.run import build_parser
from src.utils.profile_reliability import validate_profile_args
from src.utils.value_mmr import validate_value_mmr_args
from src.utils.panel_crest import validate_panel_crest_options

ROOT = Path(__file__).resolve().parents[2]
# Search and augmentation are independent switches for the four-way study.
METHODS = {'RAFT': ('exact', 'baseline'), 'SHARP': ('hnsw', 'baseline'),
           'CREST': ('exact', 'crest'), 'S&C': ('hnsw', 'crest')}
ALIASES = {'RAFT_SHARP': 'SHARP', 'RAFT_CREST': 'CREST', 'RAFT_SHARP_CREST': 'S&C'}
GENERAL = {'ETTh1': 7, 'ETTh2': 7, 'ETTm1': 7, 'ETTm2': 7, 'Exchange': 8, 'Weather': 21}
WEB = {'quito_hour': ('quito_corpus_hour', 576, 288, 24),
       'extended_wikipedia': ('extended_wikipedia', 576, 7, 7),
       'tailedts_202401': ('tailedts_202401', 168, 48, 168)}


def parse_args(argv=None):
    parser = build_parser()
    for action in parser._actions:
        if action.dest in ('data', 'model_id', 'enc_in'):
            action.required = False
    parser.add_argument('--config', type=Path, help='JSON defaults; explicit CLI options take precedence')
    parser.add_argument('--dataset', choices=list(WEB) + list(GENERAL), default='extended_wikipedia')
    parser.add_argument('--method', choices=list(METHODS) + list(ALIASES), default='RAFT')
    parser.add_argument('--scale', type=int, default=200000, help='Published whole-series cohort label, not an exact window cap')
    parser.add_argument('--cohort_file', type=Path, help='Explicit compatible cohort JSON instead of a bundled cohort')
    parser.add_argument('--data_root', type=Path, default=ROOT/'data',
                        help='Root containing general/ and web/; defaults to repository data/')
    parser.add_argument('--output', type=Path, help='New output directory; existing nonempty directories are rejected')
    parser.add_argument('--cpu_threads', type=int, default=4, help='CPU threads outside retrieval, including training')
    parser.add_argument('--retrieval_cpu_threads', type=int, default=32, help='CPU/BLAS threads within complete Model.prepare_dataset')
    parser.add_argument('--dry_run', action='store_true', help='Resolve configuration without reading data or retrieving')
    parser.add_argument('--prepare_only', action='store_true', help='Verify data, cohort and fixed-test identities without retrieval')
    parser.add_argument('--save_predictions', action='store_true')
    known, _ = parser.parse_known_args(argv)
    config = json.loads(known.config.read_text()) if known.config else {}
    if not isinstance(config, dict):
        parser.error('config must be a JSON object')
    actions = {a.dest: a for a in parser._actions}
    unknown = set(config) - set(actions)
    if unknown:
        parser.error('unknown config keys: ' + ', '.join(sorted(unknown)))
    # Apply argparse types and choices to JSON values as well as CLI arguments.
    for key, value in list(config.items()):
        action = actions[key]
        try:
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                if not isinstance(value, bool): raise ValueError('expected boolean')
            elif action.type is not None:
                if isinstance(value, bool): raise ValueError('expected non-boolean value')
                value = action.type(value)
            if action.choices is not None and value not in action.choices:
                raise ValueError('outside allowed choices')
        except (ValueError, TypeError) as ex:
            parser.error(f'invalid config {key}: {ex}')
        config[key] = value
    parser.set_defaults(**config)
    known, _ = parser.parse_known_args(argv)
    web = known.dataset in WEB
    device = 'cuda' if known.use_gpu and torch.cuda.is_available() else 'cpu'
    defaults = dict(model_id=known.dataset, data='panel' if web else known.dataset if known.dataset.startswith('ETT') else 'custom',
        enc_in=1 if web else GENERAL[known.dataset], features='M', seq_len=WEB[known.dataset][1] if web else 720,
        pred_len=WEB[known.dataset][2] if web else 96, label_len=0 if web else 48, n_period=1,
        topm=40, hnsw_m=32, hnsw_ef_construction=400, hnsw_ef_search=400, hnsw_seed=None,
        mmr_candidate_pool=100, profile_reliability_k=40, train_epochs=100 if web else 10,
        patience=10 if web else 0, crest_weighting='adaptive_softmax' if web else 'gaussian',
        runtime_breakdown=True, retrieval_postprocess_device=device, mmr_device=device,
        canonical_root=str(known.data_root/'web/processed'), panel_grouped_root='',
        root_path=str(known.data_root/'general'/('ETT' if known.dataset.startswith('ETT') else 'experiment')),
        data_path=known.dataset+'.csv' if known.dataset.startswith('ETT') else known.dataset+'_wide.csv')
    parser.set_defaults(**defaults)
    parser.set_defaults(**config)
    a = parser.parse_args(argv)
    a.method = ALIASES.get(a.method, a.method)
    tokens = sys.argv[1:] if argv is None else argv
    explicit = set(config)
    for token in tokens:
        action = parser._option_string_actions.get(token.split('=')[0])
        if action is not None:
            explicit.add(action.dest)
    bundled = a.data_root/'web'
    if web and 'canonical_root' not in explicit and not Path(a.canonical_root).exists():
        if (bundled/a.dataset/'package.json').is_file():
            a.canonical_root = str(bundled)
    for key, value in zip(('retrieval_backend', 'retrieval_controller'), METHODS[a.method]):
        if key in explicit and getattr(a, key) != value:
            parser.error(key + ' conflicts with the selected method')
    a.retrieval_backend, a.retrieval_controller = METHODS[a.method]
    a.main_method = {'RAFT':'RAFT', 'SHARP':'RAFT_SHARP', 'CREST':'RAFT_CREST', 'S&C':'RAFT_SHARP_CREST'}[a.method]
    a.main_dataset = a.dataset
    # This selects production best-validation/RNG/normalization semantics; the
    # historical fixed-value CLI validator is intentionally not used here.
    a.main_protocol = True
    a.hnsw_seed = a.seed if a.hnsw_seed is None else a.hnsw_seed
    a.device = torch.device(f'cuda:{a.gpu}' if a.use_gpu and torch.cuda.is_available() else 'cpu')
    a.retrieval_only = a.retrieval_only or a.stage == 'retrieval'
    a.panel_dataset = WEB[a.dataset][0] if web else None
    a.panel_root = a.canonical_root if web else ''
    if web and a.cohort_file is None:
        a.cohort_file = ROOT/'configs/cohorts'/a.dataset/f'{a.scale}.json'
    for key in ('topm','mmr_candidate_pool','profile_reliability_k','cpu_threads','retrieval_cpu_threads','hnsw_m',
                'hnsw_ef_construction','hnsw_ef_search','hnsw_oversample_factor','retrieval_batch_size',
                'value_mmr_chunk_size','retrieval_aggregation_chunk_size','hnsw_build_batch_size',
                'hnsw_search_num_threads','train_epochs','batch_size','seq_len','pred_len'):
        if getattr(a,key) < 1: parser.error(key+' must be positive')
    if a.hnsw_build_num_threads != -1 and a.hnsw_build_num_threads < 1:
        parser.error('hnsw_build_num_threads must be -1 or positive')
    if not (0 <= a.seed < 2**32 and 0 <= a.hnsw_seed < 2**32):
        parser.error('seeds must be integers in [0, 2**32 - 1]')
    if a.retrieval_controller == 'crest' and not (a.topm <= a.mmr_candidate_pool and a.profile_reliability_k <= a.mmr_candidate_pool):
        parser.error('CREST requires final K <= candidate M and reliability R <= candidate M')
    if a.features != 'M' or a.enc_in != (1 if web else GENERAL[a.dataset]):
        parser.error('RQ1 uses features=M and the dataset channel count')
    if a.data != ('panel' if web else a.dataset if a.dataset.startswith('ETT') else 'custom'):
        parser.error('data loader must match dataset')
    if a.inverse or a.use_amp or a.itr != 1 or not a.is_training:
        parser.error('this entrypoint runs one non-AMP train/evaluate experiment; use a new invocation for each seed')
    if a.series_fraction is not None or a.max_series is not None or a.series_subset_path or a.panel_group:
        parser.error('use a whole-series cohort_file; random panel subset overrides are not supported here')
    if a.candidate_stride != 1 or a.query_stride != 1:
        parser.error('published nested protocol uses stride=1')
    if not web and a.crest_weighting != 'gaussian': parser.error('General requires gaussian weighting')
    if web and a.label_len != 0: parser.error('Panel label_len must be 0')
    if a.learning_rate <= 0 or a.patience < 0 or a.min_delta < 0: parser.error('invalid training policy')
    if a.device.type == 'cpu' and ('cuda' in (a.mmr_device,a.retrieval_postprocess_device)):
        parser.error('CUDA retrieval requested with CPU execution')
    validate_profile_args(a); validate_value_mmr_args(a)
    if web: validate_panel_crest_options(a)
    return a
