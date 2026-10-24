"""Serial panel retrieval, final RAFT tensor cache, and run-level timing."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

import torch


def prepare_panel_retrieval(model, configs, train_data, valid_data, test_data):
    """Keep the index alive through all splits and optional Recall diagnostics."""
    started = time.perf_counter()
    datasets = dict(train=train_data, valid=valid_data, test=test_data)
    tool = model.rt
    tool.prepare_dataset(train_data)
    try:
        if tool.crest_enabled:
            tool.prepare_panel_crest_state(
                train_data, cache_root=getattr(configs, 'cache_root', './retrieval-cache')
            )
        if tool.crest_enabled:
            model.reliability_configuration = {
                'formula': 'cr_q=A_q*V_q', 'neighborhood': 'stp_ranked_spaced_up_to_n',
                'min_spacing': tool.pred_len, 'shortfall_policy': 'use_available',
                'tau_v': tool.tau_v, 'value_scale': tool.value_scale.tolist(),
            }
        model.retrieval_dict = {}
        for split, data in datasets.items():
            payload = tool.retrieve_all(data, train=split == 'train', split=split)
            if getattr(data, 'forecast_scaler', None) is not None:
                from src.utils.panel_forecast import query_scale_payload_
                query_scale_payload_(payload, data)
            model.retrieval_dict[split] = payload
    finally:
        if tool.crest_enabled:
            tool.cleanup_panel_crest_state()
    panel = train_data.memory_report
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
    run_name = f'{getattr(configs, "panel_dataset", "panel")}-{run_id}'
    cache_dir = Path(getattr(configs, 'cache_root', './retrieval-cache')) / run_name
    log_dir = Path(getattr(configs, 'log_dir', './logs'))
    cache_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / 'retrieval.pt'
    write_start = time.perf_counter()
    # The baseline model addresses these CPU tensors as [period, query, H, C].
    # torch.save streams the existing storages without another full-result copy.
    torch.save(model.retrieval_dict, cache_path)
    cache_write_seconds = time.perf_counter() - write_start
    production_end = time.perf_counter()
    bundle = getattr(configs, '_panel_data_bundle', None)
    production_start = getattr(bundle, 'production_started_at', started)
    report = {
        'run_id': run_id,
        'retrieval_backend': tool.retrieval_backend,
        'exact_key_bank_seconds': tool.prepare_runtime['exact_key_bank_ms'] / 1000.0,
        'exact_search_seconds': sum(r['exact_search_ms'] for r in tool.split_runtime.values()) / 1000.0,
        'search_seconds': sum(r['hnsw_search_ms'] + r['exact_search_ms']
                              for r in tool.split_runtime.values()) / 1000.0,
        'panel': dict(panel),
        'dataset_load_seconds': panel.get('dataset_load_seconds', 0.0),
        'panel_index_setup_seconds': panel.get('panel_index_setup_seconds', 0.0),
        'hnsw_build_seconds': tool.prepare_runtime['hnsw_build_ms'] / 1000.0,
        'query_generation_seconds': sum(r['query_generation_ms'] for r in tool.split_runtime.values()) / 1000.0,
        'hnsw_search_seconds': sum(r['hnsw_search_ms'] for r in tool.split_runtime.values()) / 1000.0,
        'masking_seconds': sum(r['masking_fallback_ms'] for r in tool.split_runtime.values()) / 1000.0,
        'aggregation_seconds': sum(r['baseline_aggregation_ms'] + r['gaussian_aggregation_ms'] for r in tool.split_runtime.values()) / 1000.0,
        'final_cache_write_seconds': cache_write_seconds,
        'retrieval_wall_seconds': production_end - production_start,
        'hnsw_recall_enabled': tool.hnsw_recall_check,
        'cache_path': str(cache_path.resolve()),
        'cache_shapes': {split: list(value.shape) for split, value in model.retrieval_dict.items()},
        'configuration': {
            name: getattr(tool, name) for name in (
                'seq_len', 'pred_len', 'channels', 'n_period', 'topm',
                'temperature', 'hnsw_m', 'hnsw_ef_construction', 'hnsw_ef_search',
                'hnsw_seed', 'hnsw_build_num_threads', 'hnsw_search_num_threads',
                'hnsw_build_batch_size', 'retrieval_batch_size',
                'retrieval_aggregation_chunk_size', 'hnsw_oversample_factor',
                'candidate_search_strategy',
            )
        },
        'split_runtime': tool.split_runtime,
    }
    train_runtime = tool.split_runtime.get('train', {})
    masked = train_runtime.get('period_queries', 0)
    fallbacks = train_runtime.get('wide_fallbacks', 0)
    report.update(masked_training_queries=masked, wide_fallback_count=fallbacks,
                  fallback_ratio=fallbacks/masked if masked else 0.0)
    if tool.crest_enabled:
        report.update({
            'crest_prepare_seconds': tool.prepare_runtime['profile_bank_iqr_ms'] / 1000.0,
            **{f'crest_{name}_seconds': sum(
                r[field] for r in tool.split_runtime.values()
            ) / 1000.0 for name, field in (
                ('profile', 'profile_reliability_ms'), ('mmr', 'mmr_ms'),
                ('gaussian_aggregation', 'gaussian_aggregation_ms'),
            )},
        })
        report['configuration'].update({
            name: getattr(tool, name) for name in (
                'retrieval_controller', 'crest_weighting', 'crest_profile', 'mmr_redundancy_source',
                'mmr_candidate_pool', 'profile_sections', 'profile_reliability_k',
                'stp_weights', 'mean_profile_temperature', 'tau_v', 'sigma_min', 'sigma_max',
                'lambda_min', 'lambda_max', 'mmr_value_repr', 'mmr_paa_segments',
                'mmr_rp_dim', 'mmr_rp_seed', 'value_mmr_chunk_size', 'retrieval_seed',
            )
        })
        report['configuration'].update({
            name: str(getattr(tool, name)) for name in (
                'retrieval_postprocess_device', 'mmr_device',
            )
        })
    if tool.crest_enabled:
        report['reliability_configuration'] = model.reliability_configuration
        report['configuration'].update({
            'reliability_formula': model.reliability_configuration['formula'],
            'reliability_neighborhood': model.reliability_configuration['neighborhood'],
            'reliability_min_spacing': tool.pred_len,
            'reliability_shortfall_policy': 'use_available',
            'value_scale': model.reliability_configuration['value_scale'],
        })
    # Save production accounting before the optional potentially long diagnostic.
    report_path = log_dir / f'{run_name}.json'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    model.hnsw_recall_results = {}
    if tool.retrieval_backend == 'hnsw' and tool.hnsw_recall_check:
        recall_started = time.perf_counter()
        model.hnsw_recall_results = tool.run_hnsw_recall(datasets)
        results = [r for split in model.hnsw_recall_results.values() for r in split]
        report.update({
            'hnsw_recall_hnsw_seconds': sum(r['hnsw_diagnostic_ms'] for r in results) / 1000.0,
            'hnsw_recall_exact_seconds': sum(r['exact_reference_ms'] for r in results) / 1000.0,
            'hnsw_recall_total_seconds': time.perf_counter() - recall_started,
            'hnsw_recall_results': {
                split: [{key: (value.tolist() if key == 'query_ids' else value)
                         for key, value in result.items()
                         if key not in ('exact_labels', 'exact_similarities', 'hnsw_labels')}
                        for result in split_results]
                for split, split_results in model.hnsw_recall_results.items()
            },
        })
    report['program_wall_seconds'] = time.perf_counter() - getattr(
        configs, '_program_started_at', production_start
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    model.retrieval_runtime = report
    model.retrieval_report_path = str(report_path.resolve())
    print(
        'Panel retrieval complete: '
        f"load={report['dataset_load_seconds']:.3f}s, "
        f"build={report['hnsw_build_seconds'] + report['exact_key_bank_seconds']:.3f}s, "
        f"query={report['query_generation_seconds']:.3f}s, "
        f"search={report['search_seconds']:.3f}s, "
        f"aggregation={report['aggregation_seconds']:.3f}s, "
        f"wall={report['retrieval_wall_seconds']:.3f}s; "
        f"Recall={'on' if tool.hnsw_recall_check else 'off'}"
    )
    if tool.crest_enabled:
        print(
            'Panel CREST: '
            f"prepare={report['crest_prepare_seconds']:.3f}s, "
            f"profile={report['crest_profile_seconds']:.3f}s, "
            f"MMR={report['crest_mmr_seconds']:.3f}s, "
            f"Gaussian/aggregation={report['crest_gaussian_aggregation_seconds']:.3f}s"
        )
    print(f'Final cache: {cache_path.resolve()}')
    print(f'Runtime report: {report_path.resolve()}')
    del model.rt
