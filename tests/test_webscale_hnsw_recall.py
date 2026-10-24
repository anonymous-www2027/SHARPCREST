from unittest import mock

import numpy as np
import pytest
import torch

from src.common.webscale_ts import InMemorySeriesBank, PanelWindowDataset, WindowManifest
from src.layers.Retrieval import RetrievalTool
from src.run import build_parser


def recall_fixture(cache_dir='', seed=0, enabled=True, topm=3):
    generator = np.random.default_rng(123)
    series_count = 4
    length = 48
    series_ids = np.array([2, 7, 19, 31], dtype=np.int64)
    values = generator.normal(size=(series_count * length, 1)).astype(np.float32)
    offsets = np.arange(series_count + 1, dtype=np.int64) * length
    lengths = np.full(series_count, length, dtype=np.int64)
    bank = InMemorySeriesBank(values, series_ids, offsets, lengths)
    candidates = WindowManifest.from_bounds(
        bank, seq_len=6, pred_len=2, stride=1, end_bound=36
    )
    train_queries = WindowManifest.from_bounds(
        bank, seq_len=6, pred_len=2, stride=1, end_bound=36
    )
    valid_queries = WindowManifest.from_bounds(
        bank, seq_len=6, pred_len=2, stride=1,
        start_bound=34, end_bound=48,
    )
    report = {
        'dataset_identifier': '/synthetic/panel',
        'panel_dataset': 'synthetic',
        'candidate_stride': 1,
        'query_stride': 1,
    }
    datasets = {
        'train': PanelWindowDataset(
            bank, train_queries, candidates, split='train', memory_report=report
        ),
        'valid': PanelWindowDataset(
            bank, valid_queries, candidates, split='valid', memory_report=report
        ),
    }
    tool = RetrievalTool(
        seq_len=6,
        pred_len=2,
        channels=1,
        n_period=1,
        topm=topm,
        hnsw_m=8,
        hnsw_ef_construction=80,
        hnsw_ef_search=40,
        hnsw_seed=5,
        retrieval_backend='hnsw',
        retrieval_controller='baseline',
        retrieval_batch_size=11,
        hnsw_build_batch_size=13,
        hnsw_recall_check=enabled,
        hnsw_recall_queries=9,
        hnsw_recall_seed=seed,
        hnsw_recall_splits=('train', 'valid'),
        hnsw_recall_candidate_chunk_size=17,
        hnsw_recall_query_batch_size=4,
        hnsw_recall_cache_dir=str(cache_dir),
    )
    tool.prepare_dataset(datasets['train'])
    return tool, datasets


def full_exact_reference(tool, dataset, query_ids, train):
    query_windows = tool._panel_windows_for_ids(dataset, query_ids)
    all_ids = np.arange(tool.n_train, dtype=np.int64)
    candidate_series, candidate_starts = tool.panel_candidate_manifest.resolve_ids(
        all_ids
    )
    candidate_windows = tool.panel_series_bank.get_windows(
        candidate_series, candidate_starts, tool.seq_len
    )
    query_keys = tool._key_for_period(query_windows, 1)
    candidate_keys = tool._key_for_period(torch.from_numpy(candidate_windows), 1)
    similarities = query_keys @ candidate_keys.T
    if train:
        query_series, query_starts = dataset.query_manifest.resolve_ids(query_ids)
        width = tool.seq_len + tool.pred_len
        forbidden = (
            (candidate_series[None, :] == query_series[:, None])
            & (candidate_starts[None, :] < query_starts[:, None] + width)
            & (query_starts[:, None] < candidate_starts[None, :] + width)
        )
        similarities[torch.from_numpy(forbidden)] = float('-inf')
    scores, labels = torch.topk(similarities, k=tool.topm, dim=1)
    return labels, scores


def test_panel_batch_native_query_gather_matches_item_order():
    tool, datasets = recall_fixture(enabled=False)
    for dataset in datasets.values():
        ids = np.arange(len(dataset), dtype=np.int64)
        series, starts = dataset.query_manifest.resolve_ids(ids)
        batched = dataset.series_bank.get_windows(series, starts, tool.seq_len)
        reference = np.stack([dataset[index][1] for index in range(len(dataset))])
        assert np.array_equal(batched, reference)
        assert np.array_equal(ids, np.asarray([dataset[index][0] for index in ids]))

    dataset = datasets['valid']
    tool._active_panel_query_manifest = dataset.query_manifest
    reference_chunks = []
    for start in range(0, len(dataset), tool.retrieval_batch_size):
        end = min(start + tool.retrieval_batch_size, len(dataset))
        ids = np.arange(start, end, dtype=np.int64)
        item_x = torch.from_numpy(np.stack([
            dataset[int(query_id)][1] for query_id in ids
        ]))
        reference_chunks.append(tool.retrieve(item_x, ids, train=False))
    reference_retrieval = torch.cat(reference_chunks, dim=1)
    batch_native_retrieval = tool.retrieve_all(
        dataset, train=False, split='valid'
    )
    assert torch.equal(batch_native_retrieval, reference_retrieval)


def test_chunked_exact_matches_full_matrix_and_train_mask():
    tool, datasets = recall_fixture(enabled=False)
    for split, train in (('train', True), ('valid', False)):
        dataset = datasets[split]
        query_ids = np.array([0, len(dataset) // 2, len(dataset) - 1])
        tool._active_panel_query_manifest = dataset.query_manifest
        windows = tool._panel_windows_for_ids(dataset, query_ids)
        actual_labels, actual_scores = tool._chunked_panel_exact_topk(
            windows, query_ids, 0, train=train, candidate_chunk_size=7
        )
        expected_labels, expected_scores = full_exact_reference(
            tool, dataset, query_ids, train
        )
        assert torch.equal(actual_labels, expected_labels)
        assert torch.allclose(actual_scores, expected_scores, atol=1e-6)
        if train:
            candidate_series, candidate_starts = (
                tool.panel_candidate_manifest.resolve_ids(actual_labels.numpy())
            )
            query_series, query_starts = dataset.query_manifest.resolve_ids(
                query_ids
            )
            forbidden = (
                (candidate_series == query_series[:, None])
                & (candidate_starts < query_starts[:, None] + 8)
                & (query_starts[:, None] < candidate_starts + 8)
            )
            assert not forbidden.any()


def test_recall_sampling_cache_and_runtime_stats_are_isolated(tmp_path):
    tool, datasets = recall_fixture(cache_dir=tmp_path, seed=11)
    before = (tool._wide_fallbacks, tool._wide_searches, tool._train_queries)
    first = tool.run_hnsw_recall(datasets)
    after = (tool._wide_fallbacks, tool._wide_searches, tool._train_queries)
    assert before == after
    for split in ('train', 'valid'):
        result = first[split][0]
        assert 0.0 <= result['min_recall'] <= result['mean_recall'] <= 1.0
        assert result['exact_cache'] == 'miss'
        assert len(result['query_ids']) == 9

    with mock.patch.object(
        tool, '_chunked_panel_exact_topk', wraps=tool._chunked_panel_exact_topk
    ) as exact:
        second = tool.run_hnsw_recall(datasets)
        exact.assert_not_called()
    for split in ('train', 'valid'):
        assert second[split][0]['exact_cache'] == 'hit'
        assert np.array_equal(
            first[split][0]['query_ids'], second[split][0]['query_ids']
        )
        assert torch.equal(
            first[split][0]['exact_labels'], second[split][0]['exact_labels']
        )

    metadata = tool._recall_cache_metadata(
        datasets['valid'], 'valid', 0, first['valid'][0]['query_ids']
    )
    original_path = tool._recall_cache_path(metadata)
    tool.hnsw_ef_search = 7
    assert tool._recall_cache_path(tool._recall_cache_metadata(
        datasets['valid'], 'valid', 0, first['valid'][0]['query_ids']
    )) == original_path
    tool.search_target += 1
    assert tool._recall_cache_path(tool._recall_cache_metadata(
        datasets['valid'], 'valid', 0, first['valid'][0]['query_ids']
    )) != original_path


def test_exact_cache_key_covers_exact_reference_identity(tmp_path):
    tool, datasets = recall_fixture(cache_dir=tmp_path, seed=23)
    query_ids = tool._sample_recall_query_ids(datasets['valid'], 'valid')

    def cache_path(dataset=datasets['valid'], split='valid'):
        return tool._recall_cache_path(tool._recall_cache_metadata(
            dataset, split, 0, query_ids
        ))

    baseline = cache_path()
    tool.hnsw_m += 1
    tool.hnsw_ef_construction += 1
    tool.hnsw_ef_search += 1
    assert cache_path() == baseline

    tool.hnsw_recall_seed += 1
    assert cache_path() != baseline
    tool.hnsw_recall_seed -= 1
    tool.seq_len += 1
    assert cache_path() != baseline
    tool.seq_len -= 1
    assert cache_path(split='train') != baseline

    original_ids = datasets['valid'].series_bank.series_ids
    datasets['valid'].series_bank.series_ids = original_ids.copy()
    datasets['valid'].series_bank.series_ids[0] += 1
    assert cache_path() != baseline
    datasets['valid'].series_bank.series_ids = original_ids


def test_recall_seed_changes_samples_and_off_does_no_exact_work():
    left, datasets = recall_fixture(seed=3, enabled=False)
    ids_left = left._sample_recall_query_ids(datasets['valid'], 'valid')
    left.hnsw_recall_seed = 4
    ids_right = left._sample_recall_query_ids(datasets['valid'], 'valid')
    assert not np.array_equal(ids_left, ids_right)
    with mock.patch.object(left, '_chunked_panel_exact_topk') as exact:
        assert left.run_hnsw_recall(datasets) == {}
        exact.assert_not_called()


def test_recall_flag_does_not_change_index_count_or_retrieval(tmp_path):
    disabled, disabled_data = recall_fixture(enabled=False)
    enabled, enabled_data = recall_fixture(enabled=True)
    assert disabled.hnsw_indices[0].get_current_count() == enabled.n_train
    assert disabled.hnsw_seed == enabled.hnsw_seed

    disabled_result = disabled.retrieve_all(
        disabled_data['valid'], train=False, split='valid'
    )
    enabled_result = enabled.retrieve_all(
        enabled_data['valid'], train=False, split='valid'
    )
    assert torch.equal(disabled_result, enabled_result)


def test_recall_cli_defaults_are_off_and_validation_split():
    parser = build_parser()
    args = parser.parse_args([
        '--model_id', 'x', '--data', 'panel', '--enc_in', '1'
    ])
    assert args.hnsw_recall_check is False
    assert args.hnsw_recall_queries == 100
    assert args.hnsw_recall_splits == 'valid'
    assert args.hnsw_recall_candidate_chunk_size == 4096
    assert args.hnsw_recall_query_batch_size == 32
    assert args.hnsw_recall_device == 'cpu'
    assert args.hnsw_recall_cache_dir == ''


@pytest.mark.parametrize(
    'kwargs,message',
    [
        ({'hnsw_recall_queries': 0}, 'queries'),
        ({'hnsw_recall_candidate_chunk_size': 0}, 'candidate_chunk'),
        ({'hnsw_recall_query_batch_size': 0}, 'query_batch'),
        ({'hnsw_recall_splits': 'valid,unknown'}, 'splits'),
    ],
)
def test_recall_configuration_validation(kwargs, message):
    defaults = dict(seq_len=6, pred_len=2, channels=1, n_period=1, topm=2)
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=message):
        RetrievalTool(**defaults)


def test_recall_cuda_validation_is_clear():
    with mock.patch('torch.cuda.is_available', return_value=False):
        with pytest.raises(ValueError, match='CUDA HNSW recall'):
            RetrievalTool(
                seq_len=6, pred_len=2, channels=1, n_period=1, topm=2,
                hnsw_recall_device='cuda',
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_chunked_exact_cuda_matches_cpu():
    tool, datasets = recall_fixture(enabled=False)
    dataset = datasets['valid']
    query_ids = np.array([0, len(dataset) // 2, len(dataset) - 1])
    tool._active_panel_query_manifest = dataset.query_manifest
    windows = tool._panel_windows_for_ids(dataset, query_ids)
    cpu_labels, cpu_scores = tool._chunked_panel_exact_topk(
        windows, query_ids, 0, train=False, candidate_chunk_size=7,
        device='cpu',
    )
    cuda_labels, cuda_scores = tool._chunked_panel_exact_topk(
        windows, query_ids, 0, train=False, candidate_chunk_size=7,
        device='cuda',
    )
    assert torch.equal(cuda_labels, cpu_labels)
    assert torch.allclose(cuda_scores, cpu_scores, atol=1e-6)
