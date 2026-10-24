"""Regression coverage for the unmasked HNSW NumPy-to-Torch boundary."""

from unittest import mock

import numpy as np
import pytest
import torch

from src.common.webscale_ts import InMemorySeriesBank, PanelWindowDataset
from src.data_provider.webscale_panel_adapter import build_target_split_manifests
from src.layers.Retrieval import RetrievalTool


def make_tool(**kwargs):
    options = dict(seq_len=6, pred_len=2, channels=1, n_period=1, topm=3,
                   hnsw_ef_search=8)
    options.update(kwargs)
    return RetrievalTool(**options)


class RawIndex:
    def __init__(self, labels, distances):
        self.labels = labels
        self.distances = distances
        self.efs = []
        self.queries = []

    def set_ef(self, ef):
        self.efs.append(ef)

    def knn_query(self, queries, k):
        self.queries.append((queries, k))
        return self.labels, self.distances


def strided_outputs(batch_size, count):
    # Non-owning, non-contiguous buffers with native-like unsigned label dtype.
    labels = np.arange(batch_size * count * 2, dtype=np.uint64).reshape(batch_size, count * 2)[:, ::2]
    distances = np.linspace(0, 2, batch_size * count * 2, dtype=np.float64).reshape(batch_size, count * 2)[:, ::2]
    return labels, distances


@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('target_count', [None, 2, 5])
@pytest.mark.parametrize('ef_search', [1, 8])
def test_owned_arrays_preserve_ids_scores_target_and_ef(batch_size, target_count, ef_search):
    tool = make_tool(hnsw_ef_search=ef_search)
    count = tool.search_target if target_count is None else target_count
    raw_labels, raw_distances = strided_outputs(batch_size, count)
    expected_labels = raw_labels.astype(np.int64)
    expected_scores = 1.0 - raw_distances.astype(np.float32)
    index = RawIndex(raw_labels, raw_distances)
    queries = np.zeros((batch_size, tool.seq_len), dtype=np.float32)
    with mock.patch.object(tool, '_normalize_knn_output', wraps=tool._normalize_knn_output) as normalize:
        labels, scores = tool._unmasked_knn_query_arrays(index, queries, target_count)
    normalize.assert_called_once()
    assert index.efs == [max(ef_search, count)]
    assert len(index.queries) == 1
    assert index.queries[0][0] is queries
    assert index.queries[0][1] == count
    for array, dtype, expected in [(labels, np.int64, expected_labels), (scores, np.float32, expected_scores)]:
        assert isinstance(array, np.ndarray)
        assert array.dtype == dtype
        assert array.ndim == 2 and array.shape == (batch_size, count)
        assert array.flags.c_contiguous and array.flags.owndata
        assert array.base is None
        np.testing.assert_array_equal(array, expected)
        assert not np.shares_memory(array, raw_labels)
        assert not np.shares_memory(array, raw_distances)
    raw_labels[:] = 999
    raw_distances[:] = -5
    np.testing.assert_array_equal(labels, expected_labels)
    np.testing.assert_array_equal(scores, expected_scores)


@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('target_count', [None, 2])
def test_production_returns_previous_tensors_without_wrapper_stack_or_concatenate(batch_size, target_count):
    tool = make_tool()
    tool.panel_mode = True
    tool.n_train = 100
    count = tool.topm if target_count is None else target_count
    raw_labels, raw_distances = strided_outputs(batch_size, count)
    index = RawIndex(raw_labels, raw_distances)
    tool.hnsw_indices = [index]
    windows = torch.arange(batch_size * 6, dtype=torch.float32).reshape(batch_size, 6, 1)
    query_ids = np.arange(batch_size)
    # Exact old boundary: raw arrays -> row views -> list -> stack -> Torch.
    old_scores = tool._similarity(raw_distances)
    old_neighbors = [(raw_labels[row], old_scores[row]) for row in range(batch_size)]
    expected_labels = torch.as_tensor(np.stack([r[0] for r in old_neighbors]).astype(np.int64), dtype=torch.long)
    expected_scores = torch.as_tensor(np.stack([r[1] for r in old_neighbors]), dtype=torch.float32)
    with mock.patch.object(tool, '_unmasked_knn_query', side_effect=AssertionError('legacy wrapper')), \
         mock.patch.object(tool, '_masked_knn_query', side_effect=AssertionError('masked search')), \
         mock.patch('numpy.stack', side_effect=AssertionError('stack')), \
         mock.patch('numpy.concatenate', side_effect=AssertionError('concatenate')):
        labels, scores = tool.search_candidates(windows, 0, query_ids, False, target_count)
    assert torch.equal(labels, expected_labels)
    assert torch.equal(scores, expected_scores)
    assert labels.shape == scores.shape == (batch_size, count)
    assert labels.dtype == torch.long and scores.dtype == torch.float32
    assert len(index.queries) == 1
    raw_labels[:] = 999
    raw_distances[:] = -5
    assert torch.equal(labels, expected_labels)
    assert torch.equal(scores, expected_scores)


def test_compatibility_wrapper_preserves_historical_row_tuple_values():
    tool = make_tool()
    raw_labels, raw_distances = strided_outputs(4, 3)
    queries = np.zeros((4, 6), dtype=np.float32)
    expected_scores = tool._similarity(raw_distances)
    rows = tool._unmasked_knn_query(RawIndex(raw_labels, raw_distances), queries)
    assert isinstance(rows, list) and len(rows) == 4
    for row, pair in enumerate(rows):
        assert isinstance(pair, tuple) and len(pair) == 2
        np.testing.assert_array_equal(pair[0], raw_labels[row])
        np.testing.assert_array_equal(pair[1], expected_scores[row])


def test_search_timing_counts_one_native_query_only():
    tool = make_tool()
    tool._active_runtime = {'hnsw_search_ms': 7.0}
    index = RawIndex(*strided_outputs(2, 3))
    with mock.patch('src.layers.Retrieval.time.perf_counter', side_effect=[10.0, 10.125]):
        tool._unmasked_knn_query_arrays(index, np.zeros((2, 6), dtype=np.float32))
    assert tool._active_runtime['hnsw_search_ms'] == 132.0
    assert len(index.queries) == 1


@pytest.mark.parametrize('label_shape,distance_shape,batch,message', [
    ((2, 2), (2, 2), 2, 'candidate count mismatch'),
    ((2, 3), (2, 2), 2, 'element-count mismatch'),
    ((5,), (5,), 2, 'invalid HNSW output size'),
    ((2, 0), (2, 0), 2, 'empty neighbor dimension'),
    ((0, 3), (0, 3), 0, 'empty query batch'),
])
def test_malformed_fixed_candidate_output_fails(label_shape, distance_shape, batch, message):
    tool = make_tool()
    index = RawIndex(np.zeros(label_shape, dtype=np.int64), np.zeros(distance_shape, dtype=np.float32))
    with pytest.raises(RuntimeError, match=message):
        tool._unmasked_knn_query_arrays(index, np.zeros((batch, 6), dtype=np.float32))


def test_similarity_view_is_copied_only_when_needed():
    tool = make_tool()
    index = RawIndex(*strided_outputs(2, 3))
    view = np.arange(12, dtype=np.float32).reshape(2, 6)[:, ::2]
    queries = np.zeros((2, 6), dtype=np.float32)
    with mock.patch.object(tool, '_similarity', return_value=view):
        _, scores = tool._unmasked_knn_query_arrays(index, queries)
    assert scores.flags.c_contiguous and scores.flags.owndata
    assert scores.base is None and not np.shares_memory(scores, view)
    np.testing.assert_array_equal(scores, view)
    owned = view.copy(order='C')
    with mock.patch.object(tool, '_similarity', return_value=owned):
        _, scores = tool._unmasked_knn_query_arrays(index, queries)
    assert scores is owned


@pytest.mark.parametrize('invalid', [np.zeros((2, 3), dtype=np.float64), np.zeros(6, dtype=np.float32), [[0, 0, 0], [0, 0, 0]]])
def test_similarity_type_dtype_and_shape_are_validated(invalid):
    tool = make_tool()
    index = RawIndex(*strided_outputs(2, 3))
    with mock.patch.object(tool, '_similarity', return_value=invalid):
        with pytest.raises(RuntimeError, match='float32 NumPy'):
            tool._unmasked_knn_query_arrays(index, np.zeros((2, 6), dtype=np.float32))


def previous_unmasked_search(tool, windows, period_idx, query_ids, train, target_count=None):
    """Pre-patch HNSW unmasked flow from 68d91ab, for numerical comparison."""
    assert not train
    count = tool.search_target if target_count is None else target_count
    queries = tool._key_for_period(windows.detach().cpu().float(), tool.period_num[period_idx]).numpy().astype(np.float32, copy=False)
    index = tool.hnsw_indices[period_idx]
    index.set_ef(max(tool.hnsw_ef_search, count))
    labels, distances = index.knn_query(queries, k=count)
    similarities = tool._similarity(distances)
    rows = [(labels[row], similarities[row]) for row in range(len(labels))]
    return (
        torch.as_tensor(np.stack([item[0] for item in rows]).astype(np.int64, copy=False), dtype=torch.long),
        torch.as_tensor(np.stack([item[1] for item in rows]), dtype=torch.float32),
    )


def test_valid_test_aggregation_and_recall_match_pre_patch_on_same_indices():
    values = np.random.default_rng(18).normal(size=(4 * 64, 2)).astype(np.float32)
    bank = InMemorySeriesBank(values, np.array([1, 4, 8, 9]), np.arange(5) * 64, np.full(4, 64))
    preset = dict(expected_length=64, train_end=40, valid_end=52, test_end=64)
    candidates, manifests = build_target_split_manifests(bank, 'synthetic', preset, 6, 2, candidate_stride=2, query_stride=2)
    datasets = {split: PanelWindowDataset(bank, manifest, candidates, split=split) for split, manifest in manifests.items()}
    tool = make_tool(channels=2, n_period=2, hnsw_build_batch_size=7,
                     retrieval_batch_size=5, hnsw_recall_check=True,
                     hnsw_recall_queries=7, hnsw_recall_splits=('valid', 'test'),
                     hnsw_recall_candidate_chunk_size=9,
                     hnsw_recall_query_batch_size=3)
    tool.prepare_dataset(datasets['train'])
    indices = tuple(tool.hnsw_indices)
    with mock.patch.object(tool, 'search_candidates', side_effect=lambda *a, **kw: previous_unmasked_search(tool, *a, **kw)):
        before = {split: tool.retrieve_all(datasets[split], train=False, split=split) for split in ('valid', 'test')}
        recall_before = tool.run_hnsw_recall(datasets)
    after = {split: tool.retrieve_all(datasets[split], train=False, split=split) for split in ('valid', 'test')}
    recall_after = tool.run_hnsw_recall(datasets)
    assert tuple(tool.hnsw_indices) == indices
    for split in ('valid', 'test'):
        assert torch.equal(after[split], before[split])
        assert torch.count_nonzero(after[split]) > 0
        for old, new in zip(recall_before[split], recall_after[split]):
            for key in ('hnsw_labels', 'exact_labels', 'exact_similarities'):
                assert torch.equal(old[key], new[key])
            np.testing.assert_array_equal(old['query_ids'], new['query_ids'])
            for key in ('mean_recall', 'median_recall', 'min_recall', 'period'):
                assert old[key] == new[key]
