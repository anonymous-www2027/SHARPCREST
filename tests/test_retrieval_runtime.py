from reliability_reference import cr_reference
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.layers.Retrieval import RetrievalTool
from src.models.RAFT import Model
from src.run import build_parser
from src.utils.profile_reliability import (
    compute_iqr_scale,
    compute_mean_profile,
    distance_to_similarity,
    map_reliability,
    robust_profile_distance,
    topk_reliability,
)


class SyntheticDataset(Dataset):
    def __init__(self, count=80, offset=0, channels=1):
        generator = np.random.default_rng(100 + offset)
        self.x = generator.normal(size=(count, 6, channels)).astype(np.float32)
        self.y = generator.normal(size=(count, 2, channels)).astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return (
            index,
            self.x[index],
            self.y[index],
            np.zeros((6, 1), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
        )


def make_tool(
    controller='mean_profile', topm=4, pool=8, batch_size=40,
    postprocess_device='cpu', mmr_device='cpu',
    retrieval_seed=17,
    backend='hnsw', redundancy_source='key', value_chunk_size=8,
):
    return RetrievalTool(
        seq_len=6,
        pred_len=2,
        channels=1,
        n_period=1,
        topm=topm,
        hnsw_m=8,
        hnsw_ef_construction=40,
        hnsw_ef_search=40,
        hnsw_seed=3,
        retrieval_batch_size=batch_size,
        retrieval_backend=backend,
        retrieval_controller=controller,
        mmr_candidate_pool=pool,
        profile_sections=3,
        profile_reliability_k=topm,
        mean_profile_temperature=1.0,
        retrieval_seed=retrieval_seed,
        retrieval_aggregation_chunk_size=2,
        retrieval_postprocess_device=postprocess_device,
        mmr_device=mmr_device,
        mmr_redundancy_source=redundancy_source,
        value_mmr_chunk_size=value_chunk_size,
        runtime_breakdown=True,
    )


def make_config():
    return SimpleNamespace(
        task_name='long_term_forecast',
        device=torch.device('cpu'),
        seq_len=6,
        pred_len=2,
        enc_in=1,
        n_period=1,
        topm=4,
        hnsw_m=8,
        hnsw_ef_construction=40,
        hnsw_ef_search=40,
        hnsw_seed=3,
        retrieval_batch_size=40,
        retrieval_backend='hnsw',
        retrieval_controller='crest',
        mmr_candidate_pool=8,
        profile_sections=3,
        profile_reliability_k=4,
        mean_profile_temperature=1.0,
        sigma_min=0.05,
        sigma_max=0.30,
        lambda_min=0.30,
        lambda_max=0.90,
        hnsw_oversample_factor=5,
        seed=17,
        fusion_mode='concat_projection',
        retrieval_postprocess_device='cpu',
        mmr_device='cpu',
        mmr_redundancy_source='key',
        value_mmr_chunk_size=8,
        retrieval_aggregation_chunk_size=2,
        runtime_breakdown=False,
    )


def reference_mmr(tool, similarities, lambda_q, period_idx, query_index, train):
    """Committed implementation before incremental-redundancy optimization."""
    pool_size = similarities.numel()
    similarity_matrix = 1.0 - torch.abs(
        similarities.unsqueeze(0) - similarities.unsqueeze(1)
    )
    selected = []
    selected_mask = torch.zeros(pool_size, dtype=torch.bool)
    generator = tool._mmr_generator(period_idx, query_index, train)
    for step in range(tool.topm):
        if step == 0:
            best_position = 0
        else:
            positions = torch.tensor(selected, dtype=torch.long)
            redundancy = similarity_matrix[:, positions].max(dim=1).values
            scores = lambda_q * similarities - (1.0 - lambda_q) * redundancy
            scores[selected_mask] = float('-inf')
            probabilities = F.softmax(scores / 0.1, dim=0)
            best_position = torch.multinomial(
                probabilities, 1, generator=generator
            ).item()
        selected.append(best_position)
        selected_mask[best_position] = True
    return torch.tensor(selected, dtype=torch.long)


def reference_value_mmr(
    tool, similarities, lambda_q, values, period_idx, query_index, train
):
    normalized = tool._normalize_value_vectors(values.reshape(len(values), -1))
    value_similarity = normalized @ normalized.T
    selected = []
    selected_mask = torch.zeros(len(values), dtype=torch.bool)
    generator = tool._mmr_generator(period_idx, query_index, train)
    for step in range(tool.topm):
        if step == 0:
            position = 0
        else:
            redundancy = value_similarity[:, selected].max(dim=1).values
            scores = lambda_q * similarities - (1.0 - lambda_q) * redundancy
            scores[selected_mask] = float('-inf')
            probabilities = F.softmax(scores / 0.1, dim=0)
            position = torch.multinomial(
                probabilities, 1, generator=generator
            ).item()
        selected.append(position)
        selected_mask[position] = True
    return torch.tensor(selected)


def reference_gaussian(similarities, sigma_q):
    sigma_q = sigma_q.to(dtype=similarities.dtype)
    raw = torch.exp(
        -((1.0 - similarities) ** 2) / (2.0 * sigma_q ** 2)
    )
    denominator = raw.sum()
    if not torch.isfinite(denominator) or denominator <= 0:
        normalized = torch.zeros_like(raw)
        normalized[similarities.argmax()] = 1.0
    else:
        normalized = raw / denominator
    normalized[normalized.argmax()] += 1.0 - normalized.sum()
    return raw, normalized


def neighbor_tensors(tool, x):
    queries = tool._key_for_period(x, 1).numpy().astype(np.float32, copy=False)
    neighbors = tool._unmasked_knn_query(tool.hnsw_indices[0], queries)
    labels = torch.as_tensor(
        np.stack([row[0] for row in neighbors]).astype(np.int64)
    ).long()
    similarities = torch.as_tensor(
        np.stack([row[1] for row in neighbors])
    ).float()
    return labels, similarities


@pytest.mark.parametrize('seed', [0, 1, 17, 991])
@pytest.mark.parametrize('lambda_value', [0.3, 0.6, 0.9])
def test_incremental_mmr_is_exactly_reference(seed, lambda_value):
    tool = make_tool(topm=7, pool=19)
    assert tool.mmr_device.type == 'cpu'
    generator = torch.Generator().manual_seed(seed)
    vectors = [
        torch.linspace(0.95, -0.25, 19),
        torch.rand(19, generator=generator) * 2.0 - 1.0,
        torch.tensor([0.5] * 19),
    ]
    for query_index, similarities in enumerate(vectors, start=5):
        expected = reference_mmr(
            tool, similarities, torch.tensor(lambda_value), 0,
            query_index, True,
        )
        actual = tool.mmr_select_from_pool(
            similarities, torch.tensor(lambda_value), 0, query_index, True
        )
        assert torch.equal(actual, expected)


def test_value_patch_similarity_pearson_semantics_and_zero_variance():
    base = torch.tensor([[1.0, 3.0, -2.0, 4.0]])
    identical = RetrievalTool.value_patch_similarity(base, base)
    offset = RetrievalTool.value_patch_similarity(base, base + 19.0)
    negative = RetrievalTool.value_patch_similarity(base, -base)
    random = RetrievalTool.value_patch_similarity(
        base, torch.tensor([[0.3, -1.7, 2.2, 0.1]])
    )
    zero_variance = RetrievalTool.value_patch_similarity(
        torch.ones_like(base), base
    )
    patch = base.reshape(1, 2, 2)
    patch_offset = RetrievalTool.value_patch_similarity(patch, patch + 7.0)

    assert identical.item() == pytest.approx(1.0, abs=1e-6)
    assert offset.item() == pytest.approx(1.0, abs=1e-6)
    assert negative.item() == pytest.approx(-1.0, abs=1e-6)
    assert torch.isfinite(random).all()
    assert -1.0 <= random.item() <= 1.0
    assert torch.isfinite(zero_variance).all()
    assert zero_variance.item() == 0.0
    assert patch_offset.item() == pytest.approx(1.0, abs=1e-6)


def test_incremental_cpu_value_mmr_matches_full_matrix_reference():
    tool = make_tool(
        topm=7, pool=19, redundancy_source='value', mmr_device='cpu'
    )
    generator = torch.Generator().manual_seed(991)
    similarities = torch.rand(19, generator=generator) * 2.0 - 1.0
    values = torch.randn(19, 6, generator=generator)
    lambda_q = torch.tensor(0.63)
    expected = reference_value_mmr(
        tool, similarities, lambda_q, values, 2, 37, False
    )
    actual = tool.mmr_select_from_pool(
        similarities, lambda_q, 2, 37, False, candidate_values=values
    )
    assert torch.equal(actual, expected)


def test_cli_defaults_to_sharp_only_and_value_for_future_crest():
    parser = build_parser()
    assert parser.get_default('retrieval_backend') == 'hnsw'
    assert parser.get_default('retrieval_controller') == 'baseline'
    assert parser.get_default('mmr_redundancy_source') == 'value'
    assert parser.get_default('value_mmr_chunk_size') == 8


def test_controller_alias_and_top_m_toggle():
    baseline = make_tool(controller='baseline', topm=4, pool=10)
    crest = make_tool(controller='crest', topm=4, pool=10)
    alias = make_tool(controller='mean_profile', topm=4, pool=10)
    assert baseline.retrieval_controller == 'baseline'
    assert not baseline.crest_enabled
    assert baseline.search_target == 4
    assert crest.retrieval_controller == 'crest'
    assert alias.retrieval_controller == 'crest'
    assert crest.crest_enabled and alias.crest_enabled
    assert crest.search_target == alias.search_target == 10


def test_crest_off_has_no_profile_mmr_gaussian_or_value_norm_overhead():
    data = SyntheticDataset()
    tool = make_tool(
        controller='baseline', redundancy_source='value', batch_size=13
    )
    tool.prepare_dataset(data)
    assert tool.profile_bank is None
    assert tool.profile_scale is None
    assert tool.profile_bank_postprocess is None
    assert tool.profile_scale_postprocess is None
    assert tool.value_norm_banks == []
    assert tool.prepare_runtime['profile_bank_iqr_ms'] == 0.0
    assert tool.prepare_runtime['value_norm_bank_ms'] == 0.0

    x = torch.from_numpy(data.x[:13])
    with mock.patch.object(
        tool, 'compute_profile_reliability_batch',
        side_effect=AssertionError('CREST profile must be inactive'),
    ), mock.patch.object(
        tool, 'mmr_select_from_pool',
        side_effect=AssertionError('CREST MMR must be inactive'),
    ), mock.patch.object(
        tool, 'gaussian_weights_batch',
        side_effect=AssertionError('CREST Gaussian must be inactive'),
    ):
        prediction = tool.retrieve(x, torch.arange(13), train=False)
    assert torch.isfinite(prediction).all()
    tool.retrieve_all(data, train=False, split='baseline')
    runtime = tool.split_runtime['baseline']
    assert runtime['profile_reliability_ms'] == 0.0
    assert runtime['mmr_ms'] == 0.0
    assert runtime['gaussian_aggregation_ms'] == 0.0
    assert runtime['baseline_aggregation_ms'] > 0.0


def test_crest_off_ignores_inactive_cuda_postprocess_options():
    with mock.patch('torch.cuda.is_available', return_value=False):
        tool = make_tool(
            controller='baseline',
            postprocess_device='cuda',
            mmr_device='cuda',
            redundancy_source='value',
        )
    assert not tool.crest_enabled
    assert tool.search_target == tool.topm


def test_exact_and_hnsw_search_share_interface_target_and_mask_semantics():
    data = SyntheticDataset()
    exact = make_tool(controller='baseline', backend='exact')
    hnsw = make_tool(controller='baseline', backend='hnsw')
    exact.prepare_dataset(data)
    hnsw.prepare_dataset(data)
    query = torch.from_numpy(data.x[20:25])
    query_ids = np.arange(20, 25)

    for train in (False, True):
        exact_labels, exact_similarities = exact.search_candidates(
            query, 0, query_ids, train
        )
        hnsw_labels, hnsw_similarities = hnsw.search_candidates(
            query, 0, query_ids, train
        )
        for labels, similarities in (
            (exact_labels, exact_similarities),
            (hnsw_labels, hnsw_similarities),
        ):
            assert labels.shape == similarities.shape == (5, exact.topm)
            assert labels.dtype == torch.long
            assert similarities.dtype == torch.float32
            assert torch.isfinite(similarities).all()
        reference = exact._key_for_period(query, 1) @ exact.exact_key_banks[0].T
        if train:
            for row, query_id in enumerate(query_ids):
                left, right = exact._forbidden_bounds(query_id)
                reference[row, left:right + 1] = float('-inf')
                assert not torch.any(
                    (exact_labels[row] >= left) & (exact_labels[row] <= right)
                )
                assert not torch.any(
                    (hnsw_labels[row] >= left) & (hnsw_labels[row] <= right)
                )
        reference_similarities, reference_labels = torch.topk(
            reference, k=exact.topm, dim=1, largest=True, sorted=True
        )
        assert torch.equal(exact_labels, reference_labels)
        assert torch.equal(exact_similarities, reference_similarities)


@pytest.mark.parametrize('backend', ['exact', 'hnsw'])
@pytest.mark.parametrize(
    'controller,redundancy',
    [('baseline', 'value'), ('crest', 'key'), ('crest', 'value')],
)
def test_modular_mode_smoke(backend, controller, redundancy):
    data = SyntheticDataset()
    tool = make_tool(
        controller=controller,
        backend=backend,
        redundancy_source=redundancy,
        batch_size=13,
    )
    tool.prepare_dataset(data)
    prediction = tool.retrieve(
        torch.from_numpy(data.x[:13]), torch.arange(13), train=False
    )
    assert prediction.shape == (1, 13, 2, 1)
    assert torch.isfinite(prediction).all()


def test_batch_profile_is_querywise_reference_with_dtype_boundary():
    data = SyntheticDataset(channels=2)
    tool = RetrievalTool(
        seq_len=6, pred_len=2, channels=2, n_period=1, topm=4,
        retrieval_controller='mean_profile', mmr_candidate_pool=8,
        profile_sections=3, profile_reliability_k=4,
    )
    tool.prepare_dataset(data)
    query = torch.from_numpy(data.x[10:15])
    labels = torch.tensor([
        [0, 3, 7, 9, 11, 13, 17, 19],
        [1, 4, 8, 10, 12, 14, 18, 20],
        [2, 5, 9, 11, 13, 15, 19, 21],
        [3, 6, 10, 12, 14, 16, 20, 22],
        [4, 7, 11, 13, 15, 17, 21, 23],
    ])
    actual = tool.compute_profile_reliability_batch(
        query, labels, train=False
    )

    reference_rows = []
    for row in range(len(query)):
        query_profile = compute_mean_profile(query[row:row + 1], 3)[0]
        distance = robust_profile_distance(
            query_profile, tool.profile_bank[labels[row]], tool.profile_scale
        )
        similarity = distance_to_similarity(distance, 1.0)
        reliability = cr_reference(similarity, tool.raw_future_bank[labels[row]], 4, torch.from_numpy(data.y), tool=tool, candidate_labels=labels[row])
        sigma_q, lambda_q = map_reliability(
            reliability, 0.05, 0.30, 0.30, 0.90
        )
        reference_rows.append(
            (reliability, sigma_q, lambda_q, distance, similarity)
        )
    for index, batch_value in enumerate(actual):
        expected = torch.stack([row[index] for row in reference_rows])
        torch.testing.assert_close(batch_value.cpu(), expected)

    train_actual = tool.compute_profile_reliability_batch(
        query, labels, query_indices=np.arange(10, 15), train=True
    )
    for index, batch_value in enumerate(train_actual):
        assert torch.equal(batch_value.cpu(), actual[index].cpu())


def test_batch_gaussian_matches_per_query_and_exact_unit_sum():
    similarities = torch.tensor([
        [0.91, 0.70, 0.33, -0.20],
        [-1.00, -0.90, -0.80, -0.70],
        [1.00, 1.00, 1.00, 1.00],
    ])
    sigma = torch.tensor([0.21, 0.05, 0.30])
    raw, normalized = RetrievalTool.gaussian_weights_batch(similarities, sigma)
    for row in range(len(similarities)):
        expected_raw, expected_normalized = reference_gaussian(
            similarities[row], sigma[row]
        )
        assert torch.allclose(raw[row], expected_raw, rtol=0, atol=0)
        assert torch.allclose(normalized[row], expected_normalized, rtol=0, atol=0)
    assert torch.equal(normalized.sum(dim=1), torch.ones(3))


def test_baseline_end_to_end_matches_reference():
    data = SyntheticDataset()
    tool = make_tool('baseline')
    tool.prepare_dataset(data)
    x = torch.from_numpy(data.x[:7])
    result = tool.retrieve(x, torch.arange(7), train=False)
    labels, similarities = neighbor_tensors(tool, x)
    weights = F.softmax(similarities / tool.temperature, dim=1)
    values = tool.y_data_all_mg[0].flatten(start_dim=1)
    expected = (values[labels] * weights.unsqueeze(-1)).sum(dim=1)
    assert torch.equal(result[0].flatten(start_dim=1), expected)


def test_mean_profile_end_to_end_candidate_ids_and_prediction_match_reference():
    data = SyntheticDataset()
    tool = make_tool()
    tool.prepare_dataset(data)
    x = torch.from_numpy(data.x[:7])
    result = tool.retrieve(x, torch.arange(7), train=False)
    labels, similarities = neighbor_tensors(tool, x)
    values = tool.y_data_all_mg[0].flatten(start_dim=1)

    expected_ids = []
    expected_predictions = []
    for row in range(len(x)):
        query_profile = compute_mean_profile(x[row:row + 1], 3)[0]
        distance = robust_profile_distance(
            query_profile, tool.profile_bank[labels[row]], tool.profile_scale
        )
        profile_similarity = distance_to_similarity(distance, 1.0)
        reliability = cr_reference(profile_similarity, tool.raw_future_bank[labels[row]], 4, torch.from_numpy(data.y), tool=tool, candidate_labels=labels[row])
        sigma_q, lambda_q = map_reliability(
            reliability, 0.05, 0.30, 0.30, 0.90
        )
        positions = reference_mmr(
            tool, similarities[row], lambda_q, 0, row, False
        )
        selected_ids = labels[row, positions]
        selected_similarities = similarities[row, positions]
        _raw, normalized = reference_gaussian(selected_similarities, sigma_q)
        expected_ids.append(selected_ids)
        expected_predictions.append(
            (values[selected_ids] * normalized.unsqueeze(-1)).sum(dim=0)
        )

    # Exact candidate IDs prove the batch controller did not perturb stochastic MMR.
    _r, _s, lambda_batch, _d, _ps = tool.compute_profile_reliability_batch(
        x, labels, train=False
    )
    actual_ids = torch.stack([
        labels[row, tool.mmr_select_from_pool(
            similarities[row], lambda_batch[row], 0, row, False
        )]
        for row in range(len(x))
    ])
    assert torch.equal(actual_ids, torch.stack(expected_ids))
    assert torch.equal(
        result[0].flatten(start_dim=1), torch.stack(expected_predictions)
    )


def test_retrieval_does_not_advance_global_rng():
    data = SyntheticDataset()
    tool = make_tool()
    tool.prepare_dataset(data)
    x = torch.from_numpy(data.x[:5])
    torch.manual_seed(1234)
    state = torch.random.get_rng_state()
    first = tool.retrieve(x, torch.arange(5), train=False)
    assert torch.equal(state, torch.random.get_rng_state())
    torch.manual_seed(9999)
    second = tool.retrieve(x, torch.arange(5), train=False)
    assert torch.equal(first, second)


def test_masked_search_semantics_and_runtime_scalars():
    class FakeIndex:
        def __init__(self):
            self.ks = []

        def set_ef(self, _value):
            pass

        def knn_query(self, queries, k):
            self.ks.append(k)
            if len(self.ks) == 1:
                labels = np.full((len(queries), k), 250, dtype=np.int64)
            else:
                labels = np.tile(np.arange(k), (len(queries), 1)).astype(np.int64)
            return labels, np.zeros_like(labels, dtype=np.float32)

    tool = RetrievalTool(
        seq_len=2, pred_len=1, channels=1, n_period=1, topm=4,
        retrieval_controller='mean_profile', mmr_candidate_pool=10,
        profile_sections=2, profile_reliability_k=4,
        hnsw_oversample_factor=5,
    )
    tool.n_train = 500
    fake = FakeIndex()
    result = tool._masked_knn_query(
        fake, np.zeros((1, 2), dtype=np.float32), np.array([250])
    )
    assert fake.ks == [50, 15]
    assert len(result[0][0]) == 10

    baseline_tool = RetrievalTool(
        seq_len=2, pred_len=1, channels=1, n_period=1, topm=4,
        retrieval_controller='baseline', mmr_candidate_pool=10,
        profile_sections=2, profile_reliability_k=4,
        hnsw_oversample_factor=5,
    )
    baseline_tool.n_train = 500
    baseline_fake = FakeIndex()
    baseline_result = baseline_tool._masked_knn_query(
        baseline_fake, np.zeros((1, 2), dtype=np.float32), np.array([250])
    )
    assert baseline_fake.ks == [20, 9]
    assert len(baseline_result[0][0]) == 4

    data = SyntheticDataset()
    runtime_tool = make_tool(batch_size=11)
    runtime_tool.prepare_dataset(data)
    retrieval = runtime_tool.retrieve_all(data, train=False, split='test')
    assert torch.isfinite(retrieval).all()
    assert runtime_tool.prepare_runtime['prepare_total_ms'] > 0
    runtime = runtime_tool.split_runtime['test']
    assert runtime['queries'] == len(data)
    assert runtime['split_total_ms'] > 0
    assert runtime['ms_per_query'] == pytest.approx(
        runtime['split_total_ms'] / len(data)
    )


def test_model_has_only_concat_projection_forecasting_path():
    model = Model(make_config())
    assert model.rt.retrieval_backend == 'hnsw'
    assert model.rt.retrieval_controller == 'crest'
    assert model.rt.mmr_redundancy_source == 'key'
    assert model.rt.value_mmr_chunk_size == 8
    assert model.linear_pred.in_features == 4
    assert not hasattr(model, 'linear_pred_2')
    assert not hasattr(model, 'raw_gaussian_dynamic_fusion')
    model.retrieval_dict = {'valid': torch.randn(1, 3, 2, 1)}
    output = model(torch.randn(3, 6, 1), torch.arange(3), mode='valid')
    assert output.shape == (3, 2, 1)
    assert torch.isfinite(output).all()


def test_model_propagates_modular_defaults_and_value_mmr_configuration():
    config = make_config()
    config.retrieval_backend = 'exact'
    config.retrieval_controller = 'baseline'
    config.mmr_redundancy_source = 'value'
    config.value_mmr_chunk_size = 4
    model = Model(config)
    assert model.rt.retrieval_backend == 'exact'
    assert not model.rt.crest_enabled
    assert model.rt.mmr_redundancy_source == 'value'
    assert model.rt.value_mmr_chunk_size == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_optional_cuda_postprocess_preserves_candidate_ids():
    data = SyntheticDataset()
    cpu_tool = make_tool()
    cpu_tool.prepare_dataset(data)
    cuda_tool = RetrievalTool(
        seq_len=6, pred_len=2, channels=1, n_period=1, topm=4,
        hnsw_m=8, hnsw_ef_construction=40, hnsw_ef_search=40,
        hnsw_seed=3, retrieval_batch_size=40,
        retrieval_controller='mean_profile', mmr_candidate_pool=8,
        profile_sections=3, profile_reliability_k=4,
        mean_profile_temperature=1.0, retrieval_seed=17,
        retrieval_postprocess_device='cuda',
        mmr_device='cpu',
        mmr_redundancy_source='key',
        retrieval_aggregation_chunk_size=2,
    )
    cuda_tool.prepare_dataset(data)
    # Use one HNSW instance so this test isolates post-processing only.
    cuda_tool.hnsw_indices = cpu_tool.hnsw_indices
    cuda_tool.y_data_all_mg = cpu_tool.y_data_all_mg

    x = torch.from_numpy(data.x[:7])
    labels, temporal_similarity = neighbor_tensors(cpu_tool, x)
    cpu_controller = cpu_tool.compute_profile_reliability_batch(
        x, labels, train=False
    )
    cuda_controller = cuda_tool.compute_profile_reliability_batch(
        x, labels, train=False
    )
    cpu_ids = torch.stack([
        labels[row, cpu_tool.mmr_select_from_pool(
            temporal_similarity[row], cpu_controller[2][row], 0, row, False
        )]
        for row in range(len(x))
    ])
    cuda_lambda = cuda_controller[2].cpu()
    cuda_ids = torch.stack([
        labels[row, cuda_tool.mmr_select_from_pool(
            temporal_similarity[row], cuda_lambda[row], 0, row, False
        )]
        for row in range(len(x))
    ])
    assert torch.equal(cpu_ids, cuda_ids)
    assert torch.allclose(
        cpu_tool.retrieve(x, torch.arange(7), train=False),
        cuda_tool.retrieve(x, torch.arange(7), train=False),
        rtol=1e-6,
        atol=1e-6,
    )


def test_cuda_mmr_without_cuda_is_rejected():
    with mock.patch('torch.cuda.is_available', return_value=False):
        with pytest.raises(ValueError, match='CUDA MMR requested without CUDA'):
            make_tool(mmr_device='cuda')


def cuda_mmr_case(rows=17, pool=29):
    generator = torch.Generator().manual_seed(88)
    similarities = (
        torch.rand(rows, pool, generator=generator) * 2.0 - 1.0
    ).cuda()
    lambda_q = (
        torch.rand(rows, generator=generator) * 0.6 + 0.3
    ).cuda()
    query_ids = np.arange(10, 10 + 7 * rows, 7, dtype=np.int64)
    return similarities, lambda_q, query_ids


def cuda_value_mmr_case(rows=17, pool=29, value_dim=23):
    similarities, lambda_q, query_ids = cuda_mmr_case(rows, pool)
    generator = torch.Generator().manual_seed(177)
    value_bank = torch.randn(rows * pool, value_dim, generator=generator)
    value_bank = RetrievalTool._normalize_value_vectors(value_bank)
    labels = torch.arange(rows * pool).reshape(rows, pool)
    return similarities, lambda_q, query_ids, labels, value_bank


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_inverse_cdf_closes_float32_tail_before_zero_probability_suffix():
    probabilities = torch.tensor(
        [[0.3333333, 0.3333333, 0.3333333, 0.0, 0.0]],
        device='cuda',
        dtype=torch.float32,
    )
    uniform = torch.tensor([[1.0 - 1e-9]], device='cuda', dtype=torch.float64)
    unnormalized_tail = probabilities.cumsum(dim=1)[:, -1:]
    assert torch.all(unnormalized_tail < uniform)

    selected = RetrievalTool._inverse_cdf_sample(probabilities, uniform)
    assert selected.shape == (1,)
    assert selected.item() == 2
    assert probabilities[0, selected].item() > 0.0
    assert selected.item() < probabilities.shape[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_stateless_cuda_uniform_range_and_distribution_sanity():
    tool = make_tool(topm=9, pool=29, mmr_device='cuda')
    query_ids = torch.arange(4096, device='cuda', dtype=torch.int64)
    uniforms = tool._mmr_stateless_uniforms(query_ids, 3, False)

    assert uniforms.shape == (4096, 8)
    assert torch.all(uniforms > 0.0)
    assert torch.all(uniforms < 1.0)
    assert 0.4 < uniforms.mean().item() < 0.6
    assert uniforms.min().item() < 0.05
    assert uniforms.max().item() > 0.95


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_stateless_cuda_uniform_counter_sensitivity():
    query_ids = torch.tensor([0, 1, 7, 31, 4096], device='cuda')
    tool = make_tool(topm=9, pool=29, mmr_device='cuda', retrieval_seed=17)
    baseline = tool._mmr_stateless_uniforms(query_ids, 2, False)

    assert not torch.equal(
        baseline, tool._mmr_stateless_uniforms(query_ids, 3, False)
    )
    assert not torch.equal(
        baseline, tool._mmr_stateless_uniforms(query_ids, 2, True)
    )
    other_seed = make_tool(
        topm=9, pool=29, mmr_device='cuda', retrieval_seed=18
    )
    assert not torch.equal(
        baseline, other_seed._mmr_stateless_uniforms(query_ids, 2, False)
    )
    high_seed = make_tool(
        topm=9, pool=29, mmr_device='cuda',
        retrieval_seed=2 ** 32 + 17,
    )
    assert not torch.equal(
        baseline, high_seed._mmr_stateless_uniforms(query_ids, 2, False)
    )
    assert torch.all(torch.any(baseline[:, :-1] != baseline[:, 1:], dim=1))
    shifted = tool._mmr_stateless_uniforms(query_ids + 1, 2, False)
    assert torch.all(torch.any(baseline != shifted, dim=1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_stateless_cuda_uniforms_mix_large_query_high_word():
    tool = make_tool(topm=9, pool=29, mmr_device='cuda')
    query_ids = torch.tensor(
        [7, 2 ** 32 + 7, 2 ** 40 + 12345],
        device='cuda',
        dtype=torch.int64,
    )
    uniforms = tool._mmr_stateless_uniforms(query_ids, 2, False)

    assert uniforms.shape == (3, 8)
    assert torch.all((uniforms > 0.0) & (uniforms < 1.0))
    assert not torch.equal(uniforms[0], uniforms[1])
    assert not torch.equal(uniforms[1], uniforms[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_value_cuda_mmr_is_invariant_to_calls_layout_and_value_chunk_size():
    tool = make_tool(
        topm=9, pool=29, mmr_device='cuda', redundancy_source='value',
        value_chunk_size=4,
    )
    similarities, lambda_q, query_ids, labels, value_bank = (
        cuda_value_mmr_case()
    )

    def select(sim, lam, ids, candidate_labels):
        return tool.mmr_select_batch_cuda_value_chunked(
            sim, lam, candidate_labels, value_bank, 2, ids, False
        )

    selected = select(similarities, lambda_q, query_ids, labels)
    repeated = select(similarities, lambda_q, query_ids, labels)
    assert torch.equal(selected, repeated)

    for chunk_size in (8, 16):
        tool.value_mmr_chunk_size = chunk_size
        assert torch.equal(
            selected, select(similarities, lambda_q, query_ids, labels)
        )

    partitioned = torch.cat([
        select(
            similarities[start:end], lambda_q[start:end],
            query_ids[start:end], labels[start:end],
        )
        for start, end in ((0, 3), (3, 12), (12, 17))
    ])
    assert torch.equal(selected, partitioned)

    permutation = torch.tensor(
        [8, 1, 15, 3, 12, 0, 16, 5, 10, 2, 14, 6, 9, 4, 13, 7, 11]
    )
    permutation_cuda = permutation.cuda()
    permuted = select(
        similarities[permutation_cuda], lambda_q[permutation_cuda],
        query_ids[permutation.numpy()], labels[permutation],
    )
    restored = permuted[torch.argsort(permutation).cuda()]
    assert torch.equal(selected, restored)

    def aggregate(selected_positions):
        selected_similarities = similarities.gather(1, selected_positions)
        sigma = torch.linspace(0.05, 0.30, len(similarities), device='cuda')
        _raw, weights = tool.gaussian_weights_batch(
            selected_similarities, sigma
        )
        selected_labels = labels.cuda().gather(1, selected_positions)
        prediction_values = value_bank[:, :7].cuda()
        prediction = (
            prediction_values[selected_labels] * weights.unsqueeze(2)
        ).sum(dim=1)
        return weights, prediction

    weights, prediction = aggregate(selected)
    restored_weights, restored_prediction = aggregate(restored)
    assert torch.equal(weights, restored_weights)
    assert torch.equal(prediction, restored_prediction)
    assert torch.isfinite(prediction).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_batched_cuda_mmr_repeated_call_is_query_local():
    tool = make_tool(topm=9, pool=29, mmr_device='cuda')
    similarities, lambda_q, query_ids = cuda_mmr_case()
    first = tool.mmr_select_batch_cuda(
        similarities, lambda_q, 2, query_ids, False
    )

    other_similarities, other_lambda, other_ids = cuda_mmr_case(rows=11)
    tool.mmr_select_batch_cuda(
        other_similarities.flip(1), other_lambda, 4, other_ids + 1000, True
    )
    repeated = tool.mmr_select_batch_cuda(
        similarities, lambda_q, 2, query_ids, False
    )
    assert torch.equal(first, repeated)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_batched_cuda_mmr_is_invariant_to_batch_permutation():
    tool = make_tool(topm=9, pool=29, mmr_device='cuda')
    similarities, lambda_q, query_ids = cuda_mmr_case()
    original = tool.mmr_select_batch_cuda(
        similarities, lambda_q, 2, query_ids, False
    )

    permutation = torch.tensor(
        [8, 1, 15, 3, 12, 0, 16, 5, 10, 2, 14, 6, 9, 4, 13, 7, 11]
    )
    permutation_cuda = permutation.cuda()
    permuted = tool.mmr_select_batch_cuda(
        similarities[permutation_cuda],
        lambda_q[permutation_cuda],
        2,
        query_ids[permutation.numpy()],
        False,
    )
    restored = permuted[torch.argsort(permutation).cuda()]
    assert torch.equal(original, restored)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_batched_cuda_mmr_is_invariant_to_batch_partition():
    tool = make_tool(topm=9, pool=29, mmr_device='cuda')
    similarities, lambda_q, query_ids = cuda_mmr_case()
    full = tool.mmr_select_batch_cuda(
        similarities, lambda_q, 2, query_ids, False
    )

    chunks = []
    for start, end in [(0, 3), (3, 12), (12, 17)]:
        chunks.append(tool.mmr_select_batch_cuda(
            similarities[start:end], lambda_q[start:end], 2,
            query_ids[start:end], False,
        ))
    assert torch.equal(full, torch.cat(chunks, dim=0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_batched_cuda_mmr_is_independent_of_global_rng():
    tool = make_tool(topm=9, pool=29, mmr_device='cuda')
    similarities, lambda_q, query_ids = cuda_mmr_case()

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    state_before = torch.cuda.get_rng_state()
    first = tool.mmr_select_batch_cuda(
        similarities, lambda_q, 2, query_ids, False
    )
    assert torch.equal(state_before, torch.cuda.get_rng_state())

    torch.manual_seed(9999)
    torch.cuda.manual_seed_all(9999)
    state_before = torch.cuda.get_rng_state()
    second = tool.mmr_select_batch_cuda(
        similarities, lambda_q, 2, query_ids, False
    )
    assert torch.equal(state_before, torch.cuda.get_rng_state())
    assert torch.equal(first, second)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_batched_cuda_mmr_validity_and_step_semantics():
    tool = make_tool(topm=11, pool=31, mmr_device='cuda')
    generator = torch.Generator().manual_seed(88)
    similarities = (
        torch.rand(23, 31, generator=generator) * 2.0 - 1.0
    ).cuda()
    lambda_q = (
        torch.rand(23, generator=generator) * 0.6 + 0.3
    ).cuda()
    query_ids = np.arange(23)
    with mock.patch(
        'torch.multinomial',
        side_effect=AssertionError('CUDA MMR must use inverse-CDF sampling'),
    ):
        selected = tool.mmr_select_batch_cuda(
            similarities, lambda_q, 0, query_ids, False
        )
    selected_cpu = selected.cpu()
    assert selected.shape == (23, 11)
    assert torch.equal(selected_cpu[:, 0], torch.zeros(23, dtype=torch.long))
    assert ((selected_cpu >= 0) & (selected_cpu < 31)).all()
    assert all(len(torch.unique(row)) == 11 for row in selected_cpu)

    uniforms = tool._mmr_stateless_uniforms(query_ids, 0, False)
    selected_mask = torch.zeros_like(similarities, dtype=torch.bool)
    selected_mask[:, 0] = True
    running_redundancy = 1.0 - torch.abs(
        similarities - similarities[:, 0:1]
    )
    for step in range(1, tool.topm):
        scores = (
            lambda_q.unsqueeze(1) * similarities
            - (1.0 - lambda_q.unsqueeze(1)) * running_redundancy
        )
        scores.masked_fill_(selected_mask, float('-inf'))
        probabilities = F.softmax(scores / 0.1, dim=1)
        assert torch.isfinite(probabilities).all()
        assert torch.allclose(
            probabilities.sum(dim=1),
            torch.ones(23, device='cuda'),
            rtol=1e-6,
            atol=1e-6,
        )
        for prior_step in range(step):
            prior = selected[:, prior_step:prior_step + 1]
            assert torch.equal(
                probabilities.gather(1, prior),
                torch.zeros(23, 1, device='cuda'),
            )
        reference_cdf = probabilities.cumsum(dim=1)
        reference_cdf = reference_cdf / reference_cdf[:, -1:].clamp_min(
            torch.finfo(reference_cdf.dtype).tiny
        )
        reference_cdf[:, -1] = 1.0
        expected = torch.searchsorted(
            reference_cdf,
            uniforms[:, step - 1:step].contiguous(),
            right=False,
        ).squeeze(1)
        assert torch.equal(selected[:, step], expected)
        selected_mask.scatter_(1, selected[:, step:step + 1], True)
        chosen = similarities.gather(1, selected[:, step:step + 1])
        running_redundancy = torch.maximum(
            running_redundancy,
            1.0 - torch.abs(similarities - chosen),
        )
        assert torch.isfinite(running_redundancy).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
@pytest.mark.parametrize('postprocess_device', ['cpu', 'cuda'])
def test_end_to_end_cuda_mmr_retrieval_is_finite(postprocess_device):
    data = SyntheticDataset()
    tool = make_tool(
        postprocess_device=postprocess_device,
        mmr_device='cuda',
        batch_size=13,
    )
    tool.prepare_dataset(data)
    x = torch.from_numpy(data.x[:13])
    labels, similarities = neighbor_tensors(tool, x)
    controller = tool.compute_profile_reliability_batch(x, labels, train=False)
    selected = tool.mmr_select_batch_cuda(
        similarities.cuda(), controller[2], 0, np.arange(13), False
    ).cpu()
    assert all(len(torch.unique(row)) == tool.topm for row in selected)
    prediction = tool.retrieve(x, torch.arange(13), train=False)
    assert prediction.shape == (1, 13, 2, 1)
    assert torch.isfinite(prediction).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_end_to_end_value_cuda_mmr_retrieval_is_finite_and_chunk_invariant():
    data = SyntheticDataset()
    tool = make_tool(
        postprocess_device='cuda', mmr_device='cuda', batch_size=13,
        redundancy_source='value', value_chunk_size=4,
    )
    tool.prepare_dataset(data)
    x = torch.from_numpy(data.x[:13])
    first = tool.retrieve(x, torch.arange(13), train=False)
    tool.value_mmr_chunk_size = 8
    second = tool.retrieve(x, torch.arange(13), train=False)
    assert first.shape == (1, 13, 2, 1)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
