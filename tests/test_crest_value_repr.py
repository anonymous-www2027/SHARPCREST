from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src import run
from src.layers.Retrieval import RetrievalTool
from src.models.RAFT import Model
from test_retrieval_runtime import SyntheticDataset, make_config, reference_value_mmr
from src.utils.value_mmr import paa_value_patch, validate_value_mmr_args


class ValueDataset(SyntheticDataset):
    def __init__(self, count=80, offset=0):
        generator = np.random.default_rng(100 + offset)
        self.x = generator.normal(size=(count, 16, 2)).astype(np.float32)
        self.y = generator.normal(size=(count, 8, 2)).astype(np.float32)


def make_tool(**kwargs):
    options = dict(seq_len=16, pred_len=8, channels=2, n_period=2, topm=4,
                   mmr_candidate_pool=8, profile_sections=4,
                   retrieval_backend='exact', retrieval_controller='crest',
                   mmr_redundancy_source='value', retrieval_seed=17,
                   mmr_paa_segments=3, mmr_rp_dim=5, mmr_rp_seed=11)
    options.update(kwargs)
    return RetrievalTool(**options)


@pytest.mark.parametrize('channels,length,segments', [(1, 8, 4), (3, 11, 3)])
def test_paa_manual_means_remainder_and_normalized_bank(channels, length, segments):
    values = torch.arange(1, 1 + 2 * length * channels, dtype=torch.float32).reshape(2, length, channels)
    width = length // segments
    expected = torch.from_numpy(np.stack([
        values.numpy()[:, i * width: (i + 1) * width if i < segments - 1 else length].mean(1)
        for i in range(segments)], axis=1))
    actual = paa_value_patch(values, segments)
    assert torch.equal(actual, expected)
    if channels == 1:
        assert torch.equal(actual[0, :, 0], torch.tensor([1.5, 3.5, 5.5, 7.5]))
    tool = make_tool(pred_len=length, channels=channels, n_period=1,
                     mmr_value_repr='paa', mmr_paa_segments=segments)
    bank = tool._build_value_mmr_bank(values)
    centered = expected.flatten(1) - expected.flatten(1).mean(1, keepdim=True)
    assert bank.shape == (2, segments * channels)
    assert torch.equal(bank, F.normalize(centered, dim=-1, eps=1e-12))
    assert torch.allclose(bank.norm(dim=1), torch.ones(2))


def test_rp_seed_geometry_normalization_and_no_global_rng_pollution():
    values = torch.randn(1031, 8, 2, generator=torch.Generator().manual_seed(21))
    values[0] = 3  # Constant input stays zero after centering.
    tool = make_tool(mmr_value_repr='random_projection')
    state = torch.get_rng_state().clone()
    actual = tool._build_value_mmr_bank(values)
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(actual, tool._build_value_mmr_bank(values))
    other = make_tool(mmr_value_repr='random_projection', mmr_rp_seed=12)
    assert not torch.equal(actual, other._build_value_mmr_bank(values))
    generator = torch.Generator(device='cpu').manual_seed(11)
    projection = torch.randn(16, 5, generator=generator) / np.sqrt(5)
    centered = values.flatten(1) - values.flatten(1).mean(1, keepdim=True)
    expected = F.normalize(centered @ projection, dim=-1, eps=1e-12)
    assert actual.shape == (1031, 5)
    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.equal(actual[0], torch.zeros(5))
    assert torch.allclose(actual[1:].norm(dim=1), torch.ones(1030), atol=1e-6)


@pytest.mark.parametrize('backend', ['exact', 'hnsw'])
@pytest.mark.parametrize('source,mode', [('key', 'raw'), ('value', 'raw'),
                                        ('value', 'paa'), ('value', 'random_projection')])
def test_all_modes_precompute_once_then_retrieve_all_splits(backend, source, mode):
    tool = make_tool(retrieval_backend=backend, mmr_redundancy_source=source, mmr_value_repr=mode)
    data = ValueDataset()
    with mock.patch.object(tool, '_build_value_mmr_bank', wraps=tool._build_value_mmr_bank) as build:
        tool.prepare_dataset(data)
        assert build.call_count == (2 if source == 'value' else 0)
    if source == 'value':
        dim = {'raw': 16, 'paa': 6, 'random_projection': 5}[mode]
        assert all(bank.shape == (80, dim) for bank in tool.value_norm_banks)
        for bank in tool.value_norm_banks:
            assert torch.allclose(bank.norm(dim=1), torch.ones(80), atol=1e-6)
    else:
        assert tool.value_norm_banks == []
    with mock.patch.object(tool, '_build_value_mmr_bank', side_effect=AssertionError), \
            mock.patch.object(tool, '_normalize_value_vectors', side_effect=AssertionError):
        for split in ('train', 'valid', 'test'):
            result = tool.retrieve_all(data, train=split == 'train', split=split)
            assert result.shape == (2, 80, 8, 2)
            assert torch.isfinite(result).all()


@pytest.mark.parametrize('backend', ['exact', 'hnsw'])
def test_baseline_skips_all_crest_preparation_and_matches_similarity_aggregation(backend):
    data = ValueDataset()
    tool = make_tool(retrieval_backend=backend, retrieval_controller='baseline',
                     crest_profile='invalid', stp_state_weight=float('nan'),
                     mmr_value_repr='random_projection', mmr_rp_dim=-1, mmr_rp_seed=-1)
    with mock.patch('src.layers.Retrieval.compute_stp_profile', side_effect=AssertionError), \
            mock.patch('src.layers.Retrieval.compute_mean_profile', side_effect=AssertionError), \
            mock.patch.object(tool, '_build_value_mmr_bank', side_effect=AssertionError):
        tool.prepare_dataset(data)
    assert tool.profile_bank is None and tool.value_norm_banks == []
    query = torch.from_numpy(data.x[30:35])
    ids = np.arange(30, 35)
    for train in (False, True):
        reference = []
        for period_idx, values in enumerate(tool.y_data_all_mg):
            labels, similarity = tool.search_candidates(query, period_idx, ids, train)
            weights = F.softmax(similarity / tool.temperature, dim=1)
            reference.append((values.flatten(1)[labels] * weights.unsqueeze(-1)).sum(1))
        with mock.patch.object(tool, 'compute_profile_reliability_batch', side_effect=AssertionError), \
                mock.patch.object(tool, 'mmr_select_from_pool', side_effect=AssertionError), \
                mock.patch.object(tool, 'gaussian_weights_batch', side_effect=AssertionError):
            actual = tool.retrieve(query, ids, train=train)
        assert torch.equal(actual, torch.stack(reference).reshape(2, 5, 8, 2))


def test_raw_bank_selection_and_retrieval_reference():
    data = ValueDataset()
    tool = make_tool(crest_profile='mean')
    tool.prepare_dataset(data)
    query = torch.from_numpy(data.x[30:35])
    ids = np.arange(30, 35)
    predictions = []
    for period_idx, values in enumerate(tool.y_data_all_mg):
        assert torch.equal(tool.value_norm_banks[period_idx], tool._normalized_value_bank(values))
        labels, similarities = tool.search_candidates(query, period_idx, ids, True)
        _, sigma, lam, _, _ = tool.compute_profile_reliability_batch(query, labels, ids, True)
        positions = []
        for row in range(5):
            expected = reference_value_mmr(tool, similarities[row], lam[row],
                                           values[labels[row]], period_idx, ids[row], True)
            actual = tool.mmr_select_from_pool(similarities[row], lam[row], period_idx, ids[row], True,
                                               tool.value_norm_banks[period_idx][labels[row]], True)
            assert torch.equal(actual, expected)
            positions.append(actual)
        positions = torch.stack(positions)
        _, weights = tool.gaussian_weights_batch(similarities.gather(1, positions), sigma)
        assert torch.allclose(weights.sum(1), torch.ones(5))
        predictions.append((values.flatten(1)[labels.gather(1, positions)] * weights.unsqueeze(-1)).sum(1))
    assert torch.equal(tool.retrieve(query, ids, True), torch.stack(predictions).reshape(2, 5, 8, 2))


@pytest.mark.parametrize('mode,params', [
    ('random_projection', {'mmr_rp_dim': 0}),
    ('random_projection', {'mmr_rp_dim': 16}),
    ('random_projection', {'mmr_rp_dim': 17}),
    ('random_projection', {'mmr_rp_dim': 1.5}),
    ('random_projection', {'mmr_rp_seed': -1}),
    ('random_projection', {'mmr_rp_seed': 2**63}),
    ('random_projection', {'mmr_rp_seed': 1.2}),
    ('paa', {'mmr_paa_segments': 1}), ('paa', {'mmr_paa_segments': 5}),
    ('paa', {'mmr_paa_segments': 2.5}),
])
def test_active_validation_and_inactive_options(mode, params):
    with pytest.raises(ValueError):
        make_tool(mmr_value_repr=mode, **params)
    args = run.build_parser().parse_args(['--model_id', 'test', '--data', 'custom', '--enc_in', '2',
                                         '--pred_len', '8', '--retrieval_controller', 'crest'])
    args.mmr_value_repr = mode
    args.mmr_rp_dim = 5
    args.mmr_paa_segments = 3
    vars(args).update(params)
    with pytest.raises(ValueError):
        validate_value_mmr_args(args)
    make_tool(mmr_value_repr='raw', **params)
    make_tool(mmr_value_repr=mode, mmr_redundancy_source='key', **params)
    make_tool(mmr_value_repr=mode, retrieval_controller='baseline', **params)
    for controller, source, representation in [('baseline', 'value', mode), ('crest', 'key', mode),
                                              ('crest', 'value', 'raw')]:
        args.retrieval_controller, args.mmr_redundancy_source, args.mmr_value_repr = controller, source, representation
        validate_value_mmr_args(args)


def test_cli_defaults_forwarding_and_only_active_setting_fields():
    args = run.build_parser().parse_args(['--model_id', 'test', '--data', 'custom', '--enc_in', '2'])
    assert (args.mmr_value_repr, args.mmr_paa_segments, args.mmr_rp_dim, args.mmr_rp_seed) == ('raw', 8, 32, 0)
    baseline = run.experiment_setting(args, 0)
    args.mmr_value_repr = 'paa'
    assert run.experiment_setting(args, 0) == baseline
    args.retrieval_controller = 'crest'
    args.mmr_redundancy_source = 'key'
    key = run.experiment_setting(args, 0)
    args.mmr_value_repr = 'random_projection'
    assert run.experiment_setting(args, 0) == key
    args.mmr_redundancy_source = 'value'
    settings = set()
    for mode, dim, seed, segments in [('raw', 32, 0, 8), ('paa', 32, 0, 8), ('paa', 32, 0, 9),
                                     ('random_projection', 32, 0, 8), ('random_projection', 31, 0, 8),
                                     ('random_projection', 32, 1, 8)]:
        args.mmr_value_repr, args.mmr_rp_dim, args.mmr_rp_seed, args.mmr_paa_segments = mode, dim, seed, segments
        settings.add(run.experiment_setting(args, 0))
    assert len(settings) == 6
    config = make_config()
    config.mmr_redundancy_source, config.mmr_value_repr = 'value', 'random_projection'
    config.mmr_rp_dim, config.mmr_rp_seed = 1, 19
    tool = Model(config).rt
    assert (tool.mmr_value_repr, tool.mmr_rp_dim, tool.mmr_rp_seed) == ('random_projection', 1, 19)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('mode', ['raw', 'paa', 'random_projection'])
def test_cuda_all_representations_stateless_selection_and_pipeline(mode):
    data = ValueDataset()
    tool = make_tool(mmr_value_repr=mode, mmr_device='cuda', retrieval_postprocess_device='cuda')
    tool.prepare_dataset(data)
    cpu = make_tool(mmr_value_repr=mode)
    cpu.prepare_dataset(data)
    query = torch.from_numpy(data.x[30:37])
    ids = np.arange(30, 37)
    for period_idx, bank in enumerate(tool.value_norm_banks):
        assert torch.equal(bank, cpu.value_norm_banks[period_idx])
        labels, similarities = tool.search_candidates(query, period_idx, ids, False)
        controller = tool.compute_profile_reliability_batch(query, labels)
        cpu_controller = cpu.compute_profile_reliability_batch(query, labels)
        for a, b in zip(controller, cpu_controller):
            assert torch.allclose(a.cpu(), b, atol=2e-6, rtol=2e-6)
        def select(labels=labels, sims=similarities, lam=controller[2], query_ids=ids):
            return tool.mmr_select_batch_cuda_value_chunked(sims.cuda(), lam, labels, bank,
                                                           period_idx, query_ids, False)
        state = torch.cuda.get_rng_state().clone()
        selected = select()
        assert torch.equal(state, torch.cuda.get_rng_state())
        tool.value_mmr_chunk_size = 1
        assert torch.equal(selected, select())
        permutation = torch.tensor([3, 1, 6, 0, 5, 2, 4])
        permuted = select(labels[permutation], similarities[permutation], controller[2][permutation.cuda()], ids[permutation])
        assert torch.equal(selected, permuted[torch.argsort(permutation).cuda()])
        assert torch.equal(selected[:, 0], torch.zeros(7, dtype=torch.long, device='cuda'))
        assert all(len(row.unique()) == tool.topm for row in selected)
        # CPU and CUDA intentionally use different sampling RNGs. Check both
        # paths' validity; equality of stochastic selections is not a contract.
        cpu_positions = cpu.mmr_select_from_pool(similarities[0], cpu_controller[2][0], period_idx,
                                                 ids[0], False, bank[labels[0]], True)
        assert len(cpu_positions.unique()) == tool.topm
    first = tool.retrieve(query, ids, False)
    tool.value_mmr_chunk_size = 4
    assert torch.equal(first, tool.retrieve(query, ids, False))
    assert first.shape == (2, 7, 8, 2) and torch.isfinite(first).all()
