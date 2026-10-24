"""Product reliability, train-only scaling, and unchanged-neighborhood checks."""

from unittest import mock

import numpy as np
import pytest
import torch

from src.utils.candidate_reliability import future_consistency, panel_value_iqr_scale, value_iqr_scale
from src.utils.profile_reliability import map_reliability
from src.layers.Retrieval import RetrievalTool
from test_retrieval_runtime import SyntheticDataset
from reliability_fixtures import fixture, make_tool, windows
from src.common.webscale_ts import WindowManifest
from reliability_reference import cr_reference


def test_pair_average_excludes_self_and_keeps_level_information():
    futures = torch.tensor([[[[0.], [1.]], [[2.], [3.]], [[4.], [5.]]]])
    expected = (2 * np.exp(-2.) + np.exp(-4.)) / 3
    actual = future_consistency(futures, torch.ones(1))
    assert actual.item() == pytest.approx(expected, rel=1e-6)
    assert future_consistency(futures[:, :1], torch.ones(1)).item() == 1.
    same = futures[:, :1].expand(-1, 3, -1, -1)
    assert future_consistency(same, torch.ones(1)).item() == 1.
    assert future_consistency(futures, torch.ones(1), tau_v=2).item() > actual.item()


@pytest.mark.parametrize('tau_v', [0., -1., float('nan'), float('inf')])
def test_invalid_temperature(tau_v):
    with pytest.raises(ValueError, match='tau_v'):
        future_consistency(torch.zeros(1, 2, 2, 1), torch.ones(1), tau_v)


@pytest.mark.parametrize('stride', [1, 3, 8, 11])
def test_panel_scale_matches_expanded_candidate_future_population(stride):
    data = fixture()['train']
    manifest = WindowManifest.from_bounds(data.series_bank, 16, 8, stride=stride, end_bound=72)
    sid, start = manifest.resolve_ids(np.arange(len(manifest)))
    full = torch.from_numpy(data.series_bank.get_windows(sid, start + 16, 8))
    assert torch.equal(panel_value_iqr_scale(data.series_bank, manifest), value_iqr_scale(full))
    # Poison all held-out observations; training-only scale must not change.
    expected = panel_value_iqr_scale(data.series_bank, manifest)
    for series in data.series_bank.series_ids:
        data.series_bank.get_series(series)[72:] = 1e9
    assert torch.equal(panel_value_iqr_scale(data.series_bank, manifest), expected)


def test_general_product_changes_with_candidate_futures_not_A_q():
    data = SyntheticDataset(channels=2)
    tool = RetrievalTool(seq_len=6, pred_len=2, channels=2, n_period=1, topm=2,
                         retrieval_controller='crest', retrieval_backend='exact',
                         mmr_candidate_pool=4, profile_sections=3, profile_reliability_k=2)
    tool.prepare_dataset(data)
    similarity = torch.tensor([[.1, .9, .8, .7]])
    labels = torch.tensor([[0, 2, 4, 6]])
    cr_q, sigma_q, lambda_q = tool._candidate_reliability(similarity, labels)
    components = {k: v.clone() for k, v in tool.last_reliability_components.items()}
    assert components['A_q'].item() == pytest.approx(.85)
    expected = cr_reference(similarity[0], tool.raw_future_bank[labels[0]], 2,
                            torch.from_numpy(data.y), tool=tool, candidate_labels=labels[0])
    torch.testing.assert_close(cr_q[0], expected)
    torch.testing.assert_close(cr_q, components['A_q'] * components['V_q'])
    mapped = map_reliability(cr_q, .05, .30, .30, .90)
    assert torch.equal(sigma_q, mapped[0]) and torch.equal(lambda_q, mapped[1])
    # Equal future shapes with different levels must affect V_q.
    tool.raw_future_bank[4] = tool.raw_future_bank[2] + 100
    changed, _, _ = tool._candidate_reliability(similarity, labels)
    assert torch.equal(tool.last_reliability_components['A_q'], components['A_q'])
    assert changed.item() < cr_q.item()
    # The neighborhood is still STP top-N, not retrieval top-N.
    tool.raw_future_bank[0] = 1e9
    again, _, _ = tool._candidate_reliability(similarity, labels)
    assert torch.equal(changed, again)


@pytest.mark.parametrize('backend', ['exact', 'hnsw'])
@pytest.mark.parametrize('weighting', ['gaussian', 'adaptive_softmax'])
def test_panel_uses_training_candidates_and_cr_for_aggregation(tmp_path, backend, weighting):
    datasets = fixture()
    tool = make_tool(retrieval_backend=backend, crest_weighting=weighting)
    tool.prepare_dataset(datasets['train'])
    tool.prepare_panel_crest_state(datasets['train'], tmp_path)
    try:
        data = datasets['valid']
        tool._active_panel_query_manifest = data.query_manifest
        x = windows(data, [0, 1, 2])
        with mock.patch.object(tool, 'crest_weights', wraps=tool.crest_weights) as aggregate:
            predictions = tool.retrieve(x, np.arange(3), train=False)
        assert torch.isfinite(predictions).all()
        assert torch.equal(aggregate.call_args.args[1], tool.last_reliability_components['cr_q'])
        assert torch.all(tool.last_reliability_components['cr_q'] <= tool.last_reliability_components['A_q'])
        full = windows(datasets['train'], np.arange(tool.n_train), candidate=True, future=True)
        labels, _ = tool.search_candidates(x, 0, np.arange(3), False)
        result = tool._panel_profile_reliability(x, labels)
        expected = torch.stack([cr_reference(result[4][r], full[labels[r]],
                               tool.profile_reliability_k, full, tool=tool, candidate_labels=labels[r]) for r in range(3)])
        torch.testing.assert_close(result[0], expected)
        # Query future is absent from the controller API and only train IDs are gathered.
        original = tool.panel_series_bank.get_windows
        def train_only(series, starts, length):
            assert np.all(starts + length <= 72)
            return original(series, starts, length)
        with mock.patch.object(tool.panel_series_bank, 'get_windows', side_effect=train_only):
            tool._panel_profile_reliability(x, labels)
    finally:
        tool.cleanup_panel_crest_state()


def test_chunk_size_does_not_change_controller():
    data = SyntheticDataset()
    tool = RetrievalTool(seq_len=6, pred_len=2, channels=1, n_period=1, topm=3,
                         retrieval_controller='crest', retrieval_backend='exact',
                         mmr_candidate_pool=6, profile_sections=3)
    tool.prepare_dataset(data)
    x = torch.from_numpy(data.x[:5])
    labels, _ = tool.search_candidates(x, 0, np.arange(5), False)
    reference = tool.compute_profile_reliability_batch(x, labels)
    tool.retrieval_aggregation_chunk_size = 1
    actual = tool.compute_profile_reliability_batch(x, labels)
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b)


def test_cli_model_and_result_identity():
    from src.run import build_parser, experiment_setting
    from src.models.RAFT import Model
    from test_retrieval_runtime import make_config
    from src.utils.profile_reliability import validate_profile_args
    args = build_parser().parse_args(['--model_id', 'cr-test', '--data', 'custom',
                                     '--enc_in', '1', '--retrieval_controller', 'crest'])
    first = experiment_setting(args, 0)
    args.tau_v = 2.0
    assert experiment_setting(args, 0) != first
    config = make_config()
    config.tau_v = 2.0
    model = Model(config)
    assert model.rt.tau_v == 2.0
    data = SyntheticDataset()
    model.prepare_dataset(data, data, data)
    assert model.reliability_configuration['formula'] == 'cr_q=A_q*V_q'
    assert model.reliability_configuration['tau_v'] == 2.0
    args.tau_v = float('nan')
    with pytest.raises(ValueError, match='tau_v'):
        validate_profile_args(args)
