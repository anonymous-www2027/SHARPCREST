"""Temporal separation of reliability neighbors, independent of query masking."""

import numpy as np
import pytest
import torch

from src.utils.candidate_reliability import spaced_neighborhood, future_consistency
from reliability_fixtures import fixture, make_tool, windows


def select(scores, series, starts, k, spacing):
    positions, valid = spaced_neighborhood(torch.tensor([scores]), np.array([series]),
                                          np.array([starts]), k, spacing)
    return positions[0][valid[0]].tolist()


def test_same_series_overlap_boundary_and_cross_series():
    assert select([.9, .8, .7, .6], [1, 1, 1, 2], [100, 110, 148, 100], 4, 48) == [0, 2, 3]


def test_checks_all_previously_selected_neighbors():
    # Third item is far from the most recent neighbor but too close to the first.
    assert select([.9, .8, .7, .6], [1]*4, [100, 300, 110, 200], 4, 48) == [0, 1, 3]


def test_checks_full_pool_in_stp_order_without_relaxation():
    assert select([.1, .9, .8, .7], [0]*4, [200, 100, 110, 160], 2, 48) == [1, 3]
    assert select([.1, .9, .8, .7], [0]*4, [101, 100, 110, 120], 4, 48) == [1]
    # Ties follow existing pool order deterministically.
    assert select([1., 1., 1.], [0]*3, [0, 1, 48], 3, 48) == [0, 2]


def test_padding_never_enters_future_consistency():
    future = torch.tensor([[[[0.]], [[2.]], [[1000.]]],
                           [[[1.]], [[100.]], [[500.]]]])
    valid = torch.tensor([[True, True, False], [True, False, False]])
    actual = future_consistency(future, torch.ones(1), valid_mask=valid)
    torch.testing.assert_close(actual, torch.tensor([np.exp(-2), 1.], dtype=torch.float32))
    with pytest.raises(ValueError, match='at least one'):
        future_consistency(future, torch.ones(1), valid_mask=torch.zeros_like(valid))


@pytest.mark.parametrize('seed', range(5))
def test_vectorized_selection_matches_independent_greedy(seed):
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(9, 15)).astype(np.float32)
    series = rng.integers(0, 3, size=scores.shape)
    starts = rng.integers(0, 40, size=scores.shape)
    positions, valid = spaced_neighborhood(torch.from_numpy(scores), series, starts, 7, 10)
    for row in range(len(scores)):
        expected = []
        for p in np.argsort(-scores[row]):
            if all(series[row, p] != series[row, q] or abs(starts[row, p] - starts[row, q]) >= 10
                   for q in expected):
                expected.append(int(p))
                if len(expected) == 7:
                    break
        assert positions[row][valid[row]].tolist() == expected


def test_panel_uses_series_coordinates_and_actual_stride(tmp_path):
    data = fixture()['train']  # candidate stride=3, H=8
    tool = make_tool(retrieval_backend='exact', profile_reliability_k=4)
    tool.prepare_dataset(data)
    tool.prepare_panel_crest_state(data, tmp_path)
    try:
        m = data.candidate_manifest
        ids = m.encode(np.array([0, 0, 0, 1]), np.array([0, 3, 9, 0]))
        # Labels 0,1,3,17: label distance is not timestamp distance.
        scores = torch.tensor([[.9, .8, .7, .6]])
        labels = torch.from_numpy(ids).unsqueeze(0)
        cr_q, _, _ = tool._candidate_reliability(scores, labels)
        components = tool.last_reliability_components
        assert components['N_q_size'].item() == 3
        assert components['A_q'].item() == pytest.approx((.9 + .7 + .6)/3)
        torch.testing.assert_close(cr_q, components['A_q'] * components['V_q'])
        # Only one neighbor survives a cluster; rejected scores do not dilute A_q.
        clustered = torch.tensor([[0, 1, 2, 1]])
        tool._candidate_reliability(scores, clustered)
        components = tool.last_reliability_components
        assert components['N_q_size'].item() == 1
        assert components['A_q'].item() == pytest.approx(.9)
        assert components['V_q'].item() == 1.
        torch.testing.assert_close(components['cr_q'], components['A_q'])
    finally:
        tool.cleanup_panel_crest_state()
