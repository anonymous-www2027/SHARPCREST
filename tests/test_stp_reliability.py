"""Mathematical and API contracts for the standalone STP utility."""

import numpy as np
import pytest
import torch

from src.utils.profile_reliability import section_boundaries
from src.utils.stp_reliability import (
    STP_FEATURE_NAMES,
    compute_stp_iqr_scale,
    compute_stp_profile,
    robust_stp_distance,
    stp_distance_to_similarity,
)


def numpy_profile(data, sections):
    """Independent float64 reference with explicit per-window/channel OLS."""
    data = np.asarray(data, dtype=np.float64)
    result = np.empty((data.shape[0], sections, data.shape[2], 3))
    for b, (start, end) in enumerate(section_boundaries(data.shape[1], sections)):
        t = np.arange(end - start, dtype=np.float64)
        for n in range(data.shape[0]):
            for c in range(data.shape[2]):
                x = data[n, start:end, c]
                alpha = np.sum((t - t.mean()) * (x - x.mean())) / np.sum((t - t.mean()) ** 2)
                beta = x.mean() - alpha * t.mean()
                residual = x - (alpha * t + beta)
                result[n, b, c] = (
                    np.median(x),
                    np.quantile(residual, 0.75) - np.quantile(residual, 0.25),
                    alpha * (len(t) - 1),
                )
    return result


def repeated_segment(values):
    return torch.tensor(values, dtype=torch.float32).repeat(2).view(1, -1, 1)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_shape_dtype_and_feature_order(dtype):
    data = torch.randn(4, 19, 5, generator=torch.Generator().manual_seed(11), dtype=dtype)
    result = compute_stp_profile(data, 3)
    assert result.shape == (4, 3, 5, 3)
    assert result.dtype == dtype
    assert result.device == data.device
    assert STP_FEATURE_NAMES == ('state', 'dispersion', 'transition')
    assert torch.isfinite(result).all()


def test_existing_boundaries_and_final_remainder():
    assert section_boundaries(11, 3) == [(0, 3), (3, 6), (6, 11)]
    data = torch.arange(11, dtype=torch.float32).view(1, 11, 1)
    result = compute_stp_profile(data, 3)[0, :, 0]
    torch.testing.assert_close(result, torch.tensor([[1., 0., 2.], [4., 0., 2.], [8., 0., 4.]]))


@pytest.mark.parametrize('values, expected', [
    ([1, 2, 3, 4], [2.5, 0, 3]),
    ([0, 2, 4, 6, 8], [4, 0, 8]),
    ([8, 6, 4, 2, 0], [4, 0, -8]),
    ([5, 5, 5, 5, 5], [5, 0, 0]),
    ([2, 8], [5, 0, 6]),
])
def test_exact_segment_features(values, expected):
    profile = compute_stp_profile(repeated_segment(values), 2)
    torch.testing.assert_close(profile[0, :, 0], torch.tensor([expected, expected], dtype=torch.float32))


def test_irregular_fluctuation_after_detrending():
    linear = compute_stp_profile(repeated_segment([0, 2, 4, 6, 8]), 2)
    # Noise is symmetric, so the fitted slope and endpoints are unchanged.
    noisy = compute_stp_profile(repeated_segment([0, 4, 0, 8, 8]), 2)
    assert torch.all(noisy[..., 1] > linear[..., 1])
    torch.testing.assert_close(noisy[..., 2], linear[..., 2])


def test_same_distribution_different_temporal_order():
    a = compute_stp_profile(repeated_segment([8, 9, 10, 11, 12]), 2)
    b = compute_stp_profile(repeated_segment([10, 8, 12, 9, 11]), 2)
    torch.testing.assert_close(a[..., 0], b[..., 0])
    assert torch.all(b[..., 1] > a[..., 1])
    assert not torch.allclose(a[..., 2], b[..., 2])


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_translation_separates_state_from_trend_and_dispersion(dtype):
    data = torch.randn(3, 23, 4, generator=torch.Generator().manual_seed(12), dtype=dtype)
    offset = torch.tensor([12., -5., 0.25, 32.], dtype=dtype)
    before = compute_stp_profile(data, 4)
    after = compute_stp_profile(data + offset, 4)
    torch.testing.assert_close(after[..., 0], before[..., 0] + offset, atol=5e-6, rtol=1e-6)
    torch.testing.assert_close(after[..., 1:], before[..., 1:], atol=5e-6, rtol=1e-6)


@pytest.mark.parametrize('channels', [2, 5])
def test_multivariate_independence(channels):
    time = np.arange(15, dtype=np.float64)
    data = np.stack([
        10 * c + (c - 2) * time + c * np.sin(time)
        for c in range(channels)
    ], axis=-1)[None]
    profile = compute_stp_profile(torch.from_numpy(data), 3)
    np.testing.assert_allclose(profile.numpy(), numpy_profile(data, 3), atol=1e-12, rtol=1e-12)
    for c in range(channels):
        separate = compute_stp_profile(torch.from_numpy(data[:, :, c:c + 1]), 3)
        torch.testing.assert_close(profile[:, :, c:c + 1], separate)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('length, sections', [(8, 2), (19, 3), (23, 5)])
def test_numpy_reference(dtype, length, sections):
    data = torch.tensor(np.random.default_rng(72).normal(size=(4, length, 3)), dtype=dtype)
    # A non-contiguous input also exercises vectorized channel/time handling.
    data = data.transpose(1, 2).contiguous().transpose(1, 2)
    assert not data.is_contiguous()
    actual = compute_stp_profile(data, sections)
    tol = 1e-6 if dtype == torch.float32 else 1e-12
    np.testing.assert_allclose(actual.numpy(), numpy_profile(data.numpy(), sections), atol=tol, rtol=tol)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_training_coordinate_iqr_scale(dtype):
    profiles = torch.tensor(np.random.default_rng(8).normal(size=(8, 3, 4, 3)), dtype=dtype)
    profiles[..., 2] *= 12
    profiles[:, 0, 0, 1] = 7  # A constant coordinate uses eps exactly.
    eps = 1e-5
    scale = compute_stp_iqr_scale(profiles, eps)
    expected = np.quantile(profiles.numpy(), 0.75, axis=0) - np.quantile(profiles.numpy(), 0.25, axis=0) + eps
    assert scale.shape == (3, 4, 3)
    assert scale.dtype == dtype
    assert scale.device == profiles.device
    np.testing.assert_allclose(scale.numpy(), expected, atol=1e-6, rtol=1e-6)
    assert scale[0, 0, 1].item() == pytest.approx(eps)


def test_one_training_candidate_uses_epsilon():
    scale = compute_stp_iqr_scale(torch.ones(1, 2, 1, 3))
    torch.testing.assert_close(scale, torch.full((2, 1, 3), 1e-8))


def test_identical_query_candidate():
    profile = compute_stp_profile(torch.randn(1, 12, 2), 3)
    distance, components = robust_stp_distance(
        profile[0], profile, compute_stp_iqr_scale(profile), [0.2, 0.3, 0.5], return_components=True,
    )
    assert distance.shape == (1,)
    assert components.shape == (1, 3)
    assert torch.equal(distance, torch.zeros_like(distance))
    assert torch.equal(components, torch.zeros_like(components))


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_single_feature_distance(axis):
    query = torch.zeros(2, 2, 3)
    candidates = query.repeat(2, 1, 1, 1)
    candidates[0, 0, 0, axis] = 8
    candidates[1, :, :, axis] = -4
    scale = torch.ones_like(query)
    scale[..., axis] = 2
    distance, components = robust_stp_distance(query, candidates, scale, [1 / 3] * 3, return_components=True)
    expected = torch.zeros(2, 3)
    expected[:, axis] = torch.tensor([1., 2.])
    torch.testing.assert_close(components, expected)
    torch.testing.assert_close(distance, torch.tensor([1 / 3, 2 / 3]))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_weighted_distance_and_batch_parity(dtype):
    rng = np.random.default_rng(9)
    query = torch.tensor(rng.normal(size=(2, 3, 4, 3)), dtype=dtype)
    candidates = torch.tensor(rng.normal(size=(2, 5, 3, 4, 3)), dtype=dtype)
    scale = torch.tensor(rng.uniform(0.1, 3, size=(3, 4, 3)), dtype=dtype)
    weights = [0.1, 0.25, 0.65]
    distance, components = robust_stp_distance(query, candidates, scale, weights, return_components=True)
    assert distance.shape == (2, 5)
    assert components.shape == (2, 5, 3)
    assert distance.dtype == dtype
    expected_components = (np.abs(candidates.numpy() - query.numpy()[:, None]) / scale.numpy()).mean(axis=(2, 3))
    np.testing.assert_allclose(components.numpy(), expected_components, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(distance.numpy(), (expected_components * weights).sum(axis=-1), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(distance, robust_stp_distance(query, candidates, scale, weights))
    for b in range(2):
        single, single_components = robust_stp_distance(query[b], candidates[b], scale, weights, return_components=True)
        torch.testing.assert_close(single, distance[b])
        torch.testing.assert_close(single_components, components[b])


def test_zero_weights_and_weight_tolerance():
    query = torch.zeros(2, 1, 3)
    candidates = torch.ones(1, 2, 1, 3)
    candidates[..., 1:] = 50
    result = robust_stp_distance(query, candidates, torch.ones_like(query), [1., 0., 0.])
    torch.testing.assert_close(result, torch.ones(1))
    robust_stp_distance(query, candidates, torch.ones_like(query), [0.3333333] * 3)


def test_similarity_formula_and_bounds():
    distance = torch.tensor([0., 0.5, 2., 20., 1e30])
    actual = stp_distance_to_similarity(distance, 2.)
    torch.testing.assert_close(actual, torch.exp(-distance / 2))
    assert actual[0] == 1
    assert torch.all(actual[:-1] > actual[1:])
    assert torch.isfinite(actual).all()
    assert torch.all((actual >= 0) & (actual <= 1))


def test_similarity_zero_distance_with_tiny_positive_temperature():
    actual = stp_distance_to_similarity(torch.tensor([0., 1.]), 1e-100)
    torch.testing.assert_close(actual, torch.tensor([1., 0.]))


def test_nonfinite_output_convention_without_input_mutation():
    data = repeated_segment([float('nan'), float('inf'), -float('inf'), 4])
    original = data.clone()
    assert torch.isfinite(compute_stp_profile(data, 2)).all()
    torch.testing.assert_close(data, original, equal_nan=True)
    scale = compute_stp_iqr_scale(torch.full((2, 2, 1, 3), float('nan')))
    torch.testing.assert_close(scale, torch.full_like(scale, 1e-8))
    query = torch.zeros(2, 1, 3)
    candidates = torch.full((1, 2, 1, 3), float('inf'))
    distance, components = robust_stp_distance(query, candidates, torch.ones_like(query), [1., 0., 0.], return_components=True)
    assert torch.isfinite(distance).all() and torch.isfinite(components).all()
    similarity = stp_distance_to_similarity(torch.tensor([float('nan'), float('inf'), -float('inf')]), 1.)
    torch.testing.assert_close(similarity, torch.tensor([0., 0., 1.]))


@pytest.mark.parametrize('shape, sections', [
    ((1, 8, 1), 1), ((1, 8, 1), 0), ((1, 8, 1), -1),
    ((1, 3, 1), 4), ((1, 5, 1), 3), ((1, 8, 1), 2.5),
    ((1, 8, 1), True), ((8, 1), 2), ((1, 8, 1, 1), 2), ((1, 8, 0), 2),
    ((0, 8, 1), 2),
])
def test_invalid_profile_arguments(shape, sections):
    with pytest.raises(ValueError):
        compute_stp_profile(torch.zeros(shape), sections)


@pytest.mark.parametrize('shape', [(2, 2, 1), (0, 2, 1, 3), (2, 1, 1, 3), (2, 2, 0, 3), (2, 2, 1, 4)])
def test_invalid_scale_shapes(shape):
    with pytest.raises(ValueError, match='shape'):
        compute_stp_iqr_scale(torch.zeros(shape))


@pytest.mark.parametrize('eps', [0, -1, float('nan'), float('inf'), 1e-100, 1e100])
def test_invalid_epsilon(eps):
    with pytest.raises(ValueError, match='eps'):
        compute_stp_iqr_scale(torch.zeros(2, 2, 1, 3), eps)


@pytest.mark.parametrize('weights', [
    [-0.1, 0.6, 0.5], [0.5, 0.5], [0.25] * 4, [0.2] * 3,
    [float('nan'), 0.5, 0.5], [float('inf'), 0, 0], [[0.2, 0.3, 0.5]],
    [0, 0, 0], [1.00001, 0, 0], None, ['bad', 0, 1],
])
def test_invalid_weights(weights):
    query = torch.zeros(2, 1, 3)
    with pytest.raises(ValueError, match='weights'):
        robust_stp_distance(query, query.unsqueeze(0), torch.ones_like(query), weights)


@pytest.mark.parametrize('query_shape, candidate_shape, scale_shape', [
    ((2, 1), (3, 2, 1), (2, 1)),
    ((2, 1, 3), (2, 1, 3), (2, 1, 3)),
    ((2, 1, 3), (4, 3, 1, 3), (2, 1, 3)),
    ((2, 1, 3), (4, 2, 2, 3), (2, 1, 3)),
    ((2, 1, 4), (4, 2, 1, 4), (2, 1, 4)),
    ((2, 1, 3), (4, 2, 1, 3), (1, 1, 3)),
    ((2, 1, 3), (4, 2, 1, 3), (1, 2, 1, 3)),
    ((2, 2, 1, 3), (1, 4, 2, 1, 3), (2, 1, 3)),
    ((1, 1, 3), (4, 1, 1, 3), (1, 1, 3)),
    ((2, 0, 3), (4, 2, 0, 3), (2, 0, 3)),
])
def test_invalid_distance_shapes(query_shape, candidate_shape, scale_shape):
    with pytest.raises(ValueError):
        robust_stp_distance(torch.zeros(query_shape), torch.zeros(candidate_shape), torch.ones(scale_shape), [0.2, 0.3, 0.5])


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_invalid_distance_scale(value):
    query = torch.zeros(2, 1, 3)
    with pytest.raises(ValueError, match='scale'):
        robust_stp_distance(query, query.unsqueeze(0), torch.full_like(query, value), [0.2, 0.3, 0.5])


@pytest.mark.parametrize('temperature', [0, -1, float('nan'), float('inf')])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError, match='temperature'):
        stp_distance_to_similarity(torch.zeros(2), temperature)


@pytest.mark.parametrize('dtype', [torch.int64, torch.float16, torch.bfloat16])
def test_unsupported_dtypes_are_explicit(dtype):
    with pytest.raises(ValueError, match='dtype'):
        compute_stp_profile(torch.ones(1, 8, 1, dtype=dtype), 2)
    with pytest.raises(ValueError, match='dtype'):
        compute_stp_iqr_scale(torch.ones(2, 2, 1, 3, dtype=dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_cuda_pipeline_matches_cpu(dtype):
    data = torch.randn(8, 19, 3, generator=torch.Generator().manual_seed(15), dtype=dtype)
    cpu = compute_stp_profile(data, 3)
    gpu = compute_stp_profile(data.cuda(), 3)
    torch.testing.assert_close(gpu.cpu(), cpu)
    cpu_scale = compute_stp_iqr_scale(cpu)
    gpu_scale = compute_stp_iqr_scale(gpu)
    torch.testing.assert_close(gpu_scale.cpu(), cpu_scale)
    cpu_distance, cpu_components = robust_stp_distance(cpu[0], cpu, cpu_scale, [0.2, 0.3, 0.5], return_components=True)
    gpu_distance, gpu_components = robust_stp_distance(gpu[0], gpu, gpu_scale, [0.2, 0.3, 0.5], return_components=True)
    torch.testing.assert_close(gpu_distance.cpu(), cpu_distance)
    torch.testing.assert_close(gpu_components.cpu(), cpu_components)
    torch.testing.assert_close(stp_distance_to_similarity(gpu_distance, 1.).cpu(), stp_distance_to_similarity(cpu_distance, 1.))
