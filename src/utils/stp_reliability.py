"""Segment Temporal Profiles for CREST state agreement.

Features and their weights follow the order: state (level), dispersion,
transition.
"""

import math
from numbers import Integral

import torch

from src.utils.profile_reliability import section_boundaries


STP_FEATURE_NAMES = ('state', 'dispersion', 'transition')
STP_EPS = 1e-8


def _validate_tensor(tensor, name):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f'{name} must be a torch tensor')
    if tensor.dtype not in (torch.float32, torch.float64):
        raise ValueError(f'{name} must have float32 or float64 dtype')


def compute_stp_profile(data, sections):
    """Return ``[N,S,C,3]`` profiles from past windows ``[N,L,C]``.

    State is the interpolated median. Transition is the total change of an OLS
    fitted line; dispersion is its residual IQR. Section boundaries, including
    the final remainder, are inherited from the existing mean-profile utility.
    """
    _validate_tensor(data, 'data')
    if data.ndim != 3 or data.shape[0] < 1 or data.shape[2] < 1:
        raise ValueError('data must have shape [N,L,C] with N,C >= 1')
    if isinstance(sections, bool) or not isinstance(sections, Integral):
        raise ValueError('sections must be an integer')
    profiles = []
    # Local summaries retain state differences that shape-only keys suppress.
    for start, end in section_boundaries(data.shape[1], sections):
        segment = data[:, start:end, :]
        length = end - start
        time = torch.arange(length, device=data.device, dtype=data.dtype)
        centered_time = (time - time.mean()).view(1, length, 1)
        centered = segment - segment.mean(dim=1, keepdim=True)
        slope = (
            (centered_time * centered).sum(dim=1)
            / centered_time.square().sum()
        )
        # Centered form is algebraically x - (alpha*t + beta), avoiding a
        # second subtraction of the absolute level when forming residuals.
        residual = centered - slope.unsqueeze(1) * centered_time
        quartiles = torch.quantile(
            residual, residual.new_tensor([0.25, 0.75]), dim=1,
        )
        profiles.append(torch.stack((
            torch.quantile(segment, 0.5, dim=1),
            quartiles[1] - quartiles[0],
            slope * (length - 1),
        ), dim=-1))
    return torch.nan_to_num(
        torch.stack(profiles, dim=1), nan=0.0, posinf=0.0, neginf=0.0,
    )


def compute_stp_iqr_scale(train_profiles, eps=STP_EPS):
    """Return coordinate-wise ``[S,C,3]`` training-bank IQR plus ``eps``.

    Callers must supply training candidate profiles only. Query/held-out
    profiles must not participate in scale fitting. Constant coordinates use
    eps; non-finite scale outputs fall back to eps as in the mean-profile helper.
    """
    _validate_tensor(train_profiles, 'train_profiles')
    if (train_profiles.ndim != 4 or train_profiles.shape[-1] != 3
            or train_profiles.shape[0] < 1 or train_profiles.shape[1] < 2
            or train_profiles.shape[2] < 1):
        raise ValueError(
            'train_profiles must have shape [N,S,C,3], N,C >= 1, S >= 2'
        )
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and > 0')
    typed_eps = train_profiles.new_tensor(eps)
    if not torch.isfinite(typed_eps) or typed_eps <= 0:
        raise ValueError('eps must be positive and finite in the profile dtype')
    quartiles = torch.quantile(
        train_profiles, train_profiles.new_tensor([0.25, 0.75]), dim=0,
    )
    scale = quartiles[1] - quartiles[0] + eps
    return torch.nan_to_num(scale, nan=eps, posinf=eps, neginf=eps).clamp_min(eps)


def robust_stp_distance(
    query_profile, candidate_profiles, scale, weights, *, return_components=False,
):
    """Compute feature-wise robust L1 distances and their weighted sum.

    Accepts ``[S,C,3]`` / ``[M,S,C,3]`` or ``[B,S,C,3]`` /
    ``[B,M,S,C,3]`` query/candidates. Scale must be exactly ``[S,C,3]``
    (already including eps). Candidates and scale are converted to the query's
    device/dtype. Output is ``[M]`` or ``[B,M]`` in that dtype/device.

    Weights are required, nonnegative, finite, length three, and must sum to one
    within absolute tolerance 1e-6; they are never implicitly normalized.
    With ``return_components=True``, return ``(distance, components)`` where
    components has an additional final feature axis in STP_FEATURE_NAMES order.
    Each component averages over segments/channels, never over features.
    """
    for name, tensor in (
        ('query_profile', query_profile),
        ('candidate_profiles', candidate_profiles),
        ('scale', scale),
    ):
        _validate_tensor(tensor, name)
    if not ((query_profile.ndim == 3 and candidate_profiles.ndim == 4)
            or (query_profile.ndim == 4 and candidate_profiles.ndim == 5)):
        raise ValueError(
            'profile shapes must be [S,C,3]/[M,S,C,3] or '
            '[B,S,C,3]/[B,M,S,C,3]'
        )
    feature_shape = query_profile.shape[-3:]
    if (feature_shape[-1] != 3 or feature_shape[0] < 2 or feature_shape[1] < 1
            or candidate_profiles.shape[-3:] != feature_shape
            or scale.shape != feature_shape):
        raise ValueError(
            'query, candidates and scale must have matching [S,C,3], '
            'S >= 2, C >= 1'
        )
    if (query_profile.ndim == 4
            and query_profile.shape[0] != candidate_profiles.shape[0]):
        raise ValueError('query and candidate batch sizes must match')
    # Only three weights use float64, to validate before a float32 cast can
    # hide negative values or a materially incorrect sum.
    try:
        weight_tensor = torch.as_tensor(
            weights, device=query_profile.device, dtype=torch.float64,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(
            'weights must be three finite nonnegative numbers summing to 1'
        ) from exc
    if (weight_tensor.shape != (3,) or not torch.isfinite(weight_tensor).all()
            or (weight_tensor < 0).any()
            or not torch.isclose(
                weight_tensor.sum(), weight_tensor.new_tensor(1.0),
                rtol=0, atol=1e-6,
            )):
        raise ValueError(
            'weights must be three finite nonnegative numbers summing to 1'
        )
    candidates = candidate_profiles.to(query_profile)
    scale = scale.to(query_profile)
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('scale must be finite and > 0 in the query dtype')
    query = query_profile.unsqueeze(-4)
    # Scale each coordinate by training IQR, then average over segments and
    # channels before combining the three feature distances (Section 5.1).
    components = ((candidates - query).abs() / scale).mean(dim=(-3, -2))
    components = torch.nan_to_num(
        components, nan=0.0, posinf=torch.finfo(components.dtype).max,
    )
    distance = (components * weight_tensor.to(query_profile)).sum(dim=-1)
    distance = torch.nan_to_num(distance, nan=0.0)
    if return_components:
        return distance, components
    return distance


def stp_distance_to_similarity(distance, temperature):
    """Return exp(-distance / temperature), finite and clamped to [0,1].

    Non-finite similarity outputs use the existing mean-profile helper's
    convention: NaN -> 0, +Inf -> 1, -Inf -> 0.
    """
    _validate_tensor(distance, 'distance')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and > 0')
    similarity = torch.exp(-distance / temperature)
    # Preserve exp(0) even if a positive temperature underflows in this dtype.
    similarity = similarity.masked_fill(distance == 0, 1.0)
    return torch.nan_to_num(
        similarity, nan=0.0, posinf=1.0, neginf=0.0,
    ).clamp(0.0, 1.0)
