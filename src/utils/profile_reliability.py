"""Mean-profile reliability utilities for profile-aware retrieval."""

import math

import torch


PROFILE_EPS = 1e-8


def section_boundaries(length, sections):
    if sections < 2:
        raise ValueError('profile_sections must be >= 2')
    if length < sections:
        raise ValueError('seq_len must be >= profile_sections')
    section_size = length // sections
    if section_size < 2:
        raise ValueError('each profile section must contain at least two time steps')
    return [
        (idx * section_size,
         (idx + 1) * section_size if idx < sections - 1 else length)
        for idx in range(sections)
    ]


def compute_mean_profile(data, sections):
    """Return section-wise means for raw past windows shaped ``[N, L, C]``."""
    if data.ndim != 3:
        raise ValueError('profile input must have shape [N, L, C]')
    means = [
        data[:, start:end, :].mean(dim=1)
        for start, end in section_boundaries(data.shape[1], sections)
    ]
    profile = torch.stack(means, dim=1)
    return torch.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)


def compute_iqr_scale(train_profiles, eps=PROFILE_EPS):
    """Coordinate-wise robust scale calculated only from the training bank."""
    profiles = train_profiles.to(torch.float64)
    scale = (
        torch.quantile(profiles, 0.75, dim=0)
        - torch.quantile(profiles, 0.25, dim=0)
        + eps
    )
    return torch.nan_to_num(
        scale, nan=eps, posinf=eps, neginf=eps
    ).clamp_min(eps)


def robust_profile_distance(query_profile, candidate_profiles, scale):
    """Robust-scaled mean L1 distance for one query or a query batch.

    Accepted shapes are ``[W, C]``/``[M, W, C]`` and
    ``[B, W, C]``/``[B, M, W, C]``.  Runtime callers keep the profile bank and
    scale in float64, so the ``to`` calls below become no-ops instead of being
    repeated conversions for every query.
    """
    query = query_profile.to(torch.float64)
    candidates = candidate_profiles.to(device=query.device, dtype=torch.float64)
    scale = scale.to(device=query.device, dtype=torch.float64).clamp_min(PROFILE_EPS)
    if query.ndim == 2 and candidates.ndim == 3:
        query = query.unsqueeze(0)
        reduce_dims = (1, 2)
    elif query.ndim == 3 and candidates.ndim == 4:
        query = query.unsqueeze(1)
        reduce_dims = (2, 3)
    else:
        raise ValueError(
            'profile shapes must be [W,C]/[M,W,C] or '
            '[B,W,C]/[B,M,W,C]'
        )
    distance = (
        (candidates - query).abs() / scale
    ).mean(dim=reduce_dims)
    return torch.nan_to_num(
        distance, nan=0.0, posinf=torch.finfo(torch.float64).max
    ).float()


def distance_to_similarity(distance, temperature):
    if temperature <= 0:
        raise ValueError('mean_profile_temperature must be > 0')
    similarity = torch.exp(-distance / temperature)
    return torch.nan_to_num(
        similarity, nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)


def topk_reliability(similarity, k):
    """Estimate candidate reliability from profile-based state agreement."""
    if similarity.ndim not in (1, 2):
        raise ValueError('profile similarity must have shape [M] or [B,M]')
    if not 1 <= k <= similarity.shape[-1]:
        raise ValueError('profile reliability k must be within the candidate pool')
    return torch.topk(similarity, k, dim=-1).values.mean(dim=-1)


def map_reliability(cr_q, sigma_min, sigma_max, lambda_min, lambda_max):
    # Higher reliability increases emphasis on retrieval relevance and weight
    # concentration; lower reliability favors diversity and broader weighting.
    sigma_q = sigma_min + (1.0 - cr_q) * (sigma_max - sigma_min)
    lambda_q = lambda_min + cr_q * (lambda_max - lambda_min)
    return sigma_q, lambda_q


def resolved_reliability_k(profile_reliability_k, topm):
    return topm if profile_reliability_k == -1 else profile_reliability_k


def resolved_crest_profile(args):
    # Resolve before validation: the legacy controller always means Mean Profile.
    return ('mean' if args.retrieval_controller == 'mean_profile'
            else getattr(args, 'crest_profile', 'stp'))


def stp_weights_from_args(args):
    return tuple(getattr(args, f'stp_{name}_weight', default)
                 for name, default in zip(('state', 'dispersion', 'transition'), (0.5, 0.25, 0.25)))


def validate_crest_profile_options(args):
    if args.retrieval_controller == 'baseline':
        return
    profile = resolved_crest_profile(args)
    if profile not in ('stp', 'mean'):
        raise ValueError('crest_profile must be stp or mean')
    if profile == 'stp':
        weights = stp_weights_from_args(args)
        if (any(not math.isfinite(w) or w < 0 for w in weights)
                or not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-6)):
            raise ValueError('STP weights must be finite, nonnegative and sum to 1')


def validate_profile_args(args):
    if args.retrieval_controller not in ('baseline', 'crest', 'mean_profile'):
        raise ValueError('unknown retrieval_controller')
    if args.hnsw_oversample_factor < 1:
        raise ValueError('hnsw_oversample_factor must be >= 1')
    resolved_k = resolved_reliability_k(args.profile_reliability_k, args.topm)
    if args.retrieval_controller == 'baseline':
        return resolved_k
    validate_crest_profile_options(args)
    tau_v = getattr(args, "tau_v", 1.0)
    if not math.isfinite(tau_v) or tau_v <= 0:
        raise ValueError("tau_v must be finite and positive")
    if (not math.isfinite(args.mean_profile_temperature)
            or args.mean_profile_temperature <= 0):
        raise ValueError('mean_profile_temperature must be finite and > 0')
    if not 0 < args.sigma_min <= args.sigma_max < float('inf'):
        raise ValueError('sigma bounds must satisfy 0 < sigma_min <= sigma_max < inf')
    if not 0 <= args.lambda_min <= args.lambda_max <= 1:
        raise ValueError('lambda bounds must satisfy 0 <= lambda_min <= lambda_max <= 1')
    if args.mmr_candidate_pool < args.topm:
        raise ValueError('mmr_candidate_pool must be >= topm')
    section_boundaries(args.seq_len, args.profile_sections)
    if not 1 <= resolved_k <= args.mmr_candidate_pool:
        raise ValueError(
            'resolved profile_reliability_k must be in [1, mmr_candidate_pool]'
        )
    return resolved_k


def profile_setting_suffix(args):
    if args.retrieval_controller == 'baseline':
        return '_ctrlbaseline'
    k = resolved_reliability_k(args.profile_reliability_k, args.topm)
    profile = resolved_crest_profile(args)
    weights = ('_stpw' + '-'.join(str(w) for w in stp_weights_from_args(args))
               if profile == 'stp' else '')
    return (
        f'_ctrlcrest_profile{profile}{weights}_ps{args.profile_sections}_prk{k}'
        f'_mpt{args.mean_profile_temperature}'
        f'_sig{args.sigma_min}-{args.sigma_max}'
        f'_lam{args.lambda_min}-{args.lambda_max}'
        f'_pool{args.mmr_candidate_pool}'
    )
