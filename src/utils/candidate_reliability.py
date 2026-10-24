"""Past agreement and future consistency for the CREST controller."""

import math

import numpy as np
import torch


VALUE_EPS = 1e-8


def value_iqr_scale(train_futures):
    """Fit channel IQRs on training value patches before period decomposition."""
    values = train_futures.detach().cpu().numpy()
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError('training futures must be finite [candidate, H, channel]')
    quartiles = np.quantile(values, [0.25, 0.75], axis=(0, 1))
    return torch.from_numpy((quartiles[1] - quartiles[0] + VALUE_EPS).astype(np.float32))


def _weighted_quantile(values, counts, quantiles):
    """Match linear quantiles of repeated values without expanding the repeats."""
    order = np.argsort(values)
    values, counts = values[order], counts[order]
    cumulative = np.cumsum(counts, dtype=np.int64)
    ranks = np.asarray(quantiles) * (int(cumulative[-1]) - 1)
    lower, upper = np.floor(ranks).astype(np.int64), np.ceil(ranks).astype(np.int64)
    lo = values[np.searchsorted(cumulative, lower, side='right')].astype(np.float64)
    hi = values[np.searchsorted(cumulative, upper, side='right')].astype(np.float64)
    return lo + (hi - lo) * (ranks - lower)


def panel_value_iqr_scale(bank, manifest):
    """Fit exact training-patch IQRs using overlap multiplicities per timestamp."""
    values, weights = [], []
    for sid, first, count, stride in zip(
        manifest.series_ids, manifest.first_start, manifest.window_counts, manifest.stride
    ):
        first_future = int(first + manifest.seq_len)
        width = int((count - 1) * stride + manifest.pred_len)
        # Only timestamps covered by training candidate futures enter the fit.
        difference = np.zeros(width + 1, dtype=np.int64)
        starts = np.arange(int(count), dtype=np.int64) * int(stride)
        difference[starts] += 1
        difference[starts + manifest.pred_len] -= 1
        multiplicity = np.cumsum(difference[:-1])
        keep = multiplicity > 0
        values.append(bank.get_window(int(sid), first_future, width)[keep])
        weights.append(multiplicity[keep])
    values, weights = np.concatenate(values), np.concatenate(weights)
    if not np.isfinite(values).all():
        raise ValueError('training candidate futures must be finite')
    scale = []
    for channel in range(values.shape[1]):
        q25, q75 = _weighted_quantile(values[:, channel], weights, [0.25, 0.75])
        scale.append(q75 - q25 + VALUE_EPS)
    return torch.tensor(scale, dtype=torch.float32)


def spaced_neighborhood(similarity, series_ids, starts, k, min_spacing):
    """Select up to k STP-ranked candidates separated within each series."""
    scores = similarity.detach().cpu().numpy()
    series_ids, starts = np.asarray(series_ids), np.asarray(starts)
    if scores.ndim != 2 or series_ids.shape != scores.shape or starts.shape != scores.shape:
        raise ValueError('scores and candidate coordinates must have shape [B,C]')
    if not 1 <= k <= scores.shape[1] or min_spacing < 1:
        raise ValueError('invalid neighborhood size or minimum spacing')
    if not np.isfinite(scores).all():
        raise ValueError('neighborhood scores must be finite')
    order = np.argsort(-scores, axis=1, kind='stable')
    batch = len(scores)
    positions = np.zeros((batch, k), dtype=np.int64)
    selected_series = np.zeros((batch, k), dtype=np.int64)
    selected_starts = np.zeros((batch, k), dtype=np.int64)
    counts = np.zeros(batch, dtype=np.int64)
    rows, slots = np.arange(batch), np.arange(k)
    for rank in range(scores.shape[1]):
        position = order[:, rank]
        series, start = series_ids[rows, position], starts[rows, position]
        occupied = slots[None, :] < counts[:, None]
        conflict = (occupied & (selected_series == series[:, None]) &
                    (np.abs(selected_starts - start[:, None]) < min_spacing)).any(axis=1)
        accept = rows[(counts < k) & ~conflict]
        slot = counts[accept]
        positions[accept, slot] = position[accept]
        selected_series[accept, slot] = series[accept]
        selected_starts[accept, slot] = start[accept]
        counts[accept] += 1
        if np.all(counts == k):
            break
    valid = slots[None, :] < counts[:, None]
    # Repeat an accepted position for storage only; masks exclude padding from both means.
    positions = np.where(valid, positions, positions[:, :1])
    return (torch.as_tensor(positions, device=similarity.device),
            torch.as_tensor(valid, device=similarity.device))


def future_consistency(candidate_futures, value_scale, tau_v=1.0, valid_mask=None):
    """Average exp(-IQR-scaled L1 / tau_v) over distinct candidate pairs."""
    if not math.isfinite(tau_v) or tau_v <= 0:
        raise ValueError('tau_v must be finite and positive')
    if candidate_futures.ndim != 4 or candidate_futures.shape[1] < 1:
        raise ValueError('candidate futures must have shape [B, N, H, channel], N >= 1')
    if not torch.isfinite(candidate_futures).all():
        raise ValueError('candidate futures must be finite')
    value_scale = value_scale.to(candidate_futures)
    if value_scale.shape != candidate_futures.shape[-1:] or not (
        torch.isfinite(value_scale).all() and (value_scale > 0).all()
    ):
        raise ValueError('value scale must contain one finite positive value per channel')
    count = candidate_futures.shape[1]
    if valid_mask is None:
        valid_mask = torch.ones(candidate_futures.shape[:2], dtype=torch.bool,
                                device=candidate_futures.device)
    else:
        valid_mask = valid_mask.to(device=candidate_futures.device, dtype=torch.bool)
    if valid_mask.shape != candidate_futures.shape[:2] or not valid_mask.any(dim=1).all():
        raise ValueError('every query must have at least one valid candidate')
    if count == 1:
        # With no pair to compare, leave past agreement unpenalized.
        return candidate_futures.new_ones(candidate_futures.shape[0])
    scaled = (candidate_futures / value_scale).flatten(2)
    d_V = torch.cdist(scaled, scaled, p=1) / (scaled.shape[-1] * tau_v)
    pairs = torch.triu_indices(count, count, offset=1, device=d_V.device)
    pair_mask = valid_mask[:, pairs[0]] & valid_mask[:, pairs[1]]
    pair_count = pair_mask.sum(dim=-1)
    similarities = torch.exp(-d_V[:, pairs[0], pairs[1]])
    V_q = (similarities * pair_mask).sum(dim=-1) / pair_count.clamp_min(1)
    V_q = torch.where(pair_count > 0, V_q, torch.ones_like(V_q))
    return V_q.clamp(0.0, 1.0)
