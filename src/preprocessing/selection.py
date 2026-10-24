"""Population-proportional quotas with deterministic nested series selection."""
import hashlib
import math
import numpy as np
from .protocol import regimes


def scale_chain(scales):
    """Include all preceding doubling scales so separate invocations agree."""
    scales = sorted(set(scales))
    if not scales or scales[0] < 100000:
        raise ValueError('Scales must be >=100000; the first anchor is 100K')
    ladder = []; value = 100000
    while value <= max(scales):
        ladder.append(value); value *= 2
    return sorted(set(ladder + scales))


def quotas(frame, n, p):
    counts = frame.regime.value_counts().reindex(regimes(p), fill_value=0).to_numpy()
    numerator = n * counts
    quota, remainder = numerator // len(frame), numerator % len(frame)
    for j in sorted(range(len(counts)), key=lambda j: (-int(remainder[j]), j))[:n-int(quota.sum())]:
        quota[j] += 1
    return [dict(regime=r, population_count=int(c), population_proportion=float(c/len(frame)),
                 ideal_quota=float(n*c/len(frame)), integer_quota=int(q))
            for r, c, q in zip(regimes(p), counts, quota)]


def extend(frame, previous, n, p):
    """Keep the ordered prefix and fill each regime's remaining quota."""
    if not len(previous) <= n <= len(frame):
        raise ValueError('Requested cohort exceeds population or shrinks the previous cohort')
    rows = quotas(frame, n, p)
    labels = frame.set_index('logical_id').regime
    selected = list(previous)
    for row in rows:
        r = row['regime']
        existing = int((labels.loc[previous] == r).sum()) if previous else 0
        deficit = row['integer_quota'] - existing
        if deficit < 0:
            raise ValueError(f'Nested quota infeasible for {r}: target {row["integer_quota"]} < existing {existing}; '
                             'choose a larger scale. Existing series will not be removed.')
        key = f'{p["sampling_key"]}:{r}:{p["selection_seed"]}'
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:16], 'little')
        eligible = np.sort(frame.loc[(frame.regime == r) & ~frame.logical_id.isin(previous),
                                     'logical_id'].to_numpy(dtype=np.int64))
        selected.extend(np.random.default_rng(seed).choice(eligible, size=deficit, replace=False).tolist())
        row.update(existing_count=existing, added_count=deficit, subseed=str(seed))
    if len(selected) != n or len(set(selected)) != n:
        raise ValueError('Invalid nested selection')
    return selected, rows


def select_scales(frame, scales, p):
    """Size complete series using the reference L/H, then apply TSF quotas."""
    capacity = p['split']['train_end'] - p['L'] - p['H'] + 1
    previous = []; result = {}
    for scale in scale_chain(scales):
        if scale == 100000 and p['anchor_series'] is not None:
            n = p['anchor_series']
        else:
            options = [max(1, math.floor(scale/capacity)), max(1, math.ceil(scale/capacity))]
            n = min(options, key=lambda n: (abs(n*capacity-scale), n))
        selected, rows = extend(frame, previous, n, p)
        result[scale] = dict(ids=selected, quotas=rows, actual_candidate_count=n*capacity)
        previous = selected
    return result
