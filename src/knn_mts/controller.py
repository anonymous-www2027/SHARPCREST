"""CREST selects candidates with past agreement, future consistency and trajectory MMR. Gaussian weights use the selected Pearson distances."""
import math
import torch
from torch.nn import functional as F
from src.utils.candidate_reliability import spaced_neighborhood, future_consistency
from .vendor.stp_reliability import (compute_stp_profile, compute_stp_iqr_scale,
    robust_stp_distance, stp_distance_to_similarity)
from .vendor.profile_reliability import map_reliability


def stateless_uniforms(ids, k, seed, split):
    """RAFT/PIR avalanche sampling keyed by global scalar-query ID and split."""
    if split not in ('val', 'test'):
        raise ValueError('only validation/test retrieval is supported')
    mask = 0xFFFFFFFF
    ids = ids.long()[:, None]
    step = torch.arange(1, k, device=ids.device)[None, :]
    state = ((ids & mask)*10000019) & mask
    state = state ^ ((((ids >> 32) & mask)*100000007) & mask)
    state = state ^ ((step*1000000007) & mask)
    state = state ^ ((seed*1000003) & mask) ^ ({'val': 2, 'test': 3}[split]*40692)
    for _ in range(2):
        state = ((state ^ (state >> 16))*0x045D9F3B) & mask
    state = (state ^ (state >> 16)) & mask
    return (state.double()+.5)/4294967296


def gaussian_l2_weights(squared_l2, sigma):
    # Shift before division: preserves normalized Gaussian ratios while avoiding
    # all-zero underflow and cancellation from a large common distance offset.
    shifted = squared_l2.double()-squared_l2.double().min(1, keepdim=True).values
    logits = -shifted/(2*sigma.double()[:, None].square())
    return torch.softmax(logits, dim=1).to(squared_l2.dtype)


def gaussian_pearson_weights(distance, sigma):
    # distance=1-rho; reuse stable Gaussian implementation on distance squared.
    return gaussian_l2_weights(distance.double().square(), sigma).to(distance.dtype)


class L2CREST:
    def __init__(self, keys, k, pool, seed, temperature, sections=6,
                 stp_weights=(.5,.25,.25), reliability_k=50,
                 sigma_min=.05, sigma_max=.30, mmr_chunk=1024, metric='squared_l2',
                 future_lookup=None, candidate_coordinates=None, value_scale=None, horizon=None, tau_v=1.0):
        self.metric = metric
        if future_lookup is None or candidate_coordinates is None or value_scale is None or horizon is None:
            raise ValueError('CREST requires training futures, coordinates, value scale and horizon')
        if not math.isfinite(tau_v) or tau_v <= 0:
            raise ValueError('tau_v must be finite and positive')
        self.future_lookup, self.candidate_coordinates = future_lookup, candidate_coordinates
        self.value_scale, self.horizon, self.tau_v = value_scale, horizon, float(tau_v)
        self.last_reliability_components = None
        self.k, self.seed, self.temperature = k, seed, temperature
        self.sections, self.weights = sections, tuple(stp_weights)
        self.reliability_k, self.sigma_min, self.sigma_max = reliability_k, sigma_min, sigma_max
        self.mmr_chunk = mmr_chunk
        # Explicit profile preparation batches, never candidate-search chunks.
        self.profiles = torch.cat([compute_stp_profile(x[:, :, None], sections)
                                   for x in keys.split(4096)])
        self.scale = compute_stp_iqr_scale(self.profiles)

    def reliability(self, query, labels):
        profile = compute_stp_profile(query[:, :, None], self.sections)
        distance = robust_stp_distance(profile, self.profiles[labels], self.scale, self.weights)
        similarity = stp_distance_to_similarity(distance, 1.)
        series, starts = self.candidate_coordinates(labels)
        positions, valid = spaced_neighborhood(similarity, series.cpu().numpy(), starts.cpu().numpy(),
                                               self.reliability_k, self.horizon)
        N_q_size = valid.sum(1)
        A_q = (similarity.gather(1, positions) * valid).sum(1) / N_q_size
        neighbors = labels.gather(1, positions)
        consistency = []
        for begin in range(0, len(labels), min(32, self.mmr_chunk)):
            selected = neighbors[begin:begin+min(32, self.mmr_chunk)]
            futures = self.future_lookup(selected).unsqueeze(-1)
            consistency.append(future_consistency(futures, self.value_scale, self.tau_v,
                               valid_mask=valid[begin:begin+len(selected)]))
        V_q = torch.cat(consistency)
        cr_q = A_q * V_q
        sigma_q, lambda_q = map_reliability(cr_q, self.sigma_min, self.sigma_max, .30, .90)
        self.last_reliability_components = {'A_q': A_q, 'V_q': V_q, 'cr_q': cr_q,
                                           'N_q_size': N_q_size}
        return cr_q, lambda_q, sigma_q

    def select(self, distances, lambda_q, labels, future_lookup, query_ids, split):
        result = []
        for start in range(0, len(labels), self.mmr_chunk):
            end = min(start+self.mmr_chunk, len(labels))
            scores = 1-distances[start:end] if self.metric == 'pearson' else -distances[start:end]/self.temperature
            lam = lambda_q[start:end, None]
            values = future_lookup(labels[start:end])
            values = F.normalize(values-values.mean(-1, keepdim=True), dim=-1)
            redundancy = (values*values[:, :1]).sum(-1)
            selected = torch.zeros((len(values), self.k), dtype=torch.long, device=values.device)
            used = torch.zeros_like(scores, dtype=torch.bool); used[:, 0] = True
            draws = stateless_uniforms(query_ids[start:end], self.k, self.seed, split)
            rows = torch.arange(len(values), device=values.device)
            for step in range(1, self.k):
                objective = lam*scores-(1-lam)*redundancy
                probs = F.softmax(objective.masked_fill(used, -torch.inf)/.1, dim=1)
                cdf = probs.double().cumsum(1); cdf = cdf/cdf[:, -1:]; cdf[:, -1] = 1
                pos = (cdf < draws[:, step-1:step]).sum(1)
                selected[:, step] = pos; used.scatter_(1, pos[:, None], True)
                redundancy = torch.maximum(redundancy, (values*values[rows, pos][:, None]).sum(-1))
            result.append(selected)
        return torch.cat(result)
