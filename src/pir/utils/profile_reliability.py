"""Partition past windows and map candidate reliability to controller parameters."""
import torch


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


def topk_reliability(similarity, k):
    if similarity.ndim not in (1, 2):
        raise ValueError('profile similarity must have shape [M] or [B,M]')
    if not 1 <= k <= similarity.shape[-1]:
        raise ValueError('profile reliability k must be within the candidate pool')
    return torch.topk(similarity, k, dim=-1).values.mean(dim=-1)


def map_reliability(cr_q, sigma_min, sigma_max, lambda_min, lambda_max):
    sigma_q = sigma_min + (1.0 - cr_q) * (sigma_max - sigma_min)
    lambda_q = lambda_min + cr_q * (lambda_max - lambda_min)
    return sigma_q, lambda_q
