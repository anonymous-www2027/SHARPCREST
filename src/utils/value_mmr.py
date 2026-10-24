"""Configuration and time-axis PAA for candidate-value MMR representations."""

from numbers import Integral

import torch

from src.utils.profile_reliability import section_boundaries


def paa_value_patch(values, segments):
    """Reduce [N,H,C] to [N,P,C]; the final segment includes the remainder.

    Like profile sections, P >= 2 and each segment has at least two steps.
    This averages future values only; it does not compute reliability features.
    """
    if values.ndim != 3:
        raise ValueError('PAA values must have shape [N,H,C]')
    if isinstance(segments, bool) or not isinstance(segments, Integral):
        raise ValueError('mmr_paa_segments must be an integer')
    return torch.stack([
        values[:, start:end].mean(dim=1)
        for start, end in section_boundaries(values.shape[1], segments)
    ], dim=1)


def validate_value_mmr_args(args):
    """Validate active value compression options for CLI or RetrievalTool."""
    if (args.retrieval_controller == 'baseline'
            or args.mmr_redundancy_source != 'value'):
        return
    mode = getattr(args, 'mmr_value_repr', 'raw')
    if mode not in ('raw', 'paa', 'random_projection'):
        raise ValueError('mmr_value_repr must be raw, paa, or random_projection')
    if mode == 'paa':
        segments = args.mmr_paa_segments
        if isinstance(segments, bool) or not isinstance(segments, Integral):
            raise ValueError('mmr_paa_segments must be an integer')
        section_boundaries(args.pred_len, segments)
    elif mode == 'random_projection':
        dim, seed = args.mmr_rp_dim, args.mmr_rp_seed
        channels = args.channels if hasattr(args, 'channels') else args.enc_in
        if (isinstance(dim, bool) or not isinstance(dim, Integral)
                or not 1 <= dim < args.pred_len * channels):
            raise ValueError('mmr_rp_dim must be an integer with 1 <= d < pred_len * channels')
        if (isinstance(seed, bool) or not isinstance(seed, Integral)
                or not 0 <= seed < 2 ** 63):
            raise ValueError('mmr_rp_seed must be an integer in [0, 2**63)')


def value_mmr_setting_suffix(args):
    if (args.retrieval_controller == 'baseline'
            or args.mmr_redundancy_source != 'value'):
        return ''
    mode = getattr(args, 'mmr_value_repr', 'raw')
    suffix = f'_vrepr{mode}'
    if mode == 'paa':
        suffix += f'_paa{args.mmr_paa_segments}'
    elif mode == 'random_projection':
        suffix += f'_rpd{args.mmr_rp_dim}_rps{args.mmr_rp_seed}'
    return suffix
