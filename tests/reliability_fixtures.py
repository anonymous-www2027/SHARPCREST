"""Small independent panel fixtures for reliability tests."""
import numpy as np
import torch
from src.common.webscale_ts import InMemorySeriesBank, PanelWindowDataset, WindowManifest
from src.layers.Retrieval import RetrievalTool


def fixture():
    values = np.random.default_rng(12).normal(size=(3 * 120, 2)).astype(np.float32)
    bank = InMemorySeriesBank(values, np.arange(3), np.arange(4) * 120, np.full(3, 120))
    candidates = WindowManifest.from_bounds(bank, 16, 8, stride=3, end_bound=72)
    queries = {
        split: WindowManifest.from_bounds(bank, 16, 8, stride=5, start_bound=start, end_bound=end)
        for split, start, end in [('train', 0, 72), ('valid', 56, 96), ('test', 80, 120)]
    }
    return {split: PanelWindowDataset(bank, query, candidate_manifest=candidates, split=split)
            for split, query in queries.items()}


def make_tool(cls=RetrievalTool, **kwargs):
    options = dict(seq_len=16, pred_len=8, channels=2, n_period=2, topm=3,
                   retrieval_controller='crest', mmr_candidate_pool=7, profile_sections=4,
                   hnsw_build_batch_size=5, retrieval_batch_size=4,
                   value_mmr_chunk_size=2, retrieval_aggregation_chunk_size=2,
                   hnsw_build_num_threads=1, hnsw_search_num_threads=1,
                   hnsw_m=8, hnsw_ef_construction=80, hnsw_ef_search=80,
                   mmr_paa_segments=3, mmr_rp_dim=5, mmr_rp_seed=11, retrieval_seed=17)
    options.update(kwargs)
    return cls(**options)


def windows(data, ids, candidate=False, future=False):
    manifest = data.candidate_manifest if candidate else data.query_manifest
    series, starts = manifest.resolve_ids(np.asarray(ids))
    return torch.from_numpy(data.series_bank.get_windows(series, starts + (16 if future else 0), 8 if future else 16))
