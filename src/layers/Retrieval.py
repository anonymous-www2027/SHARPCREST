"""SHARP search and CREST augmentation for RAFT.

SHARP uses HNSW with adaptive two-stage search. CREST estimates reliability
from past agreement and future consistency, selects candidates by MMR scores
with trajectory redundancy, and adjusts aggregation weight concentration by reliability.
"""

import math
import time

import hnswlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.hnsw_recall import HNSWRecallMixin
from src.utils.candidate_reliability import (
    future_consistency, value_iqr_scale, panel_value_iqr_scale, spaced_neighborhood,
)
from src.utils.panel_crest import PanelSTPSidecar, validate_panel_crest_options
from src.utils.value_mmr import paa_value_patch, validate_value_mmr_args
from src.utils.stp_reliability import (
    compute_stp_profile, compute_stp_iqr_scale,
    robust_stp_distance, stp_distance_to_similarity,
)
from src.utils.profile_reliability import (
    validate_crest_profile_options,
    compute_iqr_scale,
    compute_mean_profile,
    distance_to_similarity,
    map_reliability,
    resolved_reliability_k,
    robust_profile_distance,
)


class RetrievalTool(HNSWRecallMixin):
    """Build a training archive and return augmentation tensors for each split."""

    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        n_period=3,
        temperature=0.1,
        topm=20,
        hnsw_m=16,
        hnsw_ef_construction=200,
        hnsw_ef_search=200,
        hnsw_seed=0,
        retrieval_batch_size=1024,
        retrieval_backend='hnsw',
        retrieval_controller='baseline',
        mmr_candidate_pool=100,
        profile_sections=6,
        profile_reliability_k=-1,
        mean_profile_temperature=1.0,
        sigma_min=0.05,
        sigma_max=0.30,
        lambda_min=0.30,
        lambda_max=0.90,
        hnsw_oversample_factor=5,
        candidate_search_strategy='two_stage',
        retrieval_seed=0,
        retrieval_postprocess_device='cpu',
        mmr_device='cpu',
        mmr_redundancy_source='value',
        value_mmr_chunk_size=8,
        retrieval_aggregation_chunk_size=32,
        runtime_breakdown=False,
        hnsw_build_batch_size=4096,
        memory_report=False,
        hnsw_build_num_threads=-1,
        hnsw_search_num_threads=32,
        hnsw_recall_check=False,
        hnsw_recall_queries=100,
        hnsw_recall_seed=0,
        hnsw_recall_splits=('valid',),
        hnsw_recall_candidate_chunk_size=4096,
        hnsw_recall_query_batch_size=32,
        hnsw_recall_device='cpu',
        hnsw_recall_cache_dir='',
        crest_profile='stp',
        stp_state_weight=0.5,
        stp_dispersion_weight=0.25,
        stp_transition_weight=0.25,
        mmr_value_repr='raw',
        mmr_paa_segments=8,
        mmr_rp_dim=32,
        mmr_rp_seed=0,
        panel_exact_max_candidates=6000000,
        crest_weighting="gaussian",
        tau_v=1.0,
    ):
        period_num = [16, 8, 4, 2, 1]
        if not 1 <= n_period <= len(period_num):
            raise ValueError(f'n_period must be between 1 and {len(period_num)}')
        if candidate_search_strategy not in {'two_stage', 'wide'}:
            raise ValueError(
                'candidate_search_strategy must be one of: two_stage, wide'
            )

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.n_period = n_period
        self.period_num = sorted(period_num[-n_period:], reverse=True)
        self.temperature = temperature
        self.topm = topm
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.hnsw_seed = hnsw_seed
        if retrieval_backend not in {'exact', 'hnsw'}:
            raise ValueError('retrieval_backend must be exact or hnsw')
        if retrieval_controller == 'mean_profile':
            retrieval_controller = 'crest'
            crest_profile = 'mean'
        if retrieval_controller not in {'baseline', 'crest'}:
            raise ValueError(
                'retrieval_controller must be baseline, crest, or mean_profile'
            )

        self.retrieval_batch_size = retrieval_batch_size
        self.retrieval_backend = retrieval_backend
        self.retrieval_controller = retrieval_controller
        if crest_weighting not in {"gaussian", "adaptive_softmax"}:
            raise ValueError("unsupported crest_weighting")
        self.crest_weighting = crest_weighting
        self.crest_enabled = retrieval_controller == 'crest'
        self.mmr_candidate_pool = mmr_candidate_pool
        self.crest_profile = crest_profile
        self.stp_state_weight = stp_state_weight
        self.stp_dispersion_weight = stp_dispersion_weight
        self.stp_transition_weight = stp_transition_weight
        self.stp_weights = (stp_state_weight, stp_dispersion_weight, stp_transition_weight)
        self.profile_sections = profile_sections
        self.profile_reliability_k = resolved_reliability_k(
            profile_reliability_k, topm
        )
        self.mean_profile_temperature = mean_profile_temperature
        if not math.isfinite(tau_v) or tau_v <= 0:
            raise ValueError("tau_v must be finite and positive")
        self.tau_v = float(tau_v)
        self.raw_future_bank = None
        self.value_scale = None
        self.last_reliability_components = None
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.hnsw_oversample_factor = hnsw_oversample_factor
        self.candidate_search_strategy = candidate_search_strategy
        self.retrieval_seed = retrieval_seed
        self.retrieval_postprocess_device = torch.device(
            retrieval_postprocess_device
        )
        self.mmr_device = torch.device(mmr_device)
        self.mmr_redundancy_source = mmr_redundancy_source
        self.mmr_value_repr = mmr_value_repr
        self.mmr_paa_segments = mmr_paa_segments
        self.mmr_rp_dim = mmr_rp_dim
        self.mmr_rp_seed = mmr_rp_seed
        self.value_mmr_chunk_size = value_mmr_chunk_size
        self.retrieval_aggregation_chunk_size = retrieval_aggregation_chunk_size
        self.runtime_breakdown = runtime_breakdown
        if hnsw_build_batch_size < 1:
            raise ValueError('hnsw_build_batch_size must be >= 1')
        self.hnsw_build_batch_size = int(hnsw_build_batch_size)
        self.memory_report = bool(memory_report)
        for name, value in (('build', hnsw_build_num_threads),
                            ('search', hnsw_search_num_threads)):
            if value != -1 and value < 1:
                raise ValueError(f'hnsw_{name}_num_threads must be -1 or positive')
        if retrieval_batch_size < 1 or topm < 1:
            raise ValueError('retrieval_batch_size and topm must be positive')
        self.hnsw_build_num_threads = int(hnsw_build_num_threads)
        self.hnsw_search_num_threads = int(hnsw_search_num_threads)
        self.hnsw_recall_check = bool(hnsw_recall_check)
        self.hnsw_recall_queries = int(hnsw_recall_queries)
        self.hnsw_recall_seed = int(hnsw_recall_seed)
        if isinstance(hnsw_recall_splits, str):
            hnsw_recall_splits = tuple(
                item.strip() for item in hnsw_recall_splits.split(',')
                if item.strip()
            )
        self.hnsw_recall_splits = tuple(hnsw_recall_splits)
        self.hnsw_recall_candidate_chunk_size = int(
            hnsw_recall_candidate_chunk_size
        )
        self.hnsw_recall_query_batch_size = int(hnsw_recall_query_batch_size)
        self.hnsw_recall_device = torch.device(hnsw_recall_device)
        self.hnsw_recall_cache_dir = str(hnsw_recall_cache_dir)
        if self.hnsw_recall_queries < 1:
            raise ValueError('hnsw_recall_queries must be >= 1')
        if self.hnsw_recall_candidate_chunk_size < 1:
            raise ValueError('hnsw_recall_candidate_chunk_size must be >= 1')
        if self.hnsw_recall_query_batch_size < 1:
            raise ValueError('hnsw_recall_query_batch_size must be >= 1')
        valid_recall_splits = {'train', 'valid', 'test'}
        if (not self.hnsw_recall_splits
                or any(split not in valid_recall_splits
                       for split in self.hnsw_recall_splits)):
            raise ValueError(
                'hnsw_recall_splits must be a non-empty subset of '
                'train,valid,test'
            )
        if self.hnsw_recall_device.type not in {'cpu', 'cuda'}:
            raise ValueError('hnsw_recall_device must be cpu or cuda')
        if (self.hnsw_recall_device.type == 'cuda'
                and not torch.cuda.is_available()):
            raise ValueError('CUDA HNSW recall requested without CUDA')
        self.search_target = (
            self.topm if not self.crest_enabled
            else self.mmr_candidate_pool
        )

        if self.crest_enabled and mmr_candidate_pool < topm:
            raise ValueError('mmr_candidate_pool must be >= topm')
        if hnsw_oversample_factor < 1:
            raise ValueError('hnsw_oversample_factor must be >= 1')
        if retrieval_aggregation_chunk_size < 1:
            raise ValueError('retrieval_aggregation_chunk_size must be >= 1')
        if self.retrieval_postprocess_device.type not in ('cpu', 'cuda'):
            raise ValueError('retrieval_postprocess_device must be cpu or cuda')
        if self.mmr_device.type not in ('cpu', 'cuda'):
            raise ValueError('mmr_device must be cpu or cuda')
        if mmr_redundancy_source not in {'key', 'value'}:
            raise ValueError('mmr_redundancy_source must be key or value')
        if value_mmr_chunk_size < 1:
            raise ValueError('value_mmr_chunk_size must be >= 1')
        if (self.crest_enabled
                and self.retrieval_postprocess_device.type == 'cuda'
                and not torch.cuda.is_available()):
            raise ValueError('CUDA retrieval post-processing requested without CUDA')
        if (self.crest_enabled and self.mmr_device.type == 'cuda'
                and not torch.cuda.is_available()):
            raise ValueError('CUDA MMR requested without CUDA')
        if (self.crest_enabled
                and not 1 <= self.profile_reliability_k <= mmr_candidate_pool):
            raise ValueError('profile_reliability_k must be within the candidate pool')

        for period in self.period_num:
            if seq_len % period or pred_len % period:
                raise ValueError(
                    f'seq_len ({seq_len}) and pred_len ({pred_len}) must both '
                    f'be divisible by every retrieval period; failed for {period}'
                )

        validate_crest_profile_options(self)
        validate_value_mmr_args(self)

        self.panel_exact_max_candidates = panel_exact_max_candidates
        self.n_train = 0
        self.hnsw_indices = []
        self.exact_key_banks = []
        self.y_data_all_mg = []
        self.value_norm_banks = []
        self._wide_fallbacks = 0
        self._wide_searches = 0
        self._train_queries = 0
        self.profile_bank = None
        self.profile_scale = None
        self.profile_bank_postprocess = None
        self.profile_scale_postprocess = None
        self.prepare_runtime = {}
        self.split_runtime = {}
        self._active_runtime = None
        self.panel_mode = False
        self.panel_series_bank = None
        self.panel_candidate_manifest = None
        self._active_panel_query_manifest = None
        self.panel_memory_report = None

    @staticmethod
    def _decompose_period(data, period, remove_offset=True):
        """Pool a patch by period and optionally subtract its own final value."""
        data = data.unfold(dimension=1, size=period, step=period).mean(dim=-1)
        data = data.repeat_interleave(repeats=period, dim=1)
        if remove_offset:
            data = data - data[:, -1:, :]
        return data

    def _key_for_period(self, data, period):
        # Centered unit-length keys give Pearson-equivalent cosine similarity
        # for both exact search and the HNSW archive (paper Section 4.1).
        key = self._decompose_period(data, period).reshape(data.shape[0], -1)
        key.sub_(key.mean(dim=1, keepdim=True))
        key.div_(key.norm(p=2, dim=1, keepdim=True).clamp_min_(1e-12))
        return key

    @staticmethod
    def _sample_xy(sample):
        if len(sample) != 5:
            raise ValueError(
                'SHARP retrieval expects indexed forecasting samples with five fields'
            )
        return sample[1], sample[2]

    @staticmethod
    def _synchronize(device):
        if device is not None and torch.device(device).type == 'cuda':
            torch.cuda.synchronize(torch.device(device))

    def _stage_start(self, device=None):
        if self._active_runtime is None:
            return None
        self._synchronize(device)
        return time.perf_counter()

    def _record_ms(self, name, start, device=None):
        if self._active_runtime is not None:
            self._synchronize(device)
            self._active_runtime[name] += (time.perf_counter() - start) * 1000.0

    def prepare_dataset(self, train_data):
        """Build training-only keys, values, and optional CREST state profiles."""
        if getattr(train_data, 'is_panel_dataset', False):
            if self.retrieval_backend == 'exact':
                return self._prepare_panel_exact_dataset(train_data)
            return self._prepare_panel_dataset(train_data)
        prepare_start = time.perf_counter()
        data_pack_start = time.perf_counter()
        self.n_train = len(train_data)
        if self.n_train < self.search_target:
            raise ValueError(
                f'search target ({self.search_target}) cannot exceed the number '
                f'of training patches ({self.n_train})'
            )

        train_x_array = np.empty(
            (self.n_train, self.seq_len, self.channels), dtype=np.float32
        )
        train_y_array = np.empty(
            (self.n_train, self.pred_len, self.channels), dtype=np.float32
        )
        for sample_idx in range(self.n_train):
            x, y = self._sample_xy(train_data[sample_idx])
            x = np.asarray(x)
            y = np.asarray(y)
            if x.shape != (self.seq_len, self.channels):
                raise ValueError(
                    f'train sample {sample_idx} has x shape {x.shape}; expected '
                    f'({self.seq_len}, {self.channels})'
                )
            if y.ndim != 2 or y.shape[0] < self.pred_len or y.shape[1] != self.channels:
                raise ValueError(
                    f'train sample {sample_idx} has y shape {y.shape}; expected '
                    f'at least ({self.pred_len}, {self.channels})'
                )
            train_x_array[sample_idx] = x
            train_y_array[sample_idx] = y[-self.pred_len:]

        train_x = torch.from_numpy(train_x_array)
        train_y = torch.from_numpy(train_y_array)
        del train_x_array, train_y_array
        data_pack_ms = (time.perf_counter() - data_pack_start) * 1000.0

        self.profile_bank = None
        self.profile_scale = None
        self.profile_bank_postprocess = None
        self.profile_scale_postprocess = None
        self.value_norm_banks = []
        profile_ms = 0.0
        if self.crest_enabled:
            profile_start = time.perf_counter()
            self.raw_future_bank = train_y
            self.value_scale = value_iqr_scale(train_y)
            if self.crest_profile == 'stp':
                self.profile_bank = compute_stp_profile(
                    train_x, self.profile_sections
                ).contiguous()
                self.profile_scale = compute_stp_iqr_scale(self.profile_bank).contiguous()
            else:
                self.profile_bank = compute_mean_profile(
                    train_x, self.profile_sections
                ).to(torch.float64).contiguous()
                self.profile_scale = compute_iqr_scale(self.profile_bank).contiguous()
            self.profile_bank_postprocess = self.profile_bank.to(
                self.retrieval_postprocess_device
            )
            self.profile_scale_postprocess = self.profile_scale.to(
                self.retrieval_postprocess_device
            )
            self._synchronize(self.retrieval_postprocess_device)
            profile_ms = (time.perf_counter() - profile_start) * 1000.0

        search_build_start = time.perf_counter()
        self.hnsw_indices = []
        self.exact_key_banks = []
        labels = np.arange(self.n_train, dtype=np.int64)
        # HTA links each indexed key to its subsequent value by the same ID.
        # Backend selection changes search, not the training candidate pairs.
        for period in self.period_num:
            keys = self._key_for_period(train_x, period)
            if self.retrieval_backend == 'hnsw':
                index = hnswlib.Index(space='cosine', dim=keys.shape[1])
                index.init_index(
                    max_elements=self.n_train,
                    ef_construction=self.hnsw_ef_construction,
                    M=self.hnsw_m,
                    random_seed=self.hnsw_seed,
                )
                index.add_items(keys.numpy(), labels, num_threads=self.hnsw_build_num_threads)
                index.set_num_threads(self.hnsw_search_num_threads)
                index.set_ef(max(self.hnsw_ef_search, self.search_target))
                self.hnsw_indices.append(index)
            else:
                self.exact_key_banks.append(keys.contiguous())
            del keys
        search_build_ms = (
            time.perf_counter() - search_build_start
        ) * 1000.0

        value_start = time.perf_counter()
        del train_x
        self.y_data_all_mg = [
            self._decompose_period(train_y, period) for period in self.period_num
        ]
        value_ms = (time.perf_counter() - value_start) * 1000.0
        value_norm_ms = 0.0
        if self.crest_enabled and self.mmr_redundancy_source == 'value':
            value_norm_start = time.perf_counter()
            self.value_norm_banks = [
                self._build_value_mmr_bank(values)
                for values in self.y_data_all_mg
            ]
            value_norm_ms = (
                time.perf_counter() - value_norm_start
            ) * 1000.0
        del train_y
        total_ms = (time.perf_counter() - prepare_start) * 1000.0
        self.prepare_runtime = {
            'data_pack_ms': data_pack_ms,
            'profile_bank_iqr_ms': profile_ms,
            'hnsw_build_ms': (
                search_build_ms if self.retrieval_backend == 'hnsw' else 0.0
            ),
            'exact_key_bank_ms': (
                search_build_ms if self.retrieval_backend == 'exact' else 0.0
            ),
            'value_bank_ms': value_ms,
            'value_norm_bank_ms': value_norm_ms,
            'prepare_total_ms': total_ms,
        }

    def _prepare_panel_exact_dataset(self, train_data):
        """Materialize panel past windows temporarily and retain full CPU keys."""
        if self.hnsw_recall_check:
            raise ValueError('HNSW recall requires retrieval_backend=hnsw')
        validate_panel_crest_options(self)
        limit = self.panel_exact_max_candidates
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 6000000:
            raise ValueError('panel_exact_max_candidates must be an integer in [1, 6000000]')
        prepare_start = time.perf_counter()
        self.panel_mode = True
        self.panel_series_bank = train_data.series_bank
        self.panel_candidate_manifest = train_data.candidate_manifest
        self._active_panel_query_manifest = train_data.query_manifest
        self.panel_memory_report = train_data.memory_report
        self.n_train = len(self.panel_candidate_manifest)
        if self.n_train > limit:
            raise ValueError(
                f'panel exact retrieval is limited to {limit} candidates; '
                f'got {self.n_train}; use HNSW for larger web-scale experiments'
            )
        if self.n_train < self.search_target:
            raise ValueError(
                f'search target ({self.search_target}) cannot exceed the number '
                f'of training candidates ({self.n_train})'
            )
        if self.panel_series_bank.n_channels != self.channels:
            raise ValueError('panel bank channel count does not match RetrievalTool')
        self.hnsw_indices = []
        self.exact_key_banks = []
        self.y_data_all_mg = []
        self.value_norm_banks = []
        self.profile_bank = None
        self.profile_scale = None
        self.profile_bank_postprocess = None
        self.profile_scale_postprocess = None
        pack_start = time.perf_counter()
        ids = np.arange(self.n_train, dtype=np.int64)
        series, starts = self.panel_candidate_manifest.resolve_ids(ids)
        train_x = torch.from_numpy(self.panel_series_bank.get_windows(
            series, starts, self.seq_len,
        )).float().contiguous()
        data_pack_ms = (time.perf_counter() - pack_start) * 1000.0
        key_start = time.perf_counter()
        self.exact_key_banks = [
            self._key_for_period(train_x, period).contiguous()
            for period in self.period_num
        ]
        key_ms = (time.perf_counter() - key_start) * 1000.0
        del train_x, ids, series, starts
        self.prepare_runtime = {
            'data_pack_ms': data_pack_ms,
            'profile_bank_iqr_ms': 0.0,
            'hnsw_build_ms': 0.0,
            'exact_key_bank_ms': key_ms,
            'value_bank_ms': 0.0,
            'value_norm_bank_ms': 0.0,
            'prepare_total_ms': (time.perf_counter() - prepare_start) * 1000.0,
        }
        self.panel_memory_report.update({
            'exact_candidate_count': self.n_train,
            'exact_key_dim': self.seq_len * self.channels,
            'exact_period_banks': len(self.exact_key_banks),
            'exact_key_bank_bytes': sum(bank.numel() * bank.element_size()
                                        for bank in self.exact_key_banks),
            'full_candidate_x_prepare_shape': [self.n_train, self.seq_len, self.channels],
            'full_candidate_x_prepare_bytes': self.n_train * self.seq_len * self.channels * 4,
            'largest_full_similarity_rows': 0,
            'largest_full_similarity_cols': 0,
            'largest_full_similarity_bytes': 0,
            'panel_exact_max_candidates': limit,
            'full_candidate_x_materialized': True,
            'full_candidate_x_materialized_during_prepare': True,
            'full_candidate_x_retained': False,
            'full_candidate_y_materialized': False,
            'full_similarity_matrix_per_query_batch': True,
        })

    def _panel_exact_candidates(
        self, query_windows, period_idx, query_indices, train, target_count
    ):
        """One CPU matmul against all candidates, with panel overlap masking."""
        batch_size = len(query_windows)
        score_bytes = batch_size * self.n_train * 4
        report = self.panel_memory_report
        if batch_size > report['largest_full_similarity_rows']:
            report['largest_full_similarity_rows'] = batch_size
            report['largest_full_similarity_cols'] = self.n_train
            report['largest_full_similarity_bytes'] = score_bytes
        try:
            query_keys = self._key_for_period(query_windows, self.period_num[period_idx])
            similarities = torch.matmul(query_keys, self.exact_key_banks[period_idx].T)
            if train:
                if self._active_panel_query_manifest is None:
                    raise RuntimeError('panel training mask requires an active query manifest')
                candidate_series, candidate_starts = self.panel_candidate_manifest.resolve_ids(
                    np.arange(self.n_train, dtype=np.int64)
                )
                query_series, query_starts = self._active_panel_query_manifest.resolve_ids(query_indices)
                width = self.seq_len + self.pred_len
                forbidden = (
                    (candidate_series[None, :] == query_series[:, None])
                    & (candidate_starts[None, :] < query_starts[:, None] + width)
                    & (query_starts[:, None] < candidate_starts[None, :] + width)
                )
                similarities.masked_fill_(torch.from_numpy(forbidden), float('-inf'))
                available = torch.isfinite(similarities).sum(dim=1)
                if torch.any(available < target_count):
                    row = int(torch.nonzero(available < target_count, as_tuple=False)[0])
                    raise RuntimeError(
                        'not enough non-overlapping training candidates for query '
                        f'{int(query_indices[row])}: found {int(available[row])}, '
                        f'need {target_count}'
                    )
            selected_similarities, selected_labels = torch.topk(
                similarities, k=target_count, dim=1, largest=True, sorted=True,
            )
        except (MemoryError, RuntimeError) as exc:
            if (isinstance(exc, MemoryError) or 'out of memory' in str(exc).lower()
                    or "can't allocate memory" in str(exc).lower()):
                raise RuntimeError(
                    f'panel exact allocation failed: N={self.n_train}, B={batch_size}, '
                    f'estimated similarity bytes={score_bytes}; lower --retrieval_batch_size'
                ) from exc
            raise
        return selected_labels.long(), selected_similarities.float()

    def _prepare_panel_dataset(self, train_data):
        """Build HNSW incrementally without materializing all panel windows."""
        if self.retrieval_backend != 'hnsw':
            raise NotImplementedError('panel mode supports HNSW retrieval only')
        validate_panel_crest_options(self)
        prepare_start = time.perf_counter()
        self.panel_mode = True
        if hasattr(self, '_recall_values_digest'):
            del self._recall_values_digest
        self.panel_series_bank = train_data.series_bank
        self.panel_candidate_manifest = train_data.candidate_manifest
        self._active_panel_query_manifest = train_data.query_manifest
        self.panel_memory_report = train_data.memory_report
        self.n_train = len(self.panel_candidate_manifest)
        if self.n_train < self.search_target:
            raise ValueError(
                f'search target ({self.search_target}) cannot exceed the number '
                f'of training candidates ({self.n_train})'
            )
        if self.panel_series_bank.n_channels != self.channels:
            raise ValueError('panel bank channel count does not match RetrievalTool')

        self.hnsw_indices = []
        self.exact_key_banks = []
        search_build_start = time.perf_counter()
        for _period in self.period_num:
            index = hnswlib.Index(
                space='cosine', dim=self.seq_len * self.channels
            )
            index.init_index(
                max_elements=self.n_train,
                ef_construction=self.hnsw_ef_construction,
                M=self.hnsw_m,
                random_seed=self.hnsw_seed,
            )
            self.hnsw_indices.append(index)

        largest_build_batch = 0
        for start in range(0, self.n_train, self.hnsw_build_batch_size):
            end = min(start + self.hnsw_build_batch_size, self.n_train)
            labels = np.arange(start, end, dtype=np.int64)
            series_ids, starts = self.panel_candidate_manifest.resolve_ids(labels)
            windows = self.panel_series_bank.get_windows(
                series_ids, starts, self.seq_len
            )
            train_x = torch.from_numpy(windows)
            largest_build_batch = max(largest_build_batch, len(labels))
            for period_idx, period in enumerate(self.period_num):
                keys = self._key_for_period(train_x, period)
                self.hnsw_indices[period_idx].add_items(
                    keys.numpy(), labels, num_threads=self.hnsw_build_num_threads
                )
            del train_x, windows
        for index in self.hnsw_indices:
            index.set_num_threads(self.hnsw_search_num_threads)
            index.set_ef(max(self.hnsw_ef_search, self.search_target))
        search_build_ms = (
            time.perf_counter() - search_build_start
        ) * 1000.0

        # Panel futures remain in the shared series bank and are gathered only
        # for the labels returned by HNSW.
        self.y_data_all_mg = []
        self.value_norm_banks = []
        self.profile_bank = None
        self.profile_scale = None
        self.profile_bank_postprocess = None
        self.profile_scale_postprocess = None
        total_ms = (time.perf_counter() - prepare_start) * 1000.0
        self.prepare_runtime = {
            'data_pack_ms': 0.0,
            'profile_bank_iqr_ms': 0.0,
            'hnsw_build_ms': search_build_ms,
            'exact_key_bank_ms': 0.0,
            'value_bank_ms': 0.0,
            'value_norm_bank_ms': 0.0,
            'prepare_total_ms': total_ms,
        }
        index_bytes = 0
        for index in self.hnsw_indices:
            size = getattr(index, 'index_file_size', None)
            if size is not None:
                index_bytes += int(size())
        self.panel_memory_report.update({
            'hnsw_index_bytes': index_bytes,
            'hnsw_period_indices': len(self.hnsw_indices),
            'hnsw_candidate_count': self.n_train,
            'hnsw_key_dim': self.seq_len * self.channels,
            'hnsw_m': self.hnsw_m,
            'hnsw_ef_construction': self.hnsw_ef_construction,
            'hnsw_build_num_threads': self.hnsw_build_num_threads,
            'hnsw_search_num_threads': self.hnsw_search_num_threads,
            'hnsw_build_seconds': search_build_ms / 1000.0,
            'hnsw_build_batch_size': self.hnsw_build_batch_size,
            'largest_hnsw_build_batch': largest_build_batch,
            'full_candidate_x_materialized': False,
            'full_candidate_y_materialized': False,
        })
        try:
            from src.data_provider.webscale_panel_adapter import process_rss_bytes
            import resource
            self.panel_memory_report['rss_after_hnsw_bytes'] = process_rss_bytes()
            self.panel_memory_report['peak_rss_bytes'] = int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            )
        except (ImportError, OSError, ValueError):
            pass

    def prepare_panel_crest_state(self, train_data, cache_root='./retrieval-cache'):
        """Cache candidate STP and training IQR scales after exact/HNSW setup."""
        if not self.panel_mode or not self.crest_enabled:
            raise ValueError('panel CREST preparation requires panel CREST mode')
        validate_panel_crest_options(self)
        if (train_data.series_bank is not self.panel_series_bank
                or train_data.candidate_manifest is not self.panel_candidate_manifest):
            raise ValueError('panel CREST must use the indexed training candidates')
        self.cleanup_panel_crest_state()
        started = time.perf_counter()
        shape = (self.n_train, self.profile_sections, self.channels, 3)
        self.panel_stp_sidecar = PanelSTPSidecar(cache_root, shape)
        try:
            self.profile_scale = self.panel_stp_sidecar.prepare(
                self.panel_series_bank, self.panel_candidate_manifest,
                self.seq_len, self.profile_sections, self.hnsw_build_batch_size,
            )
            self.profile_scale_postprocess = self.profile_scale.to(
                self.retrieval_postprocess_device
            )
            self.value_scale = panel_value_iqr_scale(
                self.panel_series_bank, self.panel_candidate_manifest
            )
            self._synchronize(self.retrieval_postprocess_device)
        except BaseException:
            self.cleanup_panel_crest_state()
            raise
        elapsed = (time.perf_counter() - started) * 1000.0
        self.prepare_runtime['profile_bank_iqr_ms'] = elapsed
        self.prepare_runtime['prepare_total_ms'] += elapsed
        self.panel_memory_report.update({
            'largest_panel_crest_profile_batch': min(self.n_train, self.hnsw_build_batch_size),
            'largest_panel_crest_value_batch': 0,
            'panel_crest_profile_sidecar_bytes': int(np.prod(shape)) * 4,
            'panel_crest_scale_coordinate_bytes': self.n_train * 4,
        })

    def cleanup_panel_crest_state(self):
        sidecar = getattr(self, 'panel_stp_sidecar', None)
        if sidecar is not None:
            sidecar.close()
            self.panel_stp_sidecar = None
        self._panel_rp_projection = None
        self.value_scale = None
        self.last_reliability_components = None
        self.profile_scale = None
        self.profile_scale_postprocess = None

    def _panel_profile_reliability(self, query_windows, candidate_labels):
        """Estimate cr_q from past agreement and candidate-future consistency."""
        if getattr(self, 'panel_stp_sidecar', None) is None:
            raise RuntimeError('panel CREST state has not been prepared')
        device = self.retrieval_postprocess_device
        # Query IDs belong to a separate manifest, even in the training split.
        query_profiles = compute_stp_profile(
            query_windows.to(device), self.profile_sections
        )
        candidates = self.panel_stp_sidecar.lookup(candidate_labels).to(device)
        self.panel_memory_report['largest_panel_crest_profile_batch'] = max(
            self.panel_memory_report['largest_panel_crest_profile_batch'],
            candidate_labels.numel(), len(query_windows),
        )
        distance = robust_stp_distance(
            query_profiles, candidates, self.profile_scale_postprocess, self.stp_weights,
        )
        similarity = stp_distance_to_similarity(distance, self.mean_profile_temperature)
        cr_q, sigma_q, lambda_q = self._candidate_reliability(similarity, candidate_labels)
        return cr_q, sigma_q, lambda_q, distance, similarity

    def _candidate_reliability(self, similarity, candidate_labels):
        """Combine A_q and V_q on STP-ranked, temporally separated neighbors."""
        pool_labels = candidate_labels.detach().cpu().numpy()
        if self.panel_mode:
            series, starts = self.panel_candidate_manifest.resolve_ids(pool_labels)
        else:
            # General datasets use stride-one windows of one multivariate series.
            series, starts = np.zeros_like(pool_labels), pool_labels
        positions, valid = spaced_neighborhood(
            similarity, series, starts, self.profile_reliability_k, self.pred_len,
        )
        N_q_size = valid.sum(dim=1)
        A_q = (similarity.gather(1, positions) * valid).sum(dim=1) / N_q_size
        labels = candidate_labels.detach().cpu().gather(1, positions.cpu())
        consistency = []
        for begin in range(0, len(labels), self.retrieval_aggregation_chunk_size):
            selected = labels[begin:begin + self.retrieval_aggregation_chunk_size]
            if self.panel_mode:
                series, starts = self.panel_candidate_manifest.resolve_ids(selected.numpy().reshape(-1))
                future = torch.from_numpy(self.panel_series_bank.get_windows(
                    series, starts + self.seq_len, self.pred_len,
                )).reshape(len(selected), self.profile_reliability_k, self.pred_len, self.channels)
            else:
                future = self.raw_future_bank[selected]
            consistency.append(future_consistency(
                future.to(similarity.device), self.value_scale, self.tau_v,
                valid_mask=valid[begin:begin + len(selected)],
            ))
        V_q = torch.cat(consistency).to(A_q)
        cr_q = A_q * V_q
        sigma_q, lambda_q = map_reliability(
            cr_q, self.sigma_min, self.sigma_max, self.lambda_min, self.lambda_max,
        )
        if not torch.isfinite(torch.stack([cr_q, sigma_q, lambda_q], dim=1)).all():
            raise ValueError('non-finite candidate reliability output')
        self.last_reliability_components = {
            'A_q': A_q.detach(), 'V_q': V_q.detach(), 'cr_q': cr_q.detach(),
            'N_q_size': N_q_size.detach(),
        }
        return cr_q, sigma_q, lambda_q


    def _panel_value_mmr_representation(self, labels, period):
        """Represent only this query chunk's Top-M future patches on CPU."""
        series, starts = self.panel_candidate_manifest.resolve_ids(
            labels.detach().cpu().numpy().reshape(-1)
        )
        future = self.panel_series_bank.get_windows(
            series, starts + self.seq_len, self.pred_len
        )
        values = self._decompose_period(torch.from_numpy(future), period)
        self.panel_memory_report['largest_panel_crest_value_batch'] = max(
            self.panel_memory_report['largest_panel_crest_value_batch'], labels.numel(),
        )
        if self.mmr_value_repr == 'raw':
            normalized = self._normalize_value_vectors(values.flatten(1))
        elif self.mmr_value_repr == 'paa':
            compressed = paa_value_patch(values, self.mmr_paa_segments).flatten(1)
            normalized = self._normalize_value_vectors(compressed)
        else:
            projection = getattr(self, '_panel_rp_projection', None)
            if projection is None:
                generator = torch.Generator(device='cpu').manual_seed(self.mmr_rp_seed)
                projection = torch.randn(
                    self.pred_len * self.channels, self.mmr_rp_dim,
                    generator=generator, device='cpu', dtype=values.dtype,
                ) / math.sqrt(self.mmr_rp_dim)
                self._panel_rp_projection = projection
            flattened = values.flatten(1)
            centered = flattened - flattened.mean(dim=1, keepdim=True)
            normalized = F.normalize(centered @ projection, p=2, dim=-1, eps=1e-12)
        return normalized.reshape(*labels.shape, -1)

    def mmr_select_panel_cuda_value_chunked(
        self, candidate_similarities, lambda_q, candidate_labels,
        period_idx, query_indices, train,
    ):
        """Supply bounded on-demand values to the unchanged stateless core."""
        selected = []
        for start in range(0, len(candidate_labels), self.value_mmr_chunk_size):
            end = min(start + self.value_mmr_chunk_size, len(candidate_labels))
            values = self._panel_value_mmr_representation(
                candidate_labels[start:end], self.period_num[period_idx]
            ).to(self.mmr_device)
            selected.append(self.mmr_select_batch_cuda(
                candidate_similarities[start:end], lambda_q[start:end], period_idx,
                query_indices[start:end], train,
                candidate_values=values, values_normalized=True,
            ))
            del values
        return torch.cat(selected, dim=0)

    def _panel_stp_input(self, x, query_indices):
        """Web WN-R input adapter; search and V-RAW continue to consume raw data."""
        scaler = getattr(self.panel_series_bank, 'stp_robust_scaler', None)
        if scaler is None:
            return x
        series, _ = self._active_panel_query_manifest.resolve_ids(query_indices)
        return torch.from_numpy(scaler.transform(x.detach().cpu().numpy(), series))

    def _retrieve_panel_crest(self, x, sample_index, train=True):
        """Run search, state estimation, trajectory selection, and aggregation."""
        x = x.detach().cpu().float()
        query_indices = np.asarray(sample_index, dtype=np.int64)
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.channels):
            raise ValueError(f'expected input [B, {self.seq_len}, {self.channels}]')
        if getattr(self, 'panel_stp_sidecar', None) is None:
            raise RuntimeError('panel CREST state has not been prepared')
        predictions = []
        for period_idx, period in enumerate(self.period_num):
            labels, similarities = self.search_candidates(
                x, period_idx, query_indices, train, target_count=self.search_target,
            )
            start = self._stage_start(self.retrieval_postprocess_device)
            cr_q, sigma_q, lambda_q, _, _ = self._panel_profile_reliability(
                self._panel_stp_input(x, query_indices), labels
            )
            if start is not None:
                self._record_ms('profile_reliability_ms', start, self.retrieval_postprocess_device)
            start = self._stage_start(self.mmr_device)
            if self.mmr_device.type == 'cuda':
                selected = self.mmr_select_panel_cuda_value_chunked(
                    similarities.to(self.mmr_device), lambda_q.to(self.mmr_device),
                    labels, period_idx, query_indices, train,
                ).cpu()
            else:
                lambda_cpu = lambda_q.cpu()
                chunks = []
                for begin in range(0, len(x), self.value_mmr_chunk_size):
                    end = min(begin + self.value_mmr_chunk_size, len(x))
                    values = self._panel_value_mmr_representation(labels[begin:end], period)
                    chunks.extend(self.mmr_select_from_pool(
                        similarities[row], lambda_cpu[row], period_idx,
                        query_indices[row], train, candidate_values=values[row - begin],
                        values_normalized=True,
                    ) for row in range(begin, end))
                    del values
                selected = torch.stack(chunks)
            if start is not None:
                self._record_ms('mmr_ms', start, self.mmr_device)
            start = self._stage_start(self.retrieval_postprocess_device)
            selected_labels = labels.gather(1, selected)
            selected_similarities = similarities.gather(1, selected).to(
                self.retrieval_postprocess_device
            )
            normalized = self.crest_weights(selected_similarities, cr_q, sigma_q)
            predictions.append(self._aggregate_panel_values(
                selected_labels, normalized.cpu(), period,
            ))
            if start is not None:
                self._record_ms('gaussian_aggregation_ms', start, self.retrieval_postprocess_device)
        return torch.stack(predictions).reshape(
            self.n_period, len(x), self.pred_len, self.channels,
        )

    @staticmethod
    def _normalize_value_vectors(values):
        """Mean-center and L2-normalize flattened historical value patches."""
        if values.ndim < 2:
            raise ValueError('value vectors must have at least two dimensions')
        centered = values - values.mean(dim=-1, keepdim=True)
        return F.normalize(centered, p=2, dim=-1, eps=1e-12)

    @staticmethod
    def value_patch_similarity(left, right):
        """Pearson-equivalent similarity for batch-first value patches."""
        if left.shape != right.shape or left.ndim < 2:
            raise ValueError('paired value patches must have equal batch-first shapes')
        left_normalized = RetrievalTool._normalize_value_vectors(
            left.reshape(left.shape[0], -1)
        )
        right_normalized = RetrievalTool._normalize_value_vectors(
            right.reshape(right.shape[0], -1)
        )
        return (left_normalized * right_normalized).sum(dim=-1)

    def _normalized_value_bank(self, values):
        flattened = values.flatten(start_dim=1)
        normalized = torch.empty_like(flattened)
        for start in range(0, len(flattened), 1024):
            end = min(start + 1024, len(flattened))
            normalized[start:end] = self._normalize_value_vectors(
                flattened[start:end]
            )
        return normalized

    def _build_value_mmr_bank(self, values):
        """Precompute period-specific MMR vectors once, retaining raw futures.

        RP uses the same seeded CPU Gaussian matrix for each equal-shaped bank.
        Temporary centered/projected tensors are bounded to 1024 rows. Global
        random state is untouched, and queries only index the resulting bank.
        """
        if self.mmr_value_repr == 'raw':
            return self._normalized_value_bank(values)
        if self.mmr_value_repr == 'paa':
            dimension = self.mmr_paa_segments * values.shape[2]
            projection = None
        else:
            dimension = self.mmr_rp_dim
            generator = torch.Generator(device='cpu').manual_seed(self.mmr_rp_seed)
            projection = torch.randn(
                values.shape[1] * values.shape[2], dimension,
                generator=generator, device='cpu', dtype=values.dtype,
            ) / math.sqrt(dimension)
            projection = projection.to(values.device)
        bank = values.new_empty((len(values), dimension))
        for start in range(0, len(values), 1024):
            end = min(start + 1024, len(values))
            chunk = values[start:end]
            if self.mmr_value_repr == 'paa':
                compressed = paa_value_patch(chunk, self.mmr_paa_segments).flatten(1)
                bank[start:end] = self._normalize_value_vectors(compressed)
            else:
                flattened = chunk.flatten(1)
                centered = flattened - flattened.mean(dim=1, keepdim=True)
                bank[start:end] = F.normalize(
                    centered @ projection, p=2, dim=-1, eps=1e-12,
                )
        return bank

    def _forbidden_bounds(self, query_index):
        radius = self.seq_len + self.pred_len - 1
        left = max(0, int(query_index) - radius)
        right = min(self.n_train - 1, int(query_index) + radius)
        return left, right

    @staticmethod
    def _similarity(distances):
        return 1.0 - distances.astype(np.float32, copy=False)

    @staticmethod
    def _normalize_knn_output(labels, distances, batch_size):
        """Copy hnswlib outputs into independent contiguous [batch, k] buffers."""
        labels = np.asarray(labels)
        distances = np.asarray(distances)

        if batch_size < 1:
            raise RuntimeError('HNSW post-filter received an empty query batch')
        if labels.size != distances.size:
            raise RuntimeError(
                'HNSW label/distance element-count mismatch: '
                f'labels={labels.size}, distances={distances.size}'
            )
        if labels.size % batch_size != 0:
            raise RuntimeError(
                f'invalid HNSW output size {labels.size}: cannot split across '
                f'{batch_size} queries'
            )

        k = labels.size // batch_size
        if k < 1:
            raise RuntimeError('HNSW returned an empty neighbor dimension')

        # Always detach post-processing from hnswlib-owned memory.  Using
        # independent C-contiguous buffers avoids shape/stride/view quirks from
        # propagating into boolean masking on large batches.
        labels_buffer = np.empty((batch_size, k), dtype=np.int64, order='C')
        distances_buffer = np.empty((batch_size, k), dtype=np.float32, order='C')
        labels_buffer.reshape(-1)[:] = np.asarray(
            labels, dtype=np.int64
        ).reshape(-1)
        distances_buffer.reshape(-1)[:] = np.asarray(
            distances, dtype=np.float32
        ).reshape(-1)
        return labels_buffer, distances_buffer

    def _postfilter(
        self, labels, distances, query_indices, target_count=None
    ):
        if self.panel_mode:
            return self._panel_postfilter(
                labels, distances, query_indices, target_count
            )
        target_count = self.search_target if target_count is None else target_count
        query_indices = np.asarray(query_indices, dtype=np.int64).reshape(-1)
        labels, distances = self._normalize_knn_output(
            labels, distances, len(query_indices)
        )
        filtered = []
        similarities = self._similarity(distances)
        for row, query_index in enumerate(query_indices):
            left, right = self._forbidden_bounds(query_index)
            allowed = (labels[row] < left) | (labels[row] > right)
            filtered.append((
                labels[row][allowed][:target_count],
                similarities[row][allowed][:target_count],
            ))
        return filtered

    def _panel_postfilter(
        self, labels, distances, query_indices, target_count=None
    ):
        """Remove only same-series overlapping half-open panel intervals."""
        target_count = self.search_target if target_count is None else target_count
        if self._active_panel_query_manifest is None:
            raise RuntimeError('panel training mask requires an active query manifest')
        query_indices = np.asarray(query_indices, dtype=np.int64).reshape(-1)
        labels, distances = self._normalize_knn_output(
            labels, distances, len(query_indices)
        )
        candidate_series, candidate_starts = (
            self.panel_candidate_manifest.resolve_ids(labels)
        )
        query_series, query_starts = self._active_panel_query_manifest.resolve_ids(
            query_indices
        )
        width = self.seq_len + self.pred_len
        same_series = candidate_series == query_series[:, None]
        overlaps = (
            (candidate_starts < query_starts[:, None] + width)
            & (query_starts[:, None] < candidate_starts + width)
        )
        similarities = self._similarity(distances)
        filtered = []
        for row in range(len(query_indices)):
            allowed = ~(same_series[row] & overlaps[row])
            filtered.append((
                labels[row][allowed][:target_count],
                similarities[row][allowed][:target_count],
            ))
        return filtered

    def _wide_k(self, target_count=None):
        target_count = self.search_target if target_count is None else target_count
        if self.panel_mode:
            width = self.seq_len + self.pred_len
            minimum_stride = int(self.panel_candidate_manifest.stride.min())
            max_forbidden = (2 * width - 2) // minimum_stride + 1
            return min(self.n_train, target_count + max_forbidden)
        max_forbidden = 2 * (self.seq_len + self.pred_len) - 1
        return min(self.n_train, target_count + max_forbidden)

    def _validate_masked_results(
        self, results, query_indices, target_count=None
    ):
        target_count = self.search_target if target_count is None else target_count
        for row, (ids, _) in enumerate(results):
            if len(ids) < target_count:
                raise RuntimeError(
                    'not enough non-overlapping training patches for query '
                    f'{int(query_indices[row])}: found {len(ids)}, '
                    f'need {target_count}'
                )

    def _masked_knn_query(
        self, index, queries, query_indices, target_count=None
    ):
        """Exclude overlapping training candidates with SHARP two-stage search.

        Search narrowly first; widen only for queries with too few candidates.
        The configured ef_search remains a lower bound in both stages.
        """
        target_count = self.search_target if target_count is None else target_count
        query_indices = np.asarray(query_indices, dtype=np.int64).reshape(-1)
        self._train_queries += len(query_indices)

        if self.candidate_search_strategy == 'wide':
            wide_k = self._wide_k(target_count)
            index.set_ef(max(self.hnsw_ef_search, wide_k))
            start = time.perf_counter() if self._active_runtime is not None else None
            labels, distances = index.knn_query(queries, k=wide_k)
            if start is not None:
                self._record_ms('hnsw_search_ms', start)
            start = time.perf_counter() if self._active_runtime is not None else None
            results = self._postfilter(
                labels, distances, query_indices, target_count
            )
            if start is not None:
                self._record_ms('masking_fallback_ms', start)
            self._wide_searches += len(query_indices)
            self._validate_masked_results(
                results, query_indices, target_count
            )
            return results

        narrow_k = min(
            self.n_train, self.hnsw_oversample_factor * target_count
        )
        index.set_ef(max(self.hnsw_ef_search, narrow_k))
        start = time.perf_counter() if self._active_runtime is not None else None
        labels, distances = index.knn_query(queries, k=narrow_k)
        if start is not None:
            self._record_ms('hnsw_search_ms', start)
        start = time.perf_counter() if self._active_runtime is not None else None
        results = self._postfilter(
            labels, distances, query_indices, target_count
        )
        if start is not None:
            self._record_ms('masking_fallback_ms', start)

        fallback_rows = [
            row for row, (ids, _) in enumerate(results)
            if len(ids) < target_count
        ]
        self._wide_fallbacks += len(fallback_rows)
        if fallback_rows:
            wide_k = self._wide_k(target_count)
            index.set_ef(max(self.hnsw_ef_search, wide_k))
            start = time.perf_counter() if self._active_runtime is not None else None
            wide_labels, wide_distances = index.knn_query(
                queries[fallback_rows], k=wide_k
            )
            if start is not None:
                self._record_ms('hnsw_search_ms', start)
            start = time.perf_counter() if self._active_runtime is not None else None
            wide_results = self._postfilter(
                wide_labels,
                wide_distances,
                query_indices[fallback_rows],
                target_count,
            )
            self._wide_searches += len(fallback_rows)
            for row, wide_result in zip(fallback_rows, wide_results):
                results[row] = wide_result
            if start is not None:
                self._record_ms('masking_fallback_ms', start)

        self._validate_masked_results(results, query_indices, target_count)
        return results

    def _unmasked_knn_query_arrays(self, index, queries, target_count=None):
        """Detach unmasked HNSW output into owned C-contiguous [B,K] arrays."""
        target_count = self.search_target if target_count is None else target_count
        index.set_ef(max(self.hnsw_ef_search, target_count))
        start = time.perf_counter() if self._active_runtime is not None else None
        labels, distances = index.knn_query(queries, k=target_count)
        if start is not None:
            self._record_ms('hnsw_search_ms', start)
        labels, distances = self._normalize_knn_output(
            labels, distances, len(queries)
        )
        expected_shape = (len(queries), target_count)
        if labels.shape != expected_shape:
            raise RuntimeError(
                'unmasked candidate count mismatch: '
                f'got shape {labels.shape}, expected {expected_shape}'
            )
        similarities = self._similarity(distances)
        if (not isinstance(similarities, np.ndarray)
                or similarities.dtype != np.float32
                or similarities.shape != expected_shape):
            raise RuntimeError(
                'unmasked similarities must be a float32 NumPy [B,K] array'
            )
        if (not similarities.flags.c_contiguous
                or not similarities.flags.owndata):
            similarities = similarities.copy(order='C')
        return labels, similarities

    def _unmasked_knn_query(self, index, queries, target_count=None):
        """Compatibility row-tuple interface for legacy callers."""
        labels, similarities = self._unmasked_knn_query_arrays(
            index, queries, target_count
        )
        return [(labels[row], similarities[row]) for row in range(len(labels))]

    def _exact_candidates(
        self, query_windows, period_idx, query_indices, train, target_count
    ):
        query_keys = self._key_for_period(
            query_windows, self.period_num[period_idx]
        )
        similarities = torch.matmul(
            query_keys, self.exact_key_banks[period_idx].T
        )
        if train:
            for row, query_index in enumerate(query_indices):
                left, right = self._forbidden_bounds(query_index)
                similarities[row, left:right + 1] = float('-inf')
            available = torch.isfinite(similarities).sum(dim=1)
            if torch.any(available < target_count):
                failing = int(torch.nonzero(
                    available < target_count, as_tuple=False
                )[0])
                raise RuntimeError(
                    'not enough non-overlapping training patches for query '
                    f'{int(query_indices[failing])}: found '
                    f'{int(available[failing])}, need {target_count}'
                )
        selected_similarities, selected_labels = torch.topk(
            similarities, k=target_count, dim=1, largest=True, sorted=True
        )
        return selected_labels.long(), selected_similarities.float()

    def search_candidates(
        self, query_windows, period_idx, query_indices, train,
        target_count=None,
    ):
        """Return backend-independent candidate labels and similarities."""
        target_count = self.search_target if target_count is None else target_count
        if not 1 <= target_count <= self.n_train:
            raise ValueError('search target must be within the training bank')
        query_windows = query_windows.detach().cpu().float()
        query_indices = np.asarray(query_indices, dtype=np.int64)
        if len(query_windows) != len(query_indices):
            raise ValueError('query windows and indices must have equal length')

        if self.retrieval_backend == 'exact':
            start = time.perf_counter() if self._active_runtime is not None else None
            exact_search = self._panel_exact_candidates if self.panel_mode else self._exact_candidates
            result = exact_search(
                query_windows, period_idx, query_indices, train, target_count
            )
            if start is not None:
                self._record_ms('exact_search_ms', start)
            return result

        key_start = time.perf_counter() if self.panel_mode else None
        queries = self._key_for_period(
            query_windows, self.period_num[period_idx]
        ).numpy().astype(np.float32, copy=False)
        if self.panel_mode and self._active_runtime is not None:
            self._record_ms('query_generation_ms', key_start)
        if train:
            neighbors = self._masked_knn_query(
                self.hnsw_indices[period_idx], queries, query_indices,
                target_count,
            )
        else:
            labels_array, similarities_array = self._unmasked_knn_query_arrays(
                self.hnsw_indices[period_idx], queries, target_count
            )
            return (
                torch.as_tensor(labels_array, dtype=torch.long),
                torch.as_tensor(similarities_array, dtype=torch.float32),
            )
        labels = torch.as_tensor(
            np.stack([item[0] for item in neighbors]).astype(
                np.int64, copy=False
            ),
            dtype=torch.long,
        )
        similarities = torch.as_tensor(
            np.stack([item[1] for item in neighbors]), dtype=torch.float32
        )
        return labels, similarities

    def compute_profile_reliability_batch(
        self, query_windows, candidate_labels, query_indices=None, train=False
    ):
        """Estimate General-data cr_q from past agreement and future consistency."""
        if self.profile_bank_postprocess is None:
            raise RuntimeError('profile bank has not been prepared')
        device = self.retrieval_postprocess_device
        labels_device = candidate_labels.to(device=device, dtype=torch.long)
        if train:
            if query_indices is None:
                raise ValueError('training profile lookup requires query indices')
            query_labels = torch.as_tensor(
                query_indices, device=device, dtype=torch.long
            )
            query_profiles = self.profile_bank_postprocess[query_labels]
        elif self.crest_profile == 'stp':
            query_profiles = compute_stp_profile(
                query_windows.to(device), self.profile_sections
            )
        else:
            query_profiles = compute_mean_profile(
                query_windows.to(device), self.profile_sections
            ).to(torch.float64)
        candidate_profiles = self.profile_bank_postprocess[labels_device]
        if self.crest_profile == 'stp':
            distance = robust_stp_distance(
                query_profiles, candidate_profiles, self.profile_scale_postprocess,
                self.stp_weights,
            )
            similarity = stp_distance_to_similarity(
                distance, self.mean_profile_temperature
            )
        else:
            distance = robust_profile_distance(
                query_profiles, candidate_profiles, self.profile_scale_postprocess
            )
            similarity = distance_to_similarity(
                distance, self.mean_profile_temperature
            )
        cr_q, sigma_q, lambda_q = self._candidate_reliability(similarity, candidate_labels)
        return cr_q, sigma_q, lambda_q, distance, similarity

    def _mmr_generator(self, period_idx, query_index, train):
        generator = torch.Generator(device='cpu')
        split_code = 1 if train else 2
        seed = (
            int(self.retrieval_seed)
            + 1_000_003 * int(period_idx)
            + 10_000_019 * int(query_index)
            + 100_000_007 * split_code
        ) % (2 ** 63 - 1)
        generator.manual_seed(seed)
        return generator

    def mmr_select_from_pool(
        self, candidate_similarities, lambda_q, period_idx, query_index, train,
        candidate_values=None, values_normalized=False,
    ):
        """Select candidates according to MMR scores."""
        pool_size = candidate_similarities.numel()
        normalized_values = None
        if self.mmr_redundancy_source == 'value':
            if candidate_values is None:
                raise ValueError('value MMR requires candidate historical values')
            normalized_values = candidate_values.reshape(pool_size, -1).to(
                dtype=candidate_similarities.dtype
            )
            if not values_normalized:
                normalized_values = self._normalize_value_vectors(
                    normalized_values
                )
        selected = []
        selected_mask = torch.zeros(pool_size, dtype=torch.bool)
        generator = self._mmr_generator(period_idx, query_index, train)
        running_redundancy = None
        for step in range(self.topm):
            if step == 0:
                best_position = 0
            else:
                scores = (
                    lambda_q * candidate_similarities
                    - (1.0 - lambda_q) * running_redundancy
                )
                scores[selected_mask] = float('-inf')
                probabilities = F.softmax(scores / 0.1, dim=0)
                best_position = torch.multinomial(
                    probabilities, 1, generator=generator
                ).item()
            selected.append(best_position)
            selected_mask[best_position] = True
            if self.mmr_redundancy_source == 'key':
                new_redundancy = 1.0 - torch.abs(
                    candidate_similarities - candidate_similarities[best_position]
                )
            else:
                new_redundancy = (
                    normalized_values
                    * normalized_values[best_position].unsqueeze(0)
                ).sum(dim=1)
            running_redundancy = (
                new_redundancy if running_redundancy is None
                else torch.maximum(running_redundancy, new_redundancy)
            )
        return torch.tensor(selected, dtype=torch.long)

    def _mmr_stateless_uniforms(self, query_indices, period_idx, train):
        """Return avalanche-mixed stateless uniforms for CUDA MMR counters."""
        mask32 = 0xFFFFFFFF
        two32 = 4_294_967_296
        query_ids = torch.as_tensor(
            query_indices, device=self.mmr_device, dtype=torch.int64
        ).reshape(-1, 1)
        steps = torch.arange(
            1, self.topm, device=self.mmr_device, dtype=torch.int64
        ).reshape(1, -1)
        step_words = torch.bitwise_and(steps, mask32)

        query_low = torch.bitwise_and(query_ids, mask32)
        query_high = torch.bitwise_and(
            torch.bitwise_right_shift(query_ids, 32), mask32
        )
        seed = int(self.retrieval_seed)
        seed_low = seed & mask32
        seed_high = (seed >> 32) & mask32
        period_word = int(period_idx) & mask32
        split_code = 1 if train else 2

        # Mask each product before combining components. With every operand at
        # most 2^32 - 1, the largest product is below 4.3e18 and is int64-safe.
        state = torch.bitwise_and(query_low * 10_000_019, mask32)
        state = torch.bitwise_xor(
            state,
            torch.bitwise_and(query_high * 100_000_007, mask32),
        )
        state = torch.bitwise_xor(
            state, (seed_low * 1_000_003) & mask32
        )
        state = torch.bitwise_xor(
            state, (seed_high * 1_000_000_021) & mask32
        )
        state = torch.bitwise_xor(
            state, (period_word * 48_271) & mask32
        )
        state = torch.bitwise_xor(
            state, (split_code * 40_692) & mask32
        )
        state = torch.bitwise_xor(
            state,
            torch.bitwise_and(step_words * 1_000_000_007, mask32),
        )

        # XOR shifts make this a nonlinear avalanche mixer rather than a
        # single affine modular mapping of query and step counters.
        state = torch.bitwise_xor(
            state, torch.bitwise_right_shift(state, 16)
        )
        state = torch.bitwise_and(state * 0x045D9F3B, mask32)
        state = torch.bitwise_xor(
            state, torch.bitwise_right_shift(state, 16)
        )
        state = torch.bitwise_and(state * 0x045D9F3B, mask32)
        state = torch.bitwise_and(
            torch.bitwise_xor(
                state, torch.bitwise_right_shift(state, 16)
            ),
            mask32,
        )
        return (state.to(torch.float64) + 0.5) / two32

    @staticmethod
    def _inverse_cdf_sample(probabilities, uniforms):
        """Sample rows after closing native-dtype CDF rounding gaps."""
        cdf = probabilities.cumsum(dim=1)
        total = cdf[:, -1:]
        cdf.div_(total.clamp_min(torch.finfo(cdf.dtype).tiny))
        cdf[:, -1] = 1.0
        return (cdf < uniforms).sum(dim=1)

    def mmr_select_batch_cuda(
        self, candidate_similarities, lambda_q, period_idx, query_indices, train,
        candidate_values=None, values_normalized=False,
    ):
        """Select candidates according to MMR scores on CUDA."""
        if self.mmr_device.type != 'cuda':
            raise ValueError('batched CUDA MMR requires mmr_device=cuda')
        if (candidate_similarities.device.type != 'cuda'
                or (self.mmr_device.index is not None
                    and candidate_similarities.device.index != self.mmr_device.index)):
            raise ValueError('candidate similarities are not on the configured MMR device')
        if candidate_similarities.ndim != 2 or lambda_q.ndim != 1:
            raise ValueError('CUDA MMR inputs must have shapes [B,M] and [B]')
        batch_size, pool_size = candidate_similarities.shape
        if lambda_q.shape[0] != batch_size:
            raise ValueError('CUDA MMR lambda batch size mismatch')
        if self.topm > pool_size:
            raise ValueError('CUDA MMR topm cannot exceed candidate pool size')

        normalized_values = None
        if self.mmr_redundancy_source == 'value':
            if candidate_values is None:
                raise ValueError('value MMR requires candidate historical values')
            if candidate_values.shape[:2] != (batch_size, pool_size):
                raise ValueError('CUDA value MMR inputs must start with shape [B,M]')
            if candidate_values.device != candidate_similarities.device:
                raise ValueError('CUDA value MMR inputs must share the MMR device')
            normalized_values = candidate_values.reshape(
                batch_size, pool_size, -1
            ).to(dtype=candidate_similarities.dtype)
            if not values_normalized:
                normalized_values = self._normalize_value_vectors(
                    normalized_values
                )

        # A shared stateful CUDA generator would make each row depend on prior
        # calls and batch order, so generate every query/step draw statelessly.
        uniforms = self._mmr_stateless_uniforms(
            query_indices, period_idx, train
        )
        if uniforms.shape[0] != batch_size:
            raise ValueError('CUDA MMR query index batch size mismatch')

        lambda_q = lambda_q.to(
            device=self.mmr_device, dtype=candidate_similarities.dtype
        ).unsqueeze(1)
        selected = torch.empty(
            batch_size, self.topm, device=self.mmr_device, dtype=torch.long
        )
        selected_mask = torch.zeros(
            batch_size, pool_size, device=self.mmr_device, dtype=torch.bool
        )
        first = torch.zeros(batch_size, device=self.mmr_device, dtype=torch.long)
        selected[:, 0] = first
        selected_mask.scatter_(1, first.unsqueeze(1), True)
        if self.mmr_redundancy_source == 'key':
            running_redundancy = 1.0 - torch.abs(
                candidate_similarities - candidate_similarities[:, 0:1]
            )
        else:
            running_redundancy = (
                normalized_values * normalized_values[:, 0:1]
            ).sum(dim=2)

        for step in range(1, self.topm):
            scores = (
                lambda_q * candidate_similarities
                - (1.0 - lambda_q) * running_redundancy
            )
            scores.masked_fill_(selected_mask, float('-inf'))
            probabilities = F.softmax(scores / 0.1, dim=1)
            new_position = self._inverse_cdf_sample(
                probabilities, uniforms[:, step - 1:step]
            )
            selected[:, step] = new_position
            selected_mask.scatter_(1, new_position.unsqueeze(1), True)
            if self.mmr_redundancy_source == 'key':
                chosen_similarity = candidate_similarities.gather(
                    1, new_position.unsqueeze(1)
                )
                new_redundancy = 1.0 - torch.abs(
                    candidate_similarities - chosen_similarity
                )
            else:
                chosen_value = normalized_values.gather(
                    1,
                    new_position.reshape(-1, 1, 1).expand(
                        -1, 1, normalized_values.shape[2]
                    ),
                )
                new_redundancy = (
                    normalized_values * chosen_value
                ).sum(dim=2)
            running_redundancy = torch.maximum(
                running_redundancy, new_redundancy
            )
        return selected

    def mmr_select_batch_cuda_value_chunked(
        self, candidate_similarities, lambda_q, candidate_labels, value_bank,
        period_idx, query_indices, train,
    ):
        """Run value MMR with only a bounded query chunk resident on CUDA."""
        if not self.crest_enabled or self.mmr_redundancy_source != 'value':
            raise ValueError('chunked value MMR requires CREST value mode')
        labels_cpu = candidate_labels.detach().cpu().long()
        query_indices = np.asarray(query_indices, dtype=np.int64)
        selected_chunks = []
        for start in range(0, len(candidate_similarities), self.value_mmr_chunk_size):
            end = min(start + self.value_mmr_chunk_size, len(candidate_similarities))
            candidate_values = value_bank[labels_cpu[start:end]].to(
                self.mmr_device, non_blocking=True
            )
            selected_chunks.append(self.mmr_select_batch_cuda(
                candidate_similarities[start:end],
                lambda_q[start:end],
                period_idx,
                query_indices[start:end],
                train,
                candidate_values=candidate_values,
                values_normalized=True,
            ))
        return torch.cat(selected_chunks, dim=0)

    @staticmethod
    def adaptive_softmax_weights(selected_similarities, cr_q):
        """Adjust weight concentration through a reliability-dependent temperature."""
        cr_q = cr_q.to(selected_similarities)
        tau = 0.10 + 0.05 * (1.0 - cr_q)
        weights = torch.softmax(selected_similarities / tau[:, None], dim=1)
        if (not torch.isfinite(weights).all()
                or not torch.allclose(weights.sum(1), torch.ones_like(tau), atol=1e-6, rtol=0)):
            raise ValueError('invalid adaptive softmax weights')
        return tau, weights

    def crest_weights(self, similarities, cr_q, sigma_q):
        """Combine selected candidates with reliability-dependent weights.

        High reliability concentrates weight on more similar candidates;
        low reliability spreads weight across selected trajectories.
        """
        if self.crest_weighting == 'gaussian':
            return self.gaussian_weights_batch(similarities, sigma_q)[1]
        return self.adaptive_softmax_weights(similarities, cr_q)[1]

    @staticmethod
    def gaussian_weights_batch(selected_similarities, sigma_q):
        """Adjust weight concentration through a reliability-dependent width."""
        if selected_similarities.ndim != 2 or sigma_q.ndim != 1:
            raise ValueError('Gaussian inputs must have shapes [B,K] and [B]')
        sigma_q = sigma_q.to(
            device=selected_similarities.device,
            dtype=selected_similarities.dtype,
        )
        raw = torch.exp(
            -((1.0 - selected_similarities) ** 2)
            / (2.0 * sigma_q.unsqueeze(1) ** 2)
        )
        denominator = raw.sum(dim=1)
        valid = torch.isfinite(denominator) & (denominator > 0)
        normalized = torch.zeros_like(raw)
        normalized[valid] = raw[valid] / denominator[valid].unsqueeze(1)
        invalid = ~valid
        if invalid.any():
            fallback = selected_similarities[invalid].argmax(dim=1, keepdim=True)
            normalized[invalid] = normalized[invalid].scatter(1, fallback, 1.0)
        largest = normalized.argmax(dim=1, keepdim=True)
        correction = 1.0 - normalized.sum(dim=1, keepdim=True)
        normalized.scatter_add_(1, largest, correction)
        if not torch.isfinite(raw).all() or not torch.isfinite(normalized).all():
            raise ValueError('non-finite Gaussian retrieval weights')
        return raw, normalized

    @staticmethod
    def gaussian_weights(selected_similarities, sigma_q):
        raw, normalized = RetrievalTool.gaussian_weights_batch(
            selected_similarities.unsqueeze(0), sigma_q.reshape(1)
        )
        return raw[0], normalized[0]

    def _aggregate_values(self, values, labels, weights):
        batch_size = labels.shape[0]
        prediction = torch.empty(
            batch_size, values.shape[1], dtype=values.dtype
        )
        chunk = self.retrieval_aggregation_chunk_size
        for start in range(0, batch_size, chunk):
            end = min(start + chunk, batch_size)
            prediction[start:end] = (
                values[labels[start:end]] * weights[start:end].unsqueeze(-1)
            ).sum(dim=1)
        return prediction

    def _aggregate_panel_values(self, labels, weights, period):
        """Gather only selected future windows from the shared in-memory bank."""
        batch_size, topk = labels.shape
        prediction = torch.empty(
            batch_size, self.pred_len * self.channels, dtype=torch.float32
        )
        chunk = self.retrieval_aggregation_chunk_size
        largest_value_batch = 0
        for start in range(0, batch_size, chunk):
            end = min(start + chunk, batch_size)
            flat_labels = labels[start:end].detach().cpu().numpy().reshape(-1)
            series_ids, candidate_starts = (
                self.panel_candidate_manifest.resolve_ids(flat_labels)
            )
            future = self.panel_series_bank.get_windows(
                series_ids,
                candidate_starts + self.seq_len,
                self.pred_len,
            )
            largest_value_batch = max(largest_value_batch, len(flat_labels))
            values = self._decompose_period(
                torch.from_numpy(future), period
            ).reshape(end - start, topk, -1)
            prediction[start:end] = (
                values * weights[start:end].unsqueeze(-1)
            ).sum(dim=1)
        if self.panel_memory_report is not None:
            self.panel_memory_report['largest_on_demand_future_batch'] = max(
                int(self.panel_memory_report.get(
                    'largest_on_demand_future_batch', 0
                )),
                largest_value_batch,
            )
        return prediction

    def retrieve(self, x, sample_index, train=True):
        if self.panel_mode and self.crest_enabled:
            return self._retrieve_panel_crest(x, sample_index, train)
        x = x.detach().cpu().float()
        query_indices = np.asarray(sample_index, dtype=np.int64)
        batch_size, seq_len, channels = x.shape
        if seq_len != self.seq_len or channels != self.channels:
            raise ValueError(
                f'expected input [B, {self.seq_len}, {self.channels}], '
                f'got {tuple(x.shape)}'
            )

        period_predictions = []
        for period_idx, _period in enumerate(self.period_num):
            labels, similarities = self.search_candidates(
                x,
                period_idx,
                query_indices,
                train,
                target_count=self.search_target,
            )
            if not self.crest_enabled:
                start = time.perf_counter() if self._active_runtime is not None else None
                weights = F.softmax(similarities / self.temperature, dim=1)
                if self.panel_mode:
                    prediction = self._aggregate_panel_values(
                        labels, weights, self.period_num[period_idx]
                    )
                else:
                    values = self.y_data_all_mg[period_idx].flatten(start_dim=1)
                    prediction = self._aggregate_values(values, labels, weights)
                if start is not None:
                    self._record_ms('baseline_aggregation_ms', start)
            else:
                values = self.y_data_all_mg[period_idx].flatten(start_dim=1)
                start = self._stage_start(self.retrieval_postprocess_device)
                cr_q, sigma_q, lambda_q, _distance, _profile_similarity = (
                    self.compute_profile_reliability_batch(
                        x, labels, query_indices=query_indices, train=train
                    )
                )
                if start is not None:
                    self._record_ms(
                        'profile_reliability_ms', start,
                        self.retrieval_postprocess_device,
                    )

                start = self._stage_start(self.mmr_device)
                if self.mmr_device.type == 'cpu':
                    lambda_cpu = lambda_q.detach().cpu()
                    selected_positions = torch.stack([
                        self.mmr_select_from_pool(
                            similarities[row], lambda_cpu[row], period_idx,
                            query_indices[row], train,
                            candidate_values=(
                                self.value_norm_banks[period_idx][labels[row]]
                                if self.mmr_redundancy_source == 'value'
                                else None
                            ),
                            values_normalized=(
                                self.mmr_redundancy_source == 'value'
                            ),
                        )
                        for row in range(batch_size)
                    ])
                else:
                    similarities_mmr = similarities.to(
                        self.mmr_device, non_blocking=True
                    )
                    lambda_mmr = lambda_q.to(
                        self.mmr_device,
                        dtype=similarities_mmr.dtype,
                        non_blocking=True,
                    )
                    if self.mmr_redundancy_source == 'value':
                        selected_positions = (
                            self.mmr_select_batch_cuda_value_chunked(
                                similarities_mmr,
                                lambda_mmr,
                                labels,
                                self.value_norm_banks[period_idx],
                                period_idx,
                                query_indices,
                                train,
                            )
                        )
                    else:
                        selected_positions = self.mmr_select_batch_cuda(
                            similarities_mmr, lambda_mmr, period_idx,
                            query_indices, train,
                        )
                if start is not None:
                    self._record_ms('mmr_ms', start, self.mmr_device)

                if self.mmr_device.type == 'cuda':
                    if self.retrieval_postprocess_device.type == 'cuda':
                        labels_mmr = labels.to(self.mmr_device, non_blocking=True)
                        selected_labels = labels_mmr.gather(
                            1, selected_positions
                        ).cpu()
                        selected_postprocess = similarities_mmr.gather(
                            1, selected_positions
                        )
                    else:
                        selected_positions = selected_positions.cpu()
                        selected_labels = labels.gather(1, selected_positions)
                        selected_postprocess = similarities.gather(
                            1, selected_positions
                        )
                else:
                    selected_labels = labels.gather(1, selected_positions)
                    selected_similarities = similarities.gather(
                        1, selected_positions
                    )
                    selected_postprocess = selected_similarities.to(
                        self.retrieval_postprocess_device
                    )
                start = self._stage_start(self.retrieval_postprocess_device)
                _raw, normalized = self.gaussian_weights_batch(
                    selected_postprocess, sigma_q
                )
                prediction = self._aggregate_values(
                    values, selected_labels, normalized.cpu()
                )
                if start is not None:
                    self._record_ms(
                        'gaussian_aggregation_ms', start,
                        self.retrieval_postprocess_device,
                    )
            period_predictions.append(prediction)

        return torch.stack(period_predictions).reshape(
            self.n_period, batch_size, self.pred_len, channels
        )

    def retrieve_all(self, data, train=False, split='unknown'):
        if self.panel_mode:
            if not getattr(data, 'is_panel_dataset', False):
                raise ValueError('panel retrieval requires a panel query dataset')
            if data.series_bank is not self.panel_series_bank:
                raise ValueError('panel splits must share one series bank')
            if data.candidate_manifest is not self.panel_candidate_manifest:
                raise ValueError('panel splits must share one candidate manifest')
            self._active_panel_query_manifest = data.query_manifest
        loader = None
        if not self.panel_mode:
            loader = DataLoader(
                data,
                batch_size=self.retrieval_batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )
        retrievals = torch.empty(
            self.n_period,
            len(data),
            self.pred_len,
            self.channels,
            dtype=torch.float32,
        )
        runtime = {
            'queries': len(data),
            'query_generation_ms': 0.0,
            'hnsw_search_ms': 0.0,
            'exact_search_ms': 0.0,
            'masking_fallback_ms': 0.0,
            'profile_reliability_ms': 0.0,
            'mmr_ms': 0.0,
            'baseline_aggregation_ms': 0.0,
            'gaussian_aggregation_ms': 0.0,
        }
        self._active_runtime = runtime if (self.runtime_breakdown or self.panel_mode) else None
        self._wide_fallbacks = 0
        self._wide_searches = 0
        self._train_queries = 0
        cursor = 0
        total_start = time.perf_counter()
        with torch.no_grad():
            if self.panel_mode:
                for start in range(0, len(data), self.retrieval_batch_size):
                    end = min(start + self.retrieval_batch_size, len(data))
                    query_start = time.perf_counter()
                    query_ids = np.arange(start, end, dtype=np.int64)
                    series_ids, starts = data.query_manifest.resolve_ids(
                        query_ids
                    )
                    windows = data.series_bank.get_windows(
                        series_ids, starts, self.seq_len
                    )
                    self._record_ms('query_generation_ms', query_start)
                    batch_retrieval = self.retrieve(
                        torch.from_numpy(windows), query_ids, train=train
                    )
                    if not torch.isfinite(batch_retrieval).all():
                        raise ValueError('panel retrieval contains non-finite values')
                    retrievals[:, start:end] = batch_retrieval
            else:
                for sample_index, batch_x, _batch_y, _x_mark, _y_mark in loader:
                    batch_retrieval = self.retrieve(
                        batch_x, sample_index, train=train
                    )
                    batch_size = batch_retrieval.shape[1]
                    retrievals[:, cursor:cursor + batch_size] = batch_retrieval
                    cursor += batch_size
        runtime['split_total_ms'] = (time.perf_counter() - total_start) * 1000.0
        runtime['ms_per_query'] = runtime['split_total_ms'] / max(len(data), 1)
        runtime['wide_fallbacks'] = self._wide_fallbacks
        runtime['wide_searches'] = self._wide_searches
        runtime['period_queries'] = self._train_queries
        self._active_runtime = None
        self.split_runtime[split] = runtime

        if self.panel_mode and self.panel_memory_report is not None:
            cache_bytes = int(retrievals.numel() * retrievals.element_size())
            self.panel_memory_report[f'{split}_retrieval_cache_bytes'] = cache_bytes
            self.panel_memory_report['largest_window_gather_batch'] = int(
                self.panel_series_bank.largest_window_batch
            )
            if self.memory_report:
                import json
                print(
                    f'[WebscaleMemory][retrieval-{split}] '
                    f'{json.dumps(self.panel_memory_report, sort_keys=True)}'
                )

        if train and self.retrieval_backend == 'hnsw':
            if self.candidate_search_strategy == 'wide':
                print(
                    'HNSW wide masking: '
                    f'{self._wide_searches}/{self._train_queries} period-queries '
                    'used the wide search'
                )
            else:
                print(
                    'HNSW two-stage masking: '
                    f'{self._wide_fallbacks}/{self._train_queries} period-queries '
                    'used the wide second search'
                )
        return retrievals.detach()
