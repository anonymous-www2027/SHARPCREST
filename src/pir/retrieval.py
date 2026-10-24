"""Search native PIR keys and select candidates with SHARP/CREST. Preserve PIR values, cosine softmax and beta."""
from collections import defaultdict
from contextlib import contextmanager
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from src.utils.candidate_reliability import spaced_neighborhood, future_consistency, value_iqr_scale
from src.pir.utils.stp_reliability import compute_stp_profile, compute_stp_iqr_scale
from src.pir.utils.stp_reliability import robust_stp_distance, stp_distance_to_similarity
from src.pir.utils.profile_reliability import map_reliability


METHODS = ('PIR', 'PIR_SHARP', 'PIR_CREST', 'PIR_SHARP_CREST')


class PIRRetrieval:
    def __init__(self, keys, values, metadata, args, raw_values=None):
        self.keys, self.values, self.args = keys.detach(), values.detach(), args
        self.device = keys.device
        self.n, self.length, self.channels = keys.shape
        self.k = args.retrieval_num
        self.sharp = 'SHARP' in args.retrieval_method
        self.crest = 'CREST' in args.retrieval_method
        self.target = args.crest_pool if self.crest else self.k
        self.timings = defaultdict(float)
        self.counts = defaultdict(int)
        self.controller_sums = {}
        self.component_sums = {}
        self.last_reliability_components = None
        self.tau_v = float(getattr(args, "tau_v", 1.0))
        if not math.isfinite(self.tau_v) or self.tau_v <= 0:
            raise ValueError("tau_v must be finite and positive")
        self.last = None
        self.prepared = {}
        self.split = 'train'
        self.weights = tuple(args.stp_weights)
        if args.retrieval_method not in METHODS:
            raise ValueError('unknown PIR retrieval method')
        if not (1 <= self.k <= self.target <= self.n):
            raise ValueError('candidate count must be >= pool >= K')
        if values.ndim != 3 or values.shape[0] != self.n or values.shape[2] != self.channels:
            raise ValueError('candidate key/value shapes are not aligned')
        if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
            raise ValueError('candidate keys/values must be finite')
        if args.retrieval_stride != 1:
            # Mask code also supports stride > 1 for parity/unit checks, but
            # experiment entry point freezes stride=1 in the agreed protocol.
            if args.retrieval_stride < 1:
                raise ValueError('retrieval_stride must be positive')
        if self.crest and not (1 <= args.reliability_r <= self.target):
            raise ValueError('R must be in [1, M]')
        if len(self.weights) != 3 or any(w < 0 or not math.isfinite(w) for w in self.weights) or not math.isclose(sum(self.weights), 1, abs_tol=1e-6):
            raise ValueError('STP weights must be finite, nonnegative and sum to one')
        self.sid = torch.as_tensor(metadata['series_id'], device=self.device, dtype=torch.long)
        self.starts = torch.as_tensor(metadata['start'], device=self.device, dtype=torch.long)
        if self.sid.shape != (self.n,) or self.starts.shape != (self.n,):
            raise ValueError('candidate metadata must align one-to-one with key rows')
        self.ranges = {}
        for sid in np.unique(metadata['series_id']):
            rows = np.flatnonzero(np.asarray(metadata['series_id']) == sid)
            starts = np.asarray(metadata['start'])[rows]
            if not np.array_equal(rows, np.arange(rows[0], rows[-1]+1)) or not np.array_equal(starts, np.arange(len(rows))*args.retrieval_stride):
                raise ValueError('candidate bank must contain contiguous, ordered train windows per series starting at zero')
            self.ranges[int(sid)] = (int(rows[0]), len(rows))
        ordered_series = sorted(self.ranges)
        self.range_series = torch.tensor(ordered_series, device=self.device)
        self.range_offsets = torch.tensor([self.ranges[s][0] for s in ordered_series], device=self.device)
        self.range_counts = torch.tensor([self.ranges[s][1] for s in ordered_series], device=self.device)
        self.candidate_labels = torch.arange(self.n, device=self.device)
        with self.clock('key_normalization_seconds'):
            self.unit_keys = F.normalize(self.keys.permute(2, 0, 1), dim=-1)
        self.indices = []
        if self.sharp:
            import hnswlib
            with self.clock('index_build_seconds'):
                for channel in range(self.channels):
                    index = hnswlib.Index(space='cosine', dim=self.length)
                    index.init_index(max_elements=self.n, M=args.hnsw_m,
                                     ef_construction=args.hnsw_ef_construction,
                                     random_seed=args.seed)
                    index.add_items(self.unit_keys[channel].cpu().numpy(), np.arange(self.n),
                                    num_threads=args.hnsw_build_threads)
                    index.set_num_threads(args.hnsw_threads)
                    index.set_ef(args.hnsw_ef_search)
                    self.indices.append(index)
        if self.crest:
            if raw_values is None or raw_values.shape != self.values.shape or not torch.isfinite(raw_values).all():
                raise ValueError('CREST requires aligned training futures before PIR window normalization')
            self.raw_values = raw_values.detach().to(self.device)
            with self.clock('profile_bank_seconds'):
                self.value_scale = value_iqr_scale(self.raw_values)
                self.profiles = compute_stp_profile(self.keys, args.profile_sections)
                self.scale = compute_stp_iqr_scale(self.profiles)

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @contextmanager
    def clock(self, name):
        self.sync()
        started = time.perf_counter()
        yield
        self.sync()
        self.timings[name] += time.perf_counter() - started

    def forbidden_bounds(self, series, starts, training):
        """Native PIR floor/clip mask applied within each query's own series."""
        lower = torch.full_like(starts, -1)
        upper = torch.full_like(starts, -2)
        if training:
            positions = torch.searchsorted(self.range_series, series.contiguous())
            positions = positions.clamp(max=len(self.range_series)-1)
            if (self.range_series[positions] != series).any():
                raise ValueError('training query series absent from candidate bank')
            offset, count = self.range_offsets[positions], self.range_counts[positions]
            lower = torch.minimum(((starts-self.length)//self.args.retrieval_stride).clamp_min(0), count-1)+offset
            upper = torch.minimum(((starts+self.length)//self.args.retrieval_stride).clamp_min(0), count-1)+offset
        available = self.n - (upper-lower+1).clamp_min(0)
        if (available < self.target).any():
            raise ValueError('not enough eligible train candidates after native PIR mask; do not return masked neighbors')
        return lower, upper

    @staticmethod
    def mask(scores, labels, lower, upper):
        return scores.masked_fill((labels >= lower[:, None]) & (labels <= upper[:, None]), -torch.inf)

    def exact(self, queries, lower, upper):
        """Native-style batched cosine search: [C,B,L] @ [C,L,N].

        Materialize the full score matrix; no candidate or channel chunking.
        Each channel still searches only its own keys.
        """
        # TF32 can choose different top-k boundary/order when B changes.
        # Stochastic MMR can amplify that difference. Use FP32 for cosine
        # search only; restore the caller's model-matmul policy immediately.
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            scores = torch.bmm(queries, self.unit_keys.transpose(1, 2))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        forbidden = ((self.candidate_labels[None, :] >= lower[:, None]) &
                     (self.candidate_labels[None, :] <= upper[:, None]))
        scores.masked_fill_(forbidden[None, :, :], -torch.inf)
        return scores.topk(self.target, dim=-1)

    def ann(self, query, channel, lower, upper, training):
        index = self.indices[channel]
        def search(q, budget, lo, hi):
            self.counts[self.split+'/hnsw_calls'] += 1
            # Mirrors the installed hnswlib small-call policy; requested=32
            # uses parallel search only when the query count is >128.
            kind = 'single_thread_calls' if len(q) <= self.args.hnsw_threads*4 else 'parallel_calls'
            self.counts[self.split+'/hnsw_'+kind] += 1
            with self.clock(self.split+'/hnsw_query_seconds'):
                index.set_ef(max(self.args.hnsw_ef_search, budget))
                ids, _ = index.knn_query(q.cpu().numpy(), k=budget, num_threads=self.args.hnsw_threads)
                ids = torch.as_tensor(ids.astype(np.int64), device=self.device)
            with self.clock(self.split+'/exact_rescore_seconds'):
                scores = (self.unit_keys[channel][ids]*q[:, None, :]).sum(-1)
                scores = self.mask(scores, ids, lo, hi)
                scores, positions = scores.topk(self.target, dim=1)
                return scores, ids.gather(1, positions)
        wide = min(self.n, self.target + math.ceil(2*self.length/self.args.retrieval_stride)+1)
        narrow = min(wide, max(self.target, math.ceil(self.args.sharp_c*self.target))) if training else self.target
        scores, labels = search(query, narrow, lower, upper)
        failed = ~torch.isfinite(scores).all(dim=1)
        self.counts[self.split+'/channel_queries'] += len(query)
        self.counts[self.split+'/wide_fallbacks'] += int(failed.sum())
        if failed.any():
            scores[failed], labels[failed] = search(query[failed], wide, lower[failed], upper[failed])
        if not torch.isfinite(scores).all():
            raise RuntimeError('HNSW did not return enough eligible candidates after wide fallback')
        return scores, labels

    def uniforms(self, series, starts, channel):
        """Stateless RAFT avalanche mixer extended with series/channel/split IDs."""
        mask = 0xFFFFFFFF
        state = ((series[:, None] & mask)*10000019) & mask
        state = state ^ (((series[:, None] >> 32)*100000007) & mask)
        state = state ^ (((starts[:, None] & mask)*100000007) & mask)
        state = state ^ (((starts[:, None] >> 32)*10000019) & mask)
        steps = torch.arange(1, self.k, device=self.device)[None, :]
        state = state ^ ((steps*1000000007) & mask)
        state = state ^ ((self.args.seed*1000003) & mask) ^ ((channel*48271) & mask)
        state = state ^ ({'train': 1, 'val': 2, 'test': 3}[self.split]*40692)
        for _ in range(2):
            state = ((state ^ (state >> 16))*0x045D9F3B) & mask
        state = (state ^ (state >> 16)) & mask
        return (state.double()+0.5)/4294967296

    def select(self, query_profile, scores, labels, channel, series, starts):
        with self.clock(self.split+'/reliability_seconds'):
            candidate_profiles = self.profiles[:, :, channel:channel+1][labels]
            distance = robust_stp_distance(query_profile[:, :, channel:channel+1], candidate_profiles,
                                           self.scale[:, channel:channel+1], self.weights)
            similarity = stp_distance_to_similarity(distance, 1.0)
            positions, valid = spaced_neighborhood(
                similarity, self.sid[labels].cpu().numpy(), self.starts[labels].cpu().numpy(),
                self.args.reliability_r, self.values.shape[1],
            )
            N_q_size = valid.sum(1)
            A_q = (similarity.gather(1, positions) * valid).sum(1) / N_q_size
            neighbors = labels.gather(1, positions)
            consistency = []
            for begin in range(0, len(labels), 32):
                selected = neighbors[begin:begin+32]
                futures = self.raw_values[:, :, channel:channel+1][selected]
                consistency.append(future_consistency(futures, self.value_scale[channel:channel+1],
                    self.tau_v, valid_mask=valid[begin:begin+len(selected)]))
            V_q = torch.cat(consistency)
            cr_q = A_q * V_q
            sigma_q, lambda_q = map_reliability(cr_q, .05, .30, .30, .90)
            self.last_reliability_components = {'A_q': A_q, 'V_q': V_q, 'cr_q': cr_q,
                                               'N_q_size': N_q_size}
        with self.clock(self.split+'/mmr_seconds'):
            futures = self.values[:, :, channel][labels]
            values = F.normalize(futures-futures.mean(-1, keepdim=True), dim=-1)
            redundancy = (values*values[:, :1]).sum(-1)
            chosen = torch.zeros((len(scores), self.k), device=self.device, dtype=torch.long)
            used = torch.zeros_like(scores, dtype=torch.bool)
            used[:, 0] = True
            uniforms = self.uniforms(series, starts, channel)
            for step in range(1, self.k):
                mmr = lambda_q[:, None]*scores-(1-lambda_q[:, None])*redundancy
                probs = F.softmax(mmr.masked_fill(used, -torch.inf)/.1, dim=1)
                # Accumulate in double to avoid finite-precision CDF gaps.
                cdf = probs.double().cumsum(1)
                cdf = cdf/cdf[:, -1:]
                cdf[:, -1] = 1
                pos = (cdf < uniforms[:, step-1:step]).sum(1)
                chosen[:, step] = pos
                used.scatter_(1, pos[:, None], True)
                selected_values = values[torch.arange(len(scores), device=self.device), pos]
                redundancy = torch.maximum(redundancy, (values*selected_values[:, None]).sum(-1))
            selected_scores = scores.gather(1, chosen)
            selected_labels = labels.gather(1, chosen)
            # beta's input positions retain PIR's descending cosine semantics.
            selected_scores, order = selected_scores.sort(dim=1, descending=True)
            selected_labels = selected_labels.gather(1, order)
        return selected_scores, selected_labels, torch.stack([cr_q, lambda_q, sigma_q], -1)

    @torch.no_grad()
    def retrieve(self, queries, query_metadata, split):
        if split not in ('train', 'val', 'test'):
            raise ValueError('explicit train/val/test split is required')
        self.sync()
        retrieve_started = time.perf_counter()
        self.split = split
        if queries.shape[1:] != (self.length, self.channels) or not torch.isfinite(queries).all():
            raise ValueError('query must be a finite [B,L,C] tensor matching the bank')
        ids = torch.as_tensor(query_metadata, device=self.device, dtype=torch.long)
        if ids.shape != (len(queries), 2):
            raise ValueError('queries require [series_id, past_start] metadata')
        series, starts = ids[:, 0], ids[:, 1]
        lower, upper = self.forbidden_bounds(series, starts, split == 'train')
        profile = None
        if self.crest:
            with self.clock(split+'/query_profile_seconds'):
                profile = compute_stp_profile(queries, self.args.profile_sections)
        normalized = F.normalize(queries.permute(2, 0, 1), dim=-1)
        if not self.sharp:
            with self.clock(split+'/search_total_seconds'):
                exact_scores, exact_labels = self.exact(normalized, lower, upper)
        outputs, all_scores, all_labels, controllers, components = [], [], [], [], []
        for channel in range(self.channels):
            if self.sharp:
                with self.clock(split+'/search_total_seconds'):
                    scores, labels = self.ann(normalized[channel], channel, lower, upper, split == 'train')
            else:
                scores, labels = exact_scores[channel], exact_labels[channel]
            if self.crest:
                scores, labels, controller = self.select(profile, scores, labels, channel, series, starts)
                controllers.append(controller)
                components.append(torch.stack([self.last_reliability_components[name]
                    for name in ('A_q', 'V_q', 'N_q_size')], -1))
            with self.clock(split+'/aggregation_seconds'):
                probs = F.softmax(scores, dim=1)
                output = (self.values[:, :, channel][labels]*probs[:, :, None]).sum(1)
            outputs.append(output)
            all_scores.append(scores)
            all_labels.append(labels)
        self.last = {'labels': torch.stack(all_labels, 1), 'scores': torch.stack(all_scores, 1),
                     'controller': torch.stack(controllers, 1) if controllers else None,
                     'reliability_components': torch.stack(components, 1) if components else None}
        if controllers:
            total = self.last['controller'].double().sum((0, 1))
            self.controller_sums[split] = self.controller_sums.get(split, 0)+total
            component_total = self.last['reliability_components'].double().sum((0, 1))
            self.component_sums[split] = self.component_sums.get(split, 0)+component_total
        self.sync()
        self.timings[split+'/retrieval_total_seconds'] += time.perf_counter()-retrieve_started
        self.counts[split+'/calls'] += 1
        self.counts[split+'/queries'] += len(queries)
        return torch.stack(outputs, -1), self.last['scores'], 0

    def report(self):
        return {'timings': dict(self.timings), 'counts': dict(self.counts),
                'controller_mean_per_query_visit': {split: (total/(self.counts[split+'/queries']*self.channels)).cpu().tolist() for split, total in self.controller_sums.items()},
                'controller_columns': ['cr_q', 'lambda_q', 'sigma_q'],
                'component_columns': ['A_q', 'V_q', 'N_q_size'],
                'component_mean_per_query_visit': {split: (total/(self.counts[split+'/queries']*self.channels)).cpu().tolist() for split, total in self.component_sums.items()},
                'reliability_definition': {'formula': 'cr_q=A_q*V_q', 'neighborhood': 'stp_ranked_spaced_up_to_n',
                    'min_spacing': self.values.shape[1], 'shortfall_policy': 'use_available',
                    'tau_v': self.tau_v, 'value_space': 'loader_values_before_pir_window_normalization',
                    'value_scale': self.value_scale.tolist() if self.crest else None},
                'method': self.args.retrieval_method, 'aggregation': 'native_cosine_softmax_temperature_1',
                'sigma_used_for_aggregation': False, 'candidate_count': self.n,
                'channels': self.channels, 'K': self.k, 'search_target': self.target,
                'execution': 'prepared' if self.prepared else 'inline',
                'exact_cosine_matmul_tf32': False,
                'retrieval_batch_size': getattr(self.args, 'retrieval_batch_size', None),
                'hnsw_build_threads_requested': self.args.hnsw_build_threads,
                'hnsw_search_threads_requested': self.args.hnsw_threads,
                'prepared_query_counts': {s: len(v['output']) for s, v in self.prepared.items()},
                'retrieval_prepare_total_seconds': self.timings.get('prepare_index_total_seconds', 0)+self.timings.get('prepare_queries_total_seconds', 0),
                'timing_note': ('Synchronized wall seconds; nested scopes overlap. Prepared mode: train/val/test retrieval measured once per query; cache lookup measured separately. Inline mode: retrieval accumulates over epoch visits. Small HNSW calls retain the library single-thread policy.')}

    @torch.no_grad()
    def prepare_queries(self, datasets):
        """Prepare only input-dependent retrieval. Never cache learned beta/output.

        Use an isolated loader RNG so preparation cannot change training shuffle
        or dropout RNG. Preserve native per-window normalization exactly.
        """
        from torch.utils.data import DataLoader
        from src.pir.data_provider.pir_panel import query_metadata
        with self.clock('prepare_queries_total_seconds'):
            for dataset in datasets:
                split = dataset.flag
                if split in self.prepared:
                    raise ValueError('split already prepared: '+split)
                cache = {
                    'output': self.values.new_empty((len(dataset), self.values.shape[1], self.channels)),
                    'scores': self.keys.new_empty((len(dataset), self.channels, self.k)),
                    'labels': torch.empty((len(dataset), self.channels, self.k), device=self.device, dtype=torch.long),
                    'controller': self.keys.new_empty((len(dataset), self.channels, 3)) if self.crest else None,
                    'reliability_components': self.keys.new_empty((len(dataset), self.channels, 3)) if self.crest else None,
                }
                generator = torch.Generator().manual_seed(self.args.seed)
                loader = DataLoader(dataset, batch_size=self.args.retrieval_batch_size,
                                    shuffle=False, drop_last=False, num_workers=0, generator=generator)
                with self.clock(split+'/prepare_queries_seconds'):
                    for batch, (past, _target, _past_marks, _future_marks, index) in enumerate(loader):
                        query = past.float().to(self.device)
                        mean = query.mean(1, keepdim=True)
                        query = query-mean
                        query /= torch.sqrt(query.var(1, keepdim=True, unbiased=False)+1e-5)
                        # Targets and future time features never enter retrieval.
                        output, scores, _ = self.retrieve(query, query_metadata(dataset, index), split)
                        positions = index.to(self.device)
                        cache['output'][positions] = output
                        cache['scores'][positions] = scores
                        cache['labels'][positions] = self.last['labels']
                        if self.crest:
                            cache['controller'][positions] = self.last['controller']
                            cache['reliability_components'][positions] = self.last['reliability_components']
                        if batch % 20 == 0 or batch+1 == len(loader):
                            print(f'[RetrievalPrepare] {split} batch {batch+1}/{len(loader)} queries={len(index)}', flush=True)
                self.prepared[split] = cache

    @torch.no_grad()
    def lookup(self, index, split):
        if split not in self.prepared:
            raise ValueError('retrieval split not prepared: '+split)
        with self.clock(split+'/cache_lookup_seconds'):
            index = torch.as_tensor(index, device=self.device, dtype=torch.long)
            cache = self.prepared[split]
            output, scores = cache['output'][index], cache['scores'][index]
            self.last = {'labels': cache['labels'][index], 'scores': scores,
                         'controller': cache['controller'][index] if self.crest else None,
                         'reliability_components': cache['reliability_components'][index] if self.crest else None}
        self.counts[split+'/cache_lookup_calls'] += 1
        self.counts[split+'/cache_lookup_queries'] += len(index)
        return output, scores, 0
