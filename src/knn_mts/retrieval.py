import math
from collections import Counter
import time
import numpy as np
import torch
from src.utils.candidate_reliability import _weighted_quantile, VALUE_EPS
from .controller import L2CREST, gaussian_l2_weights, gaussian_pearson_weights

METHODS = ('KNN', 'KNN_SHARP', 'KNN_CREST', 'KNN_SHARP_CREST')


def pearson_keys(windows):
    """Window-centered unit vectors; constant/near-zero norm => rho=0 convention."""
    centered = windows-windows.mean(-1, keepdim=True)
    norms = centered.norm(dim=-1, keepdim=True)
    constant = (windows.max(-1, keepdim=True).values == windows.min(-1, keepdim=True).values) | (norms <= 1e-12)
    return (centered/norms.clamp_min(1e-12)).masked_fill(constant, 0.)


def aggregate_candidate_futures(futures, weights, candidate_last, query_last, value_transfer='none'):
    """Transfer each train future using observed past only, after candidate selection.

    futures [Q,K,H], weights/candidate_last [Q,K], query_last [Q].
    Search, STP, value-MMR, weights and final mixture gate are not changed.
    """
    if value_transfer == 'query_last':
        futures = futures-candidate_last[:, :, None]+query_last[:, None, None]
    elif value_transfer != 'none':
        raise ValueError('invalid value transfer')
    return (weights[:, :, None]*futures).sum(1)


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


class RawWindowBank:
    """Train-only raw windows, Pearson or squared-L2 exact/HNSW retrieval."""
    def __init__(self, train_values, length, horizon, device, k=50, temperature=1., alpha=.2,
                 method='KNN', seed=0, crest_pool=100, profile_sections=6,
                 stp_weights=(.5,.25,.25), reliability_k=None, sigma_min=.05, sigma_max=.30,
                 mmr_chunk=1024, hnsw_m=32, hnsw_ef_construction=400, hnsw_ef_search=400,
                 hnsw_build_threads=32, hnsw_search_threads=32, metric='pearson', value_transfer='none',
                 candidate_starts=None, search_backend='auto', cpu_search_threads=32, faiss_nprobe=32,
                 candidate_series_ids=None, candidate_local_starts=None, tau_v=1.0):
        if search_backend not in ('auto','cpu_exact','faiss_cpu'):
            raise ValueError('invalid search backend')
        if search_backend!='auto' and (metric!='pearson' or 'SHARP' in method):
            raise ValueError('optional CPU backends support Pearson KNN/KNN_CREST only')
        if min(cpu_search_threads,faiss_nprobe)<1:
            raise ValueError('invalid CPU search parameters')
        self.search_backend=search_backend
        self.cpu_search_threads=cpu_search_threads
        if metric not in ('pearson', 'squared_l2'):
            raise ValueError('invalid search metric')
        if value_transfer not in ('none','query_last'):
            raise ValueError('invalid value transfer')
        self.value_transfer = value_transfer
        self.metric = metric
        self.device = torch.device(device)
        self.length, self.horizon = length, horizon
        self.k, self.temperature, self.alpha = k, temperature, alpha
        self.method, self.crest, self.sharp = method, 'CREST' in method, 'SHARP' in method
        if method not in METHODS or min(length, horizon, k) < 1 or not all(math.isfinite(x) for x in (temperature, alpha)) or temperature <= 0 or alpha < 0:
            raise ValueError('invalid retrieval parameters')
        if train_values.ndim != 2 or not torch.isfinite(train_values).all():
            raise ValueError('expected finite [train_time, variables] data')
        self.channels = train_values.shape[1]
        count = len(train_values)-length-horizon+1
        self.explicit_starts = None
        if candidate_starts is not None:
            starts = torch.as_tensor(candidate_starts, dtype=torch.long, device=self.device)
            if (self.channels != 1 or starts.ndim != 1 or not len(starts)
                    or starts.min() < 0 or starts.max()+length+horizon > len(train_values)
                    or len(torch.unique(starts)) != len(starts)):
                raise ValueError('invalid explicit univariate train candidate starts')
            self.explicit_starts = starts
            count = len(starts)
        self.candidate_series_ids = None
        self.candidate_local_starts = None
        if candidate_series_ids is not None or candidate_local_starts is not None:
            if self.explicit_starts is None or candidate_series_ids is None or candidate_local_starts is None:
                raise ValueError('panel candidate coordinates require starts, series IDs and local starts')
            self.candidate_series_ids = torch.as_tensor(candidate_series_ids, dtype=torch.long, device=self.device)
            self.candidate_local_starts = torch.as_tensor(candidate_local_starts, dtype=torch.long, device=self.device)
            if (self.candidate_series_ids.shape != (count,) or self.candidate_local_starts.shape != (count,)
                    or (self.candidate_local_starts < 0).any()):
                raise ValueError('candidate coordinates must align with the explicit window bank')
        if self.crest and self.explicit_starts is not None and self.candidate_series_ids is None:
            raise ValueError('panel CREST requires true series IDs and local starts for temporal spacing')
        self.pool = crest_pool if self.crest else k
        reliability_k = k if reliability_k is None else reliability_k
        if count < 1 or not 1 <= k <= self.pool <= count*self.channels:
            raise ValueError('K/pool exceeds the strictly train-only candidate bank')
        if self.crest and (profile_sections < 2 or length//profile_sections < 2 or not 1 <= reliability_k <= self.pool or mmr_chunk < 1
                         or not all(math.isfinite(x) for x in (*stp_weights, sigma_min, sigma_max))
                         or len(stp_weights) != 3 or min(stp_weights) < 0 or abs(sum(stp_weights)-1)>1e-6
                         or not 0 < sigma_min <= sigma_max):
            raise ValueError('invalid CREST parameters')
        if self.sharp and min(hnsw_m, hnsw_ef_construction, hnsw_ef_search, hnsw_build_threads, hnsw_search_threads) < 1:
            raise ValueError('invalid HNSW parameters')
        synchronize(self.device); started = time.perf_counter()
        self.train_values = train_values.to(self.device).contiguous()
        windows = self.train_values.unfold(0, length, 1)
        self.keys = (windows[self.explicit_starts] if self.explicit_starts is not None
                     else windows[:count]).contiguous().view(-1, length)
        self.search_keys = pearson_keys(self.keys) if metric == 'pearson' else self.keys
        self.key_norm = self.keys.square().sum(1)
        self.future_offsets = length + torch.arange(horizon, device=self.device)
        synchronize(self.device)
        self.prepare_timing = dict(key_seconds=time.perf_counter()-started, index_seconds=0., profile_seconds=0.)
        self.index = None
        self.search_backend_info = None
        if search_backend!='auto':
            from .cpu_search import ExactCPUIndex, FaissCPUIndex
            tick=time.perf_counter();cpu_keys=self.search_keys.cpu().contiguous().numpy()
            synchronize(self.device)
            self.prepare_timing['cpu_key_transfer_seconds']=time.perf_counter()-tick
            tick=time.perf_counter()
            self.index=(ExactCPUIndex(cpu_keys,cpu_search_threads) if search_backend=='cpu_exact'
                        else FaissCPUIndex(cpu_keys,cpu_search_threads,faiss_nprobe))
            self.prepare_timing['index_seconds']=time.perf_counter()-tick
            self.search_backend_info=self.index.info
        if self.sharp:
            import hnswlib
            t = time.perf_counter()
            self.index = hnswlib.Index(space='cosine' if metric == 'pearson' else 'l2', dim=length)
            self.index.init_index(max_elements=len(self.keys), M=hnsw_m,
                                  ef_construction=hnsw_ef_construction, random_seed=seed)
            self.index.add_items(self.search_keys.cpu().numpy(), np.arange(len(self.keys)), num_threads=hnsw_build_threads)
            self.index.set_ef(max(self.pool, hnsw_ef_search))
            self.hnsw_search_threads = hnsw_search_threads
            self.prepare_timing['index_seconds'] = time.perf_counter()-t
        self.controller = None
        self.reliability_configuration = None
        if self.crest:
            t = time.perf_counter()
            self.controller = L2CREST(self.keys, k, self.pool, seed, temperature, profile_sections,
                                    stp_weights, reliability_k, sigma_min, sigma_max, mmr_chunk, metric=metric,
                                    future_lookup=self.futures, candidate_coordinates=self.candidate_coordinates,
                                    value_scale=self.training_value_scale(), horizon=horizon, tau_v=tau_v)
            self.reliability_configuration = {
                'formula': 'cr_q=A_q*V_q', 'neighborhood': 'stp_ranked_spaced_up_to_n',
                'min_spacing': horizon, 'shortfall_policy': 'use_available', 'tau_v': self.controller.tau_v,
                'value_space': 'train_standardized_scalar_future_before_value_transfer',
                'value_scale': self.controller.value_scale.tolist(),
            }
            synchronize(self.device); self.prepare_timing['profile_seconds'] = time.perf_counter()-t
        self.prepare_seconds = time.perf_counter()-started
        self.timings = dict(search_seconds=0., reliability_seconds=0., mmr_seconds=0., aggregation_seconds=0.)
        self.query_count = 0
        self.query_batches = {}
        self.last = None

    def futures(self, labels):
        starts = self.candidate_past_starts(labels)
        variables = labels.remainder(self.channels)
        return self.train_values[starts[:, :, None]+self.future_offsets, variables[:, :, None]]

    def candidate_past_starts(self, labels):
        return (self.explicit_starts[labels] if self.explicit_starts is not None else
                torch.div(labels, self.channels, rounding_mode='floor'))

    def candidate_coordinates(self, labels):
        """Resolve scalar-variable IDs or panel source IDs and local window starts."""
        if self.candidate_series_ids is not None:
            return self.candidate_series_ids[labels], self.candidate_local_starts[labels]
        return labels.remainder(self.channels), self.candidate_past_starts(labels)

    def training_value_scale(self):
        """Fit scalar future IQR with exact overlap multiplicities, without an NH bank."""
        values = self.train_values.detach().cpu().numpy()
        starts = (self.explicit_starts.cpu().numpy() if self.explicit_starts is not None
                  else np.arange(len(self.keys)//self.channels)) + self.length
        difference = np.zeros(len(values)+1, dtype=np.int64)
        difference[starts] += 1
        difference[starts+self.horizon] -= 1
        weights = np.cumsum(difference[:-1])
        keep = weights > 0
        quartiles = _weighted_quantile(values[keep].reshape(-1),
                                      np.repeat(weights[keep], self.channels), [.25, .75])
        return torch.tensor([quartiles[1]-quartiles[0]+VALUE_EPS], dtype=torch.float32)

    @torch.no_grad()
    def predict(self, queries, query_ids=None, split='test'):
        if queries.ndim != 2 or queries.shape[1] != self.length:
            raise ValueError('queries must have shape [Q,L]')
        synchronize(self.device); started = time.perf_counter()
        query = queries.to(self.device).contiguous()
        if not torch.isfinite(query).all():
            raise ValueError('nonfinite query')
        if self.crest:
            if query_ids is None or len(query_ids) != len(query):
                raise ValueError('CREST requires stable scalar-series query IDs')
            query_ids = torch.as_tensor(query_ids, dtype=torch.long, device=self.device)
        search_query = pearson_keys(query) if self.metric == 'pearson' else query
        if self.sharp or self.search_backend!='auto':
            threads=self.hnsw_search_threads if self.sharp else self.cpu_search_threads
            labels_np, _ = self.index.knn_query(search_query.cpu().numpy(), k=self.pool, num_threads=threads)
            labels = torch.as_tensor(labels_np.astype(np.int64), device=self.device)
        else:
            prior_tf32 = torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                scores = search_query @ self.search_keys.T
                if self.metric == 'pearson':
                    scores.clamp_(-1, 1).neg_().add_(1)
                else:
                    scores.mul_(-2).add_(query.square().sum(1, keepdim=True)).add_(self.key_norm[None, :]).clamp_min_(0)
                _, labels = scores.topk(self.pool, largest=False, sorted=True)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = prior_tf32
        # Identical final distance computation across exact and HNSW.
        if self.metric == 'pearson':
            distances = 1-(search_query[:, None, :]*self.search_keys[labels]).sum(-1).clamp(-1, 1)
        else:
            distances = (query[:, None, :]-self.keys[labels]).square().sum(-1)
        distances, order = distances.sort(dim=1); labels = labels.gather(1, order)
        synchronize(self.device); self.timings['search_seconds'] += time.perf_counter()-started
        controllers = None
        if self.crest:
            started = time.perf_counter()
            cr_q, lambda_q, sigma_q = self.controller.reliability(query, labels)
            synchronize(self.device); self.timings['reliability_seconds'] += time.perf_counter()-started
            started = time.perf_counter()
            positions = self.controller.select(distances, lambda_q, labels, self.futures, query_ids, split)
            distances, labels = distances.gather(1, positions), labels.gather(1, positions)
            distances, order = distances.sort(dim=1); labels = labels.gather(1, order)
            controllers = torch.stack([cr_q, lambda_q, sigma_q], -1)
            synchronize(self.device); self.timings['mmr_seconds'] += time.perf_counter()-started
        started = time.perf_counter()
        gaussian = gaussian_pearson_weights if self.metric == 'pearson' else gaussian_l2_weights
        weights = gaussian(distances, sigma_q) if self.crest else torch.softmax(-distances/self.temperature, dim=1)
        starts = self.candidate_past_starts(labels)
        candidate_last = self.train_values[starts+self.length-1, labels.remainder(self.channels)]
        prediction = aggregate_candidate_futures(self.futures(labels), weights, candidate_last,
                                                 query[:, -1], self.value_transfer)
        # Same gate formula/alpha/selected-distance statistic across all methods.
        mixing = self.alpha/(distances.mean(1)+self.alpha) if self.alpha else torch.zeros(len(query), device=self.device)
        self.last = dict(weights=weights, controller=controllers, effective_candidates=1/weights.square().sum(1),
                         reliability_components=self.controller.last_reliability_components if self.crest else None)
        synchronize(self.device); self.timings['aggregation_seconds'] += time.perf_counter()-started
        self.query_count += len(query)
        self.query_batches.setdefault(split, Counter())[len(query)] += 1
        return prediction, mixing, labels, distances
