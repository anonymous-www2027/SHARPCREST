"""Optional HNSW Recall diagnostics for General and panel datasets."""

import hashlib
import json
from pathlib import Path
import random
import tempfile
import time
from zipfile import BadZipFile

import numpy as np
import torch


class HNSWRecallMixin:
    """Evaluate Recall using the host retrieval tool's prepared search state."""

    @staticmethod
    def _array_digest(*arrays):
        digest = hashlib.sha256()
        for array in arrays:
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode('ascii'))
            digest.update(str(contiguous.shape).encode('ascii'))
            digest.update(contiguous.view(np.uint8))
        return digest.hexdigest()

    def _manifest_digest(self, manifest):
        return self._array_digest(
            manifest.series_ids,
            manifest.first_start,
            manifest.window_counts,
            manifest.prefix_offsets,
            manifest.stride,
            np.asarray([manifest.seq_len, manifest.pred_len], dtype=np.int64),
        )

    def _sample_recall_query_ids(self, dataset, split):
        offsets = {'train': 0, 'valid': 1_000_003, 'test': 2_000_003}
        count = min(self.hnsw_recall_queries, len(dataset))
        if count < 1:
            raise ValueError(f'cannot sample recall queries from empty {split} split')
        generator = np.random.default_rng(
            self.hnsw_recall_seed + offsets[split]
        )
        return np.sort(generator.choice(
            len(dataset), size=count, replace=False
        )).astype(np.int64, copy=False)

    def _panel_windows_for_ids(self, dataset, query_ids):
        series_ids, starts = dataset.query_manifest.resolve_ids(query_ids)
        windows = dataset.series_bank.get_windows(
            series_ids, starts, self.seq_len
        )
        return torch.from_numpy(windows)

    def _chunked_panel_exact_topk(
        self, query_windows, query_indices, period_idx, train,
        candidate_chunk_size=None, device=None, target_count=None,
    ):
        """Streaming exact reference with no full candidate key bank."""
        if not self.panel_mode:
            raise NotImplementedError('chunked exact recall is panel-only')
        candidate_chunk_size = (
            self.hnsw_recall_candidate_chunk_size
            if candidate_chunk_size is None else int(candidate_chunk_size)
        )
        target_count = self.topm if target_count is None else int(target_count)
        device = self.hnsw_recall_device if device is None else torch.device(device)
        if candidate_chunk_size < 1:
            raise ValueError('candidate_chunk_size must be >= 1')
        if not 1 <= target_count <= self.n_train:
            raise ValueError('exact target count must be within candidate bank')

        query_indices = np.asarray(query_indices, dtype=np.int64).reshape(-1)
        query_windows = query_windows.detach().cpu().float()
        if len(query_windows) != len(query_indices):
            raise ValueError('query windows and indices must have equal length')
        period = self.period_num[period_idx]
        query_keys = self._key_for_period(query_windows, period).to(device)
        running_scores = torch.empty(
            (len(query_indices), 0), dtype=torch.float32, device=device
        )
        running_labels = torch.empty(
            (len(query_indices), 0), dtype=torch.long, device=device
        )

        query_series = query_starts = None
        if train:
            if self._active_panel_query_manifest is None:
                raise RuntimeError('train exact recall requires a query manifest')
            query_series, query_starts = (
                self._active_panel_query_manifest.resolve_ids(query_indices)
            )
        width = self.seq_len + self.pred_len
        for start in range(0, self.n_train, candidate_chunk_size):
            end = min(start + candidate_chunk_size, self.n_train)
            candidate_ids = np.arange(start, end, dtype=np.int64)
            candidate_series, candidate_starts = (
                self.panel_candidate_manifest.resolve_ids(candidate_ids)
            )
            candidate_windows = self.panel_series_bank.get_windows(
                candidate_series, candidate_starts, self.seq_len
            )
            candidate_keys = self._key_for_period(
                torch.from_numpy(candidate_windows), period
            ).to(device)
            if device.type == 'cuda':
                previous_tf32 = torch.backends.cuda.matmul.allow_tf32
                torch.backends.cuda.matmul.allow_tf32 = False
                try:
                    scores = torch.matmul(query_keys, candidate_keys.T)
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            else:
                scores = torch.matmul(query_keys, candidate_keys.T)
            if train:
                forbidden = (
                    (candidate_series[None, :] == query_series[:, None])
                    & (candidate_starts[None, :] < query_starts[:, None] + width)
                    & (query_starts[:, None] < candidate_starts[None, :] + width)
                )
                scores.masked_fill_(
                    torch.from_numpy(forbidden).to(device), float('-inf')
                )
            labels = torch.arange(start, end, device=device).expand(
                len(query_indices), -1
            )
            merged_scores = torch.cat((running_scores, scores), dim=1)
            merged_labels = torch.cat((running_labels, labels), dim=1)
            keep = min(target_count, merged_scores.shape[1])
            running_scores, positions = torch.topk(
                merged_scores, k=keep, dim=1, largest=True, sorted=True
            )
            running_labels = merged_labels.gather(1, positions)
            del candidate_windows, candidate_keys, scores

        if running_scores.shape[1] != target_count:
            raise RuntimeError('exact scan did not produce the requested top-k')
        available = torch.isfinite(running_scores).sum(dim=1)
        if torch.any(available < target_count):
            row = int(torch.nonzero(
                available < target_count, as_tuple=False
            )[0])
            raise RuntimeError(
                'not enough non-overlapping exact candidates for query '
                f'{int(query_indices[row])}: found {int(available[row])}, '
                f'need {target_count}'
            )
        return running_labels.cpu(), running_scores.cpu()

    def _recall_cache_metadata(
        self, dataset, split, period_idx, query_ids, recall_k=None
    ):
        recall_k = self.search_target if recall_k is None else recall_k
        report = dataset.memory_report
        if not hasattr(self, '_recall_values_digest'):
            self._recall_values_digest = self._array_digest(
                self.panel_series_bank.values
            )
        return {
            'selected_values_hash': self._recall_values_digest,
            'exact_key_version': 'sharp-pearson-period-v1',
            'dataset_identifier': report.get('dataset_identifier'),
            'panel_dataset': report.get('panel_dataset'),
            'selected_series_hash': self._array_digest(
                dataset.series_bank.series_ids
            ),
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'n_period': self.n_period,
            'period': self.period_num[period_idx],
            'candidate_stride': report.get('candidate_stride'),
            'query_stride': report.get('query_stride'),
            'split': split,
            'k': recall_k,
            'recall_seed': self.hnsw_recall_seed,
            'requested_queries': self.hnsw_recall_queries,
            'sampled_query_hash': self._array_digest(query_ids),
            'candidate_count': self.n_train,
            'candidate_manifest_hash': self._manifest_digest(
                self.panel_candidate_manifest
            ),
            'query_manifest_hash': self._manifest_digest(
                dataset.query_manifest
            ),
        }

    def _recall_cache_path(self, metadata):
        if not self.hnsw_recall_cache_dir:
            return None
        serialized = json.dumps(metadata, sort_keys=True, separators=(',', ':'))
        key = hashlib.sha256(serialized.encode('utf-8')).hexdigest()
        return Path(self.hnsw_recall_cache_dir) / (
            f"{metadata['split']}-p{metadata['period']}-{key}.npz"
        )

    def _load_recall_cache(self, path, metadata, query_ids, recall_k=None):
        recall_k = self.search_target if recall_k is None else recall_k
        if path is None or not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as payload:
                stored_metadata = json.loads(str(payload['metadata']))
                if str(payload['payload_digest']) != self._array_digest(
                    payload['query_ids'], payload['labels'], payload['similarities']
                ):
                    return None
                if (payload['query_ids'].dtype != np.int64
                        or payload['labels'].dtype != np.int64
                        or payload['similarities'].dtype != np.float32):
                    return None
                stored_ids = payload['query_ids'].astype(np.int64, copy=False)
                labels = payload['labels'].astype(np.int64, copy=False)
                similarities = payload['similarities'].astype(
                    np.float32, copy=False
                )
            expected_shape = (len(query_ids), recall_k)
            if (stored_metadata != metadata
                    or not np.array_equal(stored_ids, query_ids)
                    or labels.shape != expected_shape
                    or similarities.shape != expected_shape
                    or not np.isfinite(similarities).all()
                    or np.any(labels < 0) or np.any(labels >= self.n_train)
                    or np.any(np.diff(similarities, axis=1) > 0)
                    or np.any(np.diff(np.sort(labels, axis=1), axis=1) == 0)):
                return None
            return torch.from_numpy(labels.copy()), torch.from_numpy(
                similarities.copy()
            )
        except (
            OSError, ValueError, KeyError, json.JSONDecodeError, EOFError,
            BadZipFile,
        ):
            return None

    @staticmethod
    def _save_recall_cache(
        path, metadata, query_ids, labels, similarities
    ):
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=f'.{path.name}.',
                suffix='.tmp.npz',
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                np.savez_compressed(
                    temporary,
                    payload_digest=np.asarray(HNSWRecallMixin._array_digest(
                        np.asarray(query_ids, dtype=np.int64),
                        labels.numpy().astype(np.int64, copy=False),
                        similarities.numpy().astype(np.float32, copy=False),
                    )),
                    metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                    query_ids=np.asarray(query_ids, dtype=np.int64),
                    labels=labels.numpy().astype(np.int64, copy=False),
                    similarities=similarities.numpy().astype(
                        np.float32, copy=False
                    ),
                )
            temporary_path.replace(path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def _print_recall_report(self, split, result):
        print(f'[HNSWRecall][{split}]')
        for name in (
            'sampled_queries', 'k', 'seed', 'candidate_count',
            'candidate_chunk_size', 'query_batch_size', 'device', 'period',
            'configured_ef_search', 'effective_train_narrow_ef',
            'effective_valid_ef', 'effective_wide_ef', 'mean_recall',
            'median_recall', 'min_recall', 'exact_reference_ms',
            'hnsw_diagnostic_ms', 'total_diagnostic_ms', 'exact_cache',
        ):
            value = result[name]
            if name in {'mean_recall', 'median_recall', 'min_recall'}:
                value = f'{value:.6f}'
            elif name.endswith('_ms'):
                value = f'{value:.3f}'
            print(f'{name}={value}')

    def run_hnsw_recall(self, datasets):
        """Measure panel Recall@search_target after production retrieval."""
        if not self.hnsw_recall_check:
            return {}
        if not self.panel_mode or self.retrieval_backend != 'hnsw':
            raise NotImplementedError(
                'HNSW recall diagnostic supports panel HNSW mode only'
            )
        missing = set(self.hnsw_recall_splits) - set(datasets)
        if missing:
            raise ValueError(f'missing recall datasets: {sorted(missing)}')

        recall_k = self.search_target
        saved_state = (
            self._wide_fallbacks,
            self._wide_searches,
            self._train_queries,
            self._active_runtime,
            self._active_panel_query_manifest,
        )
        all_results = {}
        try:
            self._active_runtime = None
            for split in self.hnsw_recall_splits:
                dataset = datasets[split]
                self._active_panel_query_manifest = dataset.query_manifest
                query_ids = self._sample_recall_query_ids(dataset, split)
                split_results = []
                for period_idx, period in enumerate(self.period_num):
                    total_start = time.perf_counter()
                    hnsw_start = time.perf_counter()
                    hnsw_chunks = []
                    for start in range(
                        0, len(query_ids), self.hnsw_recall_query_batch_size
                    ):
                        end = min(
                            start + self.hnsw_recall_query_batch_size,
                            len(query_ids),
                        )
                        batch_ids = query_ids[start:end]
                        windows = self._panel_windows_for_ids(dataset, batch_ids)
                        labels, _similarities = self.search_candidates(
                            windows, period_idx, batch_ids,
                            train=split == 'train', target_count=recall_k,
                        )
                        hnsw_chunks.append(labels)
                    hnsw_labels = torch.cat(hnsw_chunks, dim=0)
                    hnsw_ms = (time.perf_counter() - hnsw_start) * 1000.0

                    metadata = self._recall_cache_metadata(
                        dataset, split, period_idx, query_ids, recall_k=recall_k
                    )
                    cache_path = self._recall_cache_path(metadata)
                    cached = self._load_recall_cache(
                        cache_path, metadata, query_ids, recall_k=recall_k
                    )
                    exact_start = time.perf_counter()
                    cache_status = 'hit' if cached is not None else 'miss'
                    if cached is not None:
                        exact_labels, exact_similarities = cached
                    else:
                        exact_label_chunks = []
                        exact_similarity_chunks = []
                        for start in range(
                            0, len(query_ids),
                            self.hnsw_recall_query_batch_size,
                        ):
                            end = min(
                                start + self.hnsw_recall_query_batch_size,
                                len(query_ids),
                            )
                            batch_ids = query_ids[start:end]
                            windows = self._panel_windows_for_ids(
                                dataset, batch_ids
                            )
                            labels, similarities = (
                                self._chunked_panel_exact_topk(
                                    windows,
                                    batch_ids,
                                    period_idx,
                                    train=split == 'train',
                                    target_count=recall_k,
                                )
                            )
                            exact_label_chunks.append(labels)
                            exact_similarity_chunks.append(similarities)
                        exact_labels = torch.cat(exact_label_chunks, dim=0)
                        exact_similarities = torch.cat(
                            exact_similarity_chunks, dim=0
                        )
                        self._save_recall_cache(
                            cache_path,
                            metadata,
                            query_ids,
                            exact_labels,
                            exact_similarities,
                        )
                    exact_ms = (time.perf_counter() - exact_start) * 1000.0
                    recalls = np.asarray([
                        np.intersect1d(
                            hnsw_labels[row].numpy(),
                            exact_labels[row].numpy(),
                            assume_unique=True,
                        ).size / recall_k
                        for row in range(len(query_ids))
                    ], dtype=np.float64)
                    wide_k = self._wide_k(recall_k)
                    result = {
                        'query_ids': query_ids.copy(),
                        'sampled_queries': len(query_ids),
                        'k': recall_k,
                        'seed': self.hnsw_recall_seed,
                        'candidate_count': self.n_train,
                        'candidate_chunk_size': (
                            self.hnsw_recall_candidate_chunk_size
                        ),
                        'query_batch_size': self.hnsw_recall_query_batch_size,
                        'device': str(self.hnsw_recall_device),
                        'period': period,
                        'configured_ef_search': self.hnsw_ef_search,
                        'effective_train_narrow_ef': max(
                            self.hnsw_ef_search,
                            min(
                                self.n_train,
                                self.hnsw_oversample_factor * recall_k,
                            ),
                        ),
                        'effective_valid_ef': max(
                            self.hnsw_ef_search, recall_k
                        ),
                        'effective_wide_ef': max(
                            self.hnsw_ef_search, wide_k
                        ),
                        'mean_recall': float(recalls.mean()),
                        'median_recall': float(np.median(recalls)),
                        'min_recall': float(recalls.min()),
                        'exact_reference_ms': exact_ms,
                        'hnsw_diagnostic_ms': hnsw_ms,
                        'total_diagnostic_ms': (
                            (time.perf_counter() - total_start) * 1000.0
                        ),
                        'exact_cache': cache_status,
                        'exact_labels': exact_labels,
                        'exact_similarities': exact_similarities,
                        'hnsw_labels': hnsw_labels,
                    }
                    self._print_recall_report(split, result)
                    split_results.append(result)
                if len(split_results) > 1:
                    macro = float(np.mean([
                        item['mean_recall'] for item in split_results
                    ]))
                    print(f'[HNSWRecall][{split}] macro_mean={macro:.6f}')
                all_results[split] = split_results
        finally:
            (
                self._wide_fallbacks,
                self._wide_searches,
                self._train_queries,
                self._active_runtime,
                self._active_panel_query_manifest,
            ) = saved_state
        return all_results

    def _general_recall_batch(self, dataset, start, end):
        """Read ordered forecasting inputs without creating a DataLoader RNG."""
        windows = np.empty(
            (end - start, self.seq_len, self.channels), dtype=np.float32
        )
        indices = np.empty(end - start, dtype=np.int64)
        for row, position in enumerate(range(start, end)):
            sample = dataset[position]
            x, _ = self._sample_xy(sample)
            x = np.asarray(x)
            if x.shape != (self.seq_len, self.channels):
                raise ValueError(
                    f'recall sample {position} has x shape {x.shape}; expected '
                    f'({self.seq_len}, {self.channels})'
                )
            windows[row] = x
            indices[row] = int(sample[0])
        return torch.from_numpy(windows), indices

    @torch.no_grad()
    def _chunked_general_exact_topk(
        self, train_data, query_windows, query_indices, period_idx, train,
    ):
        """Exhaustively scan training positions, retaining only running Top-K.

        Keys use production CPU preprocessing on both devices. Only the exact
        matmul/Top-K moves to the recall device, independently of CREST devices.
        As in the historical torch.topk reference, tied labels have no promised
        order; chunk sizes can select different labels at a tied K boundary.
        """
        if self.panel_mode or getattr(train_data, 'is_panel_dataset', False):
            raise ValueError('general exact recall requires general datasets')
        k = self.search_target
        if len(train_data) != self.n_train or not 1 <= k <= self.n_train:
            raise ValueError('general exact recall requires the prepared training bank')
        query_indices = np.asarray(query_indices, dtype=np.int64).reshape(-1)
        if len(query_windows) != len(query_indices):
            raise ValueError('query windows and indices must have equal length')
        device = self.hnsw_recall_device
        period = self.period_num[period_idx]
        query_keys = self._key_for_period(
            query_windows.detach().cpu().float(), period
        ).to(device)
        running_scores = torch.empty((len(query_indices), 0), device=device)
        running_labels = torch.empty(
            (len(query_indices), 0), dtype=torch.long, device=device
        )
        bounds = [self._forbidden_bounds(index) for index in query_indices] if train else []
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            if device.type == 'cuda':
                torch.backends.cuda.matmul.allow_tf32 = False
            chunk_size = self.hnsw_recall_candidate_chunk_size
            for start in range(0, self.n_train, chunk_size):
                end = min(start + chunk_size, self.n_train)
                windows, _ = self._general_recall_batch(train_data, start, end)
                candidate_keys = self._key_for_period(windows, period).to(device)
                scores = query_keys @ candidate_keys.T
                for row, (left, right) in enumerate(bounds):
                    low, high = max(start, left), min(end, right + 1)
                    if low < high:
                        scores[row, low - start:high - start] = float('-inf')
                labels = torch.arange(start, end, device=device).expand(len(query_indices), -1)
                merged_scores = torch.cat((running_scores, scores), dim=1)
                merged_labels = torch.cat((running_labels, labels), dim=1)
                running_scores, positions = torch.topk(
                    merged_scores, k=min(k, merged_scores.shape[1]),
                    dim=1, largest=True, sorted=True,
                )
                running_labels = merged_labels.gather(1, positions)
                del windows, candidate_keys, scores, merged_scores, merged_labels
            if not torch.isfinite(running_scores).all():
                raise RuntimeError(
                    'not enough finite non-overlapping exact candidates '
                    f'for general Recall@{k}'
                )
            return running_labels.cpu(), running_scores.cpu()
        finally:
            if device.type == 'cuda':
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    @torch.no_grad()
    def evaluate_general_hnsw_recall(self, train_data, datasets):
        """Full selected splits x full training bank, before the CREST controller.

        No query sampling or exact cache is used. Aggregation matches 484f2da:
        sum of intersection sizes / (queries * search_target * periods).
        Preserve RNG, HNSW ef, and runtime counters even if diagnosis fails.
        """
        if not self.hnsw_recall_check:
            return {}
        if self.retrieval_backend != 'hnsw':
            raise ValueError('HNSW recall requires retrieval_backend=hnsw')
        if self.panel_mode or getattr(train_data, 'is_panel_dataset', False):
            raise ValueError('general HNSW recall requires general datasets')
        missing = set(self.hnsw_recall_splits) - set(datasets)
        if missing:
            raise ValueError(f'missing recall datasets: {sorted(missing)}')
        if len(train_data) != self.n_train or len(self.hnsw_indices) != self.n_period:
            raise ValueError('general HNSW recall requires the prepared training bank')
        for split in self.hnsw_recall_splits:
            if getattr(datasets[split], 'is_panel_dataset', False):
                raise ValueError('general HNSW recall requires general query datasets')
            if len(datasets[split]) < 1:
                raise ValueError(f'cannot evaluate recall on empty {split} split')

        saved_state = (self._wide_fallbacks, self._wide_searches,
                       self._train_queries, self._active_runtime)
        saved_ef = [index.ef for index in self.hnsw_indices]
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        # Do not initialize CUDA for a CPU-only run; preserve every initialized
        # CUDA generator when forecasting/CREST already uses it.
        cuda_devices = (list(range(torch.cuda.device_count()))
                        if torch.cuda.is_initialized() or self.hnsw_recall_device.type == 'cuda'
                        else [])
        started = time.perf_counter()
        results = {}
        try:
            self._active_runtime = None
            with torch.random.fork_rng(devices=cuda_devices):
                print('[HNSWRecall][general] full-query evaluation; '
                      'hnsw_recall_queries/seed ignored; exact_cache=disabled')
                for split in self.hnsw_recall_splits:
                    dataset = datasets[split]
                    query_count = len(dataset)
                    per_period = {}
                    overall_hits = 0
                    overall_total = 0
                    for period_idx, period in enumerate(self.period_num):
                        hits = 0
                        exact_ms = hnsw_ms = 0.0
                        batch_size = self.hnsw_recall_query_batch_size
                        for start in range(0, query_count, batch_size):
                            end = min(start + batch_size, query_count)
                            windows, query_indices = self._general_recall_batch(dataset, start, end)
                            exact_start = time.perf_counter()
                            exact_labels, _ = self._chunked_general_exact_topk(
                                train_data, windows, query_indices, period_idx, split == 'train',
                            )
                            exact_ms += (time.perf_counter() - exact_start) * 1000.0
                            hnsw_start = time.perf_counter()
                            hnsw_labels, _ = self.search_candidates(
                                windows, period_idx, query_indices, train=split == 'train',
                                target_count=self.search_target,
                            )
                            hnsw_ms += (time.perf_counter() - hnsw_start) * 1000.0
                            hits += sum(
                                np.intersect1d(exact.numpy(), approximate.numpy()).size
                                for exact, approximate in zip(exact_labels, hnsw_labels)
                            )
                        total = query_count * self.search_target
                        recall = hits / total
                        per_period[period] = {
                            'k': self.search_target, 'queries': query_count,
                            'candidate_count': self.n_train, 'mean_recall': recall,
                            'hits': hits, 'total': total,
                            'exact_reference_ms': exact_ms, 'hnsw_diagnostic_ms': hnsw_ms,
                        }
                        overall_hits += hits
                        overall_total += total
                        print(
                            f'[HNSWRecall][general][{split}][period={period}] '
                            f'k={self.search_target} queries={query_count} '
                            f'candidate_count={self.n_train} mean_recall={recall:.6f} '
                            f'hits={hits} total={total} exact_reference_ms={exact_ms:.3f} '
                            f'hnsw_diagnostic_ms={hnsw_ms:.3f}'
                        )
                    overall = overall_hits / overall_total
                    results[split] = {
                        'overall': overall, 'per_period': per_period,
                        'queries': query_count, 'k': self.search_target,
                        'hits': overall_hits, 'total': overall_total,
                    }
                    print(
                        f'[HNSWRecall][general][{split}][overall] recall={overall:.6f} '
                        f'hits={overall_hits} total={overall_total} queries={query_count} '
                        f'k={self.search_target}'
                    )
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            (self._wide_fallbacks, self._wide_searches,
             self._train_queries, self._active_runtime) = saved_state
            for index, ef in zip(self.hnsw_indices, saved_ef):
                index.set_ef(ef)
        print('[HNSWRecall][general] HNSW Recall diagnostic time: '
              f'{(time.perf_counter() - started) * 1000.0:.3f}ms')
        return results
