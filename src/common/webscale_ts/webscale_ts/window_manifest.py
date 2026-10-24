"""O(series) coordinate manifests for potentially huge window sets."""

import numpy as np


class WindowManifest:
    def __init__(
        self, series_ids, first_start, window_counts, prefix_offsets, stride,
        seq_len, pred_len,
    ):
        self.series_ids = np.asarray(series_ids, dtype=np.int64)
        self.first_start = np.asarray(first_start, dtype=np.int64)
        self.window_counts = np.asarray(window_counts, dtype=np.int64)
        self.prefix_offsets = np.asarray(prefix_offsets, dtype=np.int64)
        stride = np.asarray(stride, dtype=np.int64)
        if stride.ndim == 0:
            stride = np.full(len(self.series_ids), int(stride), dtype=np.int64)
        self.stride = stride
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        expected = (len(self.series_ids),)
        if not all(array.shape == expected for array in (
            self.first_start, self.window_counts, self.stride
        )):
            raise ValueError('manifest series metadata must have shape [S]')
        if self.prefix_offsets.shape != (len(self.series_ids) + 1,):
            raise ValueError('prefix_offsets must have shape [S+1]')
        if np.any(self.window_counts <= 0) or np.any(self.stride <= 0):
            raise ValueError('manifest counts and strides must be positive')
        if not np.array_equal(np.diff(self.prefix_offsets), self.window_counts):
            raise ValueError('prefix offsets and window counts disagree')

    @staticmethod
    def _bound_array(value, size, name):
        array = np.asarray(value, dtype=np.int64)
        if array.ndim == 0:
            return np.full(size, int(array), dtype=np.int64)
        if array.shape != (size,):
            raise ValueError(f'{name} must be scalar or shape [S]')
        return array

    @classmethod
    def from_bounds(
        cls, series_bank, seq_len, pred_len, stride=1,
        start_bound=0, end_bound=None,
    ):
        size = len(series_bank.series_ids)
        starts = cls._bound_array(start_bound, size, 'start_bound')
        ends = (
            series_bank.lengths.copy()
            if end_bound is None
            else cls._bound_array(end_bound, size, 'end_bound')
        )
        strides = cls._bound_array(stride, size, 'stride')
        if np.any(starts < 0) or np.any(ends > series_bank.lengths):
            raise ValueError('manifest bounds lie outside a selected series')
        if np.any(strides < 1):
            raise ValueError('manifest strides must be positive')
        width = int(seq_len) + int(pred_len)
        available = ends - starts - width
        counts = np.where(available >= 0, available // strides + 1, 0)
        keep = counts > 0
        series_ids = series_bank.series_ids[keep]
        starts = starts[keep]
        strides = strides[keep]
        counts = counts[keep].astype(np.int64, copy=False)
        if not len(series_ids):
            raise ValueError('manifest bounds produce no valid windows')
        prefix = np.empty(len(series_ids) + 1, dtype=np.int64)
        prefix[0] = 0
        np.cumsum(counts, out=prefix[1:])
        return cls(
            series_ids, starts, counts, prefix, strides, seq_len, pred_len
        )

    def __len__(self):
        return int(self.prefix_offsets[-1])

    @property
    def nbytes(self):
        return int(sum(array.nbytes for array in (
            self.series_ids, self.first_start, self.window_counts,
            self.prefix_offsets, self.stride,
        )))

    def resolve_ids(self, candidate_ids):
        ids = np.asarray(candidate_ids, dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= len(self)):
            raise IndexError('window ID lies outside the manifest')
        positions = np.searchsorted(
            self.prefix_offsets, ids, side='right'
        ) - 1
        local = ids - self.prefix_offsets[positions]
        series_ids = self.series_ids[positions]
        starts = self.first_start[positions] + local * self.stride[positions]
        return series_ids.astype(np.int64, copy=False), starts.astype(
            np.int64, copy=False
        )

    def encode(self, series_ids, start_indices):
        ids = np.asarray(series_ids, dtype=np.int64)
        starts = np.asarray(start_indices, dtype=np.int64)
        if ids.shape != starts.shape:
            raise ValueError('series IDs and starts must have equal shapes')
        positions = np.searchsorted(self.series_ids, ids)
        valid = positions < len(self.series_ids)
        matched = np.zeros(ids.shape, dtype=bool)
        matched[valid] = self.series_ids[positions[valid]] == ids[valid]
        if not np.all(matched):
            raise KeyError(int(ids[~matched].reshape(-1)[0]))
        delta = starts - self.first_start[positions]
        local = delta // self.stride[positions]
        valid_start = (
            (delta >= 0)
            & (delta % self.stride[positions] == 0)
            & (local < self.window_counts[positions])
        )
        if not np.all(valid_start):
            raise KeyError('start index is not represented by the manifest')
        return (self.prefix_offsets[positions] + local).astype(
            np.int64, copy=False
        )

    def summary(self):
        return {
            'series_with_windows': int(len(self.series_ids)),
            'window_count': len(self),
            'manifest_bytes': self.nbytes,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'min_stride': int(self.stride.min()),
            'max_stride': int(self.stride.max()),
        }
