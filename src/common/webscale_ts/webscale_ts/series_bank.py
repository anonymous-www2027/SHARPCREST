"""Compact contiguous in-memory storage for selected panel series."""

import numpy as np


class InMemorySeriesBank:
    def __init__(self, values, series_ids, offsets, lengths):
        self.values = np.ascontiguousarray(values, dtype=np.float32)
        self.series_ids = np.asarray(series_ids, dtype=np.int64)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.lengths = np.asarray(lengths, dtype=np.int64)
        if self.values.ndim != 2:
            raise ValueError('values must have shape [sum_T,C]')
        if self.series_ids.ndim != 1 or self.lengths.shape != self.series_ids.shape:
            raise ValueError('series_ids and lengths must be aligned vectors')
        if self.offsets.shape != (len(self.series_ids) + 1,):
            raise ValueError('offsets must have shape [S+1]')
        if len(self.series_ids) > 1 and np.any(
            self.series_ids[1:] <= self.series_ids[:-1]
        ):
            raise ValueError('series_ids must be sorted and unique')
        if self.offsets[0] != 0 or self.offsets[-1] != len(self.values):
            raise ValueError('offsets do not cover values')
        if not np.array_equal(np.diff(self.offsets), self.lengths):
            raise ValueError('offsets and lengths disagree')
        self.largest_window_batch = 0

    @classmethod
    def from_source(cls, source, subset):
        series_ids = subset.selected_ids
        lengths = np.asarray(
            [source.get_length(int(sid)) for sid in series_ids], dtype=np.int64
        )
        channels = np.asarray(
            [source.get_n_channels(int(sid)) for sid in series_ids],
            dtype=np.int64,
        )
        if len(channels) and np.any(channels != channels[0]):
            raise ValueError('all selected series must have the same channel count')
        if np.any(lengths < 0):
            raise ValueError('series lengths must be non-negative')
        n_channels = int(channels[0]) if len(channels) else 0
        offsets = np.empty(len(series_ids) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        values = np.empty((int(offsets[-1]), n_channels), dtype=np.float32)
        for position, series_id in enumerate(series_ids):
            series = np.asarray(source.get_series(int(series_id)), dtype=np.float32)
            expected = (int(lengths[position]), n_channels)
            if series.shape != expected:
                raise ValueError(
                    f'series {int(series_id)} has shape {series.shape}; '
                    f'expected {expected}'
                )
            values[offsets[position]:offsets[position + 1]] = series
        return cls(values, series_ids, offsets, lengths)

    @property
    def n_channels(self):
        return int(self.values.shape[1])

    @property
    def total_time_steps(self):
        return int(len(self.values))

    @property
    def nbytes(self):
        return int(
            self.values.nbytes + self.series_ids.nbytes
            + self.offsets.nbytes + self.lengths.nbytes
        )

    def _positions(self, series_ids):
        ids = np.asarray(series_ids, dtype=np.int64)
        positions = np.searchsorted(self.series_ids, ids)
        valid = positions < len(self.series_ids)
        matched = np.zeros(ids.shape, dtype=bool)
        matched[valid] = self.series_ids[positions[valid]] == ids[valid]
        if not np.all(matched):
            raise KeyError(int(ids[~matched].reshape(-1)[0]))
        return positions

    def get_series(self, series_id):
        position = int(self._positions(np.asarray([series_id]))[0])
        return self.values[self.offsets[position]:self.offsets[position + 1]]

    def get_window(self, series_id, start_idx, length):
        position = int(self._positions(np.asarray([series_id]))[0])
        start_idx = int(start_idx)
        length = int(length)
        if start_idx < 0 or length < 1 or start_idx + length > self.lengths[position]:
            raise IndexError('window lies outside the selected series')
        begin = int(self.offsets[position] + start_idx)
        return self.values[begin:begin + length]

    def get_windows(self, series_ids, start_indices, length):
        ids = np.asarray(series_ids, dtype=np.int64)
        starts = np.asarray(start_indices, dtype=np.int64)
        if ids.shape != starts.shape:
            raise ValueError('series_ids and start_indices must have equal shapes')
        flat_ids = ids.reshape(-1)
        flat_starts = starts.reshape(-1)
        positions = self._positions(flat_ids)
        length = int(length)
        if length < 1:
            raise ValueError('window length must be positive')
        if np.any(flat_starts < 0) or np.any(
            flat_starts + length > self.lengths[positions]
        ):
            raise IndexError('batch contains a window outside its series')
        indices = (self.offsets[positions] + flat_starts)[:, None]
        indices = indices + np.arange(length, dtype=np.int64)[None, :]
        output = self.values[indices]
        self.largest_window_batch = max(self.largest_window_batch, len(flat_ids))
        return output.reshape(ids.shape + (length, self.n_channels))

    def summary(self):
        metadata_bytes = (
            self.series_ids.nbytes + self.offsets.nbytes + self.lengths.nbytes
        )
        return {
            'selected_series': int(len(self.series_ids)),
            'selected_time_steps': self.total_time_steps,
            'selected_scalar_values': int(self.values.size),
            'n_channels': self.n_channels,
            'values_bytes': int(self.values.nbytes),
            'metadata_bytes': int(metadata_bytes),
            'total_bytes': self.nbytes,
        }
