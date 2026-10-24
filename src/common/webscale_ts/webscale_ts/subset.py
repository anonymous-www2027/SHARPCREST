"""Deterministic, shareable selection of series IDs."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SeriesSubset:
    total_series: int
    selected_ids: np.ndarray
    seed: int
    selection_mode: str
    dataset_identifier: Optional[str] = None

    def __post_init__(self):
        ids = np.asarray(self.selected_ids, dtype=np.int64)
        if self.total_series < 0:
            raise ValueError('total_series must be non-negative')
        if ids.ndim != 1:
            raise ValueError('selected_ids must be one-dimensional')
        if len(ids) and (ids[0] < 0 or ids[-1] >= self.total_series):
            raise ValueError('selected series ID is outside the source')
        if len(ids) > 1 and np.any(ids[1:] <= ids[:-1]):
            raise ValueError('selected series IDs must be sorted and unique')
        object.__setattr__(self, 'selected_ids', ids)

    @classmethod
    def create(
        cls,
        total_series: int,
        fraction: Optional[float] = None,
        max_series: Optional[int] = None,
        seed: int = 0,
        dataset_identifier: Optional[str] = None,
    ):
        if fraction is not None and max_series is not None:
            raise ValueError('fraction and max_series are mutually exclusive')
        total_series = int(total_series)
        if total_series < 0:
            raise ValueError('total_series must be non-negative')

        if fraction is not None:
            if not 0 < fraction <= 1:
                raise ValueError('fraction must be in (0,1]')
            count = min(total_series, max(1, int(np.floor(total_series * fraction))))
            mode = 'fraction'
        elif max_series is not None:
            if max_series < 1:
                raise ValueError('max_series must be positive')
            count = min(total_series, int(max_series))
            mode = 'max_series'
        else:
            count = total_series
            mode = 'all'

        if count == total_series:
            selected = np.arange(total_series, dtype=np.int64)
        else:
            generator = np.random.default_rng(seed)
            selected = np.sort(
                generator.choice(total_series, size=count, replace=False)
            ).astype(np.int64, copy=False)
        return cls(
            total_series=total_series,
            selected_ids=selected,
            seed=int(seed),
            selection_mode=mode,
            dataset_identifier=dataset_identifier,
        )

    @property
    def selected_count(self):
        return int(len(self.selected_ids))

    @property
    def selected_fraction(self):
        return self.selected_count / self.total_series if self.total_series else 0.0

    def summary(self):
        return {
            'total_series': self.total_series,
            'selected_count': self.selected_count,
            'selected_fraction': self.selected_fraction,
            'seed': self.seed,
            'selection_mode': self.selection_mode,
            'dataset_identifier': self.dataset_identifier,
        }

    def save(self, path):
        path = Path(path)
        if path.suffix != '.npz':
            path = path.with_suffix('.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = self.summary()
        np.savez_compressed(
            path,
            selected_ids=self.selected_ids,
            metadata=np.asarray(json.dumps(metadata)),
        )
        path.with_suffix('.json').write_text(
            json.dumps(metadata, indent=2) + '\n', encoding='utf-8'
        )
        return path

    @classmethod
    def load(cls, path):
        path = Path(path)
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload['metadata']))
            selected_ids = payload['selected_ids'].astype(np.int64, copy=False)
        return cls(
            total_series=int(metadata['total_series']),
            selected_ids=selected_ids,
            seed=int(metadata['seed']),
            selection_mode=metadata['selection_mode'],
            dataset_identifier=metadata.get('dataset_identifier'),
        )
