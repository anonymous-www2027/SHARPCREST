"""Compact, model-independent panel time-series infrastructure."""

from .panel_dataset import PanelWindowDataset
from .protocols import PanelSource
from .series_bank import InMemorySeriesBank
from .subset import SeriesSubset
from .window_manifest import WindowManifest

__all__ = [
    'PanelSource',
    'SeriesSubset',
    'InMemorySeriesBank',
    'WindowManifest',
    'PanelWindowDataset',
]
