"""Repository-local bridge to the installable :mod:`webscale_ts` package."""

from .webscale_ts import (  # noqa: F401
    InMemorySeriesBank,
    PanelSource,
    PanelWindowDataset,
    SeriesSubset,
    WindowManifest,
)

__all__ = [
    'PanelSource',
    'SeriesSubset',
    'InMemorySeriesBank',
    'WindowManifest',
    'PanelWindowDataset',
]
