"""Protocols implemented by disk-backed or synthetic panel sources."""

from typing import Protocol

import numpy as np


class PanelSource(Protocol):
    """Random-access source used only while constructing a series bank."""

    @property
    def num_series(self) -> int:
        ...

    def get_length(self, series_id: int) -> int:
        ...

    def get_n_channels(self, series_id: int) -> int:
        ...

    def get_series(self, series_id: int) -> np.ndarray:
        """Return one ``[T,C]`` float32-compatible series."""
        ...
