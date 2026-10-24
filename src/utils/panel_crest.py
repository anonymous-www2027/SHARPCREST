"""Disk-backed candidate STP storage and train-fitted scales for Web-scale CREST."""

import mmap
from pathlib import Path
import tempfile

import numpy as np
import torch

from src.utils.stp_reliability import STP_EPS, compute_stp_profile


def validate_panel_crest_options(options):
    if getattr(options, 'retrieval_controller', 'baseline') == 'baseline':
        return
    if (getattr(options, 'retrieval_controller', 'baseline') != 'crest'
            or getattr(options, 'crest_profile', 'stp') != 'stp'
            or getattr(options, 'mmr_redundancy_source', 'value') != 'value'):
        raise NotImplementedError(
            'panel CREST supports crest_profile=stp and mmr_redundancy_source=value only'
        )


class PanelSTPSidecar:
    """Run-local float32 [candidate, section, channel, feature] memmap.

    Only a window chunk or one N-candidate coordinate is copied into RAM.
    Drop mapped pages after each operation on systems supporting madvise so
    sequential scans do not retain the entire sidecar in process RSS.
    """

    def __init__(self, cache_root, shape):
        root = Path(cache_root) / '.panel-crest-work'
        root.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix='stp-', dir=root)
        self.path = Path(self.directory.name) / 'profiles.f32'
        self.profiles = None
        try:
            self.profiles = np.memmap(self.path, mode='w+', dtype=np.float32, shape=shape)
        except BaseException:
            self.close()
            raise

    def _release_pages(self):
        mapping = self.profiles._mmap
        if hasattr(mapping, 'madvise') and hasattr(mmap, 'MADV_DONTNEED'):
            mapping.madvise(mmap.MADV_DONTNEED)

    def prepare(self, bank, manifest, seq_len, sections, chunk_size):
        for start in range(0, len(manifest), chunk_size):
            end = min(start + chunk_size, len(manifest))
            series, starts = manifest.resolve_ids(np.arange(start, end, dtype=np.int64))
            windows = bank.get_windows(series, starts, seq_len)
            stp_robust_scaler = getattr(bank, 'stp_robust_scaler', None)
            if stp_robust_scaler is not None:
                windows = stp_robust_scaler.transform(windows, series)
            windows = torch.from_numpy(windows)
            self.profiles[start:end] = compute_stp_profile(windows, sections).numpy()
            self.profiles.flush()
            self._release_pages()
            del windows
        scale = torch.empty(self.profiles.shape[1:], dtype=torch.float32)
        for coordinate in np.ndindex(*scale.shape):
            values = np.empty(len(manifest), dtype=np.float32)
            for start in range(0, len(manifest), chunk_size):
                end = min(start + chunk_size, len(manifest))
                values[start:end] = self.profiles[(slice(start, end),) + coordinate]
                self._release_pages()
            if len(values) <= 2 ** 24:
                quartiles = torch.quantile(torch.from_numpy(values), torch.tensor([0.25, 0.75]))
            else:
                # torch.quantile rejects larger vectors. NumPy retains linear
                # interpolation semantics without that candidate-count limit.
                quartiles = torch.from_numpy(np.quantile(values, [0.25, 0.75]).astype(np.float32))
            scale[coordinate] = quartiles[1] - quartiles[0] + STP_EPS
            del values
        return torch.nan_to_num(scale, nan=STP_EPS, posinf=STP_EPS, neginf=STP_EPS).clamp_min(STP_EPS)

    def lookup(self, labels):
        # Advanced indexing owns its output: no tensor can outlive a mapped view.
        profiles = torch.from_numpy(self.profiles[labels.detach().cpu().numpy()])
        self._release_pages()
        return profiles

    def close(self):
        if self.profiles is not None:
            self.profiles._mmap.close()
            self.profiles = None
        self.directory.cleanup()
