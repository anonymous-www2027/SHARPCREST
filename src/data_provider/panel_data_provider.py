"""Vectorized training/evaluation batches for web-scale panel datasets."""

import numpy as np
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
)

from src.data_provider.webscale_panel_adapter import build_panel_bundle


class PanelBatchDataset(Dataset):
    """Materialize one list of query IDs without per-sample dataset access.

    Length remains the number of queries; the DataLoader's sampler groups
    those queries into batches. The original panel dataset is still returned
    by the provider for retrieval preparation and split metadata.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, batch_indices):
        batch_ids = np.asarray(batch_indices, dtype=np.int64)
        if batch_ids.ndim != 1:
            raise ValueError('panel batch indices must be a one-dimensional list')
        dataset = self.dataset
        series_ids, starts = dataset.query_manifest.resolve_ids(batch_ids)
        x = dataset.series_bank.get_windows(
            series_ids, starts, dataset.seq_len
        )
        y = dataset.series_bank.get_windows(
            series_ids, starts + dataset.seq_len, dataset.pred_len
        )
        scaler = getattr(dataset, 'forecast_scaler', None)
        if scaler is not None:
            x, y = scaler.transform(x, series_ids), scaler.transform(y, series_ids)
        # Match default_collate(PanelWindowDataset samples), including the
        # list container and future-only Y even when label_len is nonzero.
        return [
            torch.from_numpy(batch_ids),
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.zeros((len(batch_ids), dataset.seq_len, 1), dtype=torch.float32),
            torch.zeros((len(batch_ids), dataset.pred_len, 1), dtype=torch.float32),
        ]


def panel_data_provider(args, flag):
    bundle = getattr(args, '_panel_data_bundle', None)
    if bundle is None:
        if getattr(args, 'main_protocol', False):
            from src.data_provider.frozen_panel import build_frozen_bundle
            bundle = build_frozen_bundle(args)
        else:
            bundle = build_panel_bundle(args)
        args._panel_data_bundle = bundle
    panel_flag = 'valid' if flag == 'val' else flag
    dataset = bundle.datasets[panel_flag]
    print(flag, len(dataset))
    # Preserve PyTorch's historical train/validation shuffle and RNG use.
    sampler = SequentialSampler(dataset) if flag == 'test' else RandomSampler(dataset)
    batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=False)
    loader = DataLoader(
        PanelBatchDataset(dataset),
        # PyTorch 1.10: disabling automatic batching passes each sampler
        # element (a whole list of IDs) directly to dataset.__getitem__.
        batch_size=None,
        sampler=batch_sampler,
        num_workers=args.num_workers,
    )
    return dataset, loader
