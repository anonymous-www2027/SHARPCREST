"""On-demand five-field dataset backed by a series bank and manifest."""

import numpy as np


class PanelWindowDataset:
    is_panel_dataset = True
    scale = False

    def __init__(
        self, series_bank, query_manifest, candidate_manifest=None,
        label_len=0, split='unknown', memory_report=None,
    ):
        if query_manifest.seq_len <= 0 or query_manifest.pred_len <= 0:
            raise ValueError('query manifest must define positive window lengths')
        self.series_bank = series_bank
        self.window_manifest = query_manifest
        self.query_manifest = query_manifest
        self.candidate_manifest = (
            query_manifest if candidate_manifest is None else candidate_manifest
        )
        self.seq_len = query_manifest.seq_len
        self.pred_len = query_manifest.pred_len
        self.label_len = int(label_len)
        self.split = split
        self.memory_report = {} if memory_report is None else memory_report

    def __len__(self):
        return len(self.query_manifest)

    def __getitem__(self, query_id):
        series_id, start = self.query_manifest.resolve_ids(
            np.asarray(query_id, dtype=np.int64)
        )
        series_id = int(series_id)
        start = int(start)
        x = self.series_bank.get_window(series_id, start, self.seq_len)
        y = self.series_bank.get_window(
            series_id, start + self.seq_len, self.pred_len
        )
        return (
            int(query_id),
            x,
            y,
            np.zeros((self.seq_len, 1), dtype=np.float32),
            np.zeros((self.pred_len, 1), dtype=np.float32),
        )

    def inverse_transform(self, data):
        return data
