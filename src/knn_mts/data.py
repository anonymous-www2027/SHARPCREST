import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DATASETS = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Exchange', 'Weather')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


class Windows(Dataset):
    def __init__(self, values, starts, length, horizon):
        self.values, self.starts = values, np.asarray(starts, dtype=np.int64)
        self.length, self.horizon = length, horizon

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        boundary = start + self.length
        return self.values[start:boundary], self.values[boundary:boundary+self.horizon], start


class GeneralData:
    def __init__(self, dataset, path, length, horizon):
        if dataset not in DATASETS or min(length, horizon) < 1:
            raise ValueError('unsupported dataset or nonpositive L/H')
        self.dataset, self.path = dataset, Path(path).resolve()
        frame = pd.read_csv(self.path)
        if 'date' not in frame:
            raise ValueError('CSV must contain date and numeric variable columns')
        dates = pd.to_datetime(frame['date'], errors='raise')
        # Match the existing General loaders: CSV row order defines time.
        # Exchange_wide has non-monotonic date labels. Sorting here would change
        # the benchmark series and split identities; record it instead.
        timestamp_audit = dict(monotonic=bool(dates.is_monotonic_increasing),
                               duplicate_labels=int(dates.duplicated().sum()),
                               order_policy='preserve CSV row order, identical to existing General loaders')
        self.columns = [c for c in frame if c != 'date']
        if dataset in ('Exchange', 'Weather') and 'OT' in self.columns:
            self.columns = [c for c in self.columns if c != 'OT'] + ['OT']
        raw = frame[self.columns].to_numpy(dtype=np.float64)
        if not raw.shape[1] or not np.isfinite(raw).all():
            raise ValueError('numeric finite values required; no implicit imputation')
        if dataset.startswith('ETT'):
            multiplier = 4 if dataset.startswith('ETTm') else 1
            train_end, valid_end, test_end = [n * 30 * 24 * multiplier for n in (12, 16, 20)]
            if len(raw) < test_end:
                raise ValueError('ETT CSV shorter than the standard 12/4/4 month split')
        else:
            train_end, test_end = int(len(raw) * .7), len(raw)
            valid_end = len(raw) - int(len(raw) * .2)
        if length + horizon > train_end or horizon > min(valid_end-train_end, test_end-valid_end):
            raise ValueError('L/H does not fit the frozen split boundaries')
        self.ends = dict(train=train_end, val=valid_end, test=test_end)
        self.mean = raw[:train_end].mean(axis=0)
        std = raw[:train_end].std(axis=0)
        self.scale = np.where(std < 1e-12, 1., std)
        self.values = torch.from_numpy(((raw[:test_end]-self.mean)/self.scale).astype(np.float32))
        self.length, self.horizon = length, horizon
        self.channels = raw.shape[1]
        self.splits = {}
        for split, begin, end in [('train', length, train_end), ('val', train_end, valid_end), ('test', valid_end, test_end)]:
            starts = np.arange(begin-length, end-horizon-length+1, dtype=np.int64)
            self.splits[split] = Windows(self.values, starts, length, horizon)
        self.audit = dict(dataset=dataset, csv=str(self.path), csv_sha256=file_hash(self.path),
                          columns=self.columns, channels=self.channels, length=length, horizon=horizon,
                          boundaries=self.ends, timestamps=timestamp_audit,
                          normalization='train-only per-variable population z-score; near-constant scale=1',
                          scaler_hash=array_hash(np.stack([self.mean, self.scale])),
                          split_windows={s: len(d) for s, d in self.splits.items()},
                          query_start_hashes={s: array_hash(d.starts) for s, d in self.splits.items()},
                          bank_candidates=len(self.splits['train'])*self.channels,
                          bank_id_rule='past_start * channels + variable_index; all variables share one bank')
