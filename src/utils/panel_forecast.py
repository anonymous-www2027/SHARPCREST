"""Small train-only model scaler; raw retrieval bank is never normalized."""
import numpy as np
import torch
from src.utils.webscale_metrics import seasonal_scale


class PanelForecastScaler:
    def __init__(self, bank, train_end, period):
        self.series_ids = bank.series_ids.copy()
        self.train_end, self.period = int(train_end), int(period)
        means, stds, qs, valid, invalid = [], [], [], [], []
        for sid in self.series_ids:
            x = bank.get_series(int(sid))[:train_end].astype(np.float64)
            if len(x) != train_end or not np.isfinite(x).all():
                raise ValueError('forecast scaler requires the finite frozen train interval')
            means.append(x.mean(0)); stds.append(x.std(0, ddof=0))
            q, v, bad = seasonal_scale(x, period)
            qs.append(q); valid.append(v); invalid.append(bad)
        self.mean, self.raw_std = np.array(means), np.array(stds)
        self.scale = self.raw_std + 1e-8
        self.mase_q, self.valid_pairs, self.invalid_pairs = np.array(qs), np.array(valid), np.array(invalid)

    def positions(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        ix = np.searchsorted(self.series_ids, ids)
        if (ix >= len(self.series_ids)).any() or not np.array_equal(self.series_ids[ix], ids):
            raise ValueError('query source identity absent from forecast scaler')
        return ix

    def transform(self, values, ids):
        ix = self.positions(ids)
        return ((np.asarray(values, dtype=np.float64)-self.mean[ix, None, :])/self.scale[ix, None, :]).astype(np.float32)

    def inverse(self, values, ids):
        ix = self.positions(ids)
        return np.asarray(values, dtype=np.float64)*self.scale[ix, None, :]+self.mean[ix, None, :]

    def save(self, path):
        np.savez(path, series_id=self.series_ids, mean=self.mean, raw_std=self.raw_std, scale=self.scale,
                 train_end=self.train_end, mase_period=self.period, mase_q=self.mase_q,
                 valid_pairs=self.valid_pairs, invalid_pairs=self.invalid_pairs)


def query_scale_payload_(payload, dataset, chunk_size=1024):
    """Scale existing [period,query,H,C] storage, with only a bounded double chunk."""
    scaler = dataset.forecast_scaler
    if payload.ndim != 4 or payload.shape[1:] != (len(dataset), dataset.pred_len, len(scaler.scale[0])):
        raise ValueError('query-scale payload shape mismatch')
    for begin in range(0, len(dataset), chunk_size):
        end = min(begin+chunk_size, len(dataset))
        ids, _ = dataset.query_manifest.resolve_ids(np.arange(begin, end))
        scale = torch.as_tensor(scaler.scale[scaler.positions(ids)], device=payload.device)
        chunk = payload[:, begin:end].double()/scale[None, :, None, :]
        if not torch.isfinite(chunk).all():
            raise ValueError('nonfinite query-scale payload')
        payload[:, begin:end].copy_(chunk)
        if not torch.isfinite(payload[:, begin:end]).all():
            raise ValueError('query-scale payload overflows model dtype')
    return payload
