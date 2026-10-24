"""Raw-space web-scale metrics, separate from legacy evaluation."""
import numpy as np


def seasonal_scale(train, period):
    train = np.asarray(train, dtype=np.float64)
    if train.ndim == 1:
        train = train[:, None]
    if period < 1:
        raise ValueError('seasonal period must be positive')
    a, b = train[period:], train[:max(0, len(train)-period)]
    valid = np.isfinite(a) & np.isfinite(b)
    counts = valid.sum(0)
    sums = np.where(valid, np.abs(a-b), 0.).sum(0, dtype=np.float64)
    q = np.full(train.shape[1], np.nan, dtype=np.float64)
    np.divide(sums, counts, out=q, where=counts > 0)
    return q, counts, len(a)-counts


def smape_errors(prediction, target):
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.shape != y.shape or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('finite aligned raw prediction/target required')
    denominator = np.abs(p)+np.abs(y)
    error = np.zeros_like(p)
    np.divide(2*np.abs(p-y), denominator, out=error, where=denominator != 0)
    return error


class WebscaleMetrics:
    """Streaming scalar pooled sums; targets come from the original raw bank."""
    def __init__(self):
        self.sums = np.zeros(4, dtype=np.float64)
        self.total = self.mase_valid = 0

    def update(self, prediction, normalized_target, raw_target, series_ids, scaler):
        p, y = np.asarray(prediction, dtype=np.float64), np.asarray(normalized_target, dtype=np.float64)
        if p.shape != y.shape or not np.isfinite(p).all() or not np.isfinite(y).all():
            raise ValueError('nonfinite or misaligned normalized prediction/target')
        raw = scaler.inverse(p, series_ids)
        target = np.asarray(raw_target, dtype=np.float64)
        sa = smape_errors(raw, target)
        q = scaler.mase_q[scaler.positions(series_ids)][:, None, :]
        valid = np.broadcast_to(np.isfinite(q) & (q > 0), target.shape)
        scaled = np.zeros_like(target)
        np.divide(np.abs(raw-target), q, out=scaled, where=valid)
        if not np.isfinite(scaled).all():
            raise ValueError('nonfinite defined MASE error')
        self.sums += [((p-y)**2).sum(dtype=np.float64), np.abs(p-y).sum(dtype=np.float64),
                      sa.sum(dtype=np.float64), scaled.sum(dtype=np.float64)]
        self.total += p.size
        self.mase_valid += int(valid.sum())

    def result(self):
        if not self.total:
            raise ValueError('empty test population')
        return dict(normalized_MSE=self.sums[0]/self.total, normalized_MAE=self.sums[1]/self.total,
                    raw_sMAPE=100*self.sums[2]/self.total,
                    raw_MASE=self.sums[3]/self.mase_valid if self.mase_valid else None,
                    test_elements=self.total, mase_valid_elements=self.mase_valid,
                    mase_coverage=self.mase_valid/self.total)
