"""Historical WN-R input coordinates for Web STP only.

fit_robust: SHARP-webscale-learned-official-zscore-wnr/experiments/zscore_common.py
robust_norm: SHARP-webscale-tsf-representative-fiveway/experiments/fixed_cache.py
Both function bodies are copied verbatim; Quito q_common.py uses these same helpers.
No forecast scaler, search key, raw bank, or candidate future is transformed here.
"""
import numpy as np


def fit_robust(bank,train_end):
    # Fit train-only robust statistics for CREST state profiles.
    q=np.stack([np.quantile(bank.get_series(int(s))[:train_end].astype(np.float64),[0,.25,.5,.75,1],axis=0) for s in bank.series_ids])
    med=q[:,2];iqr=q[:,3]-q[:,1];minimum=q[:,0];maximum=q[:,4]
    obs=np.array([np.min(v[v>0]) if np.any(v>0) else 1. for v in iqr.T])
    numeric=32*np.finfo(np.float32).eps*np.maximum(1,np.abs(med));floor=np.maximum(obs[None,:],numeric)
    return dict(median=med,iqr=iqr,minimum=minimum,maximum=maximum,safe_iqr=np.maximum(iqr,floor),floor=floor,observed_floor=obs,numeric_floor=numeric,series_ids=bank.series_ids.copy(),train_end=train_end)


def robust_norm(x,sid,fit):
 ix=np.searchsorted(fit['series_ids'],sid)
 return ((x.astype(np.float64)-fit['median'][ix,None,:])/fit['safe_iqr'][ix,None,:]).astype(np.float32)

class WebSTPRobustScaler:
    """Fit once on the selected series' train observations; transform window chunks."""
    def __init__(self, bank, train_end):
        self.fit = fit_robust(bank, train_end)

    def transform(self, windows, series_ids):
        return robust_norm(windows, series_ids, self.fit)
