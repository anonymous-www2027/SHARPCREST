"""TSF functions extracted from Quito r1.0.0; see LICENSE.quito and quito_source.json."""
# Copyright (c) 2024 Alipay. Distributed under the MIT License.
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union
import numpy as np
from scipy.signal import welch
from statsmodels.tsa.seasonal import STL

ArrayLike = Union[np.ndarray, List[float]]
_EPS = 1e-12
STL_AVAILABLE = True
TORCH_AVAILABLE = False
ARCH_AVAILABLE = False

def _to_1d_numpy(x: ArrayLike) -> np.ndarray:
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=float).squeeze()
    if x.ndim != 1:
        raise ValueError(f'Expected 1D series, got shape {x.shape}')
    return x

def spectral_entropy_welch(x: np.ndarray, fs: float=1.0, nperseg: Optional[int]=None) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(np.isnan(x)):
        return 1.0
    if np.std(x) < _EPS:
        return 0.0
    x = x - np.nanmean(x)
    x = np.nan_to_num(x, nan=0.0)
    if nperseg is None:
        nperseg = max(64, min(len(x), 1024))
    _, Pxx = welch(x, fs=fs, nperseg=nperseg, detrend='constant', scaling='density')
    Pxx = Pxx + _EPS
    Pxx /= Pxx.sum()
    H = -(Pxx * np.log(Pxx)).sum()
    Hmax = np.log(len(Pxx))
    return float(H / Hmax)

def forecastability_welch(x: np.ndarray, fs: float=1.0, nperseg: Optional[int]=None, window: Optional[int]=None) -> float:
    x = np.asarray(x, dtype=float)
    if window is None or window <= 0 or window >= len(x):
        se = spectral_entropy_welch(x, fs=fs, nperseg=nperseg)
        return float(max(0.0, min(1.0, 1.0 - se)))
    vals = []
    n = len(x)
    k = max(n - window, 0) // window + 1
    for i in range(k):
        s, e = (i * window, min((i + 1) * window, n))
        se = spectral_entropy_welch(x[s:e], fs=fs, nperseg=nperseg)
        vals.append(max(0.0, min(1.0, 1.0 - se)))
    return float(np.nanmean(vals)) if vals else np.nan

def stl_decomposition_strengths(x: np.ndarray, period: Optional[int]) -> tuple:
    if not STL_AVAILABLE or not period or period < 2 or (period >= len(x)):
        return (np.nan, np.nan)
    if np.all(np.isnan(x)):
        return (np.nan, np.nan)
    x = np.nan_to_num(x, nan=float(np.nanmedian(x)))
    try:
        res = STL(x, period=period, robust=True).fit()
        resid = res.resid
        seas = res.seasonal
        trend = res.trend
        seas_den = np.var(seas + resid) + _EPS
        seas_strength = float(max(0.0, 1.0 - np.var(resid) / seas_den))
        trend_den = np.var(trend + resid) + _EPS
        trend_strength = float(max(0.0, 1.0 - np.var(resid) / trend_den))
        return (seas_strength, trend_strength)
    except Exception:
        return (np.nan, np.nan)

def missing_ratio(x: np.ndarray) -> float:
    return float(np.mean(np.isnan(x))) if x.size else 1.0

def effective_length(x: np.ndarray) -> int:
    return int(np.sum(~np.isnan(x)))

def coefficient_of_variation(x: np.ndarray) -> float:
    y = x[~np.isnan(x)]
    if y.size == 0:
        return np.nan
    m = float(np.mean(y))
    s = float(np.std(y))
    if m == 0.0:
        return float('inf') if s > 0 else 0.0
    return float(s / abs(m))

def fill_missing(x: np.ndarray, how: str='none') -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    if how == 'none':
        return x
    if how == 'zero':
        x[np.isnan(x)] = 0.0
        return x
    if how == 'mean':
        m = np.nanmean(x)
        x[np.isnan(x)] = 0.0 if np.isnan(m) else m
        return x
    if how == 'forward':
        mask = np.isnan(x)
        idx = np.where(~mask, np.arange(len(x)), 0)
        np.maximum.accumulate(idx, out=idx)
        x[mask] = x[idx[mask]]
        x[np.isnan(x)] = 0.0
        return x
    if how == 'backward':
        mask = np.isnan(x)
        idx = np.where(~mask, np.arange(len(x)), len(x) - 1)
        for i in range(len(idx) - 2, -1, -1):
            if mask[i]:
                idx[i] = idx[i + 1]
        x[mask] = x[idx[mask]]
        x[np.isnan(x)] = 0.0
        return x
    raise ValueError(f'Unknown fill method: {how}')

@dataclass
class SeriesQuality:
    forecastability: float
    season_strength: float
    trend_strength: float
    missing_ratio: float
    eff_length: int
    cv: float
    adf_stat: float
    hurst: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

def _evaluate_series(x: ArrayLike, period: Optional[int]=None, fs: float=1.0, compute_adf: bool=False, compute_hurst: bool=True, adf_fill: str='mean', forecast_window: Optional[int]=None, fill_for_metrics: str='none') -> SeriesQuality:
    x = _to_1d_numpy(x)
    miss = missing_ratio(x)
    eff = effective_length(x)
    x_cv = fill_missing(x, how=fill_for_metrics)
    cv = coefficient_of_variation(x_cv)
    if cv == 0.0:
        return SeriesQuality(forecastability=1.0, season_strength=0.0, trend_strength=0.0, missing_ratio=miss, eff_length=eff, cv=0.0, adf_stat=-99.9, hurst=0.5)
    fcast = forecastability_welch(x, fs=fs, window=forecast_window)
    seas, trend = stl_decomposition_strengths(x, period)
    adf_val = np.nan
    if compute_adf and ARCH_AVAILABLE:
        try:
            x_adf = fill_missing(x, how=adf_fill)
            adf_val = float(ADF(x_adf).stat)
        except Exception as e:
            if 'singular regressor matrix' not in str(e):
                logger.warning(f'ADF test failed for series: {e}')
            adf_val = np.nan
    hurst_val = np.nan
    if compute_hurst:
        x_filled = fill_missing(x, how='mean')
        hurst_val = hurst_exponent(x_filled)
    return SeriesQuality(forecastability=fcast, season_strength=seas, trend_strength=trend, missing_ratio=miss, eff_length=eff, cv=cv, adf_stat=adf_val, hurst=hurst_val)
