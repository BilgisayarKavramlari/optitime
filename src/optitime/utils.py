"""Utility helpers for the OptiProphet model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class ChangepointResult:
    """Structure capturing changepoint selection output."""

    locations: List[float]
    indexes: List[int]


def infer_frequency(ds: pd.Series) -> str | None:
    """Infer the sampling frequency of a datetime series.

    Parameters
    ----------
    ds: pd.Series
        Datetime series to infer the frequency for.

    Returns
    -------
    Optional[str]
        Pandas offset alias if detected, otherwise ``None``.
    """

    if ds.empty:
        return None
    freq = pd.infer_freq(ds)
    if freq is not None:
        return freq
    # Fallback: compute mode of differences
    deltas = ds.sort_values().diff().dropna().value_counts()
    if deltas.empty:
        return None
    return pd.tseries.frequencies.to_offset(deltas.idxmax()).freqstr


def build_fourier_series(
    t: np.ndarray,
    period: float,
    order: int,
) -> np.ndarray:
    """Create Fourier series matrix for seasonality terms."""

    series = []
    for i in range(1, order + 1):
        angle = 2 * np.pi * i * t / period
        series.append(np.sin(angle))
        series.append(np.cos(angle))
    if not series:
        return np.zeros((len(t), 0))
    return np.column_stack(series)


def rolling_zscore(values: Sequence[float], window: int) -> np.ndarray:
    """Compute rolling z-score for change detection."""

    arr = np.asarray(values, dtype=float)
    if window <= 1 or arr.size == 0:
        return np.zeros_like(arr)
    means = pd.Series(arr).rolling(window=window, center=True).mean().to_numpy()
    stds = pd.Series(arr).rolling(window=window, center=True).std(ddof=0).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (arr - means) / stds
    z[np.isnan(z)] = 0.0
    return z


def select_changepoints(y: Sequence[float], n_changepoints: int) -> ChangepointResult:
    """Select changepoints using rolling z-score on second differences."""

    arr = np.asarray(y, dtype=float)
    if arr.size == 0 or n_changepoints <= 0:
        return ChangepointResult([], [])
    # second differences emphasise curvature
    second_diff = np.zeros_like(arr)
    if arr.size > 2:
        second_diff[2:] = np.abs(arr[2:] - 2 * arr[1:-1] + arr[:-2])
    z_scores = rolling_zscore(second_diff, window=max(5, arr.size // 20))
    # Avoid edges
    candidate_indexes = np.arange(2, arr.size - 2)
    if candidate_indexes.size == 0:
        return ChangepointResult([], [])
    scores = z_scores[candidate_indexes]
    top = min(n_changepoints, candidate_indexes.size)
    if top == 0:
        return ChangepointResult([], [])
    idx = np.argpartition(scores, -top)[-top:]
    sorted_idx = candidate_indexes[np.argsort(candidate_indexes[idx])]
    return ChangepointResult(locations=sorted_idx.tolist(), indexes=sorted_idx.tolist())


def ensure_datetime(series: pd.Series) -> pd.Series:
    """Convert a series to datetime and drop invalid entries."""

    converted = pd.to_datetime(series, errors="coerce")
    return converted.dropna()


def safe_clip(values: np.ndarray, lower: Iterable[float], upper: Iterable[float]) -> np.ndarray:
    """Clip an array between lower and upper bounds provided element-wise."""

    return np.minimum(np.maximum(values, np.asarray(list(lower))), np.asarray(list(upper)))


METRIC_FUNCTIONS = {
    "mae": lambda y_true, y_pred: float(np.mean(np.abs(y_true - y_pred))),
    "mse": lambda y_true, y_pred: float(np.mean((y_true - y_pred) ** 2)),
    "rmse": lambda y_true, y_pred: float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    "mape": lambda y_true, y_pred: float(
        np.mean(np.where(y_true != 0, np.abs((y_true - y_pred) / y_true), 0.0)) * 100
    ),
}


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the coefficient of determination."""

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0
