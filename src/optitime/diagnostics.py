"""Diagnostics and reporting utilities for OptiProphet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .utils import METRIC_FUNCTIONS, r2_score


@dataclass
class ForecastReport:
    """Container for diagnostic report entries."""

    metrics: Dict[str, float]
    component_strength: Dict[str, float]
    changepoints: List[pd.Timestamp]
    outliers: List[Tuple[pd.Timestamp, float]]
    comments: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "metrics": self.metrics,
            "component_strength": self.component_strength,
            "changepoints": [cp.isoformat() for cp in self.changepoints],
            "outliers": [
                {"timestamp": ts.isoformat(), "residual": float(res)}
                for ts, res in self.outliers
            ],
            "comments": self.comments,
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute a range of error metrics."""

    metrics = {name: func(y_true, y_pred) for name, func in METRIC_FUNCTIONS.items()}
    metrics["r2"] = r2_score(y_true, y_pred)
    return metrics


def component_strengths(components: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Estimate component strength based on variance contributions."""

    strengths = {}
    total = 0.0
    for values in components.values():
        total += float(np.var(values))
    if total == 0:
        return {key: 0.0 for key in components}
    for key, values in components.items():
        strengths[key] = float(np.var(values) / total)
    return strengths


def detect_outliers(timestamps: pd.Series, residuals: np.ndarray, threshold: float = 3.0):
    """Identify outliers using residual z-scores."""

    if residuals.size == 0:
        return []
    z = (residuals - np.mean(residuals)) / (np.std(residuals) or 1)
    mask = np.abs(z) >= threshold
    return list(zip(timestamps[mask], residuals[mask]))
