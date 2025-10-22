"""Public interface for the OptiProphet library."""

from .model import BACKTEST_STRATEGIES, OptiProphet
from .explainability import (
    AVAILABLE_EXPLANATION_APPROACHES,
    ExplanationConfig,
    ExplainabilityEngine,
)
from .exceptions import (
    DataValidationError,
    ForecastQualityError,
    ModelNotFitError,
    OptiProphetError,
)
from .datasets import Dataset, available_datasets, dataset_info, load_dataset

__all__ = [
    "OptiProphet",
    "BACKTEST_STRATEGIES",
    "OptiProphetError",
    "DataValidationError",
    "ForecastQualityError",
    "ModelNotFitError",
    "Dataset",
    "available_datasets",
    "dataset_info",
    "load_dataset",
    "AVAILABLE_EXPLANATION_APPROACHES",
    "ExplanationConfig",
    "ExplainabilityEngine",
]
