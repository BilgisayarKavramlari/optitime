"""Public interface for the OptiProphet library."""

from .model import BACKTEST_STRATEGIES, OptiProphet
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
]
