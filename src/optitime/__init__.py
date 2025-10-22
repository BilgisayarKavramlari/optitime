"""Public interface for the OptiProphet library."""

from .model import OptiProphet
from .exceptions import (
    DataValidationError,
    ForecastQualityError,
    ModelNotFitError,
    OptiProphetError,
)

__all__ = [
    "OptiProphet",
    "OptiProphetError",
    "DataValidationError",
    "ForecastQualityError",
    "ModelNotFitError",
]
