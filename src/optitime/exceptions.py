"""Custom exceptions for the OptiProphet library."""

from __future__ import annotations


class OptiProphetError(Exception):
    """Base class for all OptiProphet related errors."""


class DataValidationError(OptiProphetError):
    """Raised when provided data is not valid for modelling."""


class ModelNotFitError(OptiProphetError):
    """Raised when an operation requires a fitted model."""


class ForecastQualityError(OptiProphetError):
    """Raised when generated forecasts do not satisfy quality checks."""
