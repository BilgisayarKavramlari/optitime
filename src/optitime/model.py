"""OptiProphet - a Prophet-inspired forecasting library implemented in Python."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from .diagnostics import ForecastReport, compute_metrics, component_strengths, detect_outliers
from .exceptions import DataValidationError, ForecastQualityError, ModelNotFitError
from .utils import (
    ChangepointResult,
    build_fourier_series,
    infer_frequency,
    select_changepoints,
)


@dataclass
class FeatureSpec:
    """Definition of a single regression feature."""

    name: str
    component: str
    kind: str
    params: Dict[str, float | int | str]


@dataclass
class DesignMatrix:
    """Container for a constructed regression matrix."""

    matrix: np.ndarray
    index: pd.Index
    feature_names: List[str]
    specs: List[FeatureSpec]


DEFAULT_SEASONALITIES = {
    "yearly": {"period": 365.25, "order": 10},
    "weekly": {"period": 7.0, "order": 3},
    "daily": {"period": 1.0, "order": 3},
}

DEFAULT_COMPONENT_FLAGS = {
    "trend": True,
    "seasonality": True,
    "regressors": True,
    "autoregressive": True,
    "moving_average": True,
    "residual": True,
}

BACKTEST_STRATEGIES = ("expanding", "sliding", "anchored")


class OptiProphet:
    """A robust, feature-rich time series forecasting model inspired by Prophet."""

    def __init__(
        self,
        *,
        n_changepoints: int = 15,
        seasonalities: Optional[Dict[str, Dict[str, float | int]]] = None,
        seasonality_mode: str = "additive",
        regressors: Optional[Iterable[str]] = None,
        ar_order: int = 2,
        ma_order: int = 1,
        interval_width: float = 0.8,
        quantiles: Iterable[float] = (0.1, 0.9),
        min_history: int = 30,
        min_success_r2: float = 0.1,
        max_mape: Optional[float] = 35.0,
        historical_components: Optional[Mapping[str, bool]] = None,
        forecast_components: bool = True,
        default_backtest_strategy: str = "expanding",
        default_backtest_window: Optional[int] = None,
    ) -> None:
        self.n_changepoints = n_changepoints
        self.seasonality_mode = seasonality_mode
        self.seasonalities = seasonalities or DEFAULT_SEASONALITIES
        self.regressors = list(regressors) if regressors else []
        self.ar_order = int(ar_order)
        self.ma_order = int(ma_order)
        self.interval_width = float(interval_width)
        self.quantiles = sorted(float(q) for q in quantiles)
        for q in self.quantiles:
            if not 0.0 < q < 1.0:
                raise ValueError("Quantiles must be within the open interval (0, 1).")
        self.min_history = int(min_history)
        self.min_success_r2 = float(min_success_r2)
        self.max_mape = float(max_mape) if max_mape is not None else None
        self.default_backtest_strategy = default_backtest_strategy.lower()
        self.default_backtest_window = default_backtest_window

        self._historical_component_flags = DEFAULT_COMPONENT_FLAGS.copy()
        if historical_components:
            for key, value in historical_components.items():
                if key not in self._historical_component_flags:
                    raise ValueError(f"Unknown historical component '{key}'.")
                self._historical_component_flags[key] = bool(value)
        self.forecast_components = bool(forecast_components)

        if self.default_backtest_strategy not in BACKTEST_STRATEGIES:
            raise ValueError(
                f"Unsupported default_backtest_strategy '{self.default_backtest_strategy}'. "
                f"Choose from {BACKTEST_STRATEGIES}."
            )

        self.fitted_: bool = False
        self.coef_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = []
        self.feature_specs_: List[FeatureSpec] = []
        self.history_: Optional[pd.DataFrame] = None
        self.freq_: Optional[str] = None
        self._time_start: Optional[pd.Timestamp] = None
        self._time_scale: Optional[float] = None
        self._history_index: Optional[pd.Index] = None
        self._changepoints: ChangepointResult | None = None
        self._training_index: Optional[pd.Index] = None
        self._fitted_values: Optional[pd.Series] = None
        self._residuals: Optional[pd.Series] = None
        self._residual_quantiles: Dict[float, float] = {}
        self._residual_scale: float = 0.0
        self._stored_feature_matrix: Optional[np.ndarray] = None
        self.report_: Optional[ForecastReport] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "OptiProphet":
        """Fit the forecasting model to a dataset.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with at least ``ds`` and ``y`` columns.
        """

        if not isinstance(df, pd.DataFrame):
            raise DataValidationError("Input data must be a pandas DataFrame.")
        if not {"ds", "y"}.issubset(df.columns):
            raise DataValidationError("DataFrame must contain 'ds' and 'y' columns.")
        if df.empty:
            raise DataValidationError("Cannot fit model on an empty dataset.")

        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        if df["ds"].isna().any():
            raise DataValidationError("Column 'ds' contains invalid datetime values.")
        df = df.sort_values("ds").drop_duplicates(subset="ds")
        df = df.set_index("ds")

        if df["y"].isna().all():
            raise DataValidationError("Target column 'y' is entirely missing.")

        df["y"] = df["y"].interpolate(limit_direction="both")
        if df["y"].isna().any():
            raise DataValidationError("Target column 'y' cannot contain NaN values after interpolation.")

        if len(df) < self.min_history:
            raise DataValidationError(
                f"Insufficient data points ({len(df)}). At least {self.min_history} observations are required."
            )

        self.freq_ = infer_frequency(df.index.to_series())
        if self.freq_ is not None:
            full_range = pd.date_range(df.index.min(), df.index.max(), freq=self.freq_)
            df = df.reindex(full_range)
            df["y"] = df["y"].interpolate(limit_direction="both")
        df = df.ffill().bfill()

        missing_regressors = [col for col in self.regressors if col not in df.columns]
        if missing_regressors:
            raise DataValidationError(
                f"Missing required regressor columns: {', '.join(missing_regressors)}"
            )

        self.history_ = df.copy()
        self._history_index = df.index
        self._time_start = df.index.min()
        span = (df.index.max() - self._time_start).total_seconds()
        self._time_scale = span if span != 0 else 1.0

        self._changepoints = select_changepoints(df["y"].to_numpy(), self.n_changepoints)

        # Initial design without MA features to estimate residuals
        initial_design = self._build_design_matrix(df, residuals=None, store_specs=False)
        if initial_design.matrix.size == 0:
            raise DataValidationError("Feature matrix is empty. Check data and configuration.")
        y_target = df.loc[initial_design.index, "y"].to_numpy()
        coef_init = self._solve_linear_system(initial_design.matrix, y_target)
        fitted_init = initial_design.matrix @ coef_init
        fitted_series = pd.Series(fitted_init, index=initial_design.index)
        residuals_series = (df["y"] - fitted_series).fillna(0.0)

        final_design = self._build_design_matrix(df, residuals=residuals_series, store_specs=True)
        if final_design.matrix.size == 0:
            raise DataValidationError("Unable to build a valid design matrix with the current configuration.")
        y_final = df.loc[final_design.index, "y"].to_numpy()
        coef_final = self._solve_linear_system(final_design.matrix, y_final)

        self.coef_ = coef_final
        self.feature_names_ = final_design.feature_names
        self.feature_specs_ = final_design.specs
        self._training_index = final_design.index
        fitted_values = final_design.matrix @ coef_final
        self._fitted_values = pd.Series(fitted_values, index=final_design.index)
        self._residuals = df.loc[final_design.index, "y"] - self._fitted_values
        if not self._residuals.empty:
            residual_array = self._residuals.to_numpy()
            self._residual_quantiles = {
                q: float(np.quantile(residual_array, q))
                for q in self.quantiles
            }
            self._residual_scale = float(np.std(residual_array))
        else:
            self._residual_quantiles = {q: 0.0 for q in self.quantiles}
            self._residual_scale = 0.0

        self.fitted_ = True

        self.report_ = self._generate_report(df)
        if self.report_.metrics.get("r2", 0.0) < self.min_success_r2:
            raise ForecastQualityError(
                f"Model fit quality insufficient (R2={self.report_.metrics['r2']:.3f} < {self.min_success_r2})."
            )
        if self.max_mape is not None and self.report_.metrics.get("mape", 0.0) > self.max_mape:
            raise ForecastQualityError(
                f"Forecast error too high (MAPE={self.report_.metrics['mape']:.2f}% > {self.max_mape}%)."
            )

        return self

    # ------------------------------------------------------------------
    # Prediction utilities
    # ------------------------------------------------------------------
    def make_future_dataframe(
        self,
        periods: int,
        freq: Optional[str] = None,
        include_history: bool = False,
    ) -> pd.DataFrame:
        """Create a future dataframe similar to Prophet."""

        if not self.fitted_ or self.history_ is None:
            raise ModelNotFitError("Model must be fitted before building future dataframe.")
        if periods < 0:
            raise ValueError("Periods must be non-negative.")
        freq = freq or self.freq_
        if freq is None:
            raise ValueError("Frequency could not be inferred from training data.")
        last_date = self.history_.index.max()
        future_index = pd.date_range(last_date, periods=periods + 1, freq=freq)[1:]
        df = pd.DataFrame({"ds": future_index})
        if include_history:
            history_df = self.history_.reset_index().rename(columns={"index": "ds"})
            df = pd.concat([history_df[["ds"]], df], ignore_index=True)
        return df

    def predict(
        self,
        future: Optional[pd.DataFrame] = None,
        *,
        include_history: bool = True,
        backcast: bool = False,
        include_components: Optional[bool] = None,
        component_overrides: Optional[Mapping[str, bool]] = None,
        include_uncertainty: bool = True,
        quantile_subset: Optional[Iterable[float]] = None,
    ) -> pd.DataFrame:
        """Generate forecasts for future (and optionally historical) timestamps."""

        if not self.fitted_ or self.history_ is None or self.coef_ is None:
            raise ModelNotFitError("Model must be fitted before calling predict().")

        if future is None:
            future = pd.DataFrame(columns=["ds"])

        if "ds" not in future.columns:
            raise DataValidationError("Future dataframe must contain 'ds' column.")

        future_df = future.copy()
        future_df["ds"] = pd.to_datetime(future_df["ds"], errors="coerce")
        if future_df["ds"].isna().any():
            raise DataValidationError("Future dataframe contains invalid datestamps.")
        future_df = future_df.drop_duplicates(subset="ds").set_index("ds").sort_index()

        required_regressors = set(self.regressors)
        missing = required_regressors.difference(future_df.columns)
        if missing:
            raise DataValidationError(
                f"Future dataframe is missing regressors: {', '.join(sorted(missing))}"
            )

        component_flags = self._resolve_component_flags(
            base=DEFAULT_COMPONENT_FLAGS,
            master_switch=self.forecast_components,
        )
        if include_components is not None:
            component_flags = self._resolve_component_flags(
                base=component_flags,
                master_switch=include_components,
            )
        if component_overrides:
            component_flags = self._resolve_component_flags(
                base=component_flags,
                overrides=component_overrides,
            )

        if quantile_subset is not None:
            quantiles = []
            for q in quantile_subset:
                q_float = float(q)
                if not any(abs(q_float - existing) < 1e-9 for existing in self.quantiles):
                    raise ValueError(
                        f"Quantile {q_float} is not configured. Available quantiles: {self.quantiles}"
                    )
                quantiles.append(q_float)
            quantiles = sorted(quantiles)
        else:
            quantiles = list(self.quantiles)

        if include_history or backcast:
            base_history = self._compose_history_predictions(
                include_backcast=backcast,
                component_flags=component_flags,
                include_uncertainty=include_uncertainty,
                quantiles=quantiles,
            )
        else:
            base_history = pd.DataFrame()

        horizon_df = future_df.loc[future_df.index.difference(self.history_.index)]
        predictions = self._forecast_horizon(
            horizon_df,
            component_flags=component_flags,
            include_uncertainty=include_uncertainty,
            quantiles=quantiles,
        )

        result_frames = []
        if include_history:
            result_frames.append(base_history)
        if not horizon_df.empty:
            result_frames.append(predictions)
        if not result_frames:
            return base_history
        result = pd.concat(result_frames).sort_values("ds").reset_index(drop=True)
        return result

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------
    def backtest(
        self,
        horizon: int,
        step: int = 1,
        *,
        strategy: Optional[str] = None,
        window: Optional[int] = None,
        include_components: Optional[bool] = None,
        component_overrides: Optional[Mapping[str, bool]] = None,
        include_uncertainty: bool = True,
        quantile_subset: Optional[Iterable[float]] = None,
    ) -> pd.DataFrame:
        """Perform configurable backtesting with multiple re-training strategies."""

        if not self.fitted_ or self.history_ is None:
            raise ModelNotFitError("Model must be fitted before running backtest().")
        if horizon <= 0:
            raise ValueError("Horizon must be positive.")
        if step <= 0:
            raise ValueError("Step must be positive.")

        strategy_name = (strategy or self.default_backtest_strategy).lower()
        valid_strategies = set(BACKTEST_STRATEGIES)
        if strategy_name not in valid_strategies:
            raise ValueError(
                f"Unsupported backtest strategy '{strategy_name}'. Choose from {sorted(valid_strategies)}."
            )

        if quantile_subset is not None:
            quantiles = []
            for q in quantile_subset:
                q_float = float(q)
                if not any(abs(q_float - existing) < 1e-9 for existing in self.quantiles):
                    raise ValueError(
                        f"Quantile {q_float} is not configured. Available quantiles: {self.quantiles}"
                    )
                quantiles.append(q_float)
            quantiles = sorted(quantiles)
        else:
            quantiles = list(self.quantiles)

        df = self.history_.reset_index().rename(columns={"index": "ds"})
        results = []
        for start in range(self.min_history, len(df) - horizon, step):
            if strategy_name == "expanding":
                train_slice = df.iloc[: start + 1]
            elif strategy_name == "sliding":
                window_size = window or self.default_backtest_window or self.min_history
                if window_size < self.min_history:
                    window_size = self.min_history
                start_idx = max(0, start + 1 - window_size)
                train_slice = df.iloc[start_idx : start + 1]
            else:  # anchored
                window_size = window or self.default_backtest_window or self.min_history
                window_size = max(window_size, self.min_history)
                train_slice = df.iloc[:window_size]
                if len(train_slice) < window_size:
                    continue

            test_slice = df.iloc[start + 1 : start + 1 + horizon]
            model = self._clone()
            try:
                model.fit(train_slice)
            except ForecastQualityError:
                continue
            forecast = model.predict(
                test_slice[["ds"]],
                include_history=False,
                include_components=include_components,
                component_overrides=component_overrides,
                include_uncertainty=include_uncertainty,
                quantile_subset=quantiles,
            )
            merged = test_slice.merge(forecast, on="ds", how="left")
            if merged["yhat"].isna().any():
                continue
            metrics = compute_metrics(merged["y"].to_numpy(), merged["yhat"].to_numpy())
            metrics.update(
                {
                    "start": merged["ds"].min(),
                    "end": merged["ds"].max(),
                    "strategy": strategy_name,
                    "train_size": len(train_slice),
                }
            )
            results.append(metrics)
        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Analysis utilities
    # ------------------------------------------------------------------
    def history_components(
        self,
        *,
        include_components: Optional[bool] = None,
        component_overrides: Optional[Mapping[str, bool]] = None,
        include_uncertainty: bool = True,
        quantile_subset: Optional[Iterable[float]] = None,
    ) -> pd.DataFrame:
        """Return component decomposition for the training data."""

        if not self.fitted_ or self._fitted_values is None or self._residuals is None:
            raise ModelNotFitError("Model must be fitted before retrieving components.")

        if self._training_index is None:
            raise ModelNotFitError("Training index is missing; refit the model.")

        component_flags = self._resolve_component_flags(
            base=self._historical_component_flags,
        )
        if include_components is not None:
            component_flags = self._resolve_component_flags(
                base=component_flags,
                master_switch=include_components,
            )
        if component_overrides:
            component_flags = self._resolve_component_flags(
                base=component_flags,
                overrides=component_overrides,
            )

        if quantile_subset is not None:
            quantiles = []
            for q in quantile_subset:
                q_float = float(q)
                if not any(abs(q_float - existing) < 1e-9 for existing in self.quantiles):
                    raise ValueError(
                        f"Quantile {q_float} is not configured. Available quantiles: {self.quantiles}"
                    )
                quantiles.append(q_float)
            quantiles = sorted(quantiles)
        else:
            quantiles = list(self.quantiles)

        components = self._compute_component_contributions(
            index=self._training_index, feature_matrix=self._fitted_feature_matrix()
        )
        df = pd.DataFrame({"ds": self._training_index})
        for key, values in components.items():
            if component_flags.get(key, True):
                df[key] = values
        df["yhat"] = self._fitted_values.to_numpy()
        if include_uncertainty:
            quantile_columns = self._quantile_forecasts(df["yhat"].to_numpy(), quantiles=quantiles)
            for label, values in quantile_columns.items():
                df[label] = values
            lower, upper = self._confidence_intervals(df["yhat"].to_numpy())
            df["yhat_lower"] = lower
            df["yhat_upper"] = upper
        if component_flags.get("residual", True):
            df["residual"] = self._residuals.to_numpy()
        return df

    def report(self) -> Dict[str, object]:
        """Return the diagnostic report as a dictionary."""

        if self.report_ is None:
            raise ModelNotFitError("Model diagnostics are available only after fitting.")
        return self.report_.to_dict()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _solve_linear_system(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return coef

    def _clone(self) -> "OptiProphet":
        clone = OptiProphet(
            n_changepoints=self.n_changepoints,
            seasonalities=self.seasonalities,
            seasonality_mode=self.seasonality_mode,
            regressors=self.regressors,
            ar_order=self.ar_order,
            ma_order=self.ma_order,
            interval_width=self.interval_width,
            quantiles=self.quantiles,
            min_history=self.min_history,
            min_success_r2=self.min_success_r2,
            max_mape=self.max_mape,
            historical_components=self._historical_component_flags,
            forecast_components=self.forecast_components,
            default_backtest_strategy=self.default_backtest_strategy,
            default_backtest_window=self.default_backtest_window,
        )
        return clone

    def _resolve_component_flags(
        self,
        *,
        base: Mapping[str, bool],
        overrides: Optional[Mapping[str, bool]] = None,
        master_switch: Optional[bool] = None,
    ) -> Dict[str, bool]:
        flags = {key: bool(value) for key, value in base.items()}
        if master_switch is not None:
            flags = {key: bool(master_switch) and value for key, value in flags.items()}
        if overrides:
            for key, value in overrides.items():
                if key not in flags:
                    raise ValueError(f"Unknown component '{key}'.")
                flags[key] = bool(value)
        return flags

    def _compute_time_normalised(self, index: pd.Index) -> np.ndarray:
        if self._time_start is None or self._time_scale is None:
            raise ModelNotFitError("Model must be fitted before computing time values.")
        elapsed = (index - self._time_start).total_seconds()
        return elapsed / self._time_scale

    def _compute_time_days(self, index: pd.Index) -> np.ndarray:
        if self._time_start is None:
            raise ModelNotFitError("Model must be fitted before computing time values.")
        elapsed = (index - self._time_start).total_seconds()
        return elapsed / (24 * 3600)

    def _build_design_matrix(
        self,
        df: pd.DataFrame,
        *,
        residuals: Optional[pd.Series],
        store_specs: bool,
    ) -> DesignMatrix:
        index = df.index
        t_norm = self._compute_time_normalised(index)
        t_days = self._compute_time_days(index)

        columns: List[np.ndarray] = []
        names: List[str] = []
        specs: List[FeatureSpec] = []

        def add_column(values: np.ndarray, name: str, component: str, kind: str, params: Dict[str, object]):
            columns.append(values.astype(float))
            names.append(name)
            specs.append(FeatureSpec(name=name, component=component, kind=kind, params=dict(params)))

        add_column(np.ones(len(index)), "intercept", "trend", "intercept", {})
        add_column(t_norm, "trend_time", "trend", "time", {})

        if self._changepoints and self._changepoints.indexes:
            for i, cp_idx in enumerate(self._changepoints.indexes):
                cp_time = t_norm[cp_idx] if cp_idx < len(t_norm) else t_norm[-1]
                values = np.maximum(0.0, t_norm - cp_time)
                add_column(values, f"changepoint_{i}", "trend", "changepoint", {"cp_time": float(cp_time)})

        for name, config in self.seasonalities.items():
            period = float(config.get("period", 1.0))
            order = int(config.get("order", 1))
            matrix = build_fourier_series(t_days, period, order)
            for term_index in range(matrix.shape[1]):
                trig = "sin" if term_index % 2 == 0 else "cos"
                component_name = f"seasonality_{name}"
                add_column(
                    matrix[:, term_index],
                    f"{component_name}_{term_index}",
                    component_name,
                    "seasonality",
                    {
                        "period": period,
                        "order": order,
                        "harmonic": term_index // 2 + 1,
                        "trig": trig,
                    },
                )

        for reg in self.regressors:
            add_column(df[reg].to_numpy(), f"regressor_{reg}", "regressors", "regressor", {"column": reg})

        if self.ar_order > 0:
            for lag in range(1, self.ar_order + 1):
                shifted = df["y"].shift(lag).to_numpy()
                add_column(shifted, f"ar_lag_{lag}", "autoregressive", "ar", {"lag": lag})

        if residuals is not None and self.ma_order > 0:
            for lag in range(1, self.ma_order + 1):
                shifted_res = residuals.shift(lag).reindex(index).to_numpy()
                add_column(shifted_res, f"ma_lag_{lag}", "moving_average", "ma", {"lag": lag})

        feature_matrix = np.column_stack(columns) if columns else np.zeros((len(index), 0))
        mask = ~np.isnan(feature_matrix).any(axis=1)
        mask &= ~df["y"].isna().to_numpy()
        valid_matrix = feature_matrix[mask]
        valid_index = index[mask]

        if store_specs:
            self._stored_feature_matrix = valid_matrix

        selected_specs = [spec for spec in specs]
        return DesignMatrix(matrix=valid_matrix, index=valid_index, feature_names=names, specs=selected_specs)

    def _fitted_feature_matrix(self) -> np.ndarray:
        if self._stored_feature_matrix is None:
            raise ModelNotFitError("Internal feature matrix missing.")
        return self._stored_feature_matrix

    def _generate_report(self, df: pd.DataFrame) -> ForecastReport:
        fitted = self._fitted_values.to_numpy() if self._fitted_values is not None else np.array([])
        actual_index = self._fitted_values.index if self._fitted_values is not None else df.index
        actual = df.loc[actual_index, "y"].to_numpy()
        metrics = compute_metrics(actual, fitted)

        components = self._compute_component_contributions(
            index=actual_index, feature_matrix=self._fitted_feature_matrix()
        )
        strengths = component_strengths(components)
        changepoint_times = []
        if self._changepoints and self._changepoints.indexes:
            for idx in self._changepoints.indexes:
                if idx < len(self._history_index):
                    changepoint_times.append(self._history_index[idx])
        outliers = detect_outliers(actual_index.to_series(), actual - fitted)
        comments = []
        if not changepoint_times:
            comments.append("No significant changepoints detected.")
        if not outliers:
            comments.append("No prominent outliers detected.")
        return ForecastReport(metrics=metrics, component_strength=strengths, changepoints=changepoint_times, outliers=outliers, comments=comments)

    def _compose_history_predictions(
        self,
        include_backcast: bool,
        *,
        component_flags: Mapping[str, bool],
        include_uncertainty: bool,
        quantiles: Iterable[float],
    ) -> pd.DataFrame:
        history_index = self._training_index
        if history_index is None or self._fitted_values is None or self._residuals is None:
            raise ModelNotFitError("Model training artefacts missing.")

        contributions = self._compute_component_contributions(
            index=history_index, feature_matrix=self._fitted_feature_matrix()
        )
        df = pd.DataFrame({"ds": history_index})
        for name, values in contributions.items():
            if component_flags.get(name, True):
                df[name] = values
        fitted_values = self._fitted_values.to_numpy()
        df["yhat"] = fitted_values
        if include_uncertainty:
            quantile_columns = self._quantile_forecasts(fitted_values, quantiles=quantiles)
            for label, values in quantile_columns.items():
                df[label] = values
            lower, upper = self._confidence_intervals(fitted_values)
            df["yhat_lower"] = lower
            df["yhat_upper"] = upper
        if component_flags.get("residual", True):
            df["residual"] = self._residuals.to_numpy()
        if include_backcast and self.history_ is not None:
            df["y"] = self.history_.loc[history_index, "y"].to_numpy()
        return df

    def _forecast_horizon(
        self,
        future: pd.DataFrame,
        *,
        component_flags: Mapping[str, bool],
        include_uncertainty: bool,
        quantiles: Iterable[float],
    ) -> pd.DataFrame:
        if future.empty:
            return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])

        coeffs = self.coef_
        specs = self.feature_specs_
        max_ar = max([spec.params.get("lag", 0) for spec in specs if spec.kind == "ar"], default=0)
        max_ma = max([spec.params.get("lag", 0) for spec in specs if spec.kind == "ma"], default=0)

        history_values = []
        history_residuals = []
        if self._training_index is not None and self._fitted_values is not None:
            tail_index = self._training_index.sort_values()
            for lag in range(max_ar):
                idx = -lag - 1
                if abs(idx) <= len(tail_index):
                    ts = tail_index[idx]
                    history_values.append(self.history_.loc[ts, "y"])
            for lag in range(max_ma):
                idx = -lag - 1
                if abs(idx) <= len(tail_index):
                    ts = tail_index[idx]
                    history_residuals.append(self._residuals.loc[ts])
        history_values = list(reversed(history_values))
        history_residuals = list(reversed(history_residuals))

        predictions = []
        contributions_history: Dict[str, List[float]] = {}

        for timestamp, row in future.iterrows():
            feature_vector, component_values = self._feature_vector_for_timestamp(
                timestamp,
                row,
                history_values,
                history_residuals,
            )
            yhat = float(np.dot(feature_vector, coeffs))
            record = {
                "ds": timestamp,
                "yhat": yhat,
            }
            for key in ("trend", "seasonality", "regressors", "autoregressive", "moving_average"):
                if component_flags.get(key, True):
                    record[key] = component_values.get(key, 0.0)
            predictions.append(record)
            for key, value in component_values.items():
                contributions_history.setdefault(key, []).append(value)
            history_values.append(yhat)
            if len(history_values) > max_ar:
                history_values.pop(0)
            residual_estimate = 0.0
            if self._residuals is not None and not self._residuals.empty:
                residual_estimate = float(np.mean(np.abs(self._residuals)))
            history_residuals.append(residual_estimate)
            if len(history_residuals) > max_ma:
                history_residuals.pop(0)

        predictions_df = pd.DataFrame(predictions)
        if include_uncertainty and not predictions_df.empty:
            lower, upper = self._confidence_intervals(predictions_df["yhat"].to_numpy())
            predictions_df["yhat_lower"] = lower
            predictions_df["yhat_upper"] = upper
            quantile_columns = self._quantile_forecasts(
                predictions_df["yhat"].to_numpy(), quantiles=quantiles
            )
            for label, values in quantile_columns.items():
                predictions_df[label] = values
        if component_flags.get("residual", True):
            predictions_df["residual"] = np.nan
        return predictions_df

    def _feature_vector_for_timestamp(
        self,
        timestamp: pd.Timestamp,
        row: pd.Series,
        history_values: List[float],
        history_residuals: List[float],
    ) -> tuple[np.ndarray, Dict[str, float]]:
        coeffs = self.coef_
        specs = self.feature_specs_
        t_norm = self._compute_time_normalised(pd.Index([timestamp]))[0]
        t_days = self._compute_time_days(pd.Index([timestamp]))[0]
        vector = []
        component_totals: Dict[str, float] = {
            "trend": 0.0,
            "seasonality": 0.0,
            "regressors": 0.0,
            "autoregressive": 0.0,
            "moving_average": 0.0,
        }

        def get_ar_value(lag: int) -> float:
            if lag <= 0 or lag > len(history_values):
                return history_values[-1] if history_values else 0.0
            return history_values[-lag]

        def get_ma_value(lag: int) -> float:
            if lag <= 0 or lag > len(history_residuals):
                return history_residuals[-1] if history_residuals else 0.0
            return history_residuals[-lag]

        for idx, spec in enumerate(specs):
            if spec.kind == "intercept":
                value = 1.0
            elif spec.kind == "time":
                value = t_norm
            elif spec.kind == "changepoint":
                cp_time = spec.params.get("cp_time", 0.0)
                value = max(0.0, t_norm - cp_time)
            elif spec.kind == "seasonality":
                period = spec.params.get("period", 1.0)
                harmonic = spec.params.get("harmonic", 1)
                trig = spec.params.get("trig", "sin")
                angle = 2 * np.pi * harmonic * t_days / period
                value = np.sin(angle) if trig == "sin" else np.cos(angle)
            elif spec.kind == "regressor":
                column = spec.params.get("column")
                if column not in row:
                    raise DataValidationError(f"Regressor '{column}' missing for timestamp {timestamp}.")
                value = float(row[column])
            elif spec.kind == "ar":
                lag = int(spec.params.get("lag", 1))
                value = get_ar_value(lag)
            elif spec.kind == "ma":
                lag = int(spec.params.get("lag", 1))
                value = get_ma_value(lag)
            else:
                value = 0.0
            vector.append(value)
            component = spec.component
            coef = coeffs[idx] if coeffs is not None and idx < len(coeffs) else 0.0
            if component.startswith("seasonality"):
                component_totals["seasonality"] += value * coef
            elif component == "trend":
                component_totals["trend"] += value * coef
            else:
                component_totals[component] += value * coef

        return np.array(vector, dtype=float), component_totals

    def _confidence_intervals(self, yhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.quantiles:
            return yhat, yhat

        low_q = min(self.quantiles)
        high_q = max(self.quantiles)
        low_adj = self._residual_quantiles.get(low_q, 0.0)
        high_adj = self._residual_quantiles.get(high_q, 0.0)
        spread = self._residual_scale * self.interval_width
        lower = yhat + low_adj - spread
        upper = yhat + high_adj + spread
        return lower, upper

    def _quantile_forecasts(
        self,
        yhat: np.ndarray,
        *,
        quantiles: Optional[Iterable[float]] = None,
    ) -> Dict[str, np.ndarray]:
        quantile_columns: Dict[str, np.ndarray] = {}
        selected = self.quantiles if quantiles is None else list(quantiles)
        for q in selected:
            label = f"yhat_q{q:.2f}"
            adjustment = self._residual_quantiles.get(q, 0.0)
            quantile_columns[label] = yhat + adjustment
        return quantile_columns

    def _compute_component_contributions(
        self,
        *,
        index: pd.Index,
        feature_matrix: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        if self.coef_ is None:
            raise ModelNotFitError("Coefficients are unavailable before fitting.")
        component_values: Dict[str, List[float]] = {}
        for row_idx in range(feature_matrix.shape[0]):
            contributions: Dict[str, float] = {
                "trend": 0.0,
                "seasonality": 0.0,
                "regressors": 0.0,
                "autoregressive": 0.0,
                "moving_average": 0.0,
            }
            for col_idx, spec in enumerate(self.feature_specs_):
                component = spec.component
                coef = self.coef_[col_idx]
                value = feature_matrix[row_idx, col_idx]
                if component.startswith("seasonality"):
                    contributions["seasonality"] += value * coef
                elif component == "trend":
                    contributions["trend"] += value * coef
                else:
                    contributions[component] += value * coef
            for key, value in contributions.items():
                component_values.setdefault(key, []).append(value)
        return {key: np.array(values) for key, values in component_values.items()}
