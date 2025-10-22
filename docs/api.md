# OptiProphet API Overview

This document summarises the public surface of the OptiProphet library, a
Prophet-inspired engine that carries forward the forecasting and diagnostic
principles pioneered inside OptiWisdom's OptiScorer programme. Everything is
implemented in pure Python and packaged for straightforward publication to
PyPI.

## `optitime.OptiProphet`

### Constructor (`__init__`)

```python
OptiProphet(
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
)
```

Key options:

- `historical_components`: default visibility for trend, seasonality, regressor,
  autoregressive, moving-average, and residual columns returned by
  `history_components()`.
- `forecast_components`: whether `predict()` includes component columns by
  default.
- `default_backtest_strategy` / `default_backtest_window`: baseline behaviour for
  `backtest()` when no strategy or window is supplied (`"expanding"`,
  `"sliding"`, or `"anchored"`).

### `fit(df: pd.DataFrame) -> OptiProphet`

Validates that the input frame contains `ds` (timestamp) and `y` (target)
columns, performs interpolation for sparse spans, engineers features, and solves
for the regression coefficients. Extensive validation mirrors OptiScorer's
quality thresholds and raises descriptive exceptions on failure.

### `make_future_dataframe(periods: int, freq: Optional[str] = None, include_history: bool = False)`

Creates a Prophet-style future dataframe. When `include_history=True` the
returned frame contains the historical timestamps followed by the requested
horizon.

### `predict(...)`

```python
predict(
    future: Optional[pd.DataFrame] = None,
    *,
    include_history: bool = True,
    backcast: bool = False,
    include_components: Optional[bool] = None,
    component_overrides: Optional[Mapping[str, bool]] = None,
    include_uncertainty: bool = True,
    quantile_subset: Optional[Iterable[float]] = None,
)
```

Important arguments:

- `include_components`: toggles all component columns on or off at once.
- `component_overrides`: selectively hide components, e.g.
  `{"seasonality": False}`.
- `include_uncertainty`: controls whether `yhat_lower`, `yhat_upper`, and
  quantile columns (`yhat_q0.10`, etc.) are emitted.
- `quantile_subset`: return only a chosen subset of the configured quantiles.

### `history_components(...)`

```python
history_components(
    *,
    include_components: Optional[bool] = None,
    component_overrides: Optional[Mapping[str, bool]] = None,
    include_uncertainty: bool = True,
    quantile_subset: Optional[Iterable[float]] = None,
)
```

Returns component contributions, fitted values, and (optionally) residuals for
the training window. Parameters mirror `predict()`, but default visibility is
driven by the constructor's `historical_components` setting.

### `backtest(...)`

```python
backtest(
    horizon: int,
    step: int = 1,
    *,
    strategy: Optional[str] = None,
    window: Optional[int] = None,
    include_components: Optional[bool] = None,
    component_overrides: Optional[Mapping[str, bool]] = None,
    include_uncertainty: bool = True,
    quantile_subset: Optional[Iterable[float]] = None,
)
```

- `strategy`: one of the retraining approaches listed in
  `optitime.BACKTEST_STRATEGIES` (`"expanding"`, `"sliding"`, `"anchored"`).
- `window`: sliding/anchored sample size when applicable.
- Formatting parameters (`include_components`, `component_overrides`,
  `include_uncertainty`, `quantile_subset`) work identically to `predict()`.

The returned dataframe includes accuracy metrics (`mae`, `rmse`, `mape`, `r2`),
the evaluation interval (`start`, `end`), the chosen strategy, and the size of
the training sample.

### `report() -> ForecastReport`

Exposes the OptiScorer-style diagnostic report with model metrics, component
strength estimates, detected changepoints, and outlier summaries. Convert to a
plain dictionary via `model.report_.to_dict()` if required by downstream tools.

## Dataset utilities

- `optitime.datasets.available_datasets()` – list the bundled dataset
  identifiers.
- `optitime.datasets.dataset_info(name)` – return metadata (description,
  frequency, start/end).
- `optitime.datasets.load_dataset(name)` – load the CSV resource as a sorted
  `pandas.DataFrame`.

## Public constants

- `optitime.BACKTEST_STRATEGIES` – tuple of supported backtest strategy names.

Worked examples live in `README.md`, `tests/run_sales_example.py`, and
`tests/run_airlines_visuals.py` for quick reference.
