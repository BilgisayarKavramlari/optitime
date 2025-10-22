# OptiProphet

OptiProphet is a from-scratch, Prophet-inspired forecasting library written entirely in Python. It blends classic trend/seasonality decomposition with autoregressive and moving-average enrichments, dynamic changepoint detection, and extensive diagnostics so you can trust the signals hidden inside your time series. The project is engineered with upcoming PyPI distribution in mind.

## Why OptiProphet?

| Capability | What it gives you |
| --- | --- |
| **Prophet-style trend & seasonality** | Piecewise-linear trend with automatic changepoints, additive seasonalities via Fourier terms. |
| **AR/MA enrichments** | Captures short-term autocorrelation by reusing recent values and residual shocks. |
| **External regressors** | Inject business covariates and time-varying interactions directly into the model. |
| **Uncertainty modelling** | Quantile-aware prediction intervals derived from in-sample dispersion. |
| **Backtesting & diagnostics** | Rolling-origin backtests, outlier surfacing, component strength analysis, and performance metrics. |
| **Robustness toolkit** | Automatic interpolation for sparse data, changepoint detection, and outlier reporting to survive structural breaks. |

All logic is implemented without relying on Prophet or other probabilistic frameworks—only `numpy` and `pandas` are required.

## Installation

The project uses a standard `pyproject.toml` layout so it is ready for PyPI packaging.

```bash
pip install optitime-prophet  # once published
```

For local development:

```bash
git clone https://github.com/optiwisdom/optitime.git
cd optitime
pip install -e .
```

## Quick start

```python
import pandas as pd
from optitime import OptiProphet

# Load a time series with columns ds (timestamp) and y (value)
data = pd.read_csv("sales.csv", parse_dates=["ds"])

model = OptiProphet(
    n_changepoints=20,
    ar_order=3,
    ma_order=2,
    regressors=["promo", "price_index"],
)

model.fit(data)

# Forecast 30 periods into the future
future = model.make_future_dataframe(periods=30, include_history=False)
future["promo"] = 0  # supply regressors for the horizon
future["price_index"] = 1.0
forecast = model.predict(future)
print(forecast.tail())

# The returned frame includes component contributions plus quantile columns
# (e.g. `yhat_q0.10`, `yhat_q0.90`) alongside `yhat_lower`/`yhat_upper` bounds.

# Inspect decomposition of the training history
components = model.history_components()
print(components.head())

# Evaluate rolling-origin backtest
cv = model.backtest(horizon=14, step=7)
print(cv.describe())

# Fetch detailed diagnostics & quality report
print(model.report())
```

## Feature highlights

- **Bidirectional insight**: `history_components()` exposes historical trend, seasonality, residual, and regressor effects, while `predict()` projects the same structure into the future.
- **Backtest ready**: `backtest()` re-fits the model on expanding windows to quantify generalisation metrics (MAE, RMSE, MAPE, R²) on rolling horizons.
- **Error-aware**: Empty frames, missing columns, low sample counts, or under-performing fits surface as descriptive exceptions such as `DataValidationError` or `ForecastQualityError`.
- **Structural resilience**: The changepoint detector uses rolling z-scores on second derivatives to adapt to trend shifts. Large residual spikes are flagged as outliers in the diagnostic report.
- **Quantile intervals**: Forecasts include configurable lower/upper bounds (`interval_width` or explicit `quantiles`) using in-sample dispersion, while dedicated columns such as `yhat_q0.10` and `yhat_q0.90` expose raw quantile estimates for downstream pipelines.
- **Autoregression & shocks**: Short-term dynamics are captured with configurable AR and MA lags, automatically rolling forward during forecasting.
- **External signals**: Provide arbitrary regressors during both fit and predict phases to blend business drivers with the statistical core.

## Error handling

OptiProphet raises explicit errors for problematic scenarios:

- `DataValidationError`: empty dataframes, missing columns, or NaN-heavy features.
- `ModelNotFitError`: methods invoked before `fit()` completes.
- `ForecastQualityError`: triggered when R² drops below the configured threshold or the MAPE exceeds the acceptable ceiling.

These exceptions include actionable messages so automated pipelines (including GitHub Actions or CI) can fail fast without leaving stale artefacts.

## Preparing for PyPI

1. Update `pyproject.toml` metadata if publishing under a different namespace.
2. Create a source distribution and wheel: `python -m build`.
3. Upload with `twine upload dist/*` once credentials are configured.

## Development roadmap

- Bayesian residual bootstrapping for richer predictive distributions.
- Optional Torch/NumPyro backends for transfer learning under sparse conditions.
- Expanded diagnostics dashboard (streamlit) for interactive exploration.

## Contributing & PR workflow

If you plan to contribute back via pull requests, make sure your local clone
knows where to send them. Configure the Git remote once after cloning:

```bash
git remote add origin https://github.com/optiwisdom/optitime.git
git fetch origin
```

The `make_pr` helper used in this project depends on the remote named
`origin`; without it the PR tooling will raise a “failed to create new pr”
error. After the remote is configured you can run the usual contribution
pipeline:

```bash
python -m compileall src
git status
git commit -am "Describe your change"
make_pr
```

This ensures the automation has enough repository context to open a pull
request successfully.

## License

Released under the MIT License.
