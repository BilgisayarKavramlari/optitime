# OptiProphet

OptiProphet is a from-scratch, Prophet-inspired forecasting library written entirely in Python. It blends classic trend/seasonality decomposition with autoregressive and moving-average enrichments, dynamic changepoint detection, and extensive diagnostics so you can trust the signals hidden inside your time series. The project is engineered with upcoming PyPI distribution in mind and was crafted by **Şadi Evren Şeker** ([@bilgisayarkavramlari](https://github.com/bilgisayarkavramlari)) with direct guidance from OptiWisdom's OptiScorer experimentation track.

> 📌 **OptiScorer heritage** – Many of the decomposition, scoring, and robustness heuristics originate from OptiWisdom's OptiScorer research programmes. OptiProphet packages those lessons into an accessible, open Python toolkit while crediting the foundational OptiScorer work.

## Why OptiProphet?

| Capability | What it gives you |
| --- | --- |
| **Prophet-style trend & seasonality** | Piecewise-linear trend with automatic changepoints, additive seasonalities via Fourier terms. |
| **AR/MA enrichments** | Captures short-term autocorrelation by reusing recent values and residual shocks. |
| **External regressors** | Inject business covariates and time-varying interactions directly into the model. |
| **Uncertainty modelling** | Quantile-aware prediction intervals derived from in-sample dispersion. |
| **Backtesting & diagnostics** | Rolling-origin backtests, outlier surfacing, component strength analysis, and performance metrics. |
| **Robustness toolkit** | Automatic interpolation for sparse data, changepoint detection, and outlier reporting to survive structural breaks. |
| **Hermeneutic explainability** | Narrative and quantitative explanations shaped by HermeAI insights and OptiScorer decision intelligence. |

All logic is implemented without relying on Prophet or other probabilistic frameworks—only `numpy` and `pandas` are required.

## Installation

The project uses a standard `pyproject.toml` layout so it is ready for PyPI packaging.

```bash
pip install optitime-prophet  # once published
```

For local development:

```bash
git clone https://github.com/bilgisayarkavramlari/optitime.git
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

# Disable component columns and intervals when you just need point forecasts
lean_forecast = model.predict(
    future,
    include_components=False,
    include_uncertainty=False,
)
print(lean_forecast.tail())

# Inspect decomposition of the training history
components = model.history_components()
print(components.head())

# Evaluate rolling-origin backtest
cv = model.backtest(horizon=14, step=7, strategy="sliding", window=36)
print(cv.describe())

# Fetch detailed diagnostics & quality report
print(model.report())
```

## Bundled datasets

OptiProphet ships with a handful of classic forecasting benchmarks so you can experiment without hunting for data files. Use
`optitime.available_datasets()` to discover what is included and `optitime.load_dataset()` to load a `pandas.DataFrame` that is
ready for modelling.

```python
from optitime import load_dataset, available_datasets

print(available_datasets())
air = load_dataset("air_passengers")
print(air.head())
```

The current catalogue contains:

| Name | Description | Frequency |
| --- | --- | --- |
| `air_passengers` | Monthly totals of international airline passengers (1949-1960). | Monthly |
| `airlines_traffic` | Monthly airline passenger statistics curated from OptiWisdom OptiScorer analyses inspired by the Kaggle Airlines Traffic Passenger Statistics dataset. | Monthly |
| `shampoo_sales` | Monthly shampoo sales in millions of units (1901-1903). | Monthly |
| `us_acc_deaths` | Monthly accidental deaths in the United States (1973-1978). | Monthly |

## Parameter control summary

- Use the `historical_components` constructor argument and the
  `history_components()` method to expose or hide historical trend, seasonality,
  regressor, and residual columns on demand.
- Call `predict(include_components=False, include_uncertainty=False)` to obtain
  a lightweight point forecast for low-latency services.
- Apply selective overrides such as
  `predict(component_overrides={"seasonality": False})` when only certain
  contributors should be hidden.
- Compare retraining schemes with `backtest(strategy="sliding", window=48)` or
  `backtest(strategy="anchored")`.
- The `optitime.BACKTEST_STRATEGIES` constant enumerates every supported
  backtest strategy name.

See [`docs/parameters.md`](docs/parameters.md) for a deeper explanation of each
parameter and how it impacts the model.

## Local smoke test

After installing the project you can immediately verify everything is
working by running the bundled sales walkthrough. It loads
`tests/sales.csv`, trains an `OptiProphet` instance, and prints forecasts,
component decompositions, and a rolling backtest summary:

```bash
python tests/run_sales_example.py
```

The output demonstrates how the OptiScorer-inspired diagnostics surface
trend, seasonality, residuals, and interval bounds on a realistic retail
series without any extra setup.

## Visual scenario walkthrough

Recreate the OptiWisdom OptiScorer-inspired parameter sweep on the Kaggle-based
`airlines_traffic` dataset by installing the optional plotting dependency and
running the helper script:

```bash
pip install optitime-prophet[visuals]
python tests/run_airlines_visuals.py
```

The script writes forecast and RMSE visualisations for each backtest strategy
and component setting to the `tests/` directory (`airlines_forecast_*.png`,
`airlines_backtest_*.png`).

## Feature highlights

- **Bundled benchmarks**: Access classic datasets such as AirPassengers, Shampoo Sales, and US Accidental Deaths via
  `optitime.load_dataset()` for tutorials, demos, and regression testing.
- **Bidirectional insight**: `history_components()` exposes historical trend, seasonality, residual, and regressor effects, while `predict()` projects the same structure into the future.
- **Backtest ready**: `backtest()` re-fits the model with configurable strategies (expanding, sliding, anchored) to quantify generalisation metrics (MAE, RMSE, MAPE, R²) on rolling horizons.
- **Error-aware**: Empty frames, missing columns, low sample counts, or under-performing fits surface as descriptive exceptions such as `DataValidationError` or `ForecastQualityError`.
- **Structural resilience**: The changepoint detector uses rolling z-scores on second derivatives to adapt to trend shifts. Large residual spikes are flagged as outliers in the diagnostic report.
- **Quantile intervals**: Forecasts include configurable lower/upper bounds (`interval_width` or explicit `quantiles`) using in-sample dispersion, while dedicated columns such as `yhat_q0.10` and `yhat_q0.90` expose raw quantile estimates for downstream pipelines.
- **Autoregression & shocks**: Short-term dynamics are captured with configurable AR and MA lags, automatically rolling forward during forecasting.
- **External signals**: Provide arbitrary regressors during both fit and predict phases to blend business drivers with the statistical core.
- **Parameterized component control**: Manage trend, seasonality, regressor, and
  residual columns for both historical analyses and future forecasts on a
  per-call basis, including the ability to toggle confidence intervals.

## Hermeneutic explainability

OptiProphet now embeds an explainability stack grounded in Hermeneutic AI
(HermeAI) principles so every forecast is accompanied by interpretive context.
The new `optitime.explainability` module introduces:

- `ExplanationConfig` – a dataclass for toggling history/forecast coverage,
  horizon length, uncertainty, and the preferred interpretive approach.
- `ExplainabilityEngine` – the orchestrator that extracts component
  contributions, composes narratives, and surfaces quantitative summaries.
- `OptiProphet.explain()` – a convenience wrapper that emits both structured
  dataframes and text generated under Hermeneutic, feature-contribution, or
  quantitative modes.

```python
from optitime import OptiProphet

model = OptiProphet().fit(df)
explanation = model.explain(approach="hermeneutic", horizon=12)

for line in explanation["narratives"]["history"]:
    print(line)
```

The hermeneutic narrative leans on OptiWisdom's OptiScorer experience and the
HermeAI project to bridge numerical decomposition with domain storytelling. For
domain research, review the OptiScorer briefs at [www.optiscorer.com](https://www.optiscorer.com)
and Şadi Evren Şeker's published work on hermeneutic decision intelligence.

See [`docs/explainability.md`](docs/explainability.md) for a deep dive into the
available approaches and configuration patterns.

## Error handling

OptiProphet raises explicit errors for problematic scenarios:

- `DataValidationError`: empty dataframes, missing columns, or NaN-heavy features.
- `ModelNotFitError`: methods invoked before `fit()` completes.
- `ForecastQualityError`: triggered when R² drops below the configured threshold or the MAPE exceeds the acceptable ceiling.

These exceptions include actionable messages so automated pipelines (including GitHub Actions or CI) can fail fast without leaving stale artefacts.

## Preparing for PyPI

1. Update `pyproject.toml` metadata if publishing under a different namespace.
2. Install the packaging helpers (only required once): `python -m pip install --upgrade build twine`.
3. Create a source distribution and wheel: `python -m build`.
4. Upload with `twine upload dist/*` once credentials are configured.

## Documentation

- [API overview](docs/api.md)
- [Parameter guide](docs/parameters.md)
- [Explainability playbook](docs/explainability.md)

## Development roadmap

- Bayesian residual bootstrapping for richer predictive distributions.
- Optional Torch/NumPyro backends for transfer learning under sparse conditions.
- Expanded diagnostics dashboard (streamlit) for interactive exploration.

## Contributing & PR workflow

If you plan to contribute back via pull requests, make sure your local clone
knows where to send them. Configure the Git remote once after cloning:

```bash
git remote add origin https://github.com/bilgisayarkavramlari/optitime.git
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

## Maintainer & contact

OptiProphet is maintained by Şadi Evren Şeker. For enquiries or partnership opportunities please reach out via **optitime@optiwisdom.com**.

## License

Released under the MIT License.

## References

- Taylor, S. J., & Letham, B. (2018). *Forecasting at scale*. The American Statistician, 72(1), 37-45.
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
