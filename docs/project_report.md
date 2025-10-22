# OptiProphet Project Report

## 1. Executive Summary
OptiProphet is a Prophet-inspired forecasting library implemented from scratch in Python. The project extends classical additive models with changepoint-aware trend detection, Fourier-based seasonalities, autoregressive/moving-average residual modelling, and Hermeneutic AI explainability. The work is stewarded by Şadi Evren Şeker and draws heavily on OptiWisdom's OptiScorer experimentation programme for robustness heuristics, interpretability patterns, and dataset curation.

<img src="https://optiwisdom.com/wp-content/uploads/2025/10/optiprophet.png" width=300 height=300>
## 2. System Architecture Overview
- **Package layout** – All runtime code lives under `src/optitime/` following the modern Python packaging layout so editable installs (`pip install -e .`) work seamlessly.
- **Core model (`model.py`)** – Hosts the `OptiProphet` estimator, feature engineering helpers, changepoint handling, forecasting, backtesting, diagnostics integration, and the Hermeneutic explainability bridge.
- **Support modules**:
  - `utils.py` – Frequency inference, Fourier design matrices, changepoint selection, rolling statistics, and helper dataclasses.
  - `diagnostics.py` – Structures the `ForecastReport`, aggregates metrics (MAE, RMSE, MAPE, R²), and surfaces outliers/component strengths.
  - `explainability.py` – Implements `ExplanationConfig`, `ExplainabilityEngine`, approach registry, and narrative generation inspired by HermeAI/OptiScorer.
  - `exceptions.py` – Domain-specific exception types (`DataValidationError`, `ForecastQualityError`, `ModelNotFitError`).
  - `datasets/` – Bundled CSV benchmarks plus loader utilities exposed through `optitime.load_dataset()` and `optitime.available_datasets()`.
- **Tests & walkthroughs** – Live under `tests/`, including dataset integration tests, smoke-test scripts, and visual scenario runners that emit PNG diagnostics.

<img src="https://optiwisdom.com/wp-content/uploads/2025/10/optiprophet-info-graphic.png" width=500 height=500>

## 3. Technology Stack & Rationale
- **Python 3.10+** – Chosen for native dataclasses, typing support, and ecosystem maturity for scientific computing.
- **NumPy** – Provides vectorised linear algebra primitives used for solving the regression systems and generating autoregressive lag matrices.
- **pandas** – Supplies the time-series friendly DataFrame abstraction, resampling/interpolation, and convenient joins for forecasts/backtests.
- **Optional Matplotlib (`visuals` extra)** – Powers the airline walkthrough plots generated in `tests/run_airlines_visuals.py`.
- **Packaging tools (`build`, `twine`)** – Required to produce wheels/sdists and publish to PyPI as documented in the README.

These dependencies keep the footprint light while replicating Prophet-style modelling without relying on probabilistic programming frameworks.

## 4. Data Assets
Four canonical datasets (AirPassengers, Airlines Traffic, Shampoo Sales, US Accidental Deaths) ship inside the package for experimentation. Additional synthetic retail sales data (`tests/sales.csv`) and Kaggle-inspired airlines statistics underpin the smoke tests and visuals. Loaders normalise columns to `ds`/`y`, ensuring compatibility with the modelling API.

## 5. Core Algorithms & Features
1. **Trend with changepoints** – `select_changepoints()` scans second-derivative shocks in the history to place changepoints, yielding piecewise-linear trend segments similar to Prophet's approach.【F:src/optitime/utils.py†L144-L245】 Reference: Taylor & Letham (2018), *Forecasting at Scale*.
2. **Seasonality via Fourier series** – `build_fourier_series()` emits sine/cosine bases for specified periods/orders so the design matrix captures periodic effects.【F:src/optitime/utils.py†L41-L142】 Reference: Harvey (1990), *Forecasting, Structural Time Series Models and the Kalman Filter*.
3. **Autoregressive & moving-average enrichments** – Configurable `ar_order`/`ma_order` arguments add lagged `y` values and residual shocks into the regression, following Box-Jenkins methodology.【F:src/optitime/model.py†L52-L118】 Reference: Box et al. (2015), *Time Series Analysis: Forecasting and Control*.
4. **External regressors** – Arbitrary covariates can be declared in the constructor and are incorporated during both fit and predict stages for domain-aware adjustments.【F:src/optitime/model.py†L87-L186】
5. **Quantile-aware uncertainty** – Forecast intervals derive from residual dispersion; explicit quantile columns (`yhat_q{quantile}`) accompany bounds for downstream calibration.【F:src/optitime/model.py†L530-L705】
6. **Historical decomposition** – `history_components()` surfaces trend, seasonality, regressor, autoregressive, moving-average, and residual contributions conditioned on configurable visibility flags.【F:src/optitime/model.py†L708-L870】
7. **Backtesting strategies** – `backtest()` supports expanding, sliding, and anchored windowing with parametrised horizon/step/window arguments to benchmark generalisation.【F:src/optitime/model.py†L402-L582】 References: Hyndman & Athanasopoulos (2021), *Forecasting: Principles and Practice*.
8. **Diagnostics** – Outlier detection, component strength scoring, and quality thresholds (R², MAPE) are reported through `ForecastReport` to enforce data health.【F:src/optitime/diagnostics.py†L1-L191】【F:src/optitime/model.py†L872-L1006】

## 6. Hermeneutic Explainability Module
The `ExplainabilityEngine` unifies three interpretability modes—`hermeneutic`, `contribution`, and `quantitative`—selected through `ExplanationConfig` or convenience parameters on `OptiProphet.explain()`.【F:src/optitime/explainability.py†L1-L320】【F:src/optitime/model.py†L1008-L1157】

- **Hermeneutic narratives** weave OptiScorer heuristics and HermeAI framing to contextualise historical trend/seasonality/residual behaviour and future projections.
- **Contribution summaries** quantify component-level impacts per timestamp, supporting decision logs.
- **Quantitative digests** aggregate metrics, uncertainty spreads, and changepoint effects for analysts.

Outputs include structured dictionaries with DataFrames and narrative text lists for both history and forecast sections. Users can toggle coverage, horizon, inclusion of uncertainty, and textual verbosity to align with governance or audit requirements.

## 7. API Surface & Parameterisation
Public entry points exported from `optitime/__init__.py` include `OptiProphet`, dataset helpers, explainability symbols, `BACKTEST_STRATEGIES`, and documentation references for discoverability.【F:src/optitime/__init__.py†L1-L120】

Detailed parameter behaviour is catalogued in `docs/api.md` and `docs/parameters.md`, covering constructor defaults, method signatures, and usage notes for toggling components, uncertainty, and backtest schemes. README quick-start sections demonstrate canonical flows, while docs/explainability.md describes the Hermeneutic module in depth.

## 8. Testing & Quality Assurance
- **Automated tests** – `tests/test_datasets.py` runs integration suites across every bundled dataset, validating fit/predict/backtest pipelines and explainability output structures.【F:tests/test_datasets.py†L1-L220】
- **Smoke tests** – `tests/run_sales_example.py` and `tests/run_airlines_visuals.py` execute end-to-end scenarios, the latter saving diagnostic charts that highlight parameter impacts on accuracy curves.【F:tests/run_sales_example.py†L1-L160】【F:tests/run_airlines_visuals.py†L1-L220】
- **Static checks** – `python -m compileall src tests` ensures syntax integrity across the codebase.

## 9. Packaging & Distribution Workflow
The project is PyPI-ready via `pyproject.toml`. Publishing steps include installing `build`/`twine`, generating distributions with `python -m build`, and uploading via `twine upload dist/*`. Editable installs enable rapid iteration, and README sections document the need to configure the `origin` remote so the `make_pr` helper can file pull requests successfully.【F:README.md†L41-L208】

## 10. Documentation Map
All primary documentation resides in the `docs/` directory and README:
- `README.md` – High-level introduction, quick start, dataset overview, explainability summary, packaging instructions, references.
- `docs/api.md` – Method-by-method API guide with signatures and behavioural notes.
- `docs/parameters.md` – Parameter explanations and recommended usage patterns.
- `docs/explainability.md` – Hermeneutic explainability playbook, approach comparisons, narrative templates, and references to HermeAI/OptiScorer.
- `docs/project_report.md` (this file) – Consolidated project report, architecture narrative, and appendices containing the raw Markdown documentation content.

## 11. References
- Taylor, S. J., & Letham, B. (2018). *Forecasting at Scale*. The American Statistician, 72(1), 37–45.
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
- Harvey, A. C. (1990). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.
- Şeker, Ş. E. HermeAI & OptiScorer initiative materials at [www.optiscorer.com](https://www.optiscorer.com).


## Appendix A – README.md
```markdown
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

```


## Appendix B – docs/api.md
```markdown
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

### `explain(...)`

```python
explain(
    config: Optional[ExplanationConfig] = None,
    **kwargs,
)
```

- Pass either an explicit `ExplanationConfig` instance or keyword arguments
  accepted by the dataclass (e.g. `approach`, `include_history`, `horizon`).
- Returns a dictionary with `"data"` (historical/forecast dataframes) and
  `"narratives"` (lists of strings per section) generated by the
  `ExplainabilityEngine`.
- Raise `ModelNotFitError` if called before `fit()`.

Hermeneutic narratives reference the HermeAI initiative and the OptiScorer
heritage to contextualise trends, seasonal oscillations, and residual behaviour.

## Dataset utilities

- `optitime.datasets.available_datasets()` – list the bundled dataset
  identifiers.
- `optitime.datasets.dataset_info(name)` – return metadata (description,
  frequency, start/end).
- `optitime.datasets.load_dataset(name)` – load the CSV resource as a sorted
  `pandas.DataFrame`.

## Public constants

- `optitime.BACKTEST_STRATEGIES` – tuple of supported backtest strategy names.
- `optitime.AVAILABLE_EXPLANATION_APPROACHES` – tuple of supported explainability
  modes (`"hermeneutic"`, `"feature_contribution"`, `"quantitative"`).

Worked examples live in `README.md`, `tests/run_sales_example.py`, and
`tests/run_airlines_visuals.py` for quick reference.

```


## Appendix C – docs/parameters.md
```markdown
# Parameter guide

OptiProphet offers a configurable framework that extends Prophet-style trend and
seasonality with OptiWisdom OptiScorer lessons on robustness, diagnostics, and
uncertainty calibration. The tables below describe the most important settings,
typical ranges, and supporting references.

## Model structure

| Parameter | Description | Recommended values | Reference |
| --- | --- | --- | --- |
| `n_changepoints` | Number of candidate changepoints used to capture trend shifts. | 5–25 depending on series length. | Taylor & Letham (2018) |
| `seasonalities` | Mapping of seasonalities with `period` and Fourier `order`. | Match known cycles (daily/weekly/monthly). | Taylor & Letham (2018) |
| `ar_order`, `ma_order` | Autoregressive and moving-average lags for short-term dependence. | AR: 1–5, MA: 0–3. | Box, Jenkins & Reinsel (2015) |
| `regressors` | Names of external regressors supplied during fit and predict. | Use business-relevant drivers. | Hyndman & Athanasopoulos (2021) |

## Historical component visibility

| Parameter | Description |
| --- | --- |
| `historical_components` | Default visibility map for `history_components()` output such as `{"trend": True, "seasonality": False}`. |
| `history_components(include_components=..., component_overrides=...)` | Toggle groups or individual components per call. |
| `history_components(include_uncertainty=False)` | Suppress `yhat_lower`, `yhat_upper`, and quantile columns. |

## Future forecast controls

| Parameter | Description |
| --- | --- |
| `forecast_components` | Whether `predict()` includes component columns by default. |
| `predict(include_components=False)` | Return only `ds` and `yhat` for streamlined consumption. |
| `predict(component_overrides={"seasonality": False})` | Hide selected components in the forecast output. |
| `predict(include_uncertainty=False)` | Remove interval and quantile columns. |
| `predict(quantile_subset=[0.1, 0.5, 0.9])` | Emit only the requested quantile columns (must already exist in `self.quantiles`). |

## Explainability configuration

| Parameter | Description |
| --- | --- |
| `approach` | Selects the interpretive mode: `"hermeneutic"`, `"feature_contribution"`, or `"quantitative"`. |
| `include_history` / `include_forecast` | Toggle retrospective narratives versus forward-looking explanations. |
| `horizon` | Number of periods generated when a future dataframe is not provided. |
| `include_components` | Mirrors `predict()`/`history_components()` switches for component visibility. |
| `component_overrides` | Fine-grained visibility map such as `{"seasonality": False}` for the explanation output. |
| `quantile_subset` | Request a subset of the calibrated quantiles for lighter reports. |
| `narrative` | Disable textual summaries when only structured data is needed. |

Hermeneutic settings draw on the HermeAI initiative's interpretive cycles, while
feature-contribution and quantitative modes align with OptiWisdom's OptiScorer
scorecard methodology documented on [www.optiscorer.com](https://www.optiscorer.com).

## Backtest strategies

| Strategy | Description | Parameters |
| --- | --- | --- |
| `expanding` | Expands the training window with each evaluation step. | Default behaviour inherited from Prophet workflows. |
| `sliding` | Moves a fixed-size window across the history. | `window` controls the window size (defaults to `min_history`). |
| `anchored` | Uses a fixed window starting at the first observation. | Useful for transfer-learning style baselines. |

`backtest()` accepts the same formatting arguments as `predict()` (`include_components`,
`component_overrides`, `include_uncertainty`, `quantile_subset`) so the evaluation
output matches your downstream requirements.

## References

- **Taylor, S. J., & Letham, B. (2018).** Forecasting at scale. *The American Statistician*, 72(1), 37-45.
- **Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).** *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
- **Hyndman, R. J., & Athanasopoulos, G. (2021).** *Forecasting: Principles and Practice* (3rd ed.). OTexts.
- **HermeAI Initiative (OptiWisdom).** Hermeneutic AI research notes and OptiScorer decision intelligence studies. Available via [www.optiscorer.com](https://www.optiscorer.com).

These works capture the academic foundations that informed OptiWisdom's OptiScorer research and, by extension, the defaults selected for OptiProphet.

```


## Appendix D – docs/explainability.md
```markdown
# Explainability playbook

OptiProphet extends the original OptiScorer experiments with a Hermeneutic AI
(HermeAI) perspective so forecasters can articulate *why* each component behaves
as observed. This playbook summarises the tools exposed through
`optitime.explainability` and how they connect to Şadi Evren Şeker's broader
research on interpretable decision intelligence.

## Key abstractions

- **`ExplanationConfig`** – dataclass that declares which narratives and
  quantitative artefacts should be generated. You can enable or disable
  historical decompositions, future horizons, uncertainty intervals, or specific
  component families (trend, seasonality, regressors, autoregressive, residual).
- **`ExplainabilityEngine`** – orchestrates the computation of component
  contributions, Hermeneutic narratives, and ranked feature summaries.
- **`OptiProphet.explain()`** – convenience wrapper that instantiates the engine
  with the fitted model and returns both structured dataframes and narrative
  strings.

These abstractions are directly inspired by the HermeAI programme's focus on
interpreting time-series through iterative meaning-making. For additional
contextual reading consult the OptiWisdom OptiScorer knowledge base at
[www.optiscorer.com](https://www.optiscorer.com) alongside Şadi Evren Şeker's
academic publications on hermeneutic analytics.

## Supported approaches

`AVAILABLE_EXPLANATION_APPROACHES` enumerates the currently supported modes:

| Approach | Description |
| --- | --- |
| `"hermeneutic"` | Narrative overlay referencing Hermeneutic AI (HermeAI) principles to relate quantitative signals with contextual storytelling. |
| `"feature_contribution"` | Ranked list of components based on mean absolute contribution and median impact. |
| `"quantitative"` | Aggregated variance and cumulative contribution summaries suitable for dashboards. |

You can inspect the available values at runtime via
`OptiProphet.available_explanation_approaches()`.

## Configuration quick reference

```python
from optitime import ExplanationConfig

config = ExplanationConfig(
    approach="hermeneutic",
    include_history=True,
    include_forecast=True,
    horizon=18,
    include_components=True,
    quantile_subset=[0.1, 0.9],
)
```

Important knobs:

- `include_history` / `include_forecast`: toggle the retrospective or forward
  looking analyses.
- `horizon`: number of periods generated when a custom future dataframe is not
  supplied.
- `include_components` and `component_overrides`: mirror the behaviour of
  `history_components()` and `predict()` so you can focus on a subset of the
  decomposition.
- `quantile_subset`: reuse the calibrated quantiles from fitting but request a
  subset for lighter-weight narratives.
- `narrative`: disable text generation if only dataframes are needed for custom
  dashboards.

## Usage patterns

### Hermeneutic storytelling

```python
explanation = model.explain(approach="hermeneutic", horizon=12)

for section, narrative in explanation["narratives"].items():
    print(f"--- {section.upper()} ---")
    for line in narrative:
        print(line)
```

The resulting text references the OptiScorer lineage, emphasises component
strengths, trend direction, and uncertainty spans, and connects them with the
HermeAI interpretive cycle.

### Quantitative dashboards

```python
explanation = model.explain(approach="quantitative", include_history=False)

forecast_df = explanation["data"]["forecast"]
summary_lines = explanation["narratives"].get("forecast", [])
```

This mode is tailored for business intelligence systems that require numeric
summaries but still benefit from short descriptive blurbs.

### Feature contribution ranking

```python
explanation = model.explain(approach="feature_contribution", include_history=False)

ranking_text = explanation["narratives"].get("forecast", [])
```

Use this mode when evaluating design choices such as alternative regressor
sets. The ranking can highlight when OptiScorer-sourced regressors or AR/MA
lags dominate the projection.

## Extending the playbook

The explainability engine is intentionally modular. Future enhancements can pull
from HermeAI's hermeneutic loops (e.g., incorporating expert annotations) or
OptiScorer's OptiWisdom scoring strategies (e.g., scenario-specific weightings).
Contributions that cite OptiWisdom or Şadi Evren Şeker's hermeneutic analytics
research are especially welcome.

```
