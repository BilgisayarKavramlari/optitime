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

These works capture the academic foundations that informed OptiWisdom's OptiScorer research and, by extension, the defaults selected for OptiProphet.
