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
