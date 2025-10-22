"""Explainability utilities for OptiProphet forecasts and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from .diagnostics import component_strengths
if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from .model import OptiProphet


AVAILABLE_EXPLANATION_APPROACHES = ("hermeneutic", "feature_contribution", "quantitative")


@dataclass
class ExplanationConfig:
    """Configuration options for generating model explanations."""

    approach: str = "hermeneutic"
    include_history: bool = True
    include_forecast: bool = True
    horizon: int = 12
    future: Optional[pd.DataFrame] = None
    include_components: Optional[bool] = None
    component_overrides: Optional[Mapping[str, bool]] = None
    include_uncertainty: bool = True
    quantile_subset: Optional[Iterable[float]] = None
    top_component_count: int = 3
    narrative: bool = True


class ExplainabilityEngine:
    """Create human-readable explanations for OptiProphet outputs."""

    def __init__(self, model: "OptiProphet") -> None:
        self.model = model

    def generate(self, config: ExplanationConfig) -> Dict[str, object]:
        """Generate explanations for the fitted model using ``config``."""

        approach = config.approach.lower().strip()
        if approach not in AVAILABLE_EXPLANATION_APPROACHES:
            raise ValueError(
                f"Unsupported explanation approach '{config.approach}'. "
                f"Choose from {AVAILABLE_EXPLANATION_APPROACHES}."
            )

        output: Dict[str, object] = {
            "approach": approach,
            "narratives": {},
            "data": {},
        }

        if config.include_history:
            history_payload = self._history_explanation(config=config, approach=approach)
            output["data"]["history"] = history_payload["data"]
            if config.narrative:
                output["narratives"]["history"] = history_payload["narrative"]

        if config.include_forecast:
            forecast_payload = self._forecast_explanation(config=config, approach=approach)
            output["data"]["forecast"] = forecast_payload["data"]
            if config.narrative:
                output["narratives"]["forecast"] = forecast_payload["narrative"]

        return output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _history_explanation(
        self,
        *,
        config: ExplanationConfig,
        approach: str,
    ) -> Dict[str, object]:
        components_df = self.model.history_components(
            include_components=config.include_components,
            component_overrides=config.component_overrides,
            include_uncertainty=config.include_uncertainty,
            quantile_subset=config.quantile_subset,
        )

        narrative = self._build_history_narrative(components_df, approach=approach)
        return {"data": components_df, "narrative": narrative}

    def _forecast_explanation(
        self,
        *,
        config: ExplanationConfig,
        approach: str,
    ) -> Dict[str, object]:
        if config.future is not None:
            future_df = config.future
        else:
            future_df = self.model.make_future_dataframe(
                periods=config.horizon,
                include_history=False,
            )

        forecast_df = self.model.predict(
            future_df,
            include_history=False,
            include_components=config.include_components,
            component_overrides=config.component_overrides,
            include_uncertainty=config.include_uncertainty,
            quantile_subset=config.quantile_subset,
        )

        narrative = self._build_forecast_narrative(forecast_df, approach=approach)
        return {"data": forecast_df, "narrative": narrative}

    def _build_history_narrative(self, df: pd.DataFrame, *, approach: str) -> List[str]:
        if df.empty:
            return ["No historical components are available for explanation."]

        components = self._extract_component_columns(df)
        strengths = component_strengths(components)

        lines = []
        if approach == "hermeneutic":
            lines.extend(self._hermeneutic_history_story(df, strengths))
        elif approach == "feature_contribution":
            lines.append("Historical decomposition ranked by mean absolute contribution:")
            ranking = self._rank_components(df)
            for name, stats in ranking:
                lines.append(
                    f"- {name}: mean contribution {stats['mean']:.3f}, "
                    f"share {stats['share']:.1%}"
                )
        else:  # quantitative
            lines.append("Historical component strength summary:")
            for name, strength in strengths.items():
                lines.append(f"- {name}: variance share {strength:.1%}")

        return lines

    def _build_forecast_narrative(self, df: pd.DataFrame, *, approach: str) -> List[str]:
        if df.empty:
            return ["Forecast dataframe is empty; no explanation generated."]

        component_cols = self._extract_component_columns(df)
        horizon = df.shape[0]

        lines = [f"Forecast horizon analysed: {horizon} steps"]
        if approach == "hermeneutic":
            lines.extend(self._hermeneutic_forecast_story(df, component_cols))
        elif approach == "feature_contribution":
            ranking = self._rank_components(df)
            if ranking:
                lines.append("Forecast contributions ranked by absolute impact:")
                for name, stats in ranking:
                    lines.append(
                        f"- {name}: median impact {stats['median']:.3f}, "
                        f"expected share {stats['share']:.1%}"
                    )
        else:
            if component_cols:
                totals = {
                    name: float(np.sum(values))
                    for name, values in component_cols.items()
                }
                total_abs = sum(abs(value) for value in totals.values()) or 1.0
                lines.append("Aggregate component impact across the forecast horizon:")
                for name, value in totals.items():
                    lines.append(
                        f"- {name}: cumulative {value:.3f} "
                        f"({abs(value) / total_abs:.1%} of total magnitude)"
                    )
        return lines

    def _hermeneutic_history_story(
        self,
        df: pd.DataFrame,
        strengths: Mapping[str, float],
    ) -> List[str]:
        lines = [
            "Hermeneutic perspective inspired by HermeAI research emphasises interpretive loops between data and domain narratives.",
            "The OptiWisdom OptiScorer lineage guides the focus on decision-centric signals.",
        ]
        trend = df.get("trend")
        if trend is not None:
            slope = float(trend.iloc[-1] - trend.iloc[0]) if len(trend) > 1 else float(trend.iloc[0])
            direction = "upward" if slope > 0 else "downward" if slope < 0 else "stable"
            lines.append(
                f"Trend interpretation: the long-run component drifts {direction} by {slope:.3f} over the observed window."
            )
        seasonality = df[[col for col in df.columns if col.startswith("seasonality")]]
        if not seasonality.empty:
            amplitude = float(seasonality.max().max() - seasonality.min().min())
            lines.append(
                f"Seasonality interpretation: cyclical swings span approximately {amplitude:.3f} units, signalling rhythmic patterns consistent with OptiScorer field studies."
            )
        residual = df.get("residual")
        if residual is not None:
            dispersion = float(residual.std(ddof=0))
            lines.append(
                f"Residual interpretation: dispersion of {dispersion:.3f} highlights the unexplained dynamics awaiting further OptiWisdom OptiScorer calibration."
            )
        if strengths:
            ranked = sorted(strengths.items(), key=lambda item: item[1], reverse=True)
            top_components = ", ".join(f"{name} ({value:.1%})" for name, value in ranked[:3])
            lines.append(
                f"Variance emphasis: dominant components are {top_components}, aligning with Hermeneutic AI focus on contextual salience."
            )
        return lines

    def _hermeneutic_forecast_story(
        self,
        df: pd.DataFrame,
        component_cols: Mapping[str, np.ndarray],
    ) -> List[str]:
        yhat = df.get("yhat")
        lines = [
            "Future narrative follows Hermeneutic AI principles by relating projections to historical motifs."
        ]
        if yhat is not None and not yhat.empty:
            start = float(yhat.iloc[0])
            end = float(yhat.iloc[-1])
            change = end - start
            direction = "increase" if change > 0 else "decrease" if change < 0 else "stability"
            lines.append(
                f"Forecast summary: expected {direction} of {change:.3f} from the first to last horizon step."
            )
        if component_cols:
            strongest = max(
                component_cols.items(),
                key=lambda item: float(np.mean(np.abs(item[1]))) if item[1].size > 0 else 0.0,
            )
            lines.append(
                f"Dominant driver: {strongest[0]} with mean absolute impact {np.mean(np.abs(strongest[1])):.3f}."
            )
        intervals = df[[col for col in df.columns if col.startswith("yhat_")]]
        if not intervals.empty:
            spreads = intervals.max(axis=1) - intervals.min(axis=1)
            lines.append(
                f"Uncertainty span median: {float(spreads.median()):.3f}, echoing OptiScorer's emphasis on transparent risk bands."
            )
        return lines

    def _extract_component_columns(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        component_cols = {}
        for column in df.columns:
            if column in {"ds", "y", "yhat"}:
                continue
            if column.startswith("yhat_"):
                continue
            values = df[column]
            if np.issubdtype(values.dtype, np.number):
                component_cols[column] = values.to_numpy()
        return component_cols

    def _rank_components(self, df: pd.DataFrame) -> List[tuple[str, Dict[str, float]]]:
        components = self._extract_component_columns(df)
        if not components:
            return []
        ranking = []
        total = sum(np.mean(np.abs(values)) for values in components.values()) or 1.0
        for name, values in components.items():
            abs_mean = float(np.mean(np.abs(values)))
            ranking.append(
                (
                    name,
                    {
                        "mean": abs_mean,
                        "median": float(np.median(values)),
                        "share": abs_mean / total,
                    },
                )
            )
        ranking.sort(key=lambda item: item[1]["mean"], reverse=True)
        return ranking

