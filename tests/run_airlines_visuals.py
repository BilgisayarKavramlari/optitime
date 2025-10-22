"""Visual parameter sweep for the airlines traffic benchmark.

This script demonstrates how toggling OptiProphet's new controls affects
forecasts, decompositions, and backtest outcomes. It writes PNG artefacts to
the same directory so you can inspect them after running::

    python tests/run_airlines_visuals.py

Install the optional plotting dependency first if required::

    pip install optitime-prophet[visuals]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - user guidance only
    raise SystemExit(
        "matplotlib is required for the visual diagnostics script. "
        "Install it with 'pip install optitime-prophet[visuals]'"
    ) from exc

from optitime import BACKTEST_STRATEGIES, OptiProphet, load_dataset  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parent


def _plot_forecast(
    *,
    df: pd.DataFrame,
    forecast: pd.DataFrame,
    title: str,
    output_name: str,
) -> None:
    merged = forecast.merge(df[["ds", "y"]], on="ds", how="left")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(merged["ds"], merged["y"], label="Gerçek", color="#4c72b0")
    ax.plot(merged["ds"], merged["yhat"], label="Tahmin", color="#dd8452")

    if {"yhat_lower", "yhat_upper"}.issubset(merged.columns):
        ax.fill_between(
            merged["ds"],
            merged["yhat_lower"],
            merged["yhat_upper"],
            alpha=0.2,
            color="#dd8452",
            label="Tahmin Aralığı",
        )

    ax.set_title(title)
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Yolcu Sayısı")
    ax.legend()
    fig.autofmt_xdate()
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Kaydedildi: {output_path}")


def _plot_backtest(
    *,
    backtest: pd.DataFrame,
    title: str,
    output_name: str,
) -> None:
    if backtest.empty:
        print("Backtest sonucu boş; grafik oluşturulmadı.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(backtest["start"], backtest["rmse"], marker="o", label="RMSE")
    ax.set_title(title)
    ax.set_xlabel("Başlangıç")
    ax.set_ylabel("RMSE")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Kaydedildi: {output_path}")


def run_variant(
    *,
    name: str,
    df: pd.DataFrame,
    strategy: str,
    include_components: Optional[bool],
    component_overrides: Optional[Mapping[str, bool]],
    include_uncertainty: bool,
    quantile_subset: Optional[Iterable[float]],
) -> None:
    model = OptiProphet(
        n_changepoints=12,
        ar_order=3,
        ma_order=1,
        min_history=60,
        min_success_r2=-1.0,
        max_mape=None,
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=18, include_history=False)
    forecast = model.predict(
        future,
        include_components=include_components,
        component_overrides=component_overrides,
        include_uncertainty=include_uncertainty,
        quantile_subset=quantile_subset,
    )

    _plot_forecast(
        df=df,
        forecast=forecast,
        title=f"{name} - Tahmin senaryosu ({strategy})",
        output_name=f"airlines_forecast_{strategy}.png",
    )

    backtest = model.backtest(
        horizon=12,
        step=3,
        strategy=strategy,
        include_components=include_components,
        component_overrides=component_overrides,
        include_uncertainty=include_uncertainty,
        quantile_subset=quantile_subset,
    )
    _plot_backtest(
        backtest=backtest,
        title=f"{name} - Backtest RMSE ({strategy})",
        output_name=f"airlines_backtest_{strategy}.png",
    )


def main() -> None:
    df = load_dataset("airlines_traffic")
    print("OptiWisdom OptiScorer türevi airlines_traffic veri kümesi yüklendi.")
    variants = [
        {
            "name": "Expanding + Bileşenler",
            "strategy": "expanding",
            "include_components": True,
            "component_overrides": None,
            "include_uncertainty": True,
            "quantile_subset": None,
        },
        {
            "name": "Sliding + Sezonsuz",
            "strategy": "sliding",
            "include_components": True,
            "component_overrides": {"seasonality": False},
            "include_uncertainty": True,
            "quantile_subset": [0.9],
        },
    ]

    for variant in variants:
        strategy = variant["strategy"]
        if strategy not in BACKTEST_STRATEGIES:
            print(f"Atlanan senaryo: bilinmeyen strateji {strategy}")
            continue
        run_variant(df=df, **variant)


if __name__ == "__main__":
    main()
