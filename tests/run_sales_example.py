"""Quick OptiProphet demo on the bundled sales.csv dataset.

Run with:
    python tests/run_sales_example.py

The script loads the synthetic sales figures, fits the OptiProphet model,
produces a 12 month forecast, and prints component diagnostics alongside a
backtest summary so you can inspect outputs immediately after installation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Ensure the local package is importable when the project has just been cloned.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optitime import OptiProphet  # noqa: E402  (import after sys.path mutation)


def main() -> None:
    data_path = Path(__file__).resolve().parent / "sales.csv"
    df = pd.read_csv(data_path, parse_dates=["ds"])

    model = OptiProphet(
        n_changepoints=10,
        ar_order=2,
        ma_order=1,
        min_history=24,
        min_success_r2=-1.0,
        max_mape=None,
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=12, include_history=False)
    forecast = model.predict(future, include_history=False)

    print("=== Forecast (next 12 months) ===")
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].head())

    lean = model.predict(
        future,
        include_history=False,
        include_components=False,
        include_uncertainty=False,
    )
    print("\n=== Lean forecast (components & intervals disabled) ===")
    print(lean.head())

    components = model.history_components(component_overrides={"seasonality": False})
    print("\n=== Last 5 historical component decompositions (seasonality hidden) ===")
    subset = [
        col for col in ["ds", "yhat", "trend", "seasonality", "residual"] if col in components.columns
    ]
    print(components[subset].tail().reset_index(drop=True))

    backtest = model.backtest(horizon=12, step=3, strategy="sliding", window=18)
    print("\n=== Backtest summary (sliding window, first 5 rows) ===")
    print(backtest.head())


if __name__ == "__main__":
    main()
