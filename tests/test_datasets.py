import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optitime import OptiProphet, available_datasets, load_dataset


class DatasetIntegrationTest(unittest.TestCase):
    def test_forecasting_pipeline_across_datasets(self) -> None:
        dataset_names = available_datasets()
        self.assertGreaterEqual(len(dataset_names), 1)

        for name in dataset_names:
            with self.subTest(dataset=name):
                df = load_dataset(name)
                self.assertFalse(df.empty)
                self.assertGreaterEqual(len(df), 30)
                self.assertIn("ds", df.columns)
                self.assertIn("y", df.columns)

                model = OptiProphet(
                    n_changepoints=8,
                    ar_order=2,
                    ma_order=1,
                    min_history=24,
                    min_success_r2=-1.0,
                    max_mape=None,
                )

                model.fit(df)

                future = model.make_future_dataframe(periods=12, include_history=False)
                forecast = model.predict(future, include_history=False)
                self.assertEqual(len(forecast), 12)
                for column in ("yhat", "yhat_lower", "yhat_upper"):
                    self.assertIn(column, forecast.columns)

                components = model.history_components()
                for column in ("trend", "seasonality", "yhat", "residual"):
                    self.assertIn(column, components.columns)

                horizon = max(3, min(12, len(df) // 4))
                step = max(1, horizon // 3)
                backtest_results = model.backtest(horizon=horizon, step=step)
                self.assertFalse(backtest_results.empty)


if __name__ == "__main__":
    unittest.main()
