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
                forecast = model.predict(
                    future,
                    include_history=False,
                    include_components=False,
                    include_uncertainty=False,
                )
                self.assertEqual(len(forecast), 12)
                self.assertIn("yhat", forecast.columns)
                self.assertNotIn("trend", forecast.columns)
                self.assertNotIn("yhat_lower", forecast.columns)
                self.assertNotIn("yhat_upper", forecast.columns)

                components = model.history_components(
                    component_overrides={"seasonality": False},
                    include_uncertainty=False,
                )
                self.assertIn("trend", components.columns)
                self.assertNotIn("seasonality", components.columns)
                self.assertNotIn("yhat_lower", components.columns)
                self.assertIn("residual", components.columns)

                horizon = max(3, min(12, len(df) // 4))
                step = max(1, horizon // 3)
                backtest_results = model.backtest(
                    horizon=horizon,
                    step=step,
                    strategy="sliding",
                    include_components=False,
                    include_uncertainty=False,
                )
                self.assertFalse(backtest_results.empty)
                self.assertTrue((backtest_results["strategy"] == "sliding").all())

                explanation = model.explain(
                    include_history=True,
                    include_forecast=True,
                    horizon=4,
                    approach="feature_contribution",
                )
                self.assertIn("data", explanation)
                self.assertIn("history", explanation["data"])
                self.assertIn("forecast", explanation["data"])
                self.assertIn("approach", explanation)
                self.assertEqual(explanation["approach"], "feature_contribution")

    def test_airlines_traffic_specific_dataset(self) -> None:
        df = load_dataset("airlines_traffic")
        self.assertGreaterEqual(len(df), 100)
        self.assertIn("ds", df.columns)
        self.assertIn("y", df.columns)

        model = OptiProphet(
            n_changepoints=12,
            ar_order=3,
            ma_order=1,
            min_history=48,
            min_success_r2=-1.0,
            max_mape=None,
        )

        model.fit(df)
        components = model.history_components(
            include_components=True,
            quantile_subset=[0.9],
        )
        self.assertIn("trend", components.columns)
        self.assertIn("seasonality", components.columns)
        self.assertIn("yhat_q0.90", components.columns)
        self.assertNotIn("yhat_q0.10", components.columns)

        backtest_results = model.backtest(
            horizon=12,
            step=3,
            strategy="anchored",
            window=72,
            quantile_subset=[0.1],
        )
        self.assertFalse(backtest_results.empty)
        self.assertTrue((backtest_results["strategy"] == "anchored").all())

        explanation = model.explain(
            include_history=True,
            include_forecast=True,
            horizon=6,
            approach="hermeneutic",
        )
        self.assertIn("narratives", explanation)
        self.assertIn("history", explanation["narratives"])
        self.assertIsInstance(explanation["narratives"]["history"], list)
        self.assertIn("forecast", explanation["data"])


if __name__ == "__main__":
    unittest.main()
