"""Unit tests for paper-style performance-momentum features."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import pandas as pd


SCRIPT_PATH = Path(__file__).with_name("07_calculate_performance_momentum.py")
SPEC = importlib.util.spec_from_file_location("performance_momentum", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
loader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loader)


class PerformanceMomentumTest(unittest.TestCase):
    def _factor_rows(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": dates,
                "rf": 2.0,
                "mkt": 50.0,
                "smb": 10.0,
                "hml": -10.0,
                "umd": 5.0,
                "market_return": 52.0,
            }
        )

    def _synthetic_nav_rows(self) -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-01", periods=260)
        rows = []
        for scheme_name in ("Fund A", "Fund B"):
            nav = 100.0
            for trade_date in dates:
                nav *= 1.001
                rows.append(
                    {
                        "scheme_name": scheme_name,
                        "trade_date": trade_date,
                        "nav_direct": nav,
                    }
                )
        return pd.DataFrame(rows)

    def _synthetic_panel(self) -> pd.DataFrame:
        nav_rows = self._synthetic_nav_rows()
        factor_rows = self._factor_rows(nav_rows["trade_date"].drop_duplicates())
        return loader.build_regression_panel(nav_rows, factor_rows)

    def test_daily_return_skips_weekends_and_holidays(self) -> None:
        nav_rows = pd.DataFrame(
            [
                {"scheme_name": "Fund A", "trade_date": "2024-01-03", "nav_direct": 100.0},
                {"scheme_name": "Fund A", "trade_date": "2024-01-05", "nav_direct": 102.0},
                {"scheme_name": "Fund A", "trade_date": "2024-01-08", "nav_direct": 103.0},
            ]
        )
        nav_rows["trade_date"] = pd.to_datetime(nav_rows["trade_date"])
        returns = loader.compute_daily_fund_returns(nav_rows)

        friday = returns.loc[returns["trade_date"] == pd.Timestamp("2024-01-05")].iloc[0]
        monday = returns.loc[returns["trade_date"] == pd.Timestamp("2024-01-08")].iloc[0]

        self.assertEqual(friday["prior_trade_date"], pd.Timestamp("2024-01-03"))
        self.assertAlmostEqual(friday["fund_return"], 0.02)
        self.assertEqual(monday["prior_trade_date"], pd.Timestamp("2024-01-05"))
        self.assertAlmostEqual(monday["fund_return"], 103 / 102 - 1)

    def test_prepare_factor_panel_converts_percent_factors(self) -> None:
        panel = pd.DataFrame(
            [
                {
                    "scheme_name": "Fund A",
                    "trade_date": "2024-01-01",
                    "fund_return": 0.01,
                    "rf": 2.0,
                    "mkt": 50.0,
                    "smb": 10.0,
                    "hml": -10.0,
                    "umd": 5.0,
                    "market_return": 52.0,
                }
            ]
        )
        prepared = loader.prepare_factor_panel(panel)
        self.assertAlmostEqual(prepared.loc[0, "rf"], 0.02)
        self.assertAlmostEqual(prepared.loc[0, "fund_excess_return"], 0.01 - 0.02)
        self.assertAlmostEqual(prepared.loc[0, "excess_return"], 0.01 - 0.52)

    def test_rolling_factor_features_require_full_window(self) -> None:
        panel = self._synthetic_panel()
        prepared = loader.prepare_factor_panel(panel)
        fund_panel = prepared[prepared["scheme_name"] == "Fund A"]
        features = loader.compute_rolling_factor_features(fund_panel, windows=(90,))

        self.assertTrue(features.loc[:88, "const_90"].isna().all())
        self.assertTrue(features.loc[89:, "const_90"].notna().any())
        self.assertIn("t_umd_90", features.columns)

    def test_information_and_sharpe_ratios_use_full_windows(self) -> None:
        panel = self._synthetic_panel()
        prepared = loader.prepare_factor_panel(panel)
        fund_panel = prepared[prepared["scheme_name"] == "Fund A"]
        ratios = loader.compute_information_and_sharpe_ratios(
            fund_panel,
            windows={"p1y": 250},
        )

        self.assertTrue(ratios.loc[:248, "ir_p1y"].isna().all())
        self.assertTrue(ratios.loc[249:, "ir_p1y"].notna().all())
        self.assertTrue(ratios.loc[249:, "sr_p1y"].notna().all())

    def test_calculate_performance_momentum_returns_all_funds(self) -> None:
        features = loader.calculate_performance_momentum(
            self._synthetic_panel(),
            progress_every=0,
        )
        self.assertEqual(set(features["scheme_name"]), {"Fund A", "Fund B"})
        self.assertIn("const_250", features.columns)
        self.assertIn("ir_p3y", features.columns)


if __name__ == "__main__":
    unittest.main()
