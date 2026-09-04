"""Unit tests for the paper-based binary outperformance target."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import pandas as pd


SCRIPT_PATH = Path(__file__).with_name("05_define_outperformance_target.py")
SPEC = importlib.util.spec_from_file_location("outperformance_target", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
loader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loader)


class DefineOutperformanceTargetTest(unittest.TestCase):
    def test_uses_strict_benchmark_comparison_and_excludes_unmatched_months(self) -> None:
        funds = pd.DataFrame(
            [
                {"scheme_name": "Beats", "month_date": "2026-02-01", "trade_month_end": "2026-02-27", "monthly_return_percent": 3.1},
                {"scheme_name": "Ties", "month_date": "2026-02-01", "trade_month_end": "2026-02-27", "monthly_return_percent": 2.0},
                {"scheme_name": "Loses", "month_date": "2026-02-01", "trade_month_end": "2026-02-27", "monthly_return_percent": 1.2},
                {"scheme_name": "No benchmark", "month_date": "2026-03-01", "trade_month_end": "2026-03-31", "monthly_return_percent": 9.0},
            ]
        )
        benchmarks = pd.DataFrame(
            [
                {"benchmark_name": "Nifty 500 TRI", "observation_month": "2026-02-01", "month_end": "2026-02-27", "benchmark_monthly_return_percent": 2.0},
            ]
        )

        result = loader.define_outperformance_target(funds, benchmarks).set_index("scheme_name")

        self.assertEqual(set(result.index), {"Beats", "Ties", "Loses"})
        self.assertEqual(result.loc["Beats", "outperforms_benchmark"], 1)
        self.assertEqual(result.loc["Ties", "outperforms_benchmark"], 0)
        self.assertEqual(result.loc["Loses", "outperforms_benchmark"], 0)
        self.assertAlmostEqual(result.loc["Beats", "excess_return_percent"], 1.1)
        self.assertEqual(result.loc["Beats", "month_date"], pd.Timestamp("2026-01-01"))
        self.assertEqual(result.loc["Beats", "target_month"], pd.Timestamp("2026-02-01"))

    def test_includes_feature_month_returns_separately_from_target_returns(self) -> None:
        funds = pd.DataFrame(
            [
                {"scheme_name": "Timing", "month_date": "2026-06-01", "trade_month_end": "2026-06-30", "monthly_return_percent": 1.480747},
                {"scheme_name": "Timing", "month_date": "2026-07-01", "trade_month_end": "2026-07-31", "monthly_return_percent": 2.898678},
            ]
        )
        benchmarks = pd.DataFrame(
            [
                {"benchmark_name": "Nifty 500 TRI", "observation_month": "2026-06-01", "month_end": "2026-06-30", "benchmark_monthly_return_percent": 1.706893},
                {"benchmark_name": "Nifty 500 TRI", "observation_month": "2026-07-01", "month_end": "2026-07-31", "benchmark_monthly_return_percent": 2.199901},
            ]
        )

        result = loader.define_outperformance_target(funds, benchmarks)
        july_target = result.loc[result["target_month"] == pd.Timestamp("2026-07-01")].iloc[0]

        self.assertEqual(july_target["month_date"], pd.Timestamp("2026-06-01"))
        self.assertAlmostEqual(july_target["current_monthly_return_percent"], 1.480747)
        self.assertAlmostEqual(
            july_target["current_benchmark_monthly_return_percent"], 1.706893
        )
        self.assertAlmostEqual(july_target["monthly_return_percent"], 2.898678)
        self.assertAlmostEqual(
            july_target["benchmark_monthly_return_percent"], 2.199901
        )
        self.assertEqual(july_target["outperforms_benchmark"], 1)


if __name__ == "__main__":
    unittest.main()
