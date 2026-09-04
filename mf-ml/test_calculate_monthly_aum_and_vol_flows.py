"""Unit tests for monthly AUM and 12-month fund-flow volatility features."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import pandas as pd


SCRIPT_PATH = Path(__file__).with_name("06_calculate_monthly_aum_and_vol_flows.py")
SPEC = importlib.util.spec_from_file_location("aum_flow_features", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
loader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loader)


class CalculateMonthlyAumAndVolFlowsTest(unittest.TestCase):
    def test_keeps_month_end_aum_and_calculates_12_month_flow_volatility(self) -> None:
        dates = pd.date_range("2025-01-31", periods=13, freq="ME")
        rows = []
        for number, date in enumerate(dates):
            # An earlier observation confirms that the monthly final one is used.
            rows.append({"scheme_name": "Fund A", "report_date": date - pd.Timedelta(days=1), "daily_aum": 1.0, "nav_direct": 1.0})
            rows.append({"scheme_name": "Fund A", "report_date": date, "daily_aum": 100 + number * 10, "nav_direct": 10 + number})

        actual = loader.calculate_monthly_aum_and_vol_flows(pd.DataFrame(rows)).set_index("month_date")
        final_month = actual.loc[pd.Timestamp("2026-01-01")]

        self.assertEqual(final_month["monthly_aum"], 220)
        self.assertAlmostEqual(final_month["monthly_return_percent"], 100 / 21 * 100)
        expected_flows = [
            (100 + number * 10) - (100 + (number - 1) * 10) * ((10 + number) / (9 + number))
            for number in range(1, 13)
        ]
        self.assertAlmostEqual(final_month["vol_flows_12m"], pd.Series(expected_flows).std())


if __name__ == "__main__":
    unittest.main()
