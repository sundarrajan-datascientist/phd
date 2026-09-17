"""Unit tests for the daily 30-day CRISIL return calculation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import pandas as pd


SCRIPT_PATH = Path(__file__).with_name("02_load_mf_monthly_returns.py")
SPEC = importlib.util.spec_from_file_location("monthly_returns_loader", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
loader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loader)


class CalculateRollingReturnsTest(unittest.TestCase):
    def test_uses_exact_date_or_previous_available_fund_nav(self) -> None:
        daily_navs = pd.DataFrame(
            [
                # Fund A: 09-Jan is used when 10-Jan (30 days before 09-Feb)
                # is a weekend/non-reporting date.
                {"scheme_name": "Fund A", "report_date": "2026-01-09", "nav_direct": 100.0},
                {"scheme_name": "Fund A", "report_date": "2026-02-09", "nav_direct": 110.0},
                # Fund B uses an exact match and must be independent from Fund A.
                {"scheme_name": "Fund B", "report_date": "2026-01-10", "nav_direct": 200.0},
                {"scheme_name": "Fund B", "report_date": "2026-02-09", "nav_direct": 220.0},
            ]
        )
        daily_navs["report_date"] = pd.to_datetime(daily_navs["report_date"])

        actual = loader.calculate_rolling_returns(daily_navs).set_index(
            ["scheme_name", "report_date"]
        )

        fund_a = actual.loc[("Fund A", pd.Timestamp("2026-02-09"))]
        self.assertEqual(fund_a["reference_date"], pd.Timestamp("2026-01-09"))
        self.assertEqual(fund_a["reference_nav"], 100.0)
        self.assertAlmostEqual(fund_a["monthly_return_percent"], 10.0)

        fund_b = actual.loc[("Fund B", pd.Timestamp("2026-02-09"))]
        self.assertEqual(fund_b["reference_date"], pd.Timestamp("2026-01-10"))
        self.assertEqual(fund_b["reference_nav"], 200.0)
        self.assertAlmostEqual(fund_b["monthly_return_percent"], 10.0)

        # The first record for each fund has no available reference NAV.
        first_fund_a = actual.loc[("Fund A", pd.Timestamp("2026-01-09"))]
        self.assertTrue(pd.isna(first_fund_a["reference_date"]))
        self.assertTrue(pd.isna(first_fund_a["monthly_return_percent"]))


if __name__ == "__main__":
    unittest.main()
