"""Compute monthly cross-sectional Spearman ICs between predictors and next-month returns.

By default this reads mf200.crisil_fund_monthly_aum_flow_features and computes the
1-month-ahead Spearman rank correlation (IC) between each numeric predictor and
forward 1-month fund return. It reports mean/median/std and the fraction of
months with |IC| >= 0.03.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sqlalchemy import URL, create_engine, text


DEFAULT_SCHEMA = "mf200"
DEFAULT_TABLE = "crisil_fund_monthly_aum_flow_features"
MIN_CROSS_SECTION = 10
IC_THRESHOLD = 0.03


def build_engine():
    return create_engine(
        URL.create(
            "postgresql+psycopg",
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "Password@123"),
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "mfdb"),
        )
    )


def read_monthly_features(engine, schema: str, table: str) -> pd.DataFrame:
    source = f'"{schema}"."{table}"'
    sql = text(
        f"SELECT scheme_name, month_date, month_end_observation, monthly_aum, "
        f"monthly_return_percent, net_flow_amount, net_flow_percent, vol_flows_12m "
        f"FROM {source}"
    )
    df = pd.read_sql(sql, engine)
    df["month_date"] = pd.to_datetime(df["month_date"]).dt.normalize()
    for col in ["monthly_aum", "monthly_return_percent", "net_flow_amount", "net_flow_percent", "vol_flows_12m"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_monthly_cross_sectional_ic(
    df: pd.DataFrame,
    predictors: Iterable[str] | None = None,
    min_cross_section: int = MIN_CROSS_SECTION,
) -> pd.DataFrame:
    # forward 1-month return (decimal) = next month's monthly_return_percent / 100
    df = df.sort_values(["scheme_name", "month_date"]).copy()
    grouped = df.groupby("scheme_name", sort=False)
    df["forward_return"] = grouped["monthly_return_percent"].shift(-1) / 100.0

    # choose predictors automatically if not provided: numeric columns except identifiers & returns
    exclude = {"scheme_name", "month_date", "month_end_observation", "monthly_return_percent", "forward_return", "net_flow_amount"}
    if predictors is None:
        predictors = [
            c for c in df.select_dtypes(include=["number"]).columns if c not in exclude
        ]

    month_dates = sorted(df["month_date"].dropna().unique())
    records = []

    for month in month_dates:
        slice_ = df[df["month_date"] == month]
        if slice_.empty:
            continue
        row: dict = {"month_date": month}
        for pred in predictors:
            s1 = slice_["forward_return"].dropna()
            s2 = slice_.set_index(slice_.index)[pred]
            # align on index and drop NA in either
            aligned = pd.concat([slice_["forward_return"], slice_[pred]], axis=1).dropna()
            if len(aligned) < min_cross_section:
                row[pred] = np.nan
            else:
                # Spearman rank correlation using scipy
                try:
                    corr, _ = spearmanr(aligned["forward_return"], aligned[pred])
                    row[pred] = float(corr) if not np.isnan(corr) else np.nan
                except Exception:
                    row[pred] = np.nan
        records.append(row)

    ic_df = pd.DataFrame.from_records(records).sort_values("month_date")
    return ic_df


def summarize_ic(ic_df: pd.DataFrame, predictors: Iterable[str]) -> pd.DataFrame:
    rows = []
    for pred in predictors:
        series = ic_df[pred].dropna()
        if series.empty:
            rows.append(
                {
                    "predictor": pred,
                    "months": 0,
                    "mean_ic": np.nan,
                    "median_ic": np.nan,
                    "std_ic": np.nan,
                    "%_months_abs_ic_ge_0.03": np.nan,
                }
            )
            continue
        count = len(series)
        pct = (series.abs() >= IC_THRESHOLD).sum() / count * 100.0
        rows.append(
            {
                "predictor": pred,
                "months": count,
                "mean_ic": series.mean(),
                "median_ic": series.median(),
                "std_ic": series.std(ddof=1),
                "%_months_abs_ic_ge_0.03": pct,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_ic", key=lambda s: s.abs(), ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", default=os.getenv("SCHEMA", DEFAULT_SCHEMA))
    parser.add_argument("--table", default=os.getenv("TABLE", DEFAULT_TABLE))
    parser.add_argument("--min-cross", type=int, default=MIN_CROSS_SECTION)
    parser.add_argument("--out-csv", default="ic_monthly_by_predictor.csv")
    args = parser.parse_args()

    engine = build_engine()
    print(f"Reading monthly features from {args.schema}.{args.table}...")
    df = read_monthly_features(engine, args.schema, args.table)
    if df.empty:
        print("No data found.")
        return 1

    # auto detect predictors
    exclude = {"scheme_name", "month_date", "month_end_observation", "monthly_return_percent", "forward_return", "net_flow_amount"}
    predictors = [c for c in df.select_dtypes(include=["number"]).columns if c not in exclude]
    print(f"Detected predictors: {predictors}")

    # Try computing monthly ICs with automatic fallback to smaller cross-sections
    tried = []
    ic_monthly = None
    for threshold in [args.min_cross, 3, 1]:
        if threshold in tried:
            continue
        tried.append(threshold)
        print(f"Computing monthly ICs with min_cross = {threshold}...")
        candidate = compute_monthly_cross_sectional_ic(df, predictors=predictors, min_cross_section=threshold)
        # check if any predictor-month IC was produced
        non_null_counts = candidate[predictors].notna().sum().sum() if not candidate.empty else 0
        print(f"  produced {int(non_null_counts):,} non-missing IC values across all months/predictors")
        if non_null_counts > 0:
            ic_monthly = candidate
            used_min_cross = threshold
            break
        else:
            print(f"  no ICs produced with min_cross={threshold}, trying smaller threshold...")

    if ic_monthly is None:
        print("No monthly ICs could be computed even with min_cross down to 1. Exiting.")
        # still write empty outputs for transparency
        pd.DataFrame().to_csv(args.out_csv, index=False)
        pd.DataFrame().to_csv(os.path.splitext(args.out_csv)[0] + "_summary.csv", index=False)
        return 1

    summary = summarize_ic(ic_monthly, predictors)

    print(f"IC summary (top by absolute mean) — using min_cross = {used_min_cross}:")
    print(summary.to_string(index=False, float_format="{:.4f}".format))

    # Save detailed monthly ICs and summary
    ic_monthly.to_csv(args.out_csv, index=False)
    summary.to_csv(os.path.splitext(args.out_csv)[0] + "_summary.csv", index=False)
    print(f"Wrote {args.out_csv} and summary file (used min_cross = {used_min_cross}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
