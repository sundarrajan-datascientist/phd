"""
Compute monthly cross-sectional Spearman ICs for all ML predictors.

Feature sources:
    mf200.crisil_fund_monthly_aum_flow_features
    mf200.performance_momentum_features

Target source:
    mf200.fund_monthly_outperformance_targets

For each fund and month t:

    Features = information available at month t
    Forward return = fund return for target month t+1

The target mapping is taken directly from
fund_monthly_outperformance_targets.

IC is calculated cross-sectionally for each month using
Spearman rank correlation between:

    predictor(t)  <->  future fund return(t+1)

Then mean/median/std IC and percentage of months with
|IC| >= 0.03 are reported.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sqlalchemy import URL, create_engine, text


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_SCHEMA = "mf200"

AUM_FLOW_TABLE = "crisil_fund_monthly_aum_flow_features"
MOMENTUM_TABLE = "performance_momentum_features"
TARGET_TABLE = "fund_monthly_outperformance_targets"

MIN_CROSS_SECTION = 10
IC_THRESHOLD = 0.03

BENCHMARK_NAME = "Nifty 500 TRI"


# ============================================================
# DATABASE
# ============================================================

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


# ============================================================
# READ AUM / FLOW FEATURES
# ============================================================

def read_aum_flow_features(
    engine,
    schema: str,
    table: str
) -> pd.DataFrame:

    source = f'"{schema}"."{table}"'

    sql = text(
        f"""
        SELECT
            scheme_name,
            month_date,
            month_end_observation,
            monthly_aum,
            monthly_return_percent,
            net_flow_amount,
            net_flow_percent,
            vol_flows_12m
        FROM {source}
        """
    )

    df = pd.read_sql(sql, engine)

    df["month_date"] = pd.to_datetime(
        df["month_date"]
    ).dt.normalize()

    numeric_cols = [
        "monthly_aum",
        "monthly_return_percent",
        "net_flow_amount",
        "net_flow_percent",
        "vol_flows_12m",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


# ============================================================
# READ MOMENTUM FEATURES
# ============================================================

def read_momentum_features(
    engine,
    schema: str,
    table: str
) -> pd.DataFrame:

    source = f'"{schema}"."{table}"'

    sql = text(
        f"""
        SELECT *
        FROM {source}
        """
    )

    df = pd.read_sql(sql, engine)

    df["trade_date"] = pd.to_datetime(
        df["trade_date"]
    ).dt.normalize()

    # Convert trade date to observation month
    df["month_date"] = (
        df["trade_date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    # Make sure numeric columns are numeric
    identifier_cols = {
        "scheme_name",
        "trade_date",
        "month_date",
        "loaded_at",
    }

    numeric_cols = [
        c for c in df.columns
        if c not in identifier_cols
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Keep the last available trading observation in each
    # fund/month.
    # --------------------------------------------------------

    df = (
        df.sort_values(
            ["scheme_name", "month_date", "trade_date"]
        )
        .drop_duplicates(
            subset=["scheme_name", "month_date"],
            keep="last"
        )
    )

    return df


# ============================================================
# READ TARGET
# ============================================================

def read_targets(
    engine,
    schema: str,
    table: str
) -> pd.DataFrame:

    source = f'"{schema}"."{table}"'

    sql = text(
        f"""
        SELECT
            scheme_name,
            month_date,
            current_monthly_return_percent,
            current_benchmark_monthly_return_percent,
            target_month,
            trade_month_end,
            benchmark_name,
            benchmark_month_end,
            monthly_return_percent,
            benchmark_monthly_return_percent,
            excess_return_percent,
            outperforms_benchmark
        FROM {source}
        WHERE benchmark_name = :benchmark
          AND monthly_return_percent IS NOT NULL
        """
    )

    df = pd.read_sql(
        sql,
        engine,
        params={"benchmark": BENCHMARK_NAME}
    )

    df["month_date"] = pd.to_datetime(
        df["month_date"]
    ).dt.normalize()

    df["target_month"] = pd.to_datetime(
        df["target_month"]
    ).dt.normalize()

    numeric_cols = [
        "current_monthly_return_percent",
        "current_benchmark_monthly_return_percent",
        "monthly_return_percent",
        "benchmark_monthly_return_percent",
        "excess_return_percent",
        "outperforms_benchmark",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


# ============================================================
# BUILD COMPLETE FEATURE DATASET
# ============================================================

def build_feature_dataset(
    aum_flow: pd.DataFrame,
    momentum: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:

    # --------------------------------------------------------
    # Check AUM/flow duplicates
    # --------------------------------------------------------

    dup_aum = aum_flow.duplicated(
        subset=["scheme_name", "month_date"]
    ).sum()

    if dup_aum > 0:
        raise ValueError(
            f"AUM/flow table contains {dup_aum} "
            "duplicate scheme/month rows."
        )

    # --------------------------------------------------------
    # Check momentum duplicates
    # --------------------------------------------------------

    dup_momentum = momentum.duplicated(
        subset=["scheme_name", "month_date"]
    ).sum()

    if dup_momentum > 0:
        raise ValueError(
            f"Momentum table contains {dup_momentum} "
            "duplicate scheme/month rows."
        )

    # --------------------------------------------------------
    # Check target duplicates
    # --------------------------------------------------------

    dup_target = targets.duplicated(
        subset=["scheme_name", "month_date"]
    ).sum()

    if dup_target > 0:
        raise ValueError(
            f"Target table contains {dup_target} "
            "duplicate scheme/month rows."
        )

    # --------------------------------------------------------
    # Rename AUM/flow features where necessary
    # --------------------------------------------------------

    aum_flow = aum_flow.copy()

    # --------------------------------------------------------
    # Keep only the actual AUM/flow predictors
    # --------------------------------------------------------

    aum_columns = [
        "scheme_name",
        "month_date",
        "monthly_aum",
        "monthly_return_percent",
        "net_flow_amount",
        "net_flow_percent",
        "vol_flows_12m",
    ]

    aum_flow = aum_flow[aum_columns]

    # --------------------------------------------------------
    # Momentum predictors
    # --------------------------------------------------------

    momentum_identifier_cols = {
        "scheme_name",
        "trade_date",
        "month_date",
        "loaded_at",
    }

    momentum_predictors = [
        c
        for c in momentum.columns
        if c not in momentum_identifier_cols
    ]

    momentum_keep = [
        "scheme_name",
        "month_date",
    ] + momentum_predictors

    momentum = momentum[momentum_keep]

    # --------------------------------------------------------
    # Rename momentum columns if there is a collision
    # --------------------------------------------------------

    collisions = set(aum_flow.columns).intersection(
        set(momentum.columns)
    ) - {"scheme_name", "month_date"}

    if collisions:
        momentum = momentum.rename(
            columns={
                c: f"momentum_{c}"
                for c in collisions
            }
        )

        momentum_predictors = [
            f"momentum_{c}" if c in collisions else c
            for c in momentum_predictors
        ]

    # --------------------------------------------------------
    # Join AUM/flow + momentum
    # --------------------------------------------------------

    features = pd.merge(
        aum_flow,
        momentum,
        on=["scheme_name", "month_date"],
        how="outer",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Join target
    #
    # target.month_date = feature month t
    # target.target_month = future month t+1
    #
    # target.monthly_return_percent =
    #       actual return during t+1
    # --------------------------------------------------------

    target_columns = [
        "scheme_name",
        "month_date",
        "target_month",
        "monthly_return_percent",
        "benchmark_monthly_return_percent",
        "excess_return_percent",
        "outperforms_benchmark",
    ]

    targets_small = targets[target_columns].copy()

    # The AUM table also has monthly_return_percent.
    # Rename target return so there is no collision.
    targets_small = targets_small.rename(
        columns={
            "monthly_return_percent": "future_return",
            "benchmark_monthly_return_percent":
                "future_benchmark_return",
            "excess_return_percent":
                "future_excess_return",
            "outperforms_benchmark":
                "future_outperformance",
        }
    )

    features = pd.merge(
        features,
        targets_small,
        on=["scheme_name", "month_date"],
        how="inner",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Make sure future return is numeric
    # --------------------------------------------------------

    features["future_return"] = pd.to_numeric(
        features["future_return"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Determine predictor columns
    # --------------------------------------------------------

    identifier_cols = {
        "scheme_name",
        "month_date",
        "target_month",
        "future_return",
        "future_benchmark_return",
        "future_excess_return",
        "future_outperformance",
    }

    predictors = [
        c
        for c in features.select_dtypes(
            include=["number"]
        ).columns
        if c not in identifier_cols
    ]

    return features, predictors


# ============================================================
# COMPUTE MONTHLY CROSS-SECTIONAL SPEARMAN IC
# ============================================================

def compute_monthly_cross_sectional_ic(
    df: pd.DataFrame,
    predictors: Iterable[str],
    min_cross_section: int = MIN_CROSS_SECTION,
) -> pd.DataFrame:

    predictors = list(predictors)

    month_dates = sorted(
        df["month_date"].dropna().unique()
    )

    records = []

    for month in month_dates:

        month_data = df[
            df["month_date"] == month
        ].copy()

        if month_data.empty:
            continue

        row = {
            "month_date": month
        }

        for predictor in predictors:

            aligned = month_data[
                ["future_return", predictor]
            ].dropna()

            # Minimum number of funds
            if len(aligned) < min_cross_section:
                row[predictor] = np.nan
                continue

            # Need variation in both variables
            if aligned["future_return"].nunique() <= 1:
                row[predictor] = np.nan
                continue

            if aligned[predictor].nunique() <= 1:
                row[predictor] = np.nan
                continue

            try:

                corr, _ = spearmanr(
                    aligned["future_return"],
                    aligned[predictor]
                )

                row[predictor] = (
                    float(corr)
                    if not np.isnan(corr)
                    else np.nan
                )

            except Exception:
                row[predictor] = np.nan

        records.append(row)

    return (
        pd.DataFrame(records)
        .sort_values("month_date")
        .reset_index(drop=True)
    )


# ============================================================
# IC SUMMARY
# ============================================================

def summarize_ic(
    ic_df: pd.DataFrame,
    predictors: Iterable[str],
) -> pd.DataFrame:

    rows = []

    for predictor in predictors:

        series = (
            ic_df[predictor]
            .dropna()
        )

        if series.empty:

            rows.append({
                "predictor": predictor,
                "months": 0,
                "mean_ic": np.nan,
                "median_ic": np.nan,
                "std_ic": np.nan,
                "min_ic": np.nan,
                "max_ic": np.nan,
                "%_months_abs_ic_ge_0.03": np.nan,
            })

            continue

        rows.append({

            "predictor": predictor,

            "months": len(series),

            "mean_ic": series.mean(),

            "median_ic": series.median(),

            "std_ic": series.std(
                ddof=1
            ),

            "min_ic": series.min(),

            "max_ic": series.max(),

            "%_months_abs_ic_ge_0.03":
                (
                    series.abs() >= IC_THRESHOLD
                ).mean() * 100.0,
        })

    summary = pd.DataFrame(rows)

    summary["abs_mean_ic"] = (
        summary["mean_ic"].abs()
    )

    summary["ic_pass"] = np.where(
        summary["abs_mean_ic"] >= IC_THRESHOLD,
        "PASS",
        "FAIL",
    )

    return (
        summary
        .sort_values(
            "abs_mean_ic",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--schema",
        default=os.getenv(
            "SCHEMA",
            DEFAULT_SCHEMA
        ),
    )

    parser.add_argument(
        "--min-cross",
        type=int,
        default=MIN_CROSS_SECTION,
    )

    parser.add_argument(
        "--out-monthly",
        default="ic_monthly_all_features.csv",
    )

    parser.add_argument(
        "--out-summary",
        default="ic_summary_all_features.csv",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    engine = build_engine()

    # --------------------------------------------------------
    # Read tables
    # --------------------------------------------------------

    print(
        f"Reading AUM/flow features "
        f"from {args.schema}.{AUM_FLOW_TABLE}..."
    )

    aum_flow = read_aum_flow_features(
        engine,
        args.schema,
        AUM_FLOW_TABLE,
    )

    print(
        f"  rows: {len(aum_flow):,}"
    )

    print(
        f"Reading momentum features "
        f"from {args.schema}.{MOMENTUM_TABLE}..."
    )

    momentum = read_momentum_features(
        engine,
        args.schema,
        MOMENTUM_TABLE,
    )

    print(
        f"  rows: {len(momentum):,}"
    )

    print(
        f"Reading targets "
        f"from {args.schema}.{TARGET_TABLE}..."
    )

    targets = read_targets(
        engine,
        args.schema,
        TARGET_TABLE,
    )

    print(
        f"  rows: {len(targets):,}"
    )

    # --------------------------------------------------------
    # Build combined dataset
    # --------------------------------------------------------

    print(
        "Combining AUM/flow + momentum + target..."
    )

    df, predictors = build_feature_dataset(
        aum_flow,
        momentum,
        targets,
    )

    print(
        f"Combined rows: {len(df):,}"
    )

    print(
        f"Detected predictors ({len(predictors)}):"
    )

    print(predictors)

    # --------------------------------------------------------
    # Future return availability
    # --------------------------------------------------------

    print()
    print(
        "Future-return observations:"
    )

    print(
        f"  Non-null: "
        f"{df['future_return'].notna().sum():,}"
    )

    print(
        f"  Null: "
        f"{df['future_return'].isna().sum():,}"
    )

    # --------------------------------------------------------
    # Compute IC
    # --------------------------------------------------------

    print()
    print(
        f"Computing monthly cross-sectional "
        f"Spearman ICs with min_cross = "
        f"{args.min_cross}..."
    )

    ic_monthly = (
        compute_monthly_cross_sectional_ic(
            df,
            predictors,
            min_cross_section=args.min_cross,
        )
    )

    non_missing_ic = (
        ic_monthly[predictors]
        .notna()
        .sum()
        .sum()
    )

    print(
        f"Produced {int(non_missing_ic):,} "
        f"non-missing IC values."
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = summarize_ic(
        ic_monthly,
        predictors,
    )

    print()
    print(
        "===================================================="
    )

    print(
        "IC SUMMARY — ALL FEATURES"
    )

    print(
        "===================================================="
    )

    print(
        summary.to_string(
            index=False,
            float_format="{:.4f}".format,
        )
    )

    # --------------------------------------------------------
    # PASSING FEATURES
    # --------------------------------------------------------

    passing = summary[
        summary["ic_pass"] == "PASS"
    ]

    print()
    print(
        "===================================================="
    )

    print(
        f"FEATURES PASSING |IC| >= "
        f"{IC_THRESHOLD}"
    )

    print(
        "===================================================="
    )

    if passing.empty:

        print(
            "No features passed the IC threshold."
        )

    else:

        print(
            passing[
                [
                    "predictor",
                    "months",
                    "mean_ic",
                    "median_ic",
                    "std_ic",
                    "abs_mean_ic",
                ]
            ].to_string(
                index=False,
                float_format="{:.4f}".format,
            )
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    ic_monthly.to_csv(
        args.out_monthly,
        index=False,
    )

    summary.to_csv(
        args.out_summary,
        index=False,
    )

    print()
    print(
        f"Wrote monthly ICs to: "
        f"{args.out_monthly}"
    )

    print(
        f"Wrote IC summary to: "
        f"{args.out_summary}"
    )

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    raise SystemExit(main())