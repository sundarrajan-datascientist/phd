"""
Create the monthly feature matrix for the mutual-fund ML model.

The feature matrix combines:
1. Daily performance/momentum characteristics, reduced to the
   last available trading day of each calendar month.
2. Monthly AUM and flow characteristics.

The 22 model features are selected separately using the
2019-2023 development period and are NOT re-selected here.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import URL, create_engine, text


# ============================================================
# Run timestamp
# ============================================================

RUN_TIMESTAMP_FILE = Path("D:\\PycharmProjects\\mf-ml\\run_timestamp.txt")

if not RUN_TIMESTAMP_FILE.exists():
    raise FileNotFoundError(
        f"Run timestamp file not found: {RUN_TIMESTAMP_FILE.resolve()}"
    )

RUN_TIMESTAMP = RUN_TIMESTAMP_FILE.read_text().strip()

if not RUN_TIMESTAMP:
    raise ValueError("run_timestamp.txt is empty")

print("Run timestamp:", RUN_TIMESTAMP)


# ============================================================
# Database configuration
# ============================================================

SCHEMA = "mf200"

PERFORMANCE_TABLE = (
    f"performance_momentum_features_2013_2019_20260912_003517"
)

# AUM/flow table is intentionally still the existing table.
# We have not performed AUM anomaly detection yet.
AUM_FLOW_TABLE = "crisil_fund_monthly_aum_flow_features"

TARGET_TABLE = f"feature_matrix_2013_2019_{RUN_TIMESTAMP}"


# ============================================================
# Database connection
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
# Create feature matrix
# ============================================================

def create_feature_matrix(engine) -> pd.DataFrame:

    sql = f"""
    WITH monthly_performance AS (
        SELECT *
        FROM (
            SELECT
                p.*,
                DATE_TRUNC('month', p.trade_date)::date AS month_date,

                ROW_NUMBER() OVER (
                    PARTITION BY
                        p.scheme_name,
                        DATE_TRUNC('month', p.trade_date)
                    ORDER BY p.trade_date DESC
                ) AS rn

            FROM mf100.{PERFORMANCE_TABLE} p

            WHERE p.trade_date >= DATE '2013-01-01'
              AND p.trade_date <  DATE '2020-01-01'
        ) x

        WHERE rn = 1
    )

    SELECT
        p.scheme_name,
        p.month_date,
        p.trade_date AS performance_date,

        -- Fundamentals
        a.monthly_aum AS tna,
        a.vol_flows_12m AS vol_flows,

        -- Performance / momentum
        p.const_90,
        p.const_120,
        p.const_180,
        p.const_250,
        p.const_750,

        p.r2_90,
        p.r2_120,
        p.r2_180,
        p.r2_250,
        p.r2_750,

        p.t_const_90,
        p.t_const_120,
        p.t_const_180,
        p.t_const_250,
        p.t_const_750,

        p.t_mkt_90,
        p.t_mkt_120,
        p.t_mkt_180,
        p.t_mkt_250,
        p.t_mkt_750,

        p.t_smb_90,
        p.t_smb_120,
        p.t_smb_180,
        p.t_smb_250,
        p.t_smb_750,

        p.t_hml_90,
        p.t_hml_120,
        p.t_hml_180,
        p.t_hml_250,
        p.t_hml_750,

        p.t_umd_90,
        p.t_umd_120,
        p.t_umd_180,
        p.t_umd_250,
        p.t_umd_750,

        p.ir_p6m,
        p.ir_p1y,
        p.ir_p3y,

        p.sr_p6m,
        p.sr_p1y,
        p.sr_p3y

    FROM monthly_performance p

    left JOIN {SCHEMA}.{AUM_FLOW_TABLE} a
        ON  a.scheme_name = p.scheme_name
        AND a.month_date  = p.month_date

    ORDER BY
        p.month_date,
        p.scheme_name
    """

    return pd.read_sql(text(sql), engine)


# ============================================================
# Write table
# ============================================================

def write_feature_matrix(
    engine,
    features: pd.DataFrame,
) -> None:

    # Replace the table on each run.
    # The timestamped name makes each run independently traceable.
    features.to_sql(
        TARGET_TABLE,
        engine,
        schema=SCHEMA,
        if_exists="replace",
        index=False,
        method=None,
        chunksize=5000,
    )


# ============================================================
# Main
# ============================================================

def main() -> int:

    engine = build_engine()

    print()
    print("Source tables:")
    print(f"  Performance : {SCHEMA}.{PERFORMANCE_TABLE}")
    print(f"  AUM / Flow  : {SCHEMA}.{AUM_FLOW_TABLE}")
    print()
    print(f"Target table:")
    print(f"  {SCHEMA}.{TARGET_TABLE}")
    print()

    features = create_feature_matrix(engine)

    print(f"Created {len(features):,} feature-matrix rows.")

    if features.empty:
        raise RuntimeError("Feature matrix is empty.")

    print(
        f"Funds: {features['scheme_name'].nunique():,}"
    )

    print(
        f"Months: {features['month_date'].min()} "
        f"to {features['month_date'].max()}"
    )

    print()
    print("Feature matrix sample:")
    print(features.head(5).to_string(index=False))

    write_feature_matrix(engine, features)

    print()
    print(
        f"Saved {len(features):,} rows into "
        f"{SCHEMA}.{TARGET_TABLE}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())