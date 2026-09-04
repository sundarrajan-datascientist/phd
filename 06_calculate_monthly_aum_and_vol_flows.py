"""Create monthly AUM and trailing 12-month flow-volatility features.

For each fund and calendar month, the final available ``dailyAUM`` and
``navDirect`` observation from ``crisil_fund_daily_raw`` is retained.  Net fund
flow is then calculated as::

    flow_t = AUM_t - AUM_(t-1) * (1 + monthly_return_t)

where ``monthly_return_t`` is the change in the retained monthly NAV.  The
``vol_flows_12m`` feature is the sample standard deviation of the most recent
12 monthly net-flow amounts.  It is reported only when all 12 months are
consecutive, avoiding a misleading volatility estimate across data gaps.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from sqlalchemy import URL, create_engine, text


DEFAULT_SCHEMA = "mf200"
SOURCE_TABLE = "crisil_fund_daily_raw"
DEFAULT_TARGET_TABLE = "crisil_fund_monthly_aum_flow_features"
FLOW_WINDOW_MONTHS = 12


def quote_identifier(identifier: str) -> str:
    """Return a safely quoted PostgreSQL identifier."""
    if not identifier or "\x00" in identifier:
        raise ValueError("Schema and table names must be non-empty identifiers")
    return '"' + identifier.replace('"', '""') + '"'


def build_engine():
    """Build a PostgreSQL engine from DB_* environment variables."""
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


def read_daily_aum_and_nav(engine, schema: str, source_table: str) -> pd.DataFrame:
    """Read only the raw fields required for monthly AUM and flow features."""
    source = f"{quote_identifier(schema)}.{quote_identifier(source_table)}"
    frame = pd.read_sql(
        text(
            f'''SELECT "schemeName" AS scheme_name, report_date,
                       "dailyAUM" AS daily_aum, "navDirect" AS nav_direct
                FROM {source}
                WHERE "schemeName" IS NOT NULL
                  AND report_date IS NOT NULL
                  AND "dailyAUM" IS NOT NULL
                  AND "navDirect" IS NOT NULL
                  AND benchmark = 'Nifty 500 TRI'
                '''
        ),
        engine,
    )
    frame["report_date"] = pd.to_datetime(frame["report_date"]).dt.normalize()
    frame["daily_aum"] = pd.to_numeric(frame["daily_aum"], errors="coerce")
    frame["nav_direct"] = pd.to_numeric(frame["nav_direct"], errors="coerce")
    return frame.dropna(subset=["daily_aum", "nav_direct"])


def calculate_monthly_aum_and_vol_flows(daily_rows: pd.DataFrame) -> pd.DataFrame:
    """Return one monthly AUM/flow feature row per fund.

    The final raw row in each calendar month is used.  The first monthly row
    for a fund has no prior AUM or return and hence no flow.  ``vol_flows_12m``
    needs 12 consecutive monthly flow observations.
    """
    required_columns = {"scheme_name", "report_date", "daily_aum", "nav_direct"}
    missing_columns = required_columns.difference(daily_rows.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    rows = daily_rows.copy()
    rows["report_date"] = pd.to_datetime(rows["report_date"], errors="coerce").dt.normalize()
    rows["daily_aum"] = pd.to_numeric(rows["daily_aum"], errors="coerce")
    rows["nav_direct"] = pd.to_numeric(rows["nav_direct"], errors="coerce")
    rows = rows.dropna(subset=["scheme_name", "report_date", "daily_aum", "nav_direct"])
    rows = rows.sort_values(["scheme_name", "report_date"]).drop_duplicates(
        ["scheme_name", "report_date"], keep="last"
    )
    rows["month_date"] = rows["report_date"].dt.to_period("M").dt.to_timestamp()
    monthly = rows.groupby(["scheme_name", "month_date"], as_index=False).tail(1).copy()
    monthly = monthly.rename(
        columns={"report_date": "month_end_observation", "daily_aum": "monthly_aum"}
    )
    monthly = monthly.sort_values(["scheme_name", "month_date"])
    grouped = monthly.groupby("scheme_name", sort=False)
    monthly["previous_month_aum"] = grouped["monthly_aum"].shift(1)
    monthly["previous_month_nav"] = grouped["nav_direct"].shift(1)
    # monthly return as a decimal and percent (keep percent for compatibility)
    monthly["monthly_return"] = (monthly["nav_direct"] / monthly["previous_month_nav"] - 1)
    monthly["monthly_return_percent"] = monthly["monthly_return"] * 100
    monthly.loc[monthly["previous_month_nav"] == 0, "monthly_return"] = pd.NA
    monthly.loc[monthly["previous_month_nav"] == 0, "monthly_return_percent"] = pd.NA

    # Net flow amount (rupee) and relative net flow (decimal)
    monthly["net_flow_amount"] = monthly["monthly_aum"] - (
        monthly["previous_month_aum"] * (1 + monthly["monthly_return"]) 
    )
    monthly.loc[monthly["previous_month_aum"] <= 0, "net_flow_amount"] = pd.NA
    monthly["net_flow"] = monthly["net_flow_amount"] / monthly["previous_month_aum"]
    monthly.loc[monthly["previous_month_aum"] <= 0, "net_flow"] = pd.NA
    monthly["net_flow_percent"] = monthly["net_flow"] * 100

    monthly["month_number"] = monthly["month_date"].dt.year * 12 + monthly["month_date"].dt.month

    monthly["month_gap"] = (
            monthly["month_number"]
            - grouped["month_number"].shift(1)
    )

    # Flow can only be calculated when the previous observation
    # belongs to the immediately preceding calendar month.
    monthly.loc[
        monthly["month_gap"] != 1,
        ["previous_month_aum", "previous_month_nav",
         "monthly_return", "monthly_return_percent", "net_flow_amount", "net_flow", "net_flow_percent"]
    ] = pd.NA

    # Vol_Flows is the sample stddev of the past 12 monthly relative flows (decimal)
    # Vol_Flows = sample standard deviation of the past 12 monthly
    # net flow amounts, as defined in the paper.
    monthly["vol_flows_12m"] = grouped["net_flow_amount"].transform(
        lambda values: values.rolling(
            FLOW_WINDOW_MONTHS,
            min_periods=FLOW_WINDOW_MONTHS
        ).std()
    )

    month_span = grouped["month_number"].transform(
        lambda values: values - values.shift(FLOW_WINDOW_MONTHS - 1)
    )
    monthly.loc[month_span != FLOW_WINDOW_MONTHS - 1, "vol_flows_12m"] = pd.NA

    return monthly[
        [
            "scheme_name",
            "month_date",
            "month_end_observation",
            "monthly_aum",
            "monthly_return_percent",
            "net_flow_amount",
            "net_flow_percent",
            "vol_flows_12m",
        ]
    ].reset_index(drop=True)


def write_features(engine, schema: str, target_table: str, features: pd.DataFrame) -> int:
    """Create the feature table if needed and upsert all calculated rows."""
    quoted_schema = quote_identifier(schema)
    target = f"{quoted_schema}.{quote_identifier(target_table)}"
    records = [
        {
            "scheme_name": row.scheme_name,
            "month_date": row.month_date.date(),
            "month_end_observation": row.month_end_observation.date(),
            "monthly_aum": row.monthly_aum,
            "monthly_return_percent": row.monthly_return_percent if pd.notna(row.monthly_return_percent) else None,
            "net_flow_amount": row.net_flow_amount if pd.notna(row.net_flow_amount) else None,
            "net_flow_percent": row.net_flow_percent if pd.notna(row.net_flow_percent) else None,
            "vol_flows_12m": row.vol_flows_12m if pd.notna(row.vol_flows_12m) else None,
        }
        for row in features.itertuples(index=False)
    ]
    with engine.begin() as connection:
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {quoted_schema}"))
        connection.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {target} (
                    scheme_name TEXT NOT NULL,
                    month_date DATE NOT NULL,
                    month_end_observation DATE NOT NULL,
                    monthly_aum NUMERIC(18, 6) NOT NULL,
                    monthly_return_percent NUMERIC(12, 6),
                    net_flow_amount NUMERIC(18, 6),
                    net_flow_percent NUMERIC(12, 6),
                    vol_flows_12m NUMERIC(18, 6),
                    loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (scheme_name, month_date)
                )
                """
            )
        )
        if not records:
            return 0
        result = connection.execute(
            text(
                f"""
                INSERT INTO {target} (
                    scheme_name, month_date, month_end_observation, monthly_aum,
                    monthly_return_percent, net_flow_amount, net_flow_percent,
                    vol_flows_12m
                ) VALUES (
                    :scheme_name, :month_date, :month_end_observation, :monthly_aum,
                    :monthly_return_percent, :net_flow_amount, :net_flow_percent,
                    :vol_flows_12m
                )
                ON CONFLICT (scheme_name, month_date) DO UPDATE SET
                    month_end_observation = EXCLUDED.month_end_observation,
                    monthly_aum = EXCLUDED.monthly_aum,
                    monthly_return_percent = EXCLUDED.monthly_return_percent,
                    net_flow_amount = EXCLUDED.net_flow_amount,
                    net_flow_percent = EXCLUDED.net_flow_percent,
                    vol_flows_12m = EXCLUDED.vol_flows_12m,
                    loaded_at = NOW()
                """
            ),
            records,
        )
    return result.rowcount


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", default=os.getenv("SCHEMA", DEFAULT_SCHEMA))
    parser.add_argument("--source-table", default=SOURCE_TABLE)
    parser.add_argument("--target-table", default=os.getenv("TABLE", DEFAULT_TARGET_TABLE))
    args = parser.parse_args()

    engine = build_engine()
    daily_rows = read_daily_aum_and_nav(engine, args.schema, args.source_table)
    print(f"Loaded {len(daily_rows):,} raw AUM/NAV rows.")
    features = calculate_monthly_aum_and_vol_flows(daily_rows)
    print(f"Created {len(features):,} monthly AUM/flow rows.")
    print(f"Rows with a complete 12-month Vol_Flows value: {features['vol_flows_12m'].notna().sum():,}.")
    if not features.empty:
        print("Feature sample:")
        print(features.head(10).to_string(index=False))
    row_count = write_features(engine, args.schema, args.target_table, features)
    print(f"Upserted {row_count} rows into {args.schema}.{args.target_table}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
