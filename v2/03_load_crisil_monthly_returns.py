"""Create one monthly return record per fund from daily rolling-return rows.

The input is ``crisil_fund_rolling_30d_returns`` produced by
``02_load_mf_monthly_returns.py``. For each fund/calendar month, this
script retains the last available daily return row. No return calculation is
performed here.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from sqlalchemy import URL, create_engine, text
from pathlib import Path

RUN_TIMESTAMP_FILE = Path("D:\\PycharmProjects\\mf-ml\\run_timestamp.txt")

if not RUN_TIMESTAMP_FILE.exists():
    raise FileNotFoundError(
        f"Run timestamp file not found: {RUN_TIMESTAMP_FILE.resolve()}"
    )

RUN_TIMESTAMP = RUN_TIMESTAMP_FILE.read_text().strip()

if not RUN_TIMESTAMP:
    raise ValueError("run_timestamp.txt is empty")

print("Run timestamp:", RUN_TIMESTAMP)

DEFAULT_SCHEMA = "mf200"

SOURCE_TABLE = f"crisil_fund_rolling_30d_returns_{RUN_TIMESTAMP}"

DEFAULT_TARGET_TABLE = (
    f"crisil_fund_monthly_return_rows_{RUN_TIMESTAMP}"
)

# DEFAULT_SCHEMA = "mf200"
# SOURCE_TABLE = "crisil_fund_rolling_30d_returns"
# DEFAULT_TARGET_TABLE = "crisil_fund_monthly_return_rows"


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


def read_daily_returns(engine, schema: str, source_table: str) -> pd.DataFrame:
    """Read the already-calculated daily return rows."""
    source = f"{quote_identifier(schema)}.{quote_identifier(source_table)}"
    frame = pd.read_sql(
        text(
            f"""
            SELECT scheme_name, report_date, monthly_return_percent
            FROM {source}
            """
        ),
        engine,
    )
    frame["report_date"] = pd.to_datetime(frame["report_date"]).dt.normalize()
    frame["monthly_return_percent"] = pd.to_numeric(
        frame["monthly_return_percent"], errors="coerce"
    )
    return frame.dropna(subset=["scheme_name", "report_date"])


def select_monthly_rows(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Keep each fund's last available daily return row in a calendar month."""
    frame = daily_returns.copy()
    frame["month_date"] = frame["report_date"].dt.to_period("M").dt.to_timestamp()

    # Sorting followed by groupby(...).tail(1) selects the actual latest daily
    # observation; it does not recalculate or aggregate a return.
    frame = frame.sort_values(["scheme_name", "month_date", "report_date"])
    monthly = frame.groupby(["scheme_name", "month_date"], as_index=False).tail(1)
    monthly = monthly.rename(columns={"report_date": "trade_month_end"})

    # A month whose final daily row has no 30-day reference NAV is not a usable
    # monthly-return training record.
    return monthly.dropna(subset=["monthly_return_percent"])[
        ["scheme_name", "month_date", "trade_month_end", "monthly_return_percent"]
    ]


def write_monthly_rows(engine, schema: str, target_table: str, rows: pd.DataFrame) -> int:
    """Create the monthly table if needed and upsert its rows."""
    quoted_schema = quote_identifier(schema)
    target = f"{quoted_schema}.{quote_identifier(target_table)}"
    records = [
        {
            "scheme_name": row.scheme_name,
            "month_date": row.month_date.date(),
            "trade_month_end": row.trade_month_end.date(),
            "monthly_return_percent": row.monthly_return_percent,
        }
        for row in rows.itertuples(index=False)
    ]

    with engine.begin() as connection:
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {quoted_schema}"))
        connection.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {target} (
                    scheme_name TEXT NOT NULL,
                    month_date DATE NOT NULL,
                    trade_month_end DATE NOT NULL,
                    monthly_return_percent NUMERIC(12, 6) NOT NULL,
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
                    scheme_name, month_date, trade_month_end, monthly_return_percent
                ) VALUES (
                    :scheme_name, :month_date, :trade_month_end, :monthly_return_percent
                )
                ON CONFLICT (scheme_name, month_date) DO UPDATE SET
                    trade_month_end = EXCLUDED.trade_month_end,
                    monthly_return_percent = EXCLUDED.monthly_return_percent,
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
    daily_returns = read_daily_returns(engine, args.schema, args.source_table)
    monthly_rows = select_monthly_rows(daily_returns)
    row_count = write_monthly_rows(engine, args.schema, args.target_table, monthly_rows)
    print(f"Upserted {row_count} monthly return rows into {args.schema}.{args.target_table}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
