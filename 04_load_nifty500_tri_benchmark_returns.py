"""Calculate Nifty 500 TRI monthly benchmark returns from monthly close prices.

The source table stores one Nifty 500 TRI bar per month. Returns are calculated
in pandas from consecutive ``close_price`` values, then written to a dedicated
benchmark-return table for joining to fund monthly-return records.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from sqlalchemy import URL, create_engine, text


DEFAULT_SCHEMA = "mf200"
SOURCE_TABLE = "investing_nifty500_tri_monthly_2006"
DEFAULT_TARGET_TABLE = "nifty500_tri_monthly_benchmark_returns_2006"
BENCHMARK_NAME = "Nifty 500 TRI"


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


def read_monthly_prices(engine, schema: str, source_table: str) -> pd.DataFrame:
    """Read the stored benchmark month and close-price data."""
    source = f"{quote_identifier(schema)}.{quote_identifier(source_table)}"
    frame = pd.read_sql(
        text(
            f"""
            SELECT observation_month, month_end, close_price
            FROM {source}
            """
        ),
        engine,
    )
    frame["observation_month"] = pd.to_datetime(frame["observation_month"]).dt.normalize()
    frame["month_end"] = pd.to_datetime(frame["month_end"]).dt.normalize()
    frame["close_price"] = pd.to_numeric(frame["close_price"], errors="coerce")
    frame = frame.dropna(subset=["observation_month", "month_end", "close_price"])
    return frame.sort_values("observation_month").drop_duplicates(
        "observation_month", keep="last"
    )


def calculate_monthly_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate return percentage from the previous calendar month's close."""
    returns = monthly_prices.copy()
    returns["previous_month_close"] = returns["close_price"].shift(1)
    returns["benchmark_monthly_return_percent"] = (
        (returns["close_price"] / returns["previous_month_close"] - 1) * 100
    )
    returns.loc[returns["previous_month_close"] == 0, "benchmark_monthly_return_percent"] = pd.NA
    returns.insert(0, "benchmark_name", BENCHMARK_NAME)
    return returns


def write_benchmark_returns(engine, schema: str, target_table: str, returns: pd.DataFrame) -> int:
    """Create the benchmark table if needed and upsert calculated return rows."""
    quoted_schema = quote_identifier(schema)
    target = f"{quoted_schema}.{quote_identifier(target_table)}"
    records = []
    for row in returns.itertuples(index=False):
        has_previous_close = pd.notna(row.previous_month_close)
        records.append(
            {
                "benchmark_name": row.benchmark_name,
                "observation_month": row.observation_month.date(),
                "month_end": row.month_end.date(),
                "close_price": row.close_price,
                "previous_month_close": row.previous_month_close if has_previous_close else None,
                "benchmark_monthly_return_percent": (
                    row.benchmark_monthly_return_percent if has_previous_close else None
                ),
            }
        )

    with engine.begin() as connection:
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {quoted_schema}"))
        connection.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {target} (
                    benchmark_name TEXT NOT NULL,
                    observation_month DATE NOT NULL,
                    month_end DATE NOT NULL,
                    close_price NUMERIC(14, 2) NOT NULL,
                    previous_month_close NUMERIC(14, 2),
                    benchmark_monthly_return_percent NUMERIC(12, 6),
                    loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (benchmark_name, observation_month)
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
                    benchmark_name, observation_month, month_end, close_price,
                    previous_month_close, benchmark_monthly_return_percent
                ) VALUES (
                    :benchmark_name, :observation_month, :month_end, :close_price,
                    :previous_month_close, :benchmark_monthly_return_percent
                )
                ON CONFLICT (benchmark_name, observation_month) DO UPDATE SET
                    month_end = EXCLUDED.month_end,
                    close_price = EXCLUDED.close_price,
                    previous_month_close = EXCLUDED.previous_month_close,
                    benchmark_monthly_return_percent = EXCLUDED.benchmark_monthly_return_percent,
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
    prices = read_monthly_prices(engine, args.schema, args.source_table)
    returns = calculate_monthly_returns(prices)
    row_count = write_benchmark_returns(engine, args.schema, args.target_table, returns)
    print(f"Upserted {row_count} Nifty 500 TRI benchmark-return rows into {args.schema}.{args.target_table}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
