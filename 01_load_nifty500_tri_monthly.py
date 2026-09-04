"""Load the Nifty 500 TRI monthly history CSV into PostgreSQL.

The source label is retained as ``observation_month`` and the corresponding
calendar month-end is stored in ``month_end``. The loader is safe to rerun:
each observation month is stored once and updated when the source CSV changes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import URL, create_engine, text


DEFAULT_CSV = Path("data/Nifty 500 TRI Historical Monthly Data.csv")
DEFAULT_SCHEMA = "mf200"
DEFAULT_TABLE = "investing_nifty500_tri_monthly"


def quote_identifier(identifier: str) -> str:
    """Return a safely quoted PostgreSQL identifier."""
    if not identifier or "\x00" in identifier:
        raise ValueError("Schema and table names must be non-empty identifiers")
    return '"' + identifier.replace('"', '""') + '"'


def build_engine():
    """Build an engine from DB_* environment variables."""
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "Password@123")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    database = os.getenv("DB_NAME", "mfdb")
    return create_engine(
        URL.create(
            "postgresql+psycopg",
            username=user,
            password=password,
            host=host,
            port=int(port),
            database=database,
        )
    )


def read_source(csv_path: Path) -> list[dict]:
    """Read and normalize the Investing.com-style monthly CSV."""
    frame = pd.read_csv(csv_path)
    expected_columns = {"Date", "Price", "Open", "High", "Low", "Vol.", "Change %"}
    missing_columns = expected_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")

    source_date = pd.to_datetime(frame["Date"], format="%d/%m/%Y")
    result = pd.DataFrame(
        {
            # Source dates are day/month/year (for example, 01/07/2026).
            "observation_month": source_date,
            "month_end": source_date + pd.offsets.MonthEnd(0),
            "close_price": frame["Price"],
            "open_price": frame["Open"],
            "high_price": frame["High"],
            "low_price": frame["Low"],
            "volume": frame["Vol."],
            "change_percent": frame["Change %"],
        }
    )
    for column in ("close_price", "open_price", "high_price", "low_price"):
        result[column] = pd.to_numeric(
            result[column].astype("string").str.replace(",", "", regex=False),
            errors="coerce",
        )
    result["volume"] = parse_volume(frame["Vol."])
    result["change_percent"] = pd.to_numeric(
        result["change_percent"].astype("string").str.rstrip("%"), errors="coerce"
    )

    if result["observation_month"].isna().any():
        raise ValueError("CSV contains an invalid Date value")
    if result["observation_month"].duplicated().any():
        raise ValueError("CSV contains duplicate Date values")
    return result.where(pd.notna(result), None).to_dict(orient="records")


def parse_volume(values: pd.Series) -> pd.Series:
    """Convert optional Investing.com volume values (for example, 1.2M) to integers."""
    cleaned = values.astype("string").str.replace(",", "", regex=False).str.strip()
    multiplier = cleaned.str.extract(r"(?i)([KMB])$")[0].str.upper().map(
        {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    ).fillna(1)
    number = pd.to_numeric(
        cleaned.str.replace(r"(?i)[KMB]$", "", regex=True), errors="coerce"
    )
    return (number * multiplier).round().astype("Int64")


def load_rows(engine, schema: str, table: str, rows: list[dict]) -> None:
    quoted_schema = quote_identifier(schema)
    quoted_table = quote_identifier(table)
    qualified_table = f"{quoted_schema}.{quoted_table}"

    with engine.begin() as connection:
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {quoted_schema}"))
        connection.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {qualified_table} (
                    observation_month DATE PRIMARY KEY,
                    month_end DATE NOT NULL,
                    close_price NUMERIC(14, 2),
                    open_price NUMERIC(14, 2),
                    high_price NUMERIC(14, 2),
                    low_price NUMERIC(14, 2),
                    volume BIGINT,
                    change_percent NUMERIC(8, 2),
                    loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        )
        connection.execute(
            text(
                f"""
                INSERT INTO {qualified_table} (
                    observation_month, month_end, close_price, open_price, high_price, low_price,
                    volume, change_percent
                ) VALUES (
                    :observation_month, :month_end, :close_price, :open_price, :high_price, :low_price,
                    :volume, :change_percent
                )
                ON CONFLICT (observation_month) DO UPDATE SET
                    month_end = EXCLUDED.month_end,
                    close_price = EXCLUDED.close_price,
                    open_price = EXCLUDED.open_price,
                    high_price = EXCLUDED.high_price,
                    low_price = EXCLUDED.low_price,
                    volume = EXCLUDED.volume,
                    change_percent = EXCLUDED.change_percent,
                    loaded_at = NOW()
                """
            ),
            rows,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Source CSV path")
    parser.add_argument("--schema", default=os.getenv("SCHEMA", DEFAULT_SCHEMA))
    parser.add_argument("--table", default=os.getenv("TABLE", DEFAULT_TABLE))
    args = parser.parse_args()

    rows = read_source(args.csv)
    load_rows(build_engine(), args.schema, args.table, rows)
    print(f"Upserted {len(rows)} rows into {args.schema}.{args.table}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
