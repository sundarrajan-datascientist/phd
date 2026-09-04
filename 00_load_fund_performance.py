"""Load CRISIL fund-performance API responses into PostgreSQL.

Fetches every date from 01-Jan-2019 through yesterday.
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta
from typing import Iterator

import pandas as pd
from sqlalchemy import create_engine, inspect, text

from main import get_fund_performance_for_subcategories, get_subcategories


DEFAULT_START_DATE = date(2018, 12, 23)
DEFAULT_CATEGORY = 1
DEFAULT_MATURITY_TYPE = 1
DEFAULT_MFID = 0
REQUEST_DELAY_SECONDS = 0.2
SCHEMA = os.getenv("SCHEMA", "mf200")
TABLE = os.getenv("TABLE", "crisil_fund_daily_raw")

# Update these mappings if you load additional CRISIL category or maturity IDs.
CATEGORY_NAMES = {1: "Equity"}
MATURITY_NAMES = {1: "Open Ended"}


def get_report_dates(start: date, end: date) -> Iterator[date]:
    """Yield every calendar date from start through end, inclusive."""
    if start > end:
        raise ValueError("start date must be on or before end date")

    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def append_report_date(
    engine,
    schema: str,
    table: str,
    report_date: date,
    category: int,
    maturity_type: int,
    mfid: int,
    subcategories: list[dict],
) -> int:
    """Fetch and append one row per fund-performance record on a report date."""
    api_date = report_date.strftime("%d-%b-%Y")
    responses = get_fund_performance_for_subcategories(
        subcategories,
        category=category,
        report_date=api_date,
        maturity_type=maturity_type,
        mfid=mfid,
    )
    frames: list[pd.DataFrame] = []
    for result in responses:
        response = result["response"]
        records = response.get("data", []) if isinstance(response, dict) else response
        if isinstance(records, dict):
            records = [records]
        if not isinstance(records, list):
            raise RuntimeError(f"Unexpected fund-performance response: {response!r}")
        if not records:
            continue

        # Flatten every returned fund record into columns, preserving the
        # request context on each row.
        frame = pd.json_normalize(records, sep="_")
        frame.insert(0, "report_date", report_date)
        frame.insert(1, "category", category)
        frame.insert(2, "category_name", CATEGORY_NAMES.get(category, f"Unknown ({category})"))
        frame.insert(3, "subcategory_id", result["subcategory"]["id"])
        frame.insert(4, "subcategory_name", result["subcategory"]["name"])
        frame.insert(5, "maturity_type", maturity_type)
        frame.insert(
            6,
            "maturity_name",
            MATURITY_NAMES.get(maturity_type, f"Unknown ({maturity_type})"),
        )
        frame.insert(7, "mfid", mfid)
        frames.append(frame)

    if not frames:
        return 0
    frame = pd.concat(frames, ignore_index=True)
    frame.to_sql(
        name=table,
        con=engine,
        schema=schema,
        if_exists="append",
        index=False,
        method="multi",
    )
    return len(frame)


def build_engine():
    """Create an engine from environment variables, without hard-coding credentials."""

    DB_USER = "postgres"
    DB_PASSWORD = "Password%40123"
    DB_HOST = "localhost"
    DB_PORT = "5432"
    DB_NAME = "mfdb"

    engine = create_engine(
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    print("Database connection created.")

    return engine



def get_start_date(engine, schema: str, table: str) -> date:
    """Resume from the day after the latest successfully saved report date."""
    if not inspect(engine).has_table(table, schema=schema):
        return DEFAULT_START_DATE

    quoted_schema = schema.replace('"', '""')
    quoted_table = table.replace('"', '""')
    with engine.connect() as connection:
        latest_report_date = connection.execute(
            text(
                f'SELECT max(report_date) '
                f'FROM "{quoted_schema}"."{quoted_table}"'
            )
        ).scalar_one()

    if latest_report_date is None:
        return DEFAULT_START_DATE
    if hasattr(latest_report_date, "date"):
        latest_report_date = latest_report_date.date()
    return latest_report_date + timedelta(days=1)





def main() -> int:
    end_date = date.today() - timedelta(days=1)
    engine = build_engine()

    start_date = get_start_date(engine, SCHEMA, TABLE)
    if start_date > end_date:
        print(f"Already up to date through {end_date.isoformat()}.")
        return 0
    print(f"Loading report dates from {start_date.isoformat()} through {end_date.isoformat()}.")
    subcategories = get_subcategories(DEFAULT_CATEGORY)
    print(f"Fetched {len(subcategories)} subcategories for category {DEFAULT_CATEGORY}.")
    for report_date in get_report_dates(start_date, end_date):
        count = append_report_date(
            engine,
            SCHEMA,
            TABLE,
            report_date,
            DEFAULT_CATEGORY,
            DEFAULT_MATURITY_TYPE,
            DEFAULT_MFID,
            subcategories,
        )
        print(f"{report_date.isoformat()}: saved {count} fund-performance rows")
        time.sleep(REQUEST_DELAY_SECONDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
