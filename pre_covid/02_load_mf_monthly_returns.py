"""Calculate 30-calendar-day fund returns from daily CRISIL NAV observations.

For every fund NAV date, the reference date starts at ``report_date - 30 days``.
If that date is unavailable for the fund, the lookup moves backwards one day at
a time until a NAV is found. All return calculation is performed in pandas;
SQL is used only to read the raw data and persist the final rows.
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

DEFAULT_SCHEMA = "mf100"

SOURCE_TABLE = f"direct_growth_nav"

DEFAULT_TARGET_TABLE = (
    f"mf_fund_rolling_30d_returns_{RUN_TIMESTAMP}"
)

LOOKBACK_DAYS = 30


# SOURCE_TABLE = "crisil_fund_daily_raw"
# DEFAULT_TARGET_TABLE = "crisil_fund_rolling_30d_returns"

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


# def read_daily_navs(engine, schema: str, source_table: str) -> pd.DataFrame:
#     """Read only the daily NAV fields needed for the pandas calculation."""
#     source = f"{quote_identifier(schema)}.{quote_identifier(source_table)}"
#     frame = pd.read_sql(
#         text(
#             f'''SELECT "schemeName" AS scheme_name, report_date, "navDirect" AS nav_direct
#                 FROM {source}
#                 WHERE "schemeName" IS NOT NULL
#                   AND report_date IS NOT NULL
#                   AND "navDirect" IS NOT NULL
#                   AND benchmark = :benchmark'''
#         ),
#         engine,
#         params={"benchmark": BENCHMARK},
#     )
#     frame["report_date"] = pd.to_datetime(frame["report_date"]).dt.normalize()
#     frame["nav_direct"] = pd.to_numeric(
#         frame["nav_direct"].astype("string").str.replace(r"[^0-9.-]", "", regex=True),
#         errors="coerce",
#     )
#     frame = frame.dropna(subset=["nav_direct"])
#
#     # Repeated API loads can produce identical fund/date observations. Keep one
#     # daily observation before calculating, without aggregating NAV values.
#     return frame.sort_values(["scheme_name", "report_date"]).drop_duplicates(
#         ["scheme_name", "report_date"], keep="last"
#     )

def read_daily_navs(engine, schema: str, source_table: str) -> pd.DataFrame:
    """Read cleaned daily NAVs needed for the 30-day return calculation."""

    source = f"{quote_identifier(schema)}.{quote_identifier(source_table)}"

    frame = pd.read_sql(
        text(
            f"""
            SELECT
                scheme_name,
                date as report_date,
                nav AS nav_direct
            FROM {source}
            WHERE scheme_name IS NOT NULL
              AND date IS NOT NULL
              AND nav IS NOT NULL
            ORDER BY scheme_name, report_date
            """
        ),
        engine,
    )

    frame["report_date"] = pd.to_datetime(
        frame["report_date"]
    ).dt.normalize()

    frame["nav_direct"] = pd.to_numeric(
        frame["nav_direct"],
        errors="coerce"
    )

    frame = frame.dropna(subset=["nav_direct"])

    return (
        frame
        .sort_values(["scheme_name", "report_date"])
        .drop_duplicates(
            ["scheme_name", "report_date"],
            keep="last"
        )
    )


def calculate_rolling_returns(daily_navs: pd.DataFrame) -> pd.DataFrame:
    """Calculate each fund's 30-day return by stepping backwards through dates."""
    output: list[dict] = []

    for scheme_name, fund_rows in daily_navs.groupby("scheme_name", sort=False):
        nav_by_date = dict(zip(fund_rows["report_date"], fund_rows["nav_direct"]))
        first_date = fund_rows["report_date"].iloc[0]

        for row in fund_rows.itertuples(index=False):
            reference_date = row.report_date - pd.Timedelta(days=LOOKBACK_DAYS)
            while reference_date >= first_date and reference_date not in nav_by_date:
                reference_date -= pd.Timedelta(days=1)

            if reference_date < first_date:
                reference_date = None
                reference_nav = None
                monthly_return_percent = None
            else:
                reference_nav = nav_by_date[reference_date]
                monthly_return_percent = (
                    None
                    if reference_nav == 0
                    else ((row.nav_direct / reference_nav) - 1) * 100
                )

            output.append(
                {
                    "scheme_name": scheme_name,
                    "report_date": row.report_date,
                    "nav_direct": row.nav_direct,
                    "reference_date": reference_date,
                    "reference_nav": reference_nav,
                    "monthly_return_percent": monthly_return_percent,
                }
            )

    return pd.DataFrame(output)


def write_returns(engine, schema: str, target_table: str, returns: pd.DataFrame) -> int:
    """Create the target table if needed and upsert pandas-calculated rows."""
    quoted_schema = quote_identifier(schema)
    target = f"{quoted_schema}.{quote_identifier(target_table)}"
    # pandas stores a missing datetime as NaT even after ``where(..., None)``.
    # Convert each date explicitly so psycopg receives a genuine SQL NULL.
    rows = []
    for row in returns.itertuples(index=False):
        has_reference = pd.notna(row.reference_date)
        rows.append(
            {
                "scheme_name": row.scheme_name,
                "report_date": row.report_date.date(),
                "nav_direct": row.nav_direct,
                "reference_date": row.reference_date.date() if has_reference else None,
                "reference_nav": row.reference_nav if has_reference else None,
                "monthly_return_percent": (
                    row.monthly_return_percent if has_reference else None
                ),
            }
        )

    with engine.begin() as connection:
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {quoted_schema}"))
        connection.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {target} (
                    scheme_name TEXT NOT NULL,
                    report_date DATE NOT NULL,
                    nav_direct NUMERIC(18, 6) NOT NULL,
                    reference_date DATE,
                    reference_nav NUMERIC(18, 6),
                    monthly_return_percent NUMERIC(12, 6),
                    loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (scheme_name, report_date)
                )
                """
            )
        )
        if not rows:
            return 0
        result = connection.execute(
            text(
                f"""
                INSERT INTO {target} (
                    scheme_name, report_date, nav_direct, reference_date,
                    reference_nav, monthly_return_percent
                ) VALUES (
                    :scheme_name, :report_date, :nav_direct, :reference_date,
                    :reference_nav, :monthly_return_percent
                )
                ON CONFLICT (scheme_name, report_date) DO UPDATE SET
                    nav_direct = EXCLUDED.nav_direct,
                    reference_date = EXCLUDED.reference_date,
                    reference_nav = EXCLUDED.reference_nav,
                    monthly_return_percent = EXCLUDED.monthly_return_percent,
                    loaded_at = NOW()
                """
            ),
            rows,
        )
    return result.rowcount


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", default=os.getenv("SCHEMA", DEFAULT_SCHEMA))
    parser.add_argument("--source-table", default=SOURCE_TABLE)
    parser.add_argument("--target-table", default=os.getenv("TABLE", DEFAULT_TARGET_TABLE))
    args = parser.parse_args()

    engine = build_engine()
    daily_navs = read_daily_navs(engine, args.schema, args.source_table)
    returns = calculate_rolling_returns(daily_navs)
    row_count = write_returns(engine, args.schema, args.target_table, returns)
    print(f"Upserted {row_count} daily 30-day return rows into {args.schema}.{args.target_table}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#everytime this process whole dataset, this needs to be optimized later
