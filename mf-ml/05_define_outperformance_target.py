"""Create the paper's binary mutual-fund outperformance target.

Wu et al. (2025), "Can machines learn Chinese mutual funds?", frame fund
selection as a classification problem.  For fund i's features observed in
month t, their one-month-ahead target is

    y[i, t + 1] = 1  if fund_monthly_return[i, t + 1]
                         > benchmark_monthly_return[t + 1]
                  0  otherwise.

This script applies that strict comparison to the monthly fund-return rows
created by ``03_load_crisil_monthly_returns.py`` and the Nifty 500 TRI
benchmark returns created by ``04_load_nifty500_tri_benchmark_returns.py``.
The resulting ``month_date`` is the feature/signal month and ``target_month``
is the following realized-return month.  The paper uses a Chinese equity-fund
index; Nifty 500 TRI is the corresponding benchmark already available in this
project.  The table also retains both feature-month returns to make this
one-month-ahead alignment explicit.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from sqlalchemy import URL, create_engine, text


DEFAULT_SCHEMA = "mf200"
FUND_RETURNS_TABLE = "crisil_fund_monthly_return_rows"
BENCHMARK_RETURNS_TABLE = "nifty500_tri_monthly_benchmark_returns"
DEFAULT_TARGET_TABLE = "fund_monthly_outperformance_targets"


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


def read_returns(
    engine, schema: str, fund_returns_table: str, benchmark_returns_table: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the existing fund and benchmark monthly returns."""
    fund_source = f"{quote_identifier(schema)}.{quote_identifier(fund_returns_table)}"
    benchmark_source = (
        f"{quote_identifier(schema)}.{quote_identifier(benchmark_returns_table)}"
    )

    fund_returns = pd.read_sql(
        text(
            f"""
            SELECT scheme_name, month_date, trade_month_end, monthly_return_percent
            FROM {fund_source}
            """
        ),
        engine,
    )
    benchmark_returns = pd.read_sql(
        text(
            f"""
            SELECT benchmark_name, observation_month, month_end,
                   benchmark_monthly_return_percent
            FROM {benchmark_source}
            """
        ),
        engine,
    )

    for column in ("month_date", "trade_month_end"):
        fund_returns[column] = pd.to_datetime(fund_returns[column]).dt.normalize()
    fund_returns["monthly_return_percent"] = pd.to_numeric(
        fund_returns["monthly_return_percent"], errors="coerce"
    )

    for column in ("observation_month", "month_end"):
        benchmark_returns[column] = pd.to_datetime(
            benchmark_returns[column]
        ).dt.normalize()
    benchmark_returns["benchmark_monthly_return_percent"] = pd.to_numeric(
        benchmark_returns["benchmark_monthly_return_percent"], errors="coerce"
    )
    return fund_returns, benchmark_returns


def define_outperformance_target(
    fund_returns: pd.DataFrame, benchmark_returns: pd.DataFrame
) -> pd.DataFrame:
    """Join matched realised-return months and create one-month-ahead labels.

    Months without either return are excluded because the paper's target cannot
    be determined for them.  A tied return is label 0 because a fund must
    *exceed* the benchmark to be classed as outperforming.  The returned
    ``month_date`` is shifted back one month, so January's row contains the
    target realised from February's return.
    """
    required_fund_columns = {
        "scheme_name",
        "month_date",
        "trade_month_end",
        "monthly_return_percent",
    }
    required_benchmark_columns = {
        "benchmark_name",
        "observation_month",
        "month_end",
        "benchmark_monthly_return_percent",
    }
    missing_fund = required_fund_columns.difference(fund_returns.columns)
    missing_benchmark = required_benchmark_columns.difference(benchmark_returns.columns)
    if missing_fund or missing_benchmark:
        raise ValueError(
            "Missing required columns: "
            f"fund={sorted(missing_fund)}, benchmark={sorted(missing_benchmark)}"
        )

    funds = fund_returns.copy()
    benchmarks = benchmark_returns.copy()
    for column in ("month_date", "trade_month_end"):
        funds[column] = pd.to_datetime(funds[column], errors="coerce").dt.normalize()
    funds["monthly_return_percent"] = pd.to_numeric(
        funds["monthly_return_percent"], errors="coerce"
    )
    for column in ("observation_month", "month_end"):
        benchmarks[column] = pd.to_datetime(
            benchmarks[column], errors="coerce"
        ).dt.normalize()
    benchmarks["benchmark_monthly_return_percent"] = pd.to_numeric(
        benchmarks["benchmark_monthly_return_percent"], errors="coerce"
    )

    funds = funds.dropna(
        subset=["scheme_name", "month_date", "monthly_return_percent"]
    )
    benchmarks = benchmarks.dropna(
        subset=["observation_month", "benchmark_monthly_return_percent"]
    )
    benchmarks = benchmarks.sort_values("month_end").drop_duplicates(
        "observation_month", keep="last"
    )

    targets = funds.merge(
        benchmarks,
        left_on="month_date",
        right_on="observation_month",
        how="inner",
        validate="many_to_one",
    )
    targets["excess_return_percent"] = (
        targets["monthly_return_percent"]
        - targets["benchmark_monthly_return_percent"]
    )
    targets["outperforms_benchmark"] = (
        targets["monthly_return_percent"]
        > targets["benchmark_monthly_return_percent"]
    ).astype("int8")
    targets = targets.rename(columns={"month_date": "target_month"})
    targets["month_date"] = (
        targets["target_month"] - pd.DateOffset(months=1)
    ).dt.normalize()

    # Retain the returns known in the feature month alongside the following
    # month's realised returns used to construct the label.  Left joins keep a
    # valid target for the first available return month, for which a preceding
    # feature-month return may not exist.
    current_fund_returns = funds[["scheme_name", "month_date", "monthly_return_percent"]].rename(
        columns={"monthly_return_percent": "current_monthly_return_percent"}
    )
    current_benchmark_returns = benchmarks[
        ["observation_month", "benchmark_monthly_return_percent"]
    ].rename(
        columns={
            "observation_month": "month_date",
            "benchmark_monthly_return_percent": "current_benchmark_monthly_return_percent",
        }
    )
    targets = targets.merge(
        current_fund_returns,
        on=["scheme_name", "month_date"],
        how="left",
        validate="many_to_one",
    ).merge(
        current_benchmark_returns,
        on="month_date",
        how="left",
        validate="many_to_one",
    )

    return targets[
        [
            "scheme_name",
            "month_date",
            "current_monthly_return_percent",
            "current_benchmark_monthly_return_percent",
            "target_month",
            "trade_month_end",
            "benchmark_name",
            "month_end",
            "monthly_return_percent",
            "benchmark_monthly_return_percent",
            "excess_return_percent",
            "outperforms_benchmark",
        ]
    ].sort_values(["month_date", "scheme_name"])


def write_targets(engine, schema: str, target_table: str, targets: pd.DataFrame) -> int:
    """Create the target table if needed and upsert the labelled observations."""
    quoted_schema = quote_identifier(schema)
    target = f"{quoted_schema}.{quote_identifier(target_table)}"
    records = [
        {
            "scheme_name": row.scheme_name,
            "month_date": row.month_date.date(),
            "target_month": row.target_month.date(),
            "trade_month_end": row.trade_month_end.date(),
            "benchmark_name": row.benchmark_name,
            "benchmark_month_end": row.month_end.date(),
            "monthly_return_percent": row.monthly_return_percent,
            "benchmark_monthly_return_percent": row.benchmark_monthly_return_percent,
            "current_monthly_return_percent": (
                row.current_monthly_return_percent
                if pd.notna(row.current_monthly_return_percent)
                else None
            ),
            "current_benchmark_monthly_return_percent": (
                row.current_benchmark_monthly_return_percent
                if pd.notna(row.current_benchmark_monthly_return_percent)
                else None
            ),
            "excess_return_percent": row.excess_return_percent,
            "outperforms_benchmark": int(row.outperforms_benchmark),
        }
        for row in targets.itertuples(index=False)
    ]

    with engine.begin() as connection:
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {quoted_schema}"))
        connection.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {target} (
                    scheme_name TEXT NOT NULL,
                    month_date DATE NOT NULL,
                    current_monthly_return_percent NUMERIC(12, 6),
                    current_benchmark_monthly_return_percent NUMERIC(12, 6),
                    target_month DATE NOT NULL,
                    trade_month_end DATE NOT NULL,
                    benchmark_name TEXT NOT NULL,
                    benchmark_month_end DATE NOT NULL,
                    monthly_return_percent NUMERIC(12, 6) NOT NULL,
                    benchmark_monthly_return_percent NUMERIC(12, 6) NOT NULL,
                    excess_return_percent NUMERIC(12, 6) NOT NULL,
                    outperforms_benchmark SMALLINT NOT NULL
                        CHECK (outperforms_benchmark IN (0, 1)),
                    loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (scheme_name, month_date)
                )
                """
            )
        )
        # Migrate tables created by the previous same-month version of this
        # script.  Existing labels are re-keyed to the preceding feature month.
        connection.execute(
            text(f"ALTER TABLE {target} ADD COLUMN IF NOT EXISTS target_month DATE")
        )
        connection.execute(
            text(
                f"""
                UPDATE {target}
                SET target_month = month_date,
                    month_date = (month_date - INTERVAL '1 month')::DATE
                WHERE target_month IS NULL
                """
            )
        )
        connection.execute(
            text(f"ALTER TABLE {target} ALTER COLUMN target_month SET NOT NULL")
        )
        connection.execute(
            text(
                f"ALTER TABLE {target} "
                "ADD COLUMN IF NOT EXISTS current_monthly_return_percent NUMERIC(12, 6)"
            )
        )
        connection.execute(
            text(
                f"ALTER TABLE {target} "
                "ADD COLUMN IF NOT EXISTS "
                "current_benchmark_monthly_return_percent NUMERIC(12, 6)"
            )
        )
        if not records:
            return 0
        result = connection.execute(
            text(
                f"""
                INSERT INTO {target} (
                    scheme_name, month_date, target_month, trade_month_end, benchmark_name,
                    benchmark_month_end, monthly_return_percent,
                    benchmark_monthly_return_percent, current_monthly_return_percent,
                    current_benchmark_monthly_return_percent, excess_return_percent,
                    outperforms_benchmark
                ) VALUES (
                    :scheme_name, :month_date, :target_month, :trade_month_end, :benchmark_name,
                    :benchmark_month_end, :monthly_return_percent,
                    :benchmark_monthly_return_percent, :current_monthly_return_percent,
                    :current_benchmark_monthly_return_percent, :excess_return_percent,
                    :outperforms_benchmark
                )
                ON CONFLICT (scheme_name, month_date) DO UPDATE SET
                    target_month = EXCLUDED.target_month,
                    trade_month_end = EXCLUDED.trade_month_end,
                    benchmark_name = EXCLUDED.benchmark_name,
                    benchmark_month_end = EXCLUDED.benchmark_month_end,
                    monthly_return_percent = EXCLUDED.monthly_return_percent,
                    benchmark_monthly_return_percent = EXCLUDED.benchmark_monthly_return_percent,
                    current_monthly_return_percent = EXCLUDED.current_monthly_return_percent,
                    current_benchmark_monthly_return_percent = EXCLUDED.current_benchmark_monthly_return_percent,
                    excess_return_percent = EXCLUDED.excess_return_percent,
                    outperforms_benchmark = EXCLUDED.outperforms_benchmark,
                    loaded_at = NOW()
                """
            ),
            records,
        )
    return result.rowcount


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", default=os.getenv("SCHEMA", DEFAULT_SCHEMA))
    parser.add_argument("--fund-returns-table", default=FUND_RETURNS_TABLE)
    parser.add_argument("--benchmark-returns-table", default=BENCHMARK_RETURNS_TABLE)
    parser.add_argument("--target-table", default=os.getenv("TABLE", DEFAULT_TARGET_TABLE))
    args = parser.parse_args()

    engine = build_engine()
    fund_returns, benchmark_returns = read_returns(
        engine, args.schema, args.fund_returns_table, args.benchmark_returns_table
    )
    print(
        "Loaded "
        f"{len(fund_returns):,} fund-return rows from "
        f"{args.schema}.{args.fund_returns_table}."
    )
    print(
        "Loaded "
        f"{len(benchmark_returns):,} benchmark-return rows from "
        f"{args.schema}.{args.benchmark_returns_table}."
    )
    if not fund_returns.empty:
        print("Fund-return sample:")
        print(fund_returns.head(5).to_string(index=False))
    if not benchmark_returns.empty:
        print("Benchmark-return sample:")
        print(benchmark_returns.head(5).to_string(index=False))

    targets = define_outperformance_target(fund_returns, benchmark_returns)
    print(f"Created {len(targets):,} matched fund-month target rows.")
    if not targets.empty:
        label_counts = targets["outperforms_benchmark"].value_counts().sort_index()
        print(
            "Label distribution "
            f"(0 = does not outperform, 1 = outperforms): {label_counts.to_dict()}"
        )
        print("Target sample:")
        print(targets.head(10).to_string(index=False))

    row_count = write_targets(engine, args.schema, args.target_table, targets)
    print(f"Upserted {row_count} target rows into {args.schema}.{args.target_table}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
