"""Calculate paper-style performance-momentum features for Indian mutual funds.

Refactored version:
- Processes one fund at a time instead of building a 4.9M-row feature DataFrame.
- Writes results to PostgreSQL in small fund batches, so RAM stays bounded.
- Uses shift(1) for the previous observed NAV (equivalent to stepping backward
  through calendar days while skipping weekends/holidays).
- Supports output date windows while retaining enough historical observations
  for the 750-trading-day features.
- Resumes safely by skipping funds already present in the target table.

The feature definitions and rolling windows follow the original script/paper:
90, 120, 180, 250, 750 trading days; IR/SR windows 125, 250, 750.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import URL, create_engine, text

RUN_TIMESTAMP_FILE = Path(r"D:\PycharmProjects\mf-ml\run_timestamp.txt")

if not RUN_TIMESTAMP_FILE.exists():
    raise FileNotFoundError(
        f"Run timestamp file not found: {RUN_TIMESTAMP_FILE.resolve()}"
    )

RUN_TIMESTAMP = RUN_TIMESTAMP_FILE.read_text().strip()
if not RUN_TIMESTAMP:
    raise ValueError("run_timestamp.txt is empty")

print("Run timestamp:", RUN_TIMESTAMP)

DEFAULT_SCHEMA = "mf100"
SOURCE_TABLE = "direct_growth_nav"
FACTOR_SCHEMA = "mf100"
FACTOR_TABLE = "indian_factor_returns"

REGRESSION_WINDOWS = (90, 120, 180, 250, 750)
IR_SR_WINDOWS = {"p6m": 125, "p1y": 250, "p3y": 750}
FACTOR_COLUMNS = ("mkt", "smb", "hml", "umd")
PERCENT_COLUMNS = ("rf", *FACTOR_COLUMNS, "market_return")

INPUT_SAMPLE_COLUMNS = (
    "scheme_name",
    "trade_date",
    "prior_trade_date",
    "nav_direct",
    "fund_return",
    "mkt",
    "market_return",
)

FEATURE_SAMPLE_COLUMNS = (
    "scheme_name",
    "trade_date",
    "const_250",
    "t_const_250",
    "r2_250",
    "ir_p1y",
    "sr_p1y",
)


def quote_identifier(identifier: str) -> str:
    if not identifier or "\x00" in identifier:
        raise ValueError("Schema and table names must be non-empty identifiers")
    return '"' + identifier.replace('"', '""') + '"'


def build_engine():
    return create_engine(
        URL.create(
            "postgresql+psycopg",
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "Password@123"),
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "mfdb"),
        ),
        pool_pre_ping=True,
    )


def read_daily_navs(engine, schema: str, source_table: str) -> pd.DataFrame:
    """Read only the NAV columns needed by the calculation."""
    source = f"{quote_identifier(schema)}.{quote_identifier(source_table)}"
    frame = pd.read_sql(
        text(
            f"""
            SELECT
                scheme_name,
                date AS trade_date,
                nav AS nav_direct
            FROM {source}
            WHERE scheme_name IS NOT NULL
              AND date IS NOT NULL
              AND nav IS NOT NULL
              AND date >= DATE '2013-01-01'
              AND date < DATE '2026-01-01'
            ORDER BY scheme_name, date
            """
        ),
        engine,
    )

    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame["nav_direct"] = pd.to_numeric(
        frame["nav_direct"].astype("string").str.replace(
            r"[^0-9.-]", "", regex=True
        ),
        errors="coerce",
    )
    frame = frame.dropna(subset=["scheme_name", "trade_date", "nav_direct"])
    frame = frame[frame["nav_direct"] > 0]
    frame = frame.drop_duplicates(
        ["scheme_name", "trade_date"], keep="last"
    )

    return frame


def read_factor_returns(
    engine, factor_schema: str, factor_table: str
) -> pd.DataFrame:
    factor_source = (
        f"{quote_identifier(factor_schema)}.{quote_identifier(factor_table)}"
    )
    frame = pd.read_sql(
        text(
            f"""
            SELECT
                "Date"::date AS trade_date,
                rf, mkt, smb, hml, umd, market_return
            FROM {factor_source}
            where
                "Date"::date >= DATE '2013-01-01'
                AND "Date"::date < DATE '2026-01-01'
            """
        ),
        engine,
    )
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    for column in PERCENT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["trade_date"]).drop_duplicates(
        "trade_date", keep="last"
    )
    return frame.sort_values("trade_date").reset_index(drop=True)


def compute_daily_fund_returns(fund_rows: pd.DataFrame) -> pd.DataFrame:
    """Compute return from the immediately previous observed NAV.

    Because fund_rows is sorted by date, shift(1) is exactly equivalent to
    the old calendar-day loop that walked backward until a NAV was found.
    """
    fund = fund_rows.sort_values("trade_date").copy()

    fund["prior_trade_date"] = fund["trade_date"].shift(1)
    fund["prior_nav_direct"] = fund["nav_direct"].shift(1)

    fund = fund.dropna(subset=["prior_trade_date", "prior_nav_direct"])
    fund = fund[fund["prior_nav_direct"] > 0].copy()

    fund["fund_return"] = (
        fund["nav_direct"] / fund["prior_nav_direct"] - 1.0
    )

    return fund[
        [
            "scheme_name",
            "trade_date",
            "prior_trade_date",
            "nav_direct",
            "prior_nav_direct",
            "fund_return",
        ]
    ]


def build_fund_panel(
    fund_rows: pd.DataFrame, factor_rows: pd.DataFrame
) -> pd.DataFrame:
    """Build the regression panel for one fund only."""
    with_returns = compute_daily_fund_returns(fund_rows)
    if with_returns.empty:
        return with_returns

    panel = with_returns.merge(
        factor_rows,
        on="trade_date",
        how="inner",
        validate="many_to_one",
    )

    if panel.empty:
        return panel

    panel[list(PERCENT_COLUMNS)] = (
        panel[list(PERCENT_COLUMNS)] / 100.0
    )

    panel["benchmark_return"] = panel["market_return"].fillna(
        panel["mkt"] + panel["rf"]
    )
    panel["excess_return"] = (
        panel["fund_return"] - panel["benchmark_return"]
    )
    panel["fund_excess_return"] = panel["fund_return"] - panel["rf"]
    panel["excess_rf"] = panel["fund_return"] - panel["rf"]

    return panel.sort_values("trade_date").reset_index(drop=True)


def _fit_ols_window(
    y: np.ndarray, x_with_const: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Return OLS coefficients, R-squared, and t-statistics for one window."""
    try:
        beta, _, _, _ = np.linalg.lstsq(
            x_with_const, y, rcond=None
        )
    except np.linalg.LinAlgError:
        return None

    n_obs, n_params = x_with_const.shape
    df_error = n_obs - n_params
    if df_error <= 0:
        return None

    residuals = y - x_with_const @ beta
    sigma_sq = np.sum(residuals**2) / df_error
    xtx_inv = np.linalg.pinv(x_with_const.T @ x_with_const)
    standard_errors = np.sqrt(
        np.maximum(np.diag(sigma_sq * xtx_inv), 0.0)
    )
    standard_errors[standard_errors == 0] = np.nan
    t_stats = beta / standard_errors

    tss = np.sum((y - np.mean(y)) ** 2)
    rss = np.sum(residuals**2)
    r_squared = 1.0 - (rss / tss) if tss != 0 else 0.0

    return beta, r_squared, t_stats


def compute_rolling_factor_features(
    fund_panel: pd.DataFrame,
    windows: Iterable[int] = REGRESSION_WINDOWS,
) -> pd.DataFrame:
    """Estimate rolling four-factor alphas and factor-loading t-stats."""
    fund_df = fund_panel.sort_values("trade_date").reset_index(drop=True)

    scheme_name = fund_df["scheme_name"].iloc[0]
    trade_dates = fund_df["trade_date"].to_numpy()

    y = fund_df["fund_excess_return"].to_numpy(dtype=float)
    factor_matrix = fund_df[list(FACTOR_COLUMNS)].to_numpy(dtype=float)
    x_with_const = np.column_stack(
        [np.ones(len(fund_df)), factor_matrix]
    )

    output = pd.DataFrame(
        {"scheme_name": scheme_name, "trade_date": trade_dates}
    )
    n_obs = len(fund_df)

    for window in windows:
        const_col = np.full(n_obs, np.nan)
        r2_col = np.full(n_obs, np.nan)
        t_cols = {
            name: np.full(n_obs, np.nan)
            for name in ("const", "mkt", "smb", "hml", "umd")
        }

        if n_obs >= window:
            for end_idx in range(window, n_obs + 1):
                start_idx = end_idx - window

                fit = _fit_ols_window(
                    y[start_idx:end_idx],
                    x_with_const[start_idx:end_idx],
                )
                if fit is None:
                    continue

                beta, r_squared, t_stats = fit
                row_idx = end_idx - 1

                const_col[row_idx] = beta[0]
                r2_col[row_idx] = r_squared
                t_cols["const"][row_idx] = t_stats[0]
                t_cols["mkt"][row_idx] = t_stats[1]
                t_cols["smb"][row_idx] = t_stats[2]
                t_cols["hml"][row_idx] = t_stats[3]
                t_cols["umd"][row_idx] = t_stats[4]

        output[f"const_{window}"] = const_col
        output[f"r2_{window}"] = r2_col
        output[f"t_const_{window}"] = t_cols["const"]
        output[f"t_mkt_{window}"] = t_cols["mkt"]
        output[f"t_smb_{window}"] = t_cols["smb"]
        output[f"t_hml_{window}"] = t_cols["hml"]
        output[f"t_umd_{window}"] = t_cols["umd"]

    return output


def compute_information_and_sharpe_ratios(
    fund_panel: pd.DataFrame,
    windows: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Compute trailing information and Sharpe ratios for one fund."""
    windows = windows or IR_SR_WINDOWS
    fund_df = fund_panel.sort_values("trade_date").reset_index(drop=True)

    output = fund_df[["scheme_name", "trade_date"]].copy()

    for suffix, window in windows.items():
        excess_mean = (
            fund_df["excess_return"]
            .rolling(window, min_periods=window)
            .mean()
        )
        excess_std = (
            fund_df["excess_return"]
            .rolling(window, min_periods=window)
            .std()
        )
        output[f"ir_{suffix}"] = excess_mean / excess_std

        rf_mean = (
            fund_df["excess_rf"]
            .rolling(window, min_periods=window)
            .mean()
        )
        fund_std = (
            fund_df["fund_return"]
            .rolling(window, min_periods=window)
            .std()
        )
        output[f"sr_{suffix}"] = rf_mean / fund_std

    ratio_columns = [
        col
        for col in output.columns
        if col.startswith(("ir_", "sr_"))
    ]
    output[ratio_columns] = output[ratio_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    return output


def calculate_one_fund(
    fund_panel: pd.DataFrame,
    output_start: pd.Timestamp | None = None,
    output_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Calculate features for one fund and optionally restrict output dates."""
    if fund_panel.empty:
        return pd.DataFrame()

    factor_features = compute_rolling_factor_features(fund_panel)
    ratio_features = compute_information_and_sharpe_ratios(fund_panel)

    features = factor_features.merge(
        ratio_features,
        on=["scheme_name", "trade_date"],
        how="left",
        validate="one_to_one",
    )

    if output_start is not None:
        features = features[features["trade_date"] >= output_start]
    if output_end is not None:
        features = features[features["trade_date"] <= output_end]

    return features.reset_index(drop=True)


def feature_columns() -> list[str]:
    return [
        *(f"const_{window}" for window in REGRESSION_WINDOWS),
        *(f"r2_{window}" for window in REGRESSION_WINDOWS),
        *(f"t_const_{window}" for window in REGRESSION_WINDOWS),
        *(f"t_mkt_{window}" for window in REGRESSION_WINDOWS),
        *(f"t_smb_{window}" for window in REGRESSION_WINDOWS),
        *(f"t_hml_{window}" for window in REGRESSION_WINDOWS),
        *(f"t_umd_{window}" for window in REGRESSION_WINDOWS),
        *(f"ir_{suffix}" for suffix in IR_SR_WINDOWS),
        *(f"sr_{suffix}" for suffix in IR_SR_WINDOWS),
    ]


def prepare_target_table(
    engine, schema: str, target_table: str
) -> None:
    """Create the target table. Existing rows are intentionally preserved."""
    quoted_schema = quote_identifier(schema)
    target = f"{quoted_schema}.{quote_identifier(target_table)}"
    columns = feature_columns()

    column_defs = ",\n".join(
        f"{quote_identifier(column)} DOUBLE PRECISION"
        for column in columns
    )

    with engine.begin() as connection:
        connection.execute(
            text(f"CREATE SCHEMA IF NOT EXISTS {quoted_schema}")
        )

        legacy_columns = connection.execute(
            text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = :schema
                  AND table_name = :table
                """
            ),
            {"schema": schema, "table": target_table},
        ).scalars().all()

        if legacy_columns and "scheme_name" not in legacy_columns:
            raise RuntimeError(
                f"Target table {schema}.{target_table} exists with an "
                "unexpected schema. Refusing to drop it automatically."
            )

        if legacy_columns and "nav_date" in legacy_columns:
            connection.execute(
                text(
                    f"ALTER TABLE {target} "
                    "DROP COLUMN IF EXISTS nav_date"
                )
            )

        connection.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {target} (
                    scheme_name TEXT NOT NULL,
                    trade_date DATE NOT NULL,
                    {column_defs},
                    loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (scheme_name, trade_date)
                )
                """
            )
        )


def get_completed_funds(
    engine, schema: str, target_table: str
) -> set[str]:
    """Return schemes already represented in the target table."""
    quoted_schema = quote_identifier(schema)
    target = f"{quoted_schema}.{quote_identifier(target_table)}"

    with engine.begin() as connection:
        exists = connection.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = :schema
                      AND table_name = :table
                )
                """
            ),
            {"schema": schema, "table": target_table},
        ).scalar()

        if not exists:
            return set()

        rows = connection.execute(
            text(f"SELECT DISTINCT scheme_name FROM {target}")
        ).scalars().all()

    return set(rows)


def write_feature_batch(
    engine,
    schema: str,
    target_table: str,
    features: pd.DataFrame,
) -> int:
    """Upsert one bounded batch. No giant Python records list is created."""
    if features.empty:
        return 0

    columns = feature_columns()
    insert_columns = ["scheme_name", "trade_date", *columns]

    quoted_schema = quote_identifier(schema)
    target = f"{quoted_schema}.{quote_identifier(target_table)}"

    placeholders = ", ".join(
        f":{column}" for column in insert_columns
    )

    update_clause = ",\n".join(
        f"{quote_identifier(column)} = "
        f"EXCLUDED.{quote_identifier(column)}"
        for column in columns
    )

    insert_sql = text(
        f"""
        INSERT INTO {target}
            ({", ".join(quote_identifier(c) for c in insert_columns)})
        VALUES ({placeholders})
        ON CONFLICT (scheme_name, trade_date) DO UPDATE SET
            {update_clause},
            loaded_at = NOW()
        """
    )

    batch = features[insert_columns].copy()
    batch["trade_date"] = pd.to_datetime(
        batch["trade_date"]
    ).dt.date
    batch = batch.astype(object).where(pd.notna(batch), None)

    records = batch.to_dict(orient="records")

    with engine.begin() as connection:
        row_count = 0
        chunk_size = 5000
        for start in range(0, len(records), chunk_size):
            chunk = records[start : start + chunk_size]
            result = connection.execute(insert_sql, chunk)
            row_count += result.rowcount

    return row_count


def print_feature_coverage(
    coverage: dict[str, int],
) -> None:
    print("Feature coverage in written rows:")
    for name, count in coverage.items():
        print(f"  {name}: {count:,}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--schema",
        default=os.getenv("SCHEMA", DEFAULT_SCHEMA),
    )
    parser.add_argument(
        "--source-table",
        default=os.getenv("SOURCE_TABLE", SOURCE_TABLE),
    )
    parser.add_argument(
        "--target-table",
        default=os.getenv("TABLE"),
        help="Explicit target table. Recommended when resuming a run.",
    )
    parser.add_argument(
        "--factor-schema",
        default=os.getenv("FACTOR_SCHEMA", FACTOR_SCHEMA),
    )
    parser.add_argument(
        "--factor-table",
        default=os.getenv("FACTOR_TABLE", FACTOR_TABLE),
    )
    parser.add_argument(
        "--fund-limit",
        type=int,
        default=None,
        help="Optional cap on number of funds, useful for testing.",
    )
    parser.add_argument(
        "--output-start",
        type=str,
        default=None,
        help="First feature-output date, inclusive, e.g. 2013-01-01.",
    )
    parser.add_argument(
        "--output-end",
        type=str,
        default=None,
        help="Last feature-output date, inclusive, e.g. 2019-12-31.",
    )
    parser.add_argument(
        "--lookback-calendar-days",
        type=int,
        default=1400,
        help=(
            "Calendar days of NAV history retained before output-start. "
            "1400 safely covers the 750-trading-day window."
        ),
    )
    parser.add_argument(
        "--write-batch-funds",
        type=int,
        default=25,
        help="Number of funds accumulated before one PostgreSQL upsert.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Reprocess all funds even if they already exist in target.",
    )

    args = parser.parse_args()

    output_start = (
        pd.Timestamp(args.output_start).normalize()
        if args.output_start
        else None
    )
    output_end = (
        pd.Timestamp(args.output_end).normalize()
        if args.output_end
        else None
    )

    if (
        output_start is not None
        and output_end is not None
        and output_end < output_start
    ):
        raise ValueError("--output-end cannot be earlier than --output-start")

    if args.write_batch_funds < 1:
        raise ValueError("--write-batch-funds must be >= 1")

    if args.lookback_calendar_days < 750:
        raise ValueError("--lookback-calendar-days should be >= 750")

    if args.target_table:
        target_table = args.target_table
    elif output_start is not None or output_end is not None:
        start_label = (
            output_start.strftime("%Y")
            if output_start is not None
            else "start"
        )
        end_label = (
            output_end.strftime("%Y")
            if output_end is not None
            else "end"
        )
        target_table = (
            f"performance_momentum_features_{start_label}_{end_label}_"
            f"{RUN_TIMESTAMP}"
        )
    else:
        target_table = (
            f"performance_momentum_features_2013_2025_{RUN_TIMESTAMP}"
        )

    print(f"Target table: {args.schema}.{target_table}")
    if output_start is not None or output_end is not None:
        print(
            "Feature output window: "
            f"{output_start.date() if output_start is not None else 'MIN'} "
            "to "
            f"{output_end.date() if output_end is not None else 'MAX'}"
        )

    engine = build_engine()

    # Create target first so a partially completed run can be resumed.
    prepare_target_table(engine, args.schema, target_table)

    completed_funds = (
        set()
        if args.no_resume
        else get_completed_funds(engine, args.schema, target_table)
    )

    if completed_funds:
        print(
            f"Resume mode: {len(completed_funds):,} funds already present "
            "in target; they will be skipped."
        )

    daily_navs = read_daily_navs(
        engine, args.schema, args.source_table
    )
    factor_rows = read_factor_returns(
        engine, args.factor_schema, args.factor_table
    )

    if factor_rows.empty:
        raise RuntimeError("No factor-return rows were loaded.")

    factor_min_date = factor_rows["trade_date"].min()
    factor_max_date = factor_rows["trade_date"].max()

    # Feature output cannot exist outside the factor-return date range.
    effective_output_start = output_start
    effective_output_end = output_end

    if effective_output_start is not None:
        effective_output_start = max(
            effective_output_start, factor_min_date
        )
    if effective_output_end is not None:
        effective_output_end = min(
            effective_output_end, factor_max_date
        )

    if (
        effective_output_start is not None
        and effective_output_end is not None
        and effective_output_end < effective_output_start
    ):
        raise RuntimeError(
            "Requested output period does not overlap factor-return dates."
        )

    # Keep only history needed for the requested output period.
    if effective_output_start is not None:
        history_start = (
            effective_output_start
            - pd.Timedelta(days=args.lookback_calendar_days)
        )
        daily_navs = daily_navs[
            daily_navs["trade_date"] >= history_start
        ]

    if effective_output_end is not None:
        daily_navs = daily_navs[
            daily_navs["trade_date"] <= effective_output_end
        ]

    daily_navs = daily_navs.sort_values(
        ["scheme_name", "trade_date"]
    ).reset_index(drop=True)

    if args.fund_limit is not None:
        all_funds = daily_navs["scheme_name"].drop_duplicates()
        keep_funds = set(all_funds.head(args.fund_limit))
        daily_navs = daily_navs[
            daily_navs["scheme_name"].isin(keep_funds)
        ].copy()

    total_funds = daily_navs["scheme_name"].nunique()

    print(
        f"Loaded {len(daily_navs):,} NAV rows for "
        f"{total_funds:,} funds from "
        f"{args.schema}.{args.source_table}."
    )
    print(
        f"Loaded {len(factor_rows):,} factor-return dates from "
        f"{args.factor_schema}.{args.factor_table}."
    )
    print(
        f"Factor date range: {factor_min_date.date()} "
        f"to {factor_max_date.date()}."
    )

    # A dict of small arrays avoids merging the full 5M-row panel.
    factor_lookup = factor_rows[
        ["trade_date", *PERCENT_COLUMNS]
    ].copy()

    pending_frames: list[pd.DataFrame] = []
    pending_funds: list[str] = []
    processed = 0
    skipped = 0
    written_rows = 0

    coverage = {
        f"const_{window}": 0
        for window in REGRESSION_WINDOWS
    }
    coverage.update(
        {
            f"ir_{suffix}": 0
            for suffix in IR_SR_WINDOWS
        }
    )

    for scheme_name, fund_rows in daily_navs.groupby(
        "scheme_name", sort=False
    ):
        if not args.no_resume and scheme_name in completed_funds:
            skipped += 1
            continue

        # One fund at a time: memory is bounded.
        panel = build_fund_panel(fund_rows, factor_lookup)

        if panel.empty:
            processed += 1
            print(
                f"[{processed + skipped:,}/{total_funds:,}] "
                f"{scheme_name}: no usable factor-matched rows."
            )
            continue

        features = calculate_one_fund(
            panel,
            output_start=effective_output_start,
            output_end=effective_output_end,
        )

        if not features.empty:
            pending_frames.append(features)
            pending_funds.append(scheme_name)

            for column in coverage:
                coverage[column] += int(features[column].notna().sum())

        processed += 1

        if processed % 25 == 0:
            print(
                f"Processed {processed + skipped:,} / "
                f"{total_funds:,} funds..."
            )

        if len(pending_funds) >= args.write_batch_funds:
            batch_features = pd.concat(
                pending_frames,
                ignore_index=True,
            )
            batch_rows = write_feature_batch(
                engine,
                args.schema,
                target_table,
                batch_features,
            )
            written_rows += batch_rows

            print(
                f"  CHECKPOINT: wrote {batch_rows:,} rows "
                f"for {len(pending_funds):,} funds; "
                f"total written = {written_rows:,}"
            )

            completed_funds.update(pending_funds)
            pending_frames.clear()
            pending_funds.clear()

    # Flush final partial batch.
    if pending_frames:
        batch_features = pd.concat(
            pending_frames,
            ignore_index=True,
        )
        batch_rows = write_feature_batch(
            engine,
            args.schema,
            target_table,
            batch_features,
        )
        written_rows += batch_rows

        print(
            f"  FINAL CHECKPOINT: wrote {batch_rows:,} rows "
            f"for {len(pending_funds):,} funds."
        )

    print()
    print("==============================================")
    print("Run completed")
    print("==============================================")
    print(f"Funds processed : {processed:,}")
    print(f"Funds skipped   : {skipped:,}")
    print(f"Rows written    : {written_rows:,}")
    print(f"Target          : {args.schema}.{target_table}")
    print_feature_coverage(coverage)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
