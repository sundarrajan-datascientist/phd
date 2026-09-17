"""Calculate paper-style performance-momentum features for Indian mutual funds.

Wu et al. (2025), "Can machines learn Chinese mutual funds?", define
performance momentum using rolling time-series regressions of fund returns on
factor models plus trailing information and Sharpe ratios.  For India we use
the four-factor model (market, size, value, momentum) from
``mf100.indian_factor_returns`` because RMW and CMA are unavailable.

Fund NAV and trade dates come from ``mf200.crisil_fund_daily_raw``.  Daily
returns are computed from ``navDirect`` by stepping back one calendar day at a
time until the fund's most recent prior NAV is found, which skips weekends and
holiday gaps.  Factor returns are joined on trade date from
``mf100.indian_factor_returns``.

Rolling regression windows follow the paper: 90, 120, 180, 250, and 750
trading days.  Information ratios and Sharpe ratios use 6-month (125-day),
1-year (250-day), and 3-year (750-day) windows.

    IR = mean(fund_return - benchmark_return) / std(fund_return - benchmark_return)
    SR = mean(fund_return - rf) / std(fund_return)

Factor returns and the risk-free rate in ``indian_factor_returns`` are stored
in percent and are converted to decimals before estimation.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import numpy as np
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
DEFAULT_TARGET_TABLE = f"performance_momentum_features_2006_{RUN_TIMESTAMP}"

FACTOR_SCHEMA = "mf100"
FACTOR_TABLE = "indian_factor_returns"
REGRESSION_WINDOWS = (90, 120, 180, 250, 750)
IR_SR_WINDOWS = {
    "p6m": 125,
    "p1y": 250,
    "p3y": 750,
}
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


def read_daily_navs(engine, schema: str, source_table: str) -> pd.DataFrame:
    """Read fund NAV observations from the CRISIL daily raw table."""
    source = f"{quote_identifier(schema)}.{quote_identifier(source_table)}"
    frame = pd.read_sql(
        text(
            f'''
            SELECT
                scheme_name,
                date as report_date,
                nav AS nav_direct
            FROM {source}
            WHERE scheme_name IS NOT NULL
              AND date IS NOT NULL
              AND nav IS NOT NULL
            '''
        ),
        engine,
    )
    frame["trade_date"] = pd.to_datetime(frame["report_date"]).dt.normalize()
    frame["nav_direct"] = pd.to_numeric(
        frame["nav_direct"].astype("string").str.replace(r"[^0-9.-]", "", regex=True),
        errors="coerce",
    )
    frame = frame.dropna(subset=["scheme_name", "trade_date", "nav_direct"])
    frame = frame[frame["nav_direct"] > 0]
    return frame.sort_values(["scheme_name", "trade_date"]).drop_duplicates(
        ["scheme_name", "trade_date"], keep="last"
    )


def read_factor_returns(
    engine, factor_schema: str, factor_table: str
) -> pd.DataFrame:
    """Load Indian Fama-French factor returns keyed by trade date."""
    factor_source = (
        f"{quote_identifier(factor_schema)}.{quote_identifier(factor_table)}"
    )
    frame = pd.read_sql(
        text(
            f'''
            SELECT
                "Date"::date AS trade_date,
                rf, mkt, smb, hml, umd, market_return
            FROM {factor_source}
            '''
        ),
        engine,
    )
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    for column in PERCENT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["trade_date"])


def find_prior_nav_date(
    trade_date: pd.Timestamp,
    nav_by_date: dict[pd.Timestamp, float],
    first_date: pd.Timestamp,
) -> pd.Timestamp | None:
    """Step back one day at a time until the fund's previous NAV date is found."""
    reference_date = trade_date - pd.Timedelta(days=1)
    while reference_date >= first_date and reference_date not in nav_by_date:
        reference_date -= pd.Timedelta(days=1)
    if reference_date < first_date:
        return None
    return reference_date


def compute_daily_fund_returns(daily_navs: pd.DataFrame) -> pd.DataFrame:
    """Derive daily returns using each fund's most recent prior NAV."""
    required_columns = {"scheme_name", "trade_date", "nav_direct"}
    missing_columns = required_columns.difference(daily_navs.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    output: list[dict] = []
    for scheme_name, fund_rows in daily_navs.groupby("scheme_name", sort=False):
        nav_by_date = dict(zip(fund_rows["trade_date"], fund_rows["nav_direct"]))
        first_date = fund_rows["trade_date"].iloc[0]

        for row in fund_rows.itertuples(index=False):
            prior_trade_date = find_prior_nav_date(
                row.trade_date, nav_by_date, first_date
            )
            if prior_trade_date is None:
                continue

            prior_nav_direct = nav_by_date[prior_trade_date]
            if prior_nav_direct <= 0:
                continue

            output.append(
                {
                    "scheme_name": scheme_name,
                    "trade_date": row.trade_date,
                    "prior_trade_date": prior_trade_date,
                    "nav_direct": row.nav_direct,
                    "prior_nav_direct": prior_nav_direct,
                    "fund_return": row.nav_direct / prior_nav_direct - 1,
                }
            )

    return pd.DataFrame(output)


def build_regression_panel(
    daily_navs: pd.DataFrame, factor_rows: pd.DataFrame
) -> pd.DataFrame:
    """Attach factor returns to fund rows with computed daily returns."""
    with_returns = compute_daily_fund_returns(daily_navs)
    if with_returns.empty:
        return with_returns

    panel = with_returns.merge(
        factor_rows,
        on="trade_date",
        how="inner",
        validate="many_to_one",
    )
    return panel[
        [
            "scheme_name",
            "trade_date",
            "prior_trade_date",
            "nav_direct",
            "fund_return",
            *PERCENT_COLUMNS,
        ]
    ]


def prepare_factor_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert percent factors to decimals and build regression inputs."""
    required_columns = {
        "scheme_name",
        "trade_date",
        "fund_return",
        *PERCENT_COLUMNS,
        *FACTOR_COLUMNS,
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    panel = frame.copy()
    panel[list(PERCENT_COLUMNS)] = panel[list(PERCENT_COLUMNS)] / 100.0
    panel["benchmark_return"] = panel["market_return"].fillna(
        panel["mkt"] + panel["rf"]
    )
    panel["excess_return"] = panel["fund_return"] - panel["benchmark_return"]
    panel["fund_excess_return"] = panel["fund_return"] - panel["rf"]
    panel["excess_rf"] = panel["fund_return"] - panel["rf"]
    return panel.sort_values(["scheme_name", "trade_date"]).reset_index(drop=True)


def _fit_ols_window(
    y: np.ndarray, x_with_const: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Return OLS coefficients, R-squared, and t-statistics for one window."""
    try:
        beta, _, _, _ = np.linalg.lstsq(x_with_const, y, rcond=None)
    except np.linalg.LinAlgError:
        return None

    n_obs, n_params = x_with_const.shape
    df_error = n_obs - n_params
    if df_error <= 0:
        return None

    residuals = y - x_with_const @ beta
    sigma_sq = np.sum(residuals**2) / df_error
    xtx_inv = np.linalg.pinv(x_with_const.T @ x_with_const)
    standard_errors = np.sqrt(np.maximum(np.diag(sigma_sq * xtx_inv), 0.0))
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
    required_columns = {
        "scheme_name",
        "trade_date",
        "fund_excess_return",
        *FACTOR_COLUMNS,
    }
    missing_columns = required_columns.difference(fund_panel.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    fund_df = fund_panel.sort_values("trade_date").reset_index(drop=True)
    scheme_name = fund_df["scheme_name"].iloc[0]
    trade_dates = fund_df["trade_date"].to_numpy()
    y = fund_df["fund_excess_return"].to_numpy(dtype=float)
    factor_matrix = fund_df[list(FACTOR_COLUMNS)].to_numpy(dtype=float)
    x_with_const = np.column_stack([np.ones(len(fund_df)), factor_matrix])

    output = pd.DataFrame({"scheme_name": scheme_name, "trade_date": trade_dates})
    n_obs = len(fund_df)

    for window in windows:
        const_col = np.full(n_obs, np.nan)
        r2_col = np.full(n_obs, np.nan)
        t_cols = {name: np.full(n_obs, np.nan) for name in ("const", "mkt", "smb", "hml", "umd")}

        if n_obs >= window:
            for end_idx in range(window, n_obs + 1):
                start_idx = end_idx - window
                fit = _fit_ols_window(y[start_idx:end_idx], x_with_const[start_idx:end_idx])
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
    required_columns = {
        "scheme_name",
        "trade_date",
        "fund_return",
        "excess_return",
        "excess_rf",
    }
    missing_columns = required_columns.difference(fund_panel.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

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

    ratio_columns = [col for col in output.columns if col.startswith(("ir_", "sr_"))]
    output[ratio_columns] = output[ratio_columns].replace([np.inf, -np.inf], np.nan)
    return output


def calculate_performance_momentum(
    panel: pd.DataFrame,
    progress_every: int = 25,
) -> pd.DataFrame:
    """Return daily performance-momentum features for every fund in the panel."""
    prepared = prepare_factor_panel(panel)
    feature_frames: list[pd.DataFrame] = []
    fund_names = prepared["scheme_name"].drop_duplicates().tolist()

    for index, scheme_name in enumerate(fund_names, start=1):
        fund_panel = prepared[prepared["scheme_name"] == scheme_name]
        factor_features = compute_rolling_factor_features(fund_panel)
        ratio_features = compute_information_and_sharpe_ratios(fund_panel)
        merged = factor_features.merge(
            ratio_features,
            on=["scheme_name", "trade_date"],
            how="left",
            validate="one_to_one",
        )
        feature_frames.append(merged)
        if progress_every and index % progress_every == 0:
            print(f"Processed {index:,} / {len(fund_names):,} funds...")

    if not feature_frames:
        return pd.DataFrame()

    return pd.concat(feature_frames, ignore_index=True).sort_values(
        ["scheme_name", "trade_date"]
    )


def print_feature_coverage(features: pd.DataFrame) -> None:
    """Print how many rows have each rolling-window estimate populated."""
    coverage = {
        f"const_{window}": features[f"const_{window}"].notna().sum()
        for window in REGRESSION_WINDOWS
    }
    coverage.update(
        {
            f"ir_{suffix}": features[f"ir_{suffix}"].notna().sum()
            for suffix in IR_SR_WINDOWS
        }
    )
    print("Feature coverage:")
    for name, count in coverage.items():
        print(f"  {name}: {count:,}")


def _nullable(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def write_features(
    engine,
    schema: str,
    target_table: str,
    features: pd.DataFrame,
) -> int:
    """Create the feature table if needed and upsert all calculated rows."""
    quoted_schema = quote_identifier(schema)
    target = f"{quoted_schema}.{quote_identifier(target_table)}"

    feature_columns = [
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

    with engine.begin() as connection:
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {quoted_schema}"))
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
            connection.execute(text(f"DROP TABLE {target}"))
        if legacy_columns and "nav_date" in legacy_columns:
            connection.execute(text(f"ALTER TABLE {target} DROP COLUMN IF EXISTS nav_date"))
        column_defs = ",\n".join(
            f"{quote_identifier(column)} DOUBLE PRECISION" for column in feature_columns
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

    records = [
        {
            "scheme_name": row.scheme_name,
            "trade_date": pd.Timestamp(row.trade_date).date(),
            **{column: _nullable(getattr(row, column)) for column in feature_columns},
        }
        for row in features.itertuples(index=False)
    ]
    if not records:
        return 0

    insert_columns = ["scheme_name", "trade_date", *feature_columns]
    placeholders = ", ".join(f":{column}" for column in insert_columns)
    update_clause = ",\n".join(
        f"{quote_identifier(column)} = EXCLUDED.{quote_identifier(column)}"
        for column in feature_columns
    )
    insert_sql = text(
        f"""
        INSERT INTO {target} ({", ".join(quote_identifier(c) for c in insert_columns)})
        VALUES ({placeholders})
        ON CONFLICT (scheme_name, trade_date) DO UPDATE SET
            {update_clause},
            loaded_at = NOW()
        """
    )

    row_count = 0
    chunk_size = 5000
    with engine.begin() as connection:
        for start in range(0, len(records), chunk_size):
            chunk = records[start : start + chunk_size]
            result = connection.execute(insert_sql, chunk)
            row_count += result.rowcount
    return row_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", default=os.getenv("SCHEMA", DEFAULT_SCHEMA))
    parser.add_argument("--source-table", default=os.getenv("SOURCE_TABLE", SOURCE_TABLE))
    parser.add_argument("--target-table", default=os.getenv("TABLE", DEFAULT_TARGET_TABLE))
    parser.add_argument("--factor-schema", default=os.getenv("FACTOR_SCHEMA", FACTOR_SCHEMA))
    parser.add_argument("--factor-table", default=os.getenv("FACTOR_TABLE", FACTOR_TABLE))
    parser.add_argument(
        "--fund-limit",
        type=int,
        default=None,
        help="Optional cap on the number of funds processed (for testing).",
    )
    args = parser.parse_args()

    engine = build_engine()
    daily_navs = read_daily_navs(engine, args.schema, args.source_table)
    factor_rows = read_factor_returns(engine, args.factor_schema, args.factor_table)
    if args.fund_limit is not None:
        keep_funds = daily_navs["scheme_name"].drop_duplicates().head(args.fund_limit)
        daily_navs = daily_navs[daily_navs["scheme_name"].isin(keep_funds)]

    panel = build_regression_panel(daily_navs, factor_rows)
    print(
        f"Loaded {len(daily_navs):,} NAV rows for "
        f"{daily_navs['scheme_name'].nunique():,} funds from "
        f"{args.schema}.{args.source_table}."
    )
    print(
        f"Joined to {len(factor_rows):,} factor-return dates from "
        f"{args.factor_schema}.{args.factor_table}."
    )
    print(
        f"Built {len(panel):,} regression rows with daily returns from the most "
        "recent prior NAV."
    )
    if not panel.empty:
        print(
            "Trade-date range: "
            f"{panel['trade_date'].min().date()} to {panel['trade_date'].max().date()}"
        )
        print("Input sample:")
        print(panel[list(INPUT_SAMPLE_COLUMNS)].head(5).to_string(index=False))

    features = calculate_performance_momentum(panel)
    print(f"Created {len(features):,} performance-momentum feature rows.")
    if not features.empty:
        print_feature_coverage(features)
        sample = features.dropna(subset=["const_250"], how="any")
        if sample.empty:
            sample = features
        print("Feature sample:")
        print(sample[list(FEATURE_SAMPLE_COLUMNS)].head(5).to_string(index=False))

    row_count = write_features(engine, args.schema, args.target_table, features)
    print(f"Upserted {row_count} rows into {args.schema}.{args.target_table}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
