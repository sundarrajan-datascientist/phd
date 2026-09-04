import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from scipy.stats import spearmanr

# ============================================================
# CONFIGURATION
# ============================================================


# DB_URL = "postgresql+psycopg2://YOUR_USER:YOUR_PASSWORD@YOUR_HOST:5432/YOUR_DB"
DB_URL =  "postgresql+psycopg2://postgres:Password%40123@localhost:5432/mfdb"


TABLE_FEATURES = "mf200.feature_matrix_2019_2023"

# Threshold used in the paper
IC_THRESHOLD = 0.03

START_DATE = "2019-01-01"
END_DATE   = "2024-01-01"


# ============================================================
# DATABASE
# ============================================================

engine = create_engine(DB_URL)


# ============================================================
# 43 CANDIDATE FEATURES
# ============================================================

FEATURES = [
    # Fundamentals
    "tna",
    "vol_flows",

    # Alpha / regression
    "const_90",
    "const_120",
    "const_180",
    "const_250",
    "const_750",

    # Alpha t-stat
    "t_const_90",
    "t_const_120",
    "t_const_180",
    "t_const_250",
    "t_const_750",

    # R-squared
    "r2_90",
    "r2_120",
    "r2_180",
    "r2_250",
    "r2_750",

    # Market
    "t_mkt_90",
    "t_mkt_120",
    "t_mkt_180",
    "t_mkt_250",
    "t_mkt_750",

    # SMB
    "t_smb_90",
    "t_smb_120",
    "t_smb_180",
    "t_smb_250",
    "t_smb_750",

    # HML
    "t_hml_90",
    "t_hml_120",
    "t_hml_180",
    "t_hml_250",
    "t_hml_750",

    # Momentum
    "t_umd_90",
    "t_umd_120",
    "t_umd_180",
    "t_umd_250",
    "t_umd_750",

    # Information ratio
    "ir_p6m",
    "ir_p1y",
    "ir_p3y",

    # Sharpe ratio
    "sr_p6m",
    "sr_p1y",
    "sr_p3y",
]


# ============================================================
# LOAD MONTHLY FUND RETURNS
# ============================================================
#
# We need the actual monthly fund return and benchmark return.
#
# Your AUM/flow table contains monthly NAV-based return information,
# but for the IC target we should use the actual monthly return.
#
# The query below calculates monthly NAV return directly from
# crisil_fund_daily_raw.
#

query_returns = """
WITH daily AS (
    SELECT
        "schemeName" AS scheme_name,
        report_date::date AS trade_date,
        "navDirect"::double precision AS nav_direct
    FROM mf200.crisil_fund_daily_raw
    WHERE "benchmark" = 'Nifty 500 TRI'
      AND report_date >= DATE '2018-12-01'
      AND report_date < DATE '2024-01-01'
      AND "navDirect" IS NOT NULL
),

monthly_nav AS (
    SELECT
        scheme_name,
        DATE_TRUNC('month', trade_date)::date AS month_date,
        nav_direct,
        ROW_NUMBER() OVER (
            PARTITION BY scheme_name,
                         DATE_TRUNC('month', trade_date)
            ORDER BY trade_date DESC
        ) AS rn
    FROM daily
),

month_end AS (
    SELECT
        scheme_name,
        month_date,
        nav_direct
    FROM monthly_nav
    WHERE rn = 1
),

returns AS (
    SELECT
        scheme_name,
        month_date,
        nav_direct /
            LAG(nav_direct) OVER (
                PARTITION BY scheme_name
                ORDER BY month_date
            ) - 1 AS monthly_return
    FROM month_end
)

SELECT
    scheme_name,
    month_date,
    monthly_return
FROM returns
WHERE month_date >= DATE '2019-01-01'
  AND month_date < DATE '2024-01-01'
ORDER BY scheme_name, month_date;
"""

fund_returns = pd.read_sql(query_returns, engine)

print("Fund return observations:", len(fund_returns))
print("Funds:", fund_returns["scheme_name"].nunique())
print(
    "Date range:",
    fund_returns["month_date"].min(),
    "to",
    fund_returns["month_date"].max()
)

query_benchmark = """
SELECT
    observation_month::date AS month_date,
    change_percent::double precision / 100.0 AS benchmark_return
FROM mf200.investing_nifty500_tri_monthly
WHERE observation_month >= DATE '2019-01-01'
  AND observation_month < DATE '2024-01-01'
ORDER BY observation_month;
                  """

benchmark = pd.read_sql(query_benchmark, engine)

# ============================================================
# LOAD FEATURE MATRIX
# ============================================================

feature_query = f"""
SELECT
    scheme_name,
    month_date,
    {", ".join(FEATURES)}
FROM {TABLE_FEATURES}
WHERE month_date >= DATE '{START_DATE}'
  AND month_date < DATE '{END_DATE}'
ORDER BY scheme_name, month_date;
"""

features = pd.read_sql(feature_query, engine)

features["month_date"] = pd.to_datetime(features["month_date"])
fund_returns["month_date"] = pd.to_datetime(fund_returns["month_date"])
benchmark["month_date"] = pd.to_datetime(benchmark["month_date"])


# ============================================================
# MERGE FUND RETURNS + BENCHMARK
# ============================================================

returns = fund_returns.merge(
    benchmark,
    on="month_date",
    how="inner"
)

returns["excess_return"] = (
    returns["monthly_return"]
    - returns["benchmark_return"]
)


# ============================================================
# CREATE NEXT-MONTH TARGET
# ============================================================

returns = returns.sort_values(
    ["scheme_name", "month_date"]
)

returns["next_month_return"] = (
    returns.groupby("scheme_name")["monthly_return"]
           .shift(-1)
)

returns["next_month_excess_return"] = (
    returns.groupby("scheme_name")["excess_return"]
           .shift(-1)
)


# ============================================================
# MERGE WITH FEATURES
# ============================================================

ic_data = features.merge(
    returns[
        [
            "scheme_name",
            "month_date",
            "next_month_return",
            "next_month_excess_return"
        ]
    ],
    on=["scheme_name", "month_date"],
    how="inner"
)

print("IC dataset:", ic_data.shape)


# ============================================================
# MONTHLY CROSS-SECTIONAL SPEARMAN IC
# ============================================================

ic_records = []

months = sorted(ic_data["month_date"].dropna().unique())

for month in months:

    month_data = ic_data[
        ic_data["month_date"] == month
    ]

    for feature in FEATURES:

        tmp = month_data[
            [feature, "next_month_excess_return"]
        ].dropna()

        # Need enough cross-sectional observations
        if len(tmp) < 10:
            ic = np.nan
            n_obs = len(tmp)

        else:
            ic, _ = spearmanr(
                tmp[feature],
                tmp["next_month_excess_return"]
            )
            n_obs = len(tmp)

        ic_records.append({
            "month_date": month,
            "feature": feature,
            "ic": ic,
            "n_obs": n_obs
        })


ic_monthly = pd.DataFrame(ic_records)

print(
    "Monthly IC observations:",
    len(ic_monthly)
)


# ============================================================
# IC SUMMARY
# ============================================================

ic_summary = (
    ic_monthly
    .groupby("feature")
    .agg(
        mean_ic=("ic", "mean"),
        median_ic=("ic", "median"),
        std_ic=("ic", "std"),
        min_ic=("ic", "min"),
        max_ic=("ic", "max"),
        months_available=("ic", "count"),
        positive_ic_pct=(
            "ic",
            lambda x: (x > 0).mean() * 100
        )
    )
    .reset_index()
)

ic_summary["abs_mean_ic"] = (
    ic_summary["mean_ic"].abs()
)


# ============================================================
# PAPER THRESHOLD
# ============================================================

ic_summary["selected_ic"] = (
    ic_summary["abs_mean_ic"] >= IC_THRESHOLD
)


# Sort by absolute IC
ic_summary = ic_summary.sort_values(
    "abs_mean_ic",
    ascending=False
)


print("\n==============================")
print("IC FEATURE SELECTION")
print("==============================")

print(
    ic_summary[
        [
            "feature",
            "mean_ic",
            "abs_mean_ic",
            "median_ic",
            "std_ic",
            "months_available",
            "positive_ic_pct",
            "selected_ic"
        ]
    ].to_string(index=False)
)

selected_features = (
    ic_summary.loc[
        ic_summary["selected_ic"],
        "feature"
    ]
    .tolist()
)

print("\n====================================")
print("SELECTED FEATURES | |MEAN IC| >= 0.03")
print("====================================")

for i, feature in enumerate(selected_features, 1):
    row = ic_summary[
        ic_summary["feature"] == feature
    ].iloc[0]

    print(
        f"{i:2d}. {feature:15s} "
        f"Mean IC = {row['mean_ic']:.4f}"
    )

print(
    "\nTotal selected:",
    len(selected_features),
    "of",
    len(FEATURES)
)

# ============================================================
# SAVE RESULTS
# ============================================================

ic_monthly.to_csv(
    "monthly_spearman_ic_2019_2023.csv",
    index=False
)

ic_summary.to_csv(
    "feature_selection_ic_summary_2019_2023.csv",
    index=False
)

pd.DataFrame({
    "selected_feature": selected_features
}).to_csv(
    "selected_features_ic_2019_2023.csv",
    index=False
)

print("\nFiles saved:")
print("  monthly_spearman_ic_2019_2023.csv")
print("  feature_selection_ic_summary_2019_2023.csv")
print("  selected_features_ic_2019_2023.csv")