import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# ============================================================
# CONFIGURATION
# ============================================================

# engine = create_engine(
#     "postgresql+psycopg2://postgres:Password%40123@localhost:5432/mfdb"
# )

DB_URL =  "postgresql+psycopg2://postgres:Password%40123@localhost:5432/mfdb"

TABLE_NAME = "mf200.feature_matrix_2019_2023"

CORR_THRESHOLD = 0.30

# Candidate ML features
FEATURES = [
    # Fundamentals
    "tna",
    "vol_flows",

    # Constant / alpha
    "const_90",
    "const_120",
    "const_180",
    "const_250",
    "const_750",

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

    # Market t-stat
    "t_mkt_90",
    "t_mkt_120",
    "t_mkt_180",
    "t_mkt_250",
    "t_mkt_750",

    # SMB t-stat
    "t_smb_90",
    "t_smb_120",
    "t_smb_180",
    "t_smb_250",
    "t_smb_750",

    # HML t-stat
    "t_hml_90",
    "t_hml_120",
    "t_hml_180",
    "t_hml_250",
    "t_hml_750",

    # UMD t-stat
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
# LOAD DATA
# ============================================================

engine = create_engine(DB_URL)

query = f"""
SELECT
    scheme_name,
    month_date,
    {", ".join(FEATURES)}
FROM {TABLE_NAME}
ORDER BY scheme_name, month_date
"""

df = pd.read_sql(query, engine)

print("=" * 70)
print("FEATURE MATRIX")
print("=" * 70)
print(f"Rows  : {len(df):,}")
print(f"Funds : {df['scheme_name'].nunique():,}")
print(f"From  : {df['month_date'].min()}")
print(f"To    : {df['month_date'].max()}")
print()


# ============================================================
# SPEARMAN CORRELATION
# ============================================================

X = df[FEATURES]

spearman_corr = X.corr(method="spearman")

print("=" * 70)
print("SPEARMAN CORRELATION MATRIX")
print("=" * 70)
print(spearman_corr.round(3))


# ============================================================
# EXTRACT UNIQUE HIGH-CORRELATION PAIRS
# ============================================================

pairs = []

for i in range(len(FEATURES)):
    for j in range(i + 1, len(FEATURES)):

        f1 = FEATURES[i]
        f2 = FEATURES[j]

        rho = spearman_corr.loc[f1, f2]

        if pd.notna(rho) and abs(rho) >= CORR_THRESHOLD:

            pairs.append({
                "feature_1": f1,
                "feature_2": f2,
                "spearman_rho": rho,
                "abs_rho": abs(rho)
            })

high_corr_pairs = pd.DataFrame(pairs)

if not high_corr_pairs.empty:
    high_corr_pairs = high_corr_pairs.sort_values(
        "abs_rho",
        ascending=False
    ).reset_index(drop=True)


# ============================================================
# DISPLAY HIGH-CORRELATION PAIRS
# ============================================================

print()
print("=" * 70)
print(f"HIGH CORRELATION PAIRS |rho| >= {CORR_THRESHOLD}")
print("=" * 70)

if high_corr_pairs.empty:
    print("No highly correlated feature pairs found.")
else:
    print(high_corr_pairs.to_string(index=False))

print()
print(f"Number of high-correlation pairs: {len(high_corr_pairs)}")


# ============================================================
# MISSINGNESS
# ============================================================

print()
print("=" * 70)
print("FEATURE MISSINGNESS")
print("=" * 70)

missing = pd.DataFrame({
    "feature": FEATURES,
    "missing": [df[f].isna().sum() for f in FEATURES],
})

missing["missing_pct"] = (
    missing["missing"] / len(df) * 100
)

print(
    missing.sort_values(
        "missing",
        ascending=False
    ).to_string(index=False)
)


# ============================================================
# SAVE RESULTS
# ============================================================

spearman_corr.to_csv(
    "spearman_correlation_matrix_2019_2023.csv"
)

high_corr_pairs.to_csv(
    "spearman_high_correlation_pairs_2019_2023.csv",
    index=False
)

missing.to_csv(
    "feature_missingness_2019_2023.csv",
    index=False
)

print()
print("Saved:")
print("  spearman_correlation_matrix_2019_2023.csv")
print("  spearman_high_correlation_pairs_2019_2023.csv")
print("  feature_missingness_2019_2023.csv")