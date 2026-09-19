"""
churn_utils.py
--------------
Shared utilities for the Customer Churn Prediction project.

Both the analysis notebook and the Flask API import from this module so that the
EXACT same feature-engineering and column definitions are applied to training
data and to new/unseen customer data. This is the key to avoiding train/serve
skew and data leakage: preprocessing statistics (medians, encoder categories)
are learned only inside a scikit-learn Pipeline that is fit on the training
split, while the deterministic feature engineering below is pure and
stateless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Column definitions
# --------------------------------------------------------------------------- #

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

# Raw feature columns expected from an incoming request (everything except the
# identifier and the target).
RAW_FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

# Add-on services used to derive engagement features.
ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Numeric columns AFTER feature engineering.
NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "num_addon_services",
    "avg_charges_per_month",
    "monthly_to_total_ratio",
]

# Categorical columns AFTER feature engineering.
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_group",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------------------------------- #
# Cleaning + feature engineering (stateless / leakage-free)
# --------------------------------------------------------------------------- #

def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce raw columns to the correct types.

    The public Telco dataset stores ``TotalCharges`` as text and uses blank
    strings for customers whose ``tenure`` is 0 (brand-new customers who have
    not been billed yet). We convert it to numeric and treat those blanks as 0.
    """
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip().replace("", np.nan),
        errors="coerce",
    )
    # New customers (tenure 0) have no TotalCharges yet -> 0.
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Ensure numeric raw fields are numeric.
    for col in ("tenure", "MonthlyCharges", "SeniorCitizen"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered features added.

    This function is deterministic and row-independent, so applying it to a
    single incoming customer produces the same result as applying it during
    training. No statistics are learned here.
    """
    df = clean_raw(df)

    # 1) tenure_group: buckets the customer lifetime. Churn risk is strongly
    #    concentrated in the first year, so bucketing exposes that non-linear
    #    relationship to the tree in a compact, interpretable form.
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, 72],
        labels=["0-12", "13-24", "25-48", "49-60", "61-72"],
    ).astype(object)

    # 2) num_addon_services: how many value-added services the customer uses.
    #    Higher engagement / stickiness generally lowers churn propensity.
    df["num_addon_services"] = (df[ADDON_SERVICES] == "Yes").sum(axis=1)

    # 3) avg_charges_per_month: lifetime average spend. Distinguishes long-term
    #    high-value customers from new high-monthly-charge customers.
    df["avg_charges_per_month"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    # 4) monthly_to_total_ratio: proxy for how "new" the spend is. A ratio near
    #    1 means most of the customer's spend is recent (short tenure), which
    #    correlates with higher churn risk.
    df["monthly_to_total_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1.0)

    return df


# --------------------------------------------------------------------------- #
# API input validation
# --------------------------------------------------------------------------- #

def validate_and_frame(payload: dict) -> pd.DataFrame:
    """Validate a single-customer JSON payload and return a 1-row DataFrame.

    Raises ``ValueError`` with a helpful message on invalid input so the API
    can return a 400 response.
    """
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object of customer fields.")

    missing = [c for c in RAW_FEATURE_COLUMNS if c not in payload]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    # Type checks for the numeric fields.
    for col in ("tenure", "MonthlyCharges", "SeniorCitizen"):
        try:
            float(payload[col])
        except (TypeError, ValueError):
            raise ValueError(f"Field '{col}' must be numeric.")

    row = {c: payload[c] for c in RAW_FEATURE_COLUMNS}
    return pd.DataFrame([row])
