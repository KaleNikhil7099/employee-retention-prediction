"""
ml/data_loader.py — Dataset loading and validation for RetentionAI.
Uses the real IBM HR Analytics dataset (1470 rows, 35 features).
"""

import os
import pandas as pd
import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATASET_FILENAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
DATASET_PATH = os.path.join(DATASET_DIR, DATASET_FILENAME)

# Columns to drop: constants, IDs, or target-leaking fields
DROP_COLS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]

TARGET_COL = "Attrition"


def dataset_exists() -> bool:
    return os.path.isfile(DATASET_PATH)


def load_raw_dataset() -> pd.DataFrame:
    """Load the raw IBM HR dataset. Raises FileNotFoundError if missing."""
    if not dataset_exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}\n\n"
            "Please download 'WA_Fn-UseC_-HR-Employee-Attrition.csv' from:\n"
            "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset\n"
            f"and place it in: {DATASET_DIR}"
        )
    df = pd.read_csv(DATASET_PATH)
    return df


def get_dataset_info(df: pd.DataFrame) -> dict:
    """Compute dataset summary statistics."""
    attrition_counts = df[TARGET_COL].value_counts()
    attrition_yes = int(attrition_counts.get("Yes", 0))
    attrition_no = int(attrition_counts.get("No", 0))
    total = len(df)

    return {
        "total_rows": total,
        "total_cols": len(df.columns),
        "attrition_yes": attrition_yes,
        "attrition_no": attrition_no,
        "attrition_rate": round(attrition_yes / total * 100, 2) if total > 0 else 0,
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_cols": list(df.select_dtypes(include=np.number).columns),
        "categorical_cols": list(df.select_dtypes(include="object").columns),
    }


def prepare_features(df: pd.DataFrame):
    """
    Split dataframe into X, y with proper column removal.
    Returns (X, y, categorical_cols, numeric_cols).
    Fits only on training data — caller must split before fitting transformers.
    """
    df = df.copy()

    # Encode target
    y = (df[TARGET_COL].str.strip().str.lower() == "yes").astype(int)

    # Drop target and constant/ID columns
    cols_to_drop = [TARGET_COL] + [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    categorical_cols = list(X.select_dtypes(include="object").columns)
    numeric_cols = list(X.select_dtypes(include=np.number).columns)

    return X, y, categorical_cols, numeric_cols


def get_dept_attrition(df: pd.DataFrame) -> pd.DataFrame:
    """Attrition rate by Department."""
    if "Department" not in df.columns:
        return pd.DataFrame()
    grp = df.groupby("Department")[TARGET_COL].apply(
        lambda s: (s.str.lower() == "yes").mean() * 100
    ).reset_index()
    grp.columns = ["Department", "Attrition Rate (%)"]
    return grp.sort_values("Attrition Rate (%)", ascending=False)


def get_role_attrition(df: pd.DataFrame) -> pd.DataFrame:
    """Attrition rate by JobRole."""
    if "JobRole" not in df.columns:
        return pd.DataFrame()
    grp = df.groupby("JobRole")[TARGET_COL].apply(
        lambda s: (s.str.lower() == "yes").mean() * 100
    ).reset_index()
    grp.columns = ["Job Role", "Attrition Rate (%)"]
    return grp.sort_values("Attrition Rate (%)", ascending=False)
