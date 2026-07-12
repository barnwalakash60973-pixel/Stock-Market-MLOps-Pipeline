# validators.py
import pandas as pd

from src.utils.constants import EXPECTED_DTYPES, REQUIRED_COLUMNS


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    This fxn is basically validate the data.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"])

    for col, dtype in EXPECTED_DTYPES.items():
        df[col] = pd.to_numeric(df[col], errors="raise")

        if dtype == "int64":
            df[col] = df[col].astype("int64")
        else:
            df[col] = df[col].astype("float64")

    return df
