"""
All non-HTTP helpers in one place: CSV validation, session-state
management, formatting, and CSV download building.
"""

import io
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st
from constants import PREDICTION_LABELS, REQUIRED_COLUMNS


@dataclass
class PredictionRun:
    timestamp: float
    filename: str
    rows: int
    dataframe: pd.DataFrame


HISTORY_KEY = "prediction_history"
LAST_RESULT_KEY = "last_prediction_result"


# --------------------------------------------------------------------------
# CSV validation (shape checks only — no feature engineering, that stays
# on the backend)
# --------------------------------------------------------------------------
@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]


def validate_file_extension(filename: str | None) -> bool:
    return bool(filename) and filename.lower().endswith(".csv")


def validate_dataframe_shape(df: pd.DataFrame) -> ValidationResult:
    """Check required columns are present and Date parses — nothing more."""
    errors, warnings = [], []

    if df.empty:
        errors.append("The uploaded file has no rows.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required column(s): {', '.join(missing)}")

    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra:
        warnings.append(
            f"Extra column(s) will be ignored by the backend: {', '.join(extra)}"
        )

    if "Date" in df.columns:
        try:
            pd.to_datetime(df["Date"])
        except (ValueError, TypeError):
            errors.append("The 'Date' column contains values that aren't valid dates.")

    return ValidationResult(is_valid=not errors, errors=errors, warnings=warnings)


def read_csv_preview(file_bytes: bytes) -> pd.DataFrame:
    """Parse uploaded bytes into a DataFrame for preview purposes only."""
    return pd.read_csv(io.BytesIO(file_bytes))


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def init_session_state() -> None:
    """Ensure the keys this app relies on exist, regardless of tab order."""
    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = []
    if LAST_RESULT_KEY not in st.session_state:
        st.session_state[LAST_RESULT_KEY] = None


def record_prediction_run(df: pd.DataFrame, source_filename: str) -> None:
    """Append a completed run so the Dashboard tab can show cumulative KPIs."""
    
    init_session_state()

    entry = PredictionRun(
        timestamp=time.time(),
        filename=source_filename,
        rows=len(df),
        dataframe=df,
    )
    st.session_state[HISTORY_KEY].append(entry)
    st.session_state[LAST_RESULT_KEY] = entry


def get_latest_result() -> dict[str, Any] | None:
    init_session_state()
    return st.session_state[LAST_RESULT_KEY]

def get_combined_predictions() -> pd.DataFrame:
    """Concatenate every run this session for cumulative Dashboard KPIs."""
    init_session_state()
    history = st.session_state[HISTORY_KEY]

    if not history:
        return pd.DataFrame()

    return pd.concat([e.dataframe for e in history], ignore_index=True)

def compute_label_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or "Prediction_Label" not in df.columns:
        return {label: 0 for label in PREDICTION_LABELS}
    counts = df["Prediction_Label"].value_counts().to_dict()
    return {label: int(counts.get(label, 0)) for label in PREDICTION_LABELS}


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------
def format_response_time(ms: float) -> str:
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.2f} s"


def format_timestamp(ts: float | None) -> str:
    return (
        "—" if ts is None else datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    )


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_prediction_filename(prefix: str = "predictions") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
