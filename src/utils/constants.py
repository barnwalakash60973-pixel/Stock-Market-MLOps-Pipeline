REQUIRED_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Ticker",
]

EXPECTED_DTYPES = {
    "Open": "float64",
    "High": "float64",
    "Low": "float64",
    "Close": "float64",
    "Adj Close": "float64",
    "Volume": "int64",
}

FEATURE_COLS = [
    "Return_1D",
    "Return_1D_Lag1",
    "Return_1D_Lag2",
    "Close_SMA10_Ratio",
    "Close_SMA20_Ratio",
    "Close_SMA50_Ratio",
    "Close_EMA10_Ratio",
    "RSI_14",
    "MACD",
    "MACD_Hist",
    "Stoch_K_14",
    "Volatility_20",
    "ATR_14",
    "True_Range_Pct",
    "Volume_Ratio_20",
    "BB_Width",
    "Volume_Lag1",
    "High_Low_Spread",
    "Upper_Shadow",
    "Lower_Shadow",
]

TARGET_COL = "Target"

TARGET_LABELS = [0, 1, 2]

TARGET_NAMES = [
    "Neutral",
    "Up",
    "Down",
]
