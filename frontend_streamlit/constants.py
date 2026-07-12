"""Static constants: schema, design tokens, static copy."""

# Must match the backend's expected CSV schema
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

PREDICTION_LABELS = ["Up", "Down", "Neutral"]
PROBABILITY_COLUMNS = ["Prob_Up", "Prob_Down", "Prob_Neutral"]

SAMPLE_CSV = (
    "Date,Open,High,Low,Close,Adj Close,Volume,Ticker\n"
    "2025-01-01,100,105,99,104,104,1500000,AAPL\n"
    "2025-01-02,104,108,103,107,107,1600000,AAPL\n"
)

# Dark fintech theme
COLORS = {
    "bg": "#0B1220",
    "surface": "#121B2E",
    "border": "#223049",
    "text": "#E5EAF2",
    "text_muted": "#8B98B0",
    "accent": "#22D3B8",
    "up": "#22D3B8",
    "down": "#F0537A",
    "neutral": "#F5B942",
    "info": "#5B8DEF",
}

LABEL_COLORS = {
    "Up": COLORS["up"],
    "Down": COLORS["down"],
    "Neutral": COLORS["neutral"],
}

PROJECT_NAME = "Stock Market Prediction API"
PROJECT_AUTHOR = "Akash Kumar Barnwal"
AUTHOR_EMAIL = "barnwalakash60973@gmail.com"
PROJECT_VERSION = "1.0.0"
MODEL_ALGORITHM = "CatBoost Classifier"
