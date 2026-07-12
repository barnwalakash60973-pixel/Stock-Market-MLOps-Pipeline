# schemas.py
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response returned by the health endpoint."""

    status: str


class PredictionRecord(BaseModel):
    """Prediction for a single stock/date."""

    Date: str
    Ticker: str
    Prediction: int
    Prediction_Label: str
    Prob_Neutral: float
    Prob_Up: float
    Prob_Down: float
    Target: int | None = None


class CSVPredictionResponse(BaseModel):
    """Response returned after CSV prediction."""

    message: str
    rows: int
    predictions: list[PredictionRecord]
