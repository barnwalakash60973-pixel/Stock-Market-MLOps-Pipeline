from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas import CSVPredictionResponse, HealthResponse
from app.services import ServiceError, prediction_service
from src.utils.logger import get_logger

logger = get_logger("api")

router = APIRouter(tags=["Prediction"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the API health status.",
)
def health_check() -> HealthResponse:
    return HealthResponse(status="healthy")


@router.post(
    "/predict/upload",
    response_model=CSVPredictionResponse,
    summary="Predict stock movement",
    description="""
Upload a CSV file with the following columns:

- Date
- Open
- High
- Low
- Close
- Adj Close
- Volume
- Ticker

Example:

Date,Open,High,Low,Close,Adj Close,Volume,Ticker
2025-01-01,100,105,99,104,104,1500000,AAPL
""",
)
async def predict_from_upload(
    file: UploadFile = File(
        ...,
        description="CSV file containing OHLCV stock data.",
    ),
) -> CSVPredictionResponse:
    """
    Upload raw historical OHLCV data as a CSV and receive predictions.
    """

    if file.filename is None or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    try:
        result = prediction_service.predict_csv(file)

        return CSVPredictionResponse(
            message="Prediction completed successfully.",
            rows=len(result),
            predictions=result.to_dict(orient="records"),
        )

    except ServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        )
