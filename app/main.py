# main.py
from fastapi import FastAPI

from app.routes import router
from app.services import prediction_service
from src.utils.logger import get_logger

logger = get_logger("api")

app = FastAPI(
    title="Stock Market Prediction API",
    description="REST API for predicting stock market "
    "movement using a trained CatBoost model.",
    version="1.0.0",
    contact={
        "name": "Akash Kumar Barnwal",
        "email": "barnwalakash60973@gmail.com",
    },
)

app.include_router(router)


@app.on_event("startup")
def startup_event() -> None:
    logger.info("Starting API...")

    try:
        prediction_service.load_model()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")


@app.get("/", tags=["Root"])
def root() -> dict:
    return {
        "message": "Stock Market Prediction API is running.",
        "version": "1.0.0",
        "docs": "/docs",
    }
