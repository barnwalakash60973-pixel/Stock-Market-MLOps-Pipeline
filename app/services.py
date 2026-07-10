import pandas as pd
from fastapi import UploadFile

from app.validators import validate_dataframe
from src.components.feature_engineering import (
    FeatureEngineering,
    FeatureEngineeringError,
)
from src.components.prediction import Prediction, PredictionError
from src.utils.logger import get_logger

logger = get_logger("api")


class ServiceError(Exception):
    """Raised when the prediction service cannot process a request."""

    pass


class PredictionService:
    """Handles feature engineering and prediction for uploaded CSV files."""

    def __init__(self):
        self.feature_engineer = FeatureEngineering()
        self.predictor = Prediction()

    def load_model(self) -> None:
        """Load the trained model during API startup."""
        self.predictor.load_model()

    def predict_csv(self, file: UploadFile) -> pd.DataFrame:
        try:
            raw_df = pd.read_csv(file.file)
            raw_df = validate_dataframe(raw_df)

            feature_df = self.feature_engineer.build_features(raw_df)
            # Only remove Target-missing rows when evaluating
            if "Target" in feature_df.columns:
                feature_df = feature_df.dropna(subset=["Target"])

            return self.predictor.predict(feature_df)

        except (FeatureEngineeringError, PredictionError) as e:
            raise ServiceError(str(e)) from e

        except Exception as e:
            logger.exception("Prediction service failed")
            raise ServiceError("Unexpected prediction error.") from e


prediction_service = PredictionService()
