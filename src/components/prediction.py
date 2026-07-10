from pathlib import Path

import joblib
import pandas as pd

from src.utils.config import load_config
from src.utils.constants import FEATURE_COLS, TARGET_COL, TARGET_NAMES
from src.utils.logger import get_logger

logger = get_logger("prediction")


class PredictionError(Exception):
    """Custom exception for prediction failures."""

    pass


class Prediction:
    """
    Loads the trained pipeline and the engineered feature data, filters
    to `inference_range` from config, predicts, and saves the result to
    data/predictions.
    """

    def __init__(self):
        try:
            config = load_config()

            self.model_path = config["output"]["model_path"]
            self.input_path = config["output"]["feature_data_path"]
            self.output_path = config["output"].get(
                "prediction_path",
                "data/predictions/predictions.parquet",
            )
            self.inference_range = config["train"]["inference_range"]

        except KeyError as e:
            logger.error(f"Missing required config key: {e}")
            raise PredictionError(f"Invalid config.yaml: missing key {e}") from e

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise PredictionError("Could not initialize Prediction") from e

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    def load_model(self):
        if not Path(self.model_path).exists():
            raise PredictionError(f"Model not found: {self.model_path}")

        try:
            logger.info(f"Loading model from {self.model_path}")
            return joblib.load(self.model_path)

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise PredictionError("Failed to load trained model") from e

    # ---------------------------------------------------------
    # Load feature data
    # ---------------------------------------------------------

    def load_features(self) -> pd.DataFrame:
        if not Path(self.input_path).exists():
            raise PredictionError(f"Feature file not found: {self.input_path}")

        try:
            logger.info(f"Loading feature data from {self.input_path}")
            df = pd.read_parquet(self.input_path)

            if df.empty:
                raise PredictionError("Feature file exists but is empty.")

            df["Date"] = pd.to_datetime(df["Date"])
            return df

        except PredictionError:
            raise

        except Exception as e:
            logger.error(f"Failed to read feature data: {e}")
            raise PredictionError("Failed to read existing feature data file") from e

    # ---------------------------------------------------------
    # Filter to inference range
    # ---------------------------------------------------------

    def _slice_to_inference_range(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = (df["Date"] >= pd.Timestamp(self.inference_range["start"])) & (
            df["Date"] <= pd.Timestamp(self.inference_range["end"])
        )
        sliced = df.loc[mask].sort_values(["Ticker", "Date"]).reset_index(drop=True)

        if sliced.empty:
            raise PredictionError(
                f"No feature rows found in inference_range "
                f"{self.inference_range['start']} -> {self.inference_range['end']}."
            )

        logger.info(
            f"Filtered to inference_range {self.inference_range['start']} -> "
            f"{self.inference_range['end']} ({len(sliced)} rows)"
        )
        return sliced

    # ---------------------------------------------------------
    # Predict
    # ---------------------------------------------------------

    def predict(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Predict from an already engineered feature dataframe.
        Used by FastAPI/Streamlit.
        """

        pipeline = self.load_model()

        # Training pipeline
        if df is None:
            df = self.load_features()
            df = self._slice_to_inference_range(df)

        X = df[FEATURE_COLS]

        predictions = pipeline.predict(X).ravel()

        probabilities = pipeline.predict_proba(X)

        result = df[["Date", "Ticker"]].copy()

        result["Prediction"] = predictions

        result["Prediction_Label"] = [TARGET_NAMES[int(p)] for p in predictions]

        result["Prob_Neutral"] = probabilities[:, 0]
        result["Prob_Up"] = probabilities[:, 1]
        result["Prob_Down"] = probabilities[:, 2]

        if TARGET_COL in df.columns:
            result[TARGET_COL] = df[TARGET_COL].values
        result["Date"] = result["Date"].dt.strftime("%Y-%m-%d")

        return result

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    def save_predictions(self, prediction_df: pd.DataFrame) -> str:
        try:
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            prediction_df.to_parquet(self.output_path, index=False)
            logger.info(f"Predictions saved to {self.output_path}")
            return self.output_path

        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
            raise PredictionError("Failed to persist predictions") from e

    # ---------------------------------------------------------
    # Run
    # ---------------------------------------------------------

    def run(self) -> str:
        prediction_df = self.predict()
        return self.save_predictions(prediction_df)


if __name__ == "__main__":
    try:
        predictor = Prediction()
        path = predictor.run()
        logger.info(f"Prediction file saved at: {path}")

    except PredictionError as e:
        logger.critical(f"Prediction pipeline failed: {e}")
        raise

    except Exception as e:
        logger.critical(f"Unexpected error in prediction pipeline: {e}")
        raise
