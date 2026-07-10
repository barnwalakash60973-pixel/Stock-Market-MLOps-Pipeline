# prediction_pipeline.py

from src.components.feature_engineering import FeatureEngineering
from src.components.prediction import Prediction
from src.utils.logger import get_logger

logger = get_logger("prediction_pipeline")


def main():
    try:
        logger.info("Starting Feature Engineering")
        FeatureEngineering().run()

        logger.info("Starting Prediction")
        Prediction().run()

        logger.info("Prediction pipeline completed successfully")

    except Exception as e:
        logger.exception(f"Prediction pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
