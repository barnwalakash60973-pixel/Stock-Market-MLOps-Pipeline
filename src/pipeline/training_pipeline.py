# training_pipeline.py
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.feature_engineering import FeatureEngineering
from src.components.model_training import ModelTrainer
from src.utils.logger import get_logger

logger = get_logger("training_pipeline")




def main():
    try:
        logger.info("Starting Training Pipeline")

        logger.info("Starting Data Ingestion")
        DataIngestion().run()
        logger.info("Data Ingestion completed")

        logger.info("Starting Data Validation")
        DataValidation().run()
        logger.info("Data Validation completed")

        logger.info("Starting Feature Engineering")
        FeatureEngineering().run()
        logger.info("Feature Engineering completed")

        logger.info("Starting Model Training")
        ModelTrainer().run()
        logger.info("Model Training completed")

        logger.info("Training Pipeline completed successfully")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise
if __name__ == "__main__":
    main()
    
