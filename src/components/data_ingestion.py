from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("data_ingestion")


class DataIngestionError(Exception):
    """Custom exception for data ingestion failures."""

    pass


class DataIngestion:
    """
    Downloads historical stock data from Yahoo Finance
    and stores it as a raw Parquet dataset.
    """

    def __init__(self):
        try:
            config = load_config()

            self.stocks = config["data"]["stocks"]
            self.start_date = config["data"]["start_date"]
            self.end_date = config["data"]["end_date"]
            self.output_path = config["output"]["raw_data_path"]

        except KeyError as e:
            logger.error(f"Missing required config key: {e}")
            raise DataIngestionError(f"Invalid config.yaml: missing key {e}") from e

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise DataIngestionError("Could not initialize DataIngestion") from e

    def _download_single_stock(self, stock: str) -> pd.DataFrame | None:
        """Download and clean data for a single stock. Returns None on failure."""
        try:
            logger.info(f"Downloading {stock}...")

            df = yf.download(
                stock,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=False,
            )

            if df.empty:
                logger.warning(f"No data returned for {stock}. Skipping.")
                return None

            # yfinance can return MultiIndex columns even for a single ticker
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.reset_index(inplace=True)
            df["Ticker"] = stock

            logger.info(f"{stock}: {len(df)} rows downloaded")

            return df

        except Exception as e:
            logger.error(f"Failed to download {stock}: {e}")
            return None

    def download_data(self) -> str:

        all_data: list[pd.DataFrame] = []

        logger.info(f"Stocks to download: {len(self.stocks)}")
        logger.info(f"Date range: {self.start_date} -> {self.end_date}")

        for stock in self.stocks:
            df = self._download_single_stock(stock)
            if df is not None:
                all_data.append(df)

        if not all_data:
            logger.error("No data was downloaded for any ticker.")
            raise DataIngestionError("Data ingestion failed: no tickers returned data.")

        try:
            final_df = pd.concat(all_data, ignore_index=True)
        except Exception as e:
            logger.error(f"Failed to concatenate downloaded data: {e}")
            raise DataIngestionError("Failed to merge downloaded stock data") from e

        final_df.sort_values(["Ticker", "Date"], inplace=True)
        final_df.reset_index(drop=True, inplace=True)

        logger.info(f"Rows downloaded: {len(final_df)}")
        logger.info(f"Final dataset shape: {final_df.shape}")

        try:
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

            final_df.to_parquet(self.output_path, index=False)
            logger.info(f"Data saved successfully to {self.output_path}")

        except Exception as e:
            logger.error(f"Failed to save data to {self.output_path}: {e}")
            raise DataIngestionError("Failed to persist downloaded data") from e

        return self.output_path

    def run(self, force=False):
        output_file = Path(self.output_path)

        if output_file.exists() and not force:
            logger.info(
                f"Raw data already exists at {self.output_path}. "
                "Skipping data ingestion."
            )
            return self.output_path

        logger.info("Starting data ingestion...")

        df = self.download_data()
        self.save_data(df)

        logger.info(f"Raw data saved to {self.output_path}")

        return self.output_path


if __name__ == "__main__":
    try:
        ingestion = DataIngestion()
        path = ingestion.run()
        logger.info(f"Saved to: {path}")

    except DataIngestionError as e:
        logger.critical(f"Data ingestion pipeline failed: {e}")
        raise

    except Exception as e:
        logger.critical(f"Unexpected error in data ingestion pipeline: {e}")
        raise
