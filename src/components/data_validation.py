
import json
from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.constants import REQUIRED_COLUMNS, EXPECTED_DTYPES

logger = get_logger("data_validation")


class DataValidationError(Exception):
    """Custom exception for data validation failures."""
    pass


class DataValidation:

    def __init__(self):
        try:
            config = load_config()

            self.input_path = config["output"]["raw_data_path"]
            self.report_path = config["output"].get(
                                "validation_report_path",
                                "artifacts/validation_report.json"
                            )

        except KeyError as e:
            logger.error(f"Missing required config key: {e}")
            raise DataValidationError(f"Invalid config.yaml: missing key {e}") from e

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise DataValidationError("Could not initialize DataValidation") from e

        self.df: pd.DataFrame | None = None
        self.report: dict = {}

    # ------------------------------------------------------------------
    # Loading (read-only, no downloading, no fallback)
    # ------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        if not Path(self.input_path).exists():
            logger.error(f"Data file does not exist at {self.input_path}. Run ingestion first.")
            raise DataValidationError(
                f"No data found at {self.input_path}. "
                f"This script only reads existing data — it does not download anything."
            )

        try:
            logger.info(f"Reading existing data from {self.input_path} (read-only)")

            df = pd.read_parquet(self.input_path)

            if df.empty:
                raise DataValidationError("Data file exists but is empty.")

            self.df = df
            logger.info(f"Loaded data shape: {df.shape}")
            return df

        except DataValidationError:
            raise

        except Exception as e:
            logger.error(f"Failed to read data: {e}")
            raise DataValidationError("Failed to read existing data file") from e

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def validate_schema(self) -> dict:
        missing = set(REQUIRED_COLUMNS) - set(self.df.columns)
        extra = set(self.df.columns) - set(REQUIRED_COLUMNS)

        result = {
            "passed": len(missing) == 0,
            "missing_columns": list(missing),
            "extra_columns": list(extra),
        }

        if missing:
            logger.error(f"Schema validation failed. Missing columns: {missing}")
        else:
            logger.info("Schema validation passed.")

        return result

    def check_missing_values(self) -> dict:
        missing_counts = self.df.isnull().sum()
        missing_dict = missing_counts[missing_counts > 0].to_dict()

        result = {
            "passed": len(missing_dict) == 0,
            "missing_by_column": missing_dict,
            "total_missing": int(missing_counts.sum()),
        }

        if missing_dict:
            logger.warning(f"Missing values found: {missing_dict}")
        else:
            logger.info("No missing values found.")

        return result

    def check_duplicates(self) -> dict:
        subset = ["Date", "Ticker"] if {"Date", "Ticker"}.issubset(self.df.columns) else None
        duplicate_count = int(self.df.duplicated(subset=subset).sum())

        result = {
            "passed": duplicate_count == 0,
            "duplicate_count": duplicate_count,
        }

        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate rows (subset={subset}).")
        else:
            logger.info("No duplicate rows found.")

        return result

    def validate_datatypes(self) -> dict:
        mismatches = {}

        for col, expected_dtype in EXPECTED_DTYPES.items():
            if col not in self.df.columns:
                continue

            actual_dtype = str(self.df[col].dtype)

            if actual_dtype != expected_dtype:
                mismatches[col] = {
                    "expected": expected_dtype,
                    "actual": actual_dtype,
                }

        result = {
            "passed": len(mismatches) == 0,
            "mismatches": mismatches,
        }

        if mismatches:
            logger.warning(f"Datatype mismatches found: {mismatches}")
        else:
            logger.info("Datatype validation passed.")

        return result

    def validate_price_columns(self) -> dict:
        price_cols = ["Open", "High", "Low", "Close", "Adj Close"]
        present = [c for c in price_cols if c in self.df.columns]

        issues = {}

        for col in present:
            invalid = self.df[self.df[col] <= 0]
            if not invalid.empty:
                issues[f"{col}_non_positive"] = len(invalid)

        if {"High", "Low", "Open", "Close"}.issubset(self.df.columns):
            bad_high = self.df[
                (self.df["High"] < self.df["Low"]) |
                (self.df["High"] < self.df["Open"]) |
                (self.df["High"] < self.df["Close"])
            ]
            bad_low = self.df[
                (self.df["Low"] > self.df["Open"]) |
                (self.df["Low"] > self.df["Close"])
            ]

            if not bad_high.empty:
                issues["high_lower_than_ohlc"] = len(bad_high)
            if not bad_low.empty:
                issues["low_higher_than_ohlc"] = len(bad_low)

        result = {
            "passed": len(issues) == 0,
            "issues": issues,
        }

        if issues:
            logger.warning(f"Price column validation issues: {issues}")
        else:
            logger.info("Price column validation passed.")

        return result

    def validate_volume(self) -> dict:
        if "Volume" not in self.df.columns:
            return {"passed": False, "issue": "Volume column missing"}

        negative_volume = int((self.df["Volume"] < 0).sum())
        zero_volume = int((self.df["Volume"] == 0).sum())

        result = {
            "passed": negative_volume == 0,
            "negative_volume_count": negative_volume,
            "zero_volume_count": zero_volume,
        }

        if negative_volume > 0:
            logger.error(f"Found {negative_volume} rows with negative volume.")
        if zero_volume > 0:
            logger.warning(f"Found {zero_volume} rows with zero volume (may be valid for holidays).")

        return result

    def validate_date_order(self) -> dict:
        if "Date" not in self.df.columns or "Ticker" not in self.df.columns:
            return {"passed": False, "issue": "Date or Ticker column missing"}

        unordered_tickers = []

        for ticker, group in self.df.groupby("Ticker"):
            dates = group["Date"].reset_index(drop=True)
            if not dates.is_monotonic_increasing:
                unordered_tickers.append(ticker)

        result = {
            "passed": len(unordered_tickers) == 0,
            "unordered_tickers": unordered_tickers,
        }

        if unordered_tickers:
            logger.warning(f"Dates not in order for tickers: {unordered_tickers}")
        else:
            logger.info("Date ordering validation passed.")

        return result

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_validation_report(self) -> str:
        try:
            Path(self.report_path).parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with open(self.report_path, "w") as f:
                json.dump(self.report, f, indent=4)

            logger.info(f"Validation report saved to {self.report_path}")
            return self.report_path

        except Exception as e:
            logger.error(f"Failed to write validation report: {e}")
            raise DataValidationError("Failed to generate validation report") from e

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def validate(self) -> dict:
        self.load_data()

        self.report = {
            "schema": self.validate_schema(),
            "missing_values": self.check_missing_values(),
            "duplicates": self.check_duplicates(),
            "datatypes": self.validate_datatypes(),
            "price_columns": self.validate_price_columns(),
            "volume": self.validate_volume(),
            "date_order": self.validate_date_order(),
        }

        self.generate_validation_report()

        critical_checks = ["schema",
                           "missing_values", 
                           "price_columns", 
                           "volume"]
        
        failed_critical = [c for c in critical_checks if not self.report[c]["passed"]]

        if failed_critical:
            logger.error(f"Critical validation checks failed: {failed_critical}")
            raise DataValidationError(f"Data validation failed on: {failed_critical}")

        logger.info("Data validation completed successfully.")
        return self.report
    
    def run(self) -> dict:
        """
        Runs the complete data validation pipeline.
        """
        return self.validate()


if __name__ == "__main__":
    try:
        validator = DataValidation()
        report = validator.run()

        for check, result in report.items():
            logger.info(
                f"{check}: {'PASSED' if result['passed'] else 'FAILED'}"
            )

    except DataValidationError as e:
        logger.critical(f"Data validation failed: {e}")
        raise

    except Exception as e:
        logger.critical(f"Unexpected error during validation: {e}")
        raise