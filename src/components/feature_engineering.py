from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger


logger = get_logger("feature_engineering")

EPS = 1e-8
UP_THRE = 0.01
LOW_THRE = -0.01


class FeatureEngineeringError(Exception):
    """Custom exception for feature engineering failures."""
    pass


class FeatureEngineering:
    """
    Reads validated stock data and generates leakage-safe engineered
    features (returns, trend, oscillators, volatility, volume,
    candlestick patterns) for downstream model training.

    """

    def __init__(self):
        try:
            config = load_config()

            self.input_path = config["output"]["raw_data_path"]
            self.output_path = config["output"]["feature_data_path"]

        except KeyError as e:
            logger.error(f"Missing required config key: {e}")
            raise FeatureEngineeringError(f"Invalid config.yaml: missing key {e}") from e

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise FeatureEngineeringError("Could not initialize FeatureEngineering") from e

    # ------------------------------------------------------------------
    # Loading (read-only, no downloading, no fallback)
    # ------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        input_file = Path(self.input_path)

        if not input_file.exists():
            logger.error(f"Data file does not exist at {self.input_path}. Run ingestion first.")
            raise FeatureEngineeringError(
                f"No data found at {self.input_path}. "
                f"This script only reads existing data — it does not download or regenerate it."
            )

        try:
            logger.info(f"Reading existing data from {self.input_path} (read-only)")

            df = pd.read_parquet(self.input_path)

            if df.empty:
                raise FeatureEngineeringError("Data file exists but is empty.")

            df["Date"] = pd.to_datetime(df["Date"])

            logger.info(f"Loaded data shape: {df.shape}")
            return df

        except FeatureEngineeringError:
            raise

        except Exception as e:
            logger.error(f"Failed to read data: {e}")
            raise FeatureEngineeringError("Failed to read existing data file") from e
        
    
    # ------------------------------------------------------------------
    # Static indicator helpers (pure functions, no leakage)
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
        """Wilder's RSI - causal, uses only past `window` daily changes."""
        delta = close.diff()
        gain, loss = delta.clip(lower=0.0), -delta.clip(upper=0.0)

        avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

        rs = avg_gain / (avg_loss + EPS)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Returns (macd, macd_signal, macd_hist) - all causal EWMs."""
        ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()

        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, min_periods=signal, adjust=False).mean()

        return macd, macd - macd_signal

    @staticmethod
    def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume - running cumulative sum, uses only t and t-1."""
        direction = np.sign(close.diff().fillna(0.0))
        return (direction * volume).cumsum()

    # ------------------------------------------------------------------
    # Feature blocks (each operates on the full multi-ticker frame,
    # grouping internally by "Ticker" so no leakage occurs across stocks)
    # ------------------------------------------------------------------

    def _create_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates return-based momentum features: 1D/5D/10D percent
        returns plus lagged 1-day returns (t-1, t-2, t-3).
        """
        g_price = df.groupby("Ticker")["Adj Close"]

        df["Return_1D"] = g_price.pct_change(1)
        
        r1 = df.groupby("Ticker")["Return_1D"]
        df["Return_1D_Lag1"] = r1.shift(1)
        df["Return_1D_Lag2"] = r1.shift(2)

        return df

    def _create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates trend features: price-to-moving-average ratios
        (SMA 10/20/50, EMA 10/20) and the SMA10:SMA50 ratio, all
        expressed relative to price so they're scale-independent.
        """
        price = df["Adj Close"]
        g_price = df.groupby("Ticker")["Adj Close"]

        sma10 = g_price.transform(lambda x: x.rolling(10, min_periods=10).mean())
        sma20 = g_price.transform(lambda x: x.rolling(20, min_periods=20).mean())
        sma50 = g_price.transform(lambda x: x.rolling(50, min_periods=50).mean())

        ema10 = g_price.transform(lambda x: x.ewm(span=10, min_periods=10, adjust=False).mean())
        
        df["Close_SMA10_Ratio"] = price / sma10
        df["Close_SMA20_Ratio"] = price / sma20
        df["Close_SMA50_Ratio"] = price / sma50

        df["Close_EMA10_Ratio"] = price / ema10
        
        return df

    def _create_oscillator_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates momentum oscillators: Wilder's RSI (14), MACD/signal/
        histogram (normalized by price), and the 14-period Stochastic %K.
        """
        price = df["Adj Close"]
        g_price = df.groupby("Ticker")["Adj Close"]

        df["RSI_14"] = g_price.transform(lambda x: self._rsi(x, 14))

        macd_parts = {"MACD": [],  "MACD_Hist": []}

        for _, sub in g_price:
            macd, hist = self._macd(sub)
            macd_parts["MACD"].append(macd)
            macd_parts["MACD_Hist"].append(hist)

        for col, parts in macd_parts.items():
            df[col] = pd.concat(parts).reindex(df.index) / price

        # Stochastic uses actual OHLC, not adjusted close
        low14 = df.groupby("Ticker")["Low"].transform(lambda x: x.rolling(14, min_periods=14).min())
        high14 = df.groupby("Ticker")["High"].transform(lambda x: x.rolling(14, min_periods=14).max())

        df["Stoch_K_14"] = 100 * (df["Close"] - low14) / (high14 - low14 + EPS)

        return df

    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates volatility features: 20-day return volatility, True
        Range % and ATR(14), and Bollinger Band position/width.
        SMA20/STD20 are recomputed locally here (not shared with the
        trend block) so this method has no dependency on call order.
        """
        price = df["Adj Close"]
        g_price = df.groupby("Ticker")["Adj Close"]
        prev_price = g_price.shift(1)

        high, low = df["High"], df["Low"]

        df["Volatility_20"] = (
            df.groupby("Ticker")["Return_1D"]
            .transform(lambda x: x.rolling(20, min_periods=20).std())
        )

        tr = pd.concat([
            high - low,
            (high - prev_price).abs(),
            (low - prev_price).abs(),
        ], axis=1).max(axis=1)

        df["True_Range_Pct"] = tr / price

        df["_tr_tmp"] = tr
        atr14 = df.groupby("Ticker")["_tr_tmp"].transform(lambda x: x.rolling(14, min_periods=14).mean())
        df["ATR_14"] = atr14 / price
        df.drop(columns="_tr_tmp", inplace=True)

        sma20 = g_price.transform(lambda x: x.rolling(20, min_periods=20).mean())
        std20 = g_price.transform(lambda x: x.rolling(20, min_periods=20).std())

        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20

        df["BB_Width"] = (bb_upper - bb_lower) / (sma20 + EPS)

        return df

    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates volume features: 20-day volume ratio, lagged volume
        ratio, and the 10-day slope of On-Balance Volume (normalized
        by average volume).
        """
        vol = df["Volume"]
        g = df.groupby("Ticker")

        vol_sma20 = g["Volume"].transform(lambda x: x.rolling(20, min_periods=20).mean())
        df["Volume_Ratio_20"] = vol / (vol_sma20 + EPS)
        df["Volume_Lag1"] = g["Volume"].shift(1) / (vol_sma20 + EPS)

        return df

    def _create_candlestick_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates candlestick/price-action features: high-low spread,
        body size, upper/lower shadows (all normalized by close), and
        the overnight gap relative to the previous close.
        """
        open_, high, low, close = df["Open"], df["High"], df["Low"], df["Close"]

        upper_body = np.maximum(close, open_)
        lower_body = np.minimum(close, open_)

        df["High_Low_Spread"] = (high - low) / close
        df["Upper_Shadow"] = (high - upper_body) / close
        df["Lower_Shadow"] = (lower_body - low) / close

        return df
    
    # ------------------------------------------------------------------
    # Target Build
    # ------------------------------------------------------------------
    def _create_target(
        self,
        df: pd.DataFrame,
        up_th: float,
       down_th: float
     ) -> pd.DataFrame:
        """

       Creates a 3-class target using the next day's adjusted return.

       0 = Neutral
       1 = Up
       2 = Down
    """

        df = df.copy()

        g = df.groupby("Ticker", group_keys=False, sort=False)

        next_price = g["Adj Close"].shift(-1)

        forward_return = next_price / df["Adj Close"] - 1

        df["Forward_Return_1D"] = forward_return

        df["Target"] = np.select(
          [
            forward_return > up_th,
            forward_return < down_th,
          ],
          [
            1,
            2,
          ],
          default=0,
        )

        # Last row of every ticker has no future price
        df.loc[next_price.isna(), "Target"] = np.nan
        df.drop(columns=["Forward_Return_1D"], inplace=True)

        return df

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds all model features. Does not mutate input.
        Sorted by [Ticker, Date]; warm-up periods are left as NaN.
        """
        df = df.copy()
        df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)  # leakage guard

        df = self._create_return_features(df)
        df = self._create_trend_features(df)
        df = self._create_oscillator_features(df)
        df = self._create_volatility_features(df)
        df = self._create_volume_features(df)
        df = self._create_candlestick_features(df)
        df = self._create_target(df, up_th = UP_THRE, down_th = LOW_THRE) 
  
        # Reduce memory usage
        float_cols = df.select_dtypes(include=["float64"]).columns
        df[float_cols] = df[float_cols].astype("float32")
        df["Target"] = df["Target"].astype("Int8")
        return df

    def run(self) -> str:
        raw_df = self.load_data()

        logger.info(f"Building features for {raw_df['Ticker'].nunique()} tickers")

        try:
            final_df = self.build_features(raw_df)
        except Exception as e:
            logger.error(f"Feature building failed: {e}")
            raise FeatureEngineeringError("Feature engineering failed") from e

        logger.info(f"Final feature dataset shape: {final_df.shape}")

        try:
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            final_df.to_parquet(self.output_path, index=False)
            logger.info(f"Feature data saved successfully to {self.output_path}")

        except Exception as e:
            logger.error(f"Failed to save feature data to {self.output_path}: {e}")
            raise FeatureEngineeringError("Failed to persist feature data") from e

        return self.output_path

    
    
if __name__ == "__main__":
    try:
        engineer = FeatureEngineering()
        path = engineer.run()
        df = pd.read_parquet(path)

        logger.info(f"Feature dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Saved to: {path}")

    except FeatureEngineeringError as e:
        logger.critical(f"Feature engineering pipeline failed: {e}")
        raise

    except Exception as e:
        logger.critical(f"Unexpected error in feature engineering pipeline: {e}")
        raise