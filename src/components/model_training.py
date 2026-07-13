from pathlib import Path

import joblib
import pandas as pd

try:
    from catboost import CatBoostClassifier
except ImportError as e:
    raise ImportError(
        "catboost is not installed. Install it with: pip install catboost"
    ) from e


from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from src.utils.logger import get_logger
from src.utils.config import load_config
from src.utils.mlflow_logger import MLflowLogger
from src.utils.constants import FEATURE_COLS, TARGET_COL, TARGET_LABELS, TARGET_NAMES

logger = get_logger("train")


class TrainingError(Exception):
    """Custom exception for model training failures."""

    pass


class ModelTrainer:
    """
    Trains a CatBoost classifier on engineered stock features to predict
    `Target`, validated with 5-fold expanding-window walk-forward splits
    (train 2015-2020 -> test 2021, train 2015-2021 -> test 2022, ...,
    train 2015-2024 -> test 2025).

    After walk-forward validation, a final production model is
    trained on all historical data through 2024 and used to generate
    forward-only inference predictions for 2025-2026 — no metrics are
    computed at this stage since future ground truth isn't assumed
    available.

    Within each fold, the training window is further split into a
    "fit" portion and a small, per-ticker, chronologically-last
    "early-stopping" (ES) portion, used only to tell CatBoost when to
    stop adding trees. The imputer is fit on the fit portion only.
    """

    def __init__(self):
        try:
            config = load_config()

            self.input_path = config["output"]["feature_data_path"]
            self.model_path = config["output"]["model_path"]
            self.fold_metrics_path = config["output"].get(
                "fold_metrics_path", "reports/walk_forward_metrics.txt"
            )

            self.final_metrics_path = config["output"].get(
                "final_metrics_path",
                "reports/final_model_metrics.txt",
            )

            self.random_state = config["train"].get("random_state", 42)
            self.model_params = config["train"].get("model_params", {})
            self.impute_strategy = config["train"].get("impute_strategy", "median")
            self.es_frac = config["train"].get("es_frac", 0.1)
            self.early_stopping_rounds = config["train"].get(
                "early_stopping_rounds", 50
            )

            self.walk_forward_folds = config["train"]["walk_forward_folds"]
            self.final_train_range = config["train"]["final_train_range"]
            self.evaluation_range = config["train"]["evaluation_range"]

        except KeyError as e:
            logger.error(f"Missing required config key: {e}")
            raise TrainingError(f"Invalid config.yaml: missing key {e}") from e

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise TrainingError("Could not initialize ModelTrainer") from e

    # ------------------------------------------------------------------
    # Loading (read-only, no downloading, no fallback)
    # ------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        input_file = Path(self.input_path)

        if not input_file.exists():
            logger.error(
                "Feature file does not exist at "
                f"{self.input_path}. Run feature_engineering first."
            )
            raise TrainingError(
                f"No feature data found at {self.input_path}.\n"
                "This script only reads existing data "
                "and does not build features itself."
            )
        try:
            logger.info(f"Reading feature data from {self.input_path} (read-only)")

            df = pd.read_parquet(input_file)

            if df.empty:
                raise TrainingError("Feature file exists but is empty.")

            df["Date"] = pd.to_datetime(df["Date"])

            logger.info(f"Loaded data shape: {df.shape}")
            return df

        except TrainingError:
            raise

        except Exception as e:
            logger.error(f"Failed to read feature data: {e}")
            raise TrainingError("Failed to read existing feature data file") from e

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing_features = set(FEATURE_COLS) - set(df.columns)
        if missing_features:
            raise TrainingError(f"Missing feature columns: {missing_features}")

        if TARGET_COL not in df.columns:
            raise TrainingError(f"Missing target column: '{TARGET_COL}'")

    def _slice_by_date(self, df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        mask = (df["Date"] >= pd.Timestamp(start)) & (df["Date"] <= pd.Timestamp(end))
        return df.loc[mask].sort_values(["Ticker", "Date"]).reset_index(drop=True)

    def _drop_missing_target_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops rows only where the label itself is missing (can't impute a target)."""
        before = len(df)
        df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
        dropped = before - len(df)
        logger.info(f"Dropped {dropped} rows with missing target out of {before}")
        return df

    def _inner_train_es_split(self, train_df: pd.DataFrame):
        """
        Per-ticker, time-ordered split of a training fold into a "fit"
        portion and a chronologically-last "early-stopping" (ES)
        portion. Splitting per ticker (rather than by raw row index on
        a Ticker-then-Date-sorted frame) ensures the ES set is truly
        "the most recent es_frac of dates" for every ticker, not just
        whichever tickers happen to sort last alphabetically.

        Tickers with too little history to yield at least one fit row
        and one ES row are skipped entirely and logged, rather than
        silently contributing an empty split.
        """
        fit_frames, es_frames = [], []

        for ticker, group in train_df.groupby("Ticker"):
            group = group.sort_values("Date")

            if len(group) < 2:
                logger.warning(f"""Ticker {ticker}: only {len(group)}
                      row(s) in this fold, skipping ES split.""")
                continue

            cut = max(1, int(len(group) * (1 - self.es_frac)))
            cut = min(cut, len(group) - 1)  # guarantee at least 1 row remains for ES

            fit_frames.append(group.iloc[:cut])
            es_frames.append(group.iloc[cut:])

        if not fit_frames or not es_frames:
            raise TrainingError(
                "ES split produced no usable rows for this fold (insufficient history)."
            )

        fit_df = (
            pd.concat(fit_frames).sort_values(["Ticker", "Date"]).reset_index(drop=True)
        )
        es_df = (
            pd.concat(es_frames).sort_values(["Ticker", "Date"]).reset_index(drop=True)
        )

        return fit_df, es_df

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def build_model(self) -> CatBoostClassifier:
        return CatBoostClassifier(
            random_state=self.random_state,
            verbose=False,
            **self.model_params,
        )

    # ------------------------------------------------------------------
    # Walk-forward validation (5 expanding-window folds)
    # ------------------------------------------------------------------

    def run_walk_forward(self, labeled_df: pd.DataFrame) -> list:
        fold_results = []

        for i, fold in enumerate(self.walk_forward_folds, start=1):
            train_df = self._slice_by_date(
                labeled_df, fold["train_start"], fold["train_end"]
            )
            test_df = self._slice_by_date(
                labeled_df, fold["test_start"], fold["test_end"]
            )

            if train_df.empty or test_df.empty:
                logger.warning(f"Fold {i}: empty train or test window, skipping.")
                continue

            logger.info(
                f"Fold {i}: train {fold['train_start']} -> {fold['train_end']} "
                f"""({len(train_df)} rows), test {fold['test_start']} ->
                  {fold['test_end']} """
                f"({len(test_df)} rows)"
            )

            fit_df, es_df = self._inner_train_es_split(train_df)

            X_fit, y_fit = fit_df[FEATURE_COLS], fit_df[TARGET_COL]
            X_es, y_es = es_df[FEATURE_COLS], es_df[TARGET_COL]
            X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

            imputer = SimpleImputer(strategy=self.impute_strategy)

            X_fit = imputer.fit_transform(X_fit)
            X_es = imputer.transform(X_es)
            X_test = imputer.transform(X_test)

            model = self.build_model()
            model.fit(
                X_fit,
                y_fit,
                eval_set=(X_es, y_es),
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False,
            )

            best_iteration = model.get_best_iteration()
            logger.info(f"Fold {i}: best_iteration={best_iteration}")

            y_pred = model.predict(X_test)

            metrics = {
                "fold": i,
                "train_range": f"{fold['train_start']} -> {fold['train_end']}",
                "test_range": f"{fold['test_start']} -> {fold['test_end']}",
                "best_iteration": best_iteration,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(
                    y_test, y_pred, average="macro", zero_division=0
                ),
                "recall": recall_score(
                    y_test, y_pred, average="macro", zero_division=0
                ),
                "f1_score": f1_score(y_test, y_pred, average="macro", zero_division=0),
            }

            logger.info(
                f"Fold {i} results — accuracy: {metrics['accuracy']:.4f}, "
                f"precision: {metrics['precision']:.4f}, "
                f"recall: {metrics['recall']:.4f}, "
                f"f1: {metrics['f1_score']:.4f}"
            )

            fold_results.append(metrics)

        return fold_results

    def save_fold_metrics(self, fold_results: list) -> str:
        try:
            report_dir = Path(self.fold_metrics_path).parent
            report_dir.mkdir(parents=True, exist_ok=True)

            if not fold_results:
                raise TrainingError("No valid walk-forward folds were produced.")

            lines = ["WALK-FORWARD VALIDATION METRICS", "=" * 40]

            for metrics in fold_results:
                lines.append(f"\nFold {metrics['fold']}")
                lines.append(f"  Train: {metrics['train_range']}")
                lines.append(f"  Test:  {metrics['test_range']}")
                lines.append(
                    f"  Best iteration (early stopping): {metrics['best_iteration']}"
                )
                for key in ["accuracy", "precision", "recall", "f1_score"]:
                    lines.append(f"  {key}: {metrics[key]:.4f}")

            # Average metrics
            lines.append("\n" + "=" * 40)
            lines.append("AVERAGE ACROSS ALL FOLDS")

            for key in ["accuracy", "precision", "recall", "f1_score"]:
                avg = sum(m[key] for m in fold_results) / len(fold_results)
                lines.append(f"{key}: {avg:.4f}")

            Path(self.fold_metrics_path).write_text("\n".join(lines))
            logger.info(f"Walk-forward metrics saved to {self.fold_metrics_path}")
            return self.fold_metrics_path

        except Exception as e:
            logger.error(f"Failed to save fold metrics: {e}")
            raise TrainingError("Failed to persist walk-forward metrics") from e

    # ------------------------------------------------------------------
    # Final production model + forward-only inference
    # ------------------------------------------------------------------

    def train_final_model(self, labeled_df: pd.DataFrame) -> tuple[Pipeline, int]:

        train_df = self._slice_by_date(
            labeled_df,
            self.final_train_range["train_start"],
            self.final_train_range["train_end"],
        )

        if train_df.empty:
            raise TrainingError("""Final training window produced no rows.
                   Check final_train_range in config.""")

        logger.info(
            f"Training final production pipeline on "
            f"{self.final_train_range['train_start']} -> "
            f"{self.final_train_range['train_end']} "
            f"({len(train_df)} rows)"
        )

        # --- Stage 1: find validated iteration count via ES split ---
        fit_df, es_df = self._inner_train_es_split(train_df)

        X_fit, y_fit = fit_df[FEATURE_COLS], fit_df[TARGET_COL]
        X_es, y_es = es_df[FEATURE_COLS], es_df[TARGET_COL]

        es_imputer = SimpleImputer(strategy=self.impute_strategy)
        X_fit_imputed = es_imputer.fit_transform(X_fit)
        X_es_imputed = es_imputer.transform(X_es)

        es_model = self.build_model()
        es_model.fit(
            X_fit_imputed,
            y_fit,
            eval_set=(X_es_imputed, y_es),
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=False,
        )
        best_iteration = es_model.get_best_iteration()

        logger.info(f"Final model: validated best_iteration={best_iteration}")

        # --- Stage 2: retrain on 100% of the window at that iteration count ---
        X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
        imputer = SimpleImputer(strategy=self.impute_strategy)
        X_train_imputed = imputer.fit_transform(X_train)

        final_params = {**self.model_params, "iterations": best_iteration + 1}
        model = CatBoostClassifier(
            random_state=self.random_state,
            verbose=False,
            **final_params,
        )
        model.fit(X_train_imputed, y_train)

        pipeline = Pipeline(
            [
                ("imputer", imputer),
                ("model", model),
            ]
        )

        return pipeline, best_iteration + 1

    # ====================================================
    # Evaluation on Evaluation Data
    # ===================================================

    def evaluate_final_model(
        self,
        pipeline: Pipeline,
        labeled_df: pd.DataFrame,
    ) -> dict:
        """
        After walk-forward validation, a final production model is trained.
        The model is evaluated on a separate evaluation period using
        ground-truth labels and then used for forward-only inference on a
        future inference period.

        """

        test_df = self._slice_by_date(
            labeled_df,
            self.evaluation_range["start"],
            self.evaluation_range["end"],
        )

        if test_df.empty:
            raise TrainingError(
                "Evaluation window produced no rows. "
                "Check evaluation_range in config."
            )

        X_test = test_df[FEATURE_COLS]
        y_test = test_df[TARGET_COL]

        predictions = pipeline.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "f1_score": f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y_test,
                predictions,
                labels=TARGET_LABELS,
            ).tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=TARGET_LABELS,
                target_names=TARGET_NAMES,
                zero_division=0,
            ),
        }

        logger.info(
            f"Evaluating final model on "
            f"{self.evaluation_range['start']} -> "
            f"{self.evaluation_range['end']} "
            f"({len(test_df)} rows)"
        )

        return metrics

    def save_final_metrics(self, metrics: dict) -> str:
        """
        Saves the final model evaluation metrics to a text report.
        """
        try:
            report_dir = Path(self.final_metrics_path).parent
            report_dir.mkdir(parents=True, exist_ok=True)

            lines = [
                "FINAL MODEL EVALUATION",
                "=" * 40,
                "",
                f"Accuracy : {metrics['accuracy']:.4f}",
                f"Precision: {metrics['precision']:.4f}",
                f"Recall   : {metrics['recall']:.4f}",
                f"F1 Score : {metrics['f1_score']:.4f}",
                "",
                "Confusion Matrix",
                "-" * 40,
                str(metrics["confusion_matrix"]),
                "",
                "Classification Report",
                "-" * 40,
                metrics["classification_report"],
            ]

            Path(self.final_metrics_path).write_text("\n".join(lines))

            logger.info(f"Final model metrics saved to {self.final_metrics_path}")

            return self.final_metrics_path

        except Exception as e:
            logger.error(f"Failed to save final model metrics: {e}")
            raise TrainingError("Failed to persist final model metrics") from e

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_pipeline(self, pipeline: Pipeline) -> str:
        try:
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, self.model_path)
            logger.info(
                "Final pipeline (imputer + model)"
                f"saved successfully to {self.model_path}"
            )
            return self.model_path

        except Exception as e:
            logger.error(f"Failed to save pipeline to {self.model_path}: {e}")
            raise TrainingError("Failed to persist trained pipeline") from e

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> str:
        # -----------------------------
        # Load and validate data
        # -----------------------------
        df = self.load_data()
        self._validate_columns(df)

        labeled_df = self._drop_missing_target_rows(df)

        # -----------------------------
        # Walk-forward validation
        # -----------------------------
        fold_results = self.run_walk_forward(labeled_df)
        self.save_fold_metrics(fold_results)

        # -----------------------------
        # Train final production model
        # -----------------------------
        final_pipeline, final_iterations = self.train_final_model(labeled_df)


        # -----------------------------
        # Final evaluation
        # -----------------------------
        final_metrics = self.evaluate_final_model(
            final_pipeline,
            labeled_df,
        )

        self.save_final_metrics(final_metrics)

        # -----------------------------
        # Save trained pipeline
        # -----------------------------
        model_path = self.save_pipeline(final_pipeline)

        # -----------------------------
        # Track final model with MLflow
        # -----------------------------
        tracker = MLflowLogger()

        params_to_log = {
            "impute_strategy": self.impute_strategy,
            "early_stopping_rounds": self.early_stopping_rounds,
            "es_frac": self.es_frac,
            **self.model_params,
        }

        params_to_log.setdefault("random_state", self.random_state)
        params_to_log["final_iterations"] = final_iterations

        with tracker.start_run(run_name="final_production_model"):

            tracker.log_params(params_to_log)

            tracker.log_metrics(
                {
                    "accuracy": final_metrics["accuracy"],
                    "precision": final_metrics["precision"],
                    "recall": final_metrics["recall"],
                    "f1_score": final_metrics["f1_score"],
                }
            )

            tracker.log_artifact(self.fold_metrics_path)
            tracker.log_artifact(self.final_metrics_path)

            tracker.log_model(final_pipeline)

            tracker.log_artifact(model_path)

        logger.info("MLflow tracking completed successfully.")

        return model_path


if __name__ == "__main__":
    try:
        trainer = ModelTrainer()
        path = trainer.run()
        logger.info(f"Final production pipeline saved to: {path}")

    except TrainingError as e:
        logger.critical(f"Training pipeline failed: {e}")
        raise

    except Exception as e:
        logger.critical(f"Unexpected error in training pipeline: {e}")
        raise
