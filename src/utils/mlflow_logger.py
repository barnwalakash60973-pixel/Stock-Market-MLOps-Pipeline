import mlflow
import mlflow.sklearn


class MLflowLogger:
    """Wrapper around MLflow experiment tracking."""

    def __init__(
        self,
        experiment_name: str = "Stock Market Prediction",
        tracking_uri: str = "sqlite:///mlflow.db",
    ):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str | None = None):
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics)

    def log_metric(self, name: str, value: float):
        mlflow.log_metric(name, value)

    def log_param(self, name: str, value):
        mlflow.log_param(name, value)

    def log_artifact(self, path: str):
        mlflow.log_artifact(path)

    def log_model(self, pipeline):
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="stock_prediction_pipeline",
            skops_trusted_types=[
                "catboost.core.CatBoostClassifier",
                "numpy.dtype",
           ],
        )

    def end_run(self):
        mlflow.end_run()