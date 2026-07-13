"""
App configuration. Override via environment variables, e.g.:
    export STOCK_API_BASE_URL="https://my-backend.example.com"
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    api_base_url: str = os.getenv("STOCK_API_BASE_URL", "http://localhost:8000")

    request_timeout: int = int(os.getenv("STOCK_API_TIMEOUT", "30"))
    predict_timeout: int = int(os.getenv("STOCK_API_PREDICT_TIMEOUT", "120"))
    health_timeout: int = int(os.getenv("STOCK_API_HEALTH_TIMEOUT", "45"))

    app_title: str = "Stock Market Prediction Dashboard"
    app_icon: str = "📈"
    page_layout: str = "wide"


settings = Settings()
