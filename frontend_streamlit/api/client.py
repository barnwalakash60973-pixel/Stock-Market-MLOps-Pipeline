"""
Thin HTTP client for the FastAPI backend.

This is the only module allowed to import `requests`. Everything else in
the app (streamlit_app.py, utils.py) calls into APIClient — never HTTP
directly — so the frontend stays a pure presentation layer with zero
prediction logic.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests
from api.endpoints import HEALTH, PREDICT_UPLOAD, build_url

from config import settings

logger = logging.getLogger("frontend.api")


class APIError(Exception):
    """Base class for client-side API errors."""


class APIConnectionError(APIError):
    """Backend unreachable."""


class APITimeoutError(APIError):
    """Request exceeded its timeout."""


class APIResponseError(APIError):
    """Backend returned a non-2xx response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API error {status_code}: {detail}")


@dataclass
class HealthStatus:
    is_healthy: bool
    status_text: str
    response_time_ms: float
    checked_at: float = field(default_factory=time.time)
    error: str | None = None


@dataclass
class PredictionResponse:
    message: str
    rows: int
    predictions: list[dict[str, Any]]
    response_time_ms: float


class APIClient:
    """Wraps every REST call to the FastAPI backend."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or settings.api_base_url).strip().rstrip("/")
        self._session = requests.Session()

    def check_health(self) -> HealthStatus:
        url = build_url(self.base_url, HEALTH)
        start = time.perf_counter()

        print("=" * 60)
        print("Health URL:", url)

        try:
            resp = self._session.get(url, timeout=settings.health_timeout)

            elapsed_ms = (time.perf_counter() - start) * 1000

            print("Status Code:", resp.status_code)
            print("Response:", resp.text)

            if resp.status_code == 200:
                return HealthStatus(
                True,
                resp.json().get("status", "healthy"),
                elapsed_ms,
            )

            return HealthStatus(
            False,
            "unhealthy",
            elapsed_ms,
            error=f"HTTP {resp.status_code}: {resp.text}",
        )

        except Exception as e:
            print("Exception:", repr(e))
            return HealthStatus(
            False,
            "error",
            0.0,
            error=repr(e),
        )

    def predict_csv(self, filename: str, file_bytes: bytes) -> PredictionResponse:
        """POST a CSV to /predict/upload and return parsed predictions.

        Raises:
            APIConnectionError, APITimeoutError, APIResponseError
        """
        url = build_url(self.base_url, PREDICT_UPLOAD)
        files = {"file": (filename, file_bytes, "text/csv")}
        start = time.perf_counter()
        try:
            resp = self._session.post(
                url,
                files=files,
                timeout=settings.predict_timeout,
            )

        except requests.exceptions.ConnectionError as e:
            logger.error("Prediction API connection failed: %s", e)
            raise APIConnectionError(
                f"Could not reach the API at {self.base_url}."
            ) from e

        except requests.exceptions.Timeout as e:
            logger.error("Prediction request timed out: %s", e)
            raise APITimeoutError(
                "The prediction request timed out. Please try again."
            ) from e

        except requests.exceptions.RequestException as e:
            logger.exception("Prediction request failed")
            raise APIError(str(e)) from e

        elapsed_ms = (time.perf_counter() - start) * 1000

        if resp.status_code != 200:
            raise APIResponseError(resp.status_code, self._extract_detail(resp))

        payload = resp.json()
        return PredictionResponse(
            message=payload.get("message", ""),
            rows=payload.get("rows", 0),
            predictions=payload.get("predictions", []),
            response_time_ms=elapsed_ms,
        )

    @staticmethod
    def _extract_detail(resp: requests.Response) -> str:
        try:
            return str(resp.json().get("detail", resp.text))
        except ValueError:
            return resp.text or f"HTTP {resp.status_code}"
