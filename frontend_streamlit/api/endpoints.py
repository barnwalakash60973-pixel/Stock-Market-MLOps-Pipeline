"""Backend endpoint paths — change here, not in client.py, if routes move."""

HEALTH = "/health"
PREDICT_UPLOAD = "/predict/upload"


def build_url(base_url: str, path: str) -> str:
    """Join a base URL and a path without double/missing slashes."""
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"
