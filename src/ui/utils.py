"""API helper utilities for the Streamlit UI."""

import os
from typing import Any, Dict, Optional

import requests


def get_api_url() -> str:
    """Return the base API URL from environment or default."""
    return os.getenv("API_URL", "http://localhost:8000").rstrip("/")


def get_auth_token() -> str:
    """Return the Bearer token from environment or default."""
    return os.getenv("API_AUTH_TOKEN", "change-me")


def call_api(
    method: str,
    endpoint: str,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Call the FastAPI backend.

    Parameters
    ----------
    method : str
        HTTP method ("GET" or "POST").
    endpoint : str
        API path, e.g. "/health" or "/predict".
    json_body : dict, optional
        JSON payload for POST requests.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    dict
        Parsed JSON response on success.

    Raises
    ------
    requests.ConnectionError
        When the API is unreachable.
    requests.HTTPError
        When the API returns a non-2xx status.
    """
    url = f"{get_api_url()}{endpoint}"
    headers = {
        "Authorization": f"Bearer {get_auth_token()}",
        "Content-Type": "application/json",
    }

    response = requests.request(
        method=method.upper(),
        url=url,
        headers=headers,
        json=json_body,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def check_api_health() -> Dict[str, Any]:
    """Fetch /health from the API. Returns the parsed response dict."""
    return call_api("GET", "/health")


def predict(designation: str, description: str = "") -> Dict[str, Any]:
    """Send a prediction request to POST /predict."""
    payload = {"designation": designation, "description": description}
    return call_api("POST", "/predict", json_body=payload)
