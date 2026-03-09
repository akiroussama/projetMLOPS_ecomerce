import hmac
import os

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


AUTH_TOKEN_ENV_VAR = "API_AUTH_TOKEN"
bearer_scheme = HTTPBearer(auto_error=False)


def require_prediction_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> None:
    expected_token = os.getenv(AUTH_TOKEN_ENV_VAR)
    authorization_header = request.headers.get("Authorization")

    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "AUTH_NOT_CONFIGURED",
                "message": (
                    f"API authentication token is not configured. "
                    f"Set {AUTH_TOKEN_ENV_VAR} on the server."
                ),
            },
        )

    if credentials is None and authorization_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": "INVALID_AUTH_SCHEME",
                "message": "Authorization scheme must be Bearer.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": "AUTHENTICATION_REQUIRED",
                "message": "Bearer token is required to access this route.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": "INVALID_AUTH_SCHEME",
                "message": "Authorization scheme must be Bearer.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not hmac.compare_digest(credentials.credentials, expected_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error_code": "INVALID_TOKEN",
                "message": "Provided API token is invalid.",
            },
        )
