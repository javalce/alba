import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import DecodeError, ExpiredSignatureError
from passlib.context import CryptContext

from alba.config import get_config

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

settings = get_config()


def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str):
    return pwd_context.hash(password)


def __generate_token(
    subject: str,
    expires_delta: timedelta,
    token_type: str,
    additional_claims: dict[str, Any] | None = None,
    additional_headers: dict[str, Any] | None = None,
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "iat": now,
        "jti": uuid.uuid4().hex,
        "type": token_type,
        "sub": subject,
        "nbf": now,
        "exp": now + expires_delta,
    }

    if additional_claims is not None:
        payload.update(additional_claims)

    return jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
        headers=additional_headers,
    )


def create_access_token(
    subject: str,
    expires_delta: timedelta | None = None,
    additional_claims: dict[str, Any] | None = None,
    additional_headers: dict[str, Any] | None = None,
) -> str:
    if expires_delta is None:
        if isinstance(settings.JWT_ACCESS_TOKEN_EXPIRE_SECONDS, int):
            expires_delta = timedelta(seconds=settings.JWT_ACCESS_TOKEN_EXPIRE_SECONDS)
        else:
            expires_delta = settings.JWT_ACCESS_TOKEN_EXPIRE_SECONDS
    else:
        expires_delta = expires_delta
    return __generate_token(subject, expires_delta, "access", additional_claims, additional_headers)


def create_refresh_token(
    subject: str,
    expires_delta: timedelta | None = None,
    additional_claims: dict[str, Any] | None = None,
    additional_headers: dict[str, Any] | None = None,
) -> str:
    if expires_delta is None:
        if isinstance(settings.JWT_REFRESH_TOKEN_EXPIRE_SECONDS, int):
            expires_delta = timedelta(seconds=settings.JWT_REFRESH_TOKEN_EXPIRE_SECONDS)
        else:
            expires_delta = settings.JWT_REFRESH_TOKEN_EXPIRE_SECONDS
    else:
        expires_delta = expires_delta
    return __generate_token(
        subject, expires_delta, "refresh", additional_claims, additional_headers
    )


def decode_token(
    token: str,
) -> dict[str, Any]:
    """Decode token"""
    try:
        return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except ExpiredSignatureError as ex:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token time expired: {ex}",
        ) from ex
    except DecodeError as ex:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token not valid: {ex}",
        ) from ex


TokenDependency = Annotated[str, Depends(oauth2_scheme)]


def jwt_required(token: TokenDependency) -> dict[str, Any]:
    payload = decode_token(token)

    if payload["type"] != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Wrong token: 'type' is not 'access'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


def jwt_refresh_required(token: TokenDependency) -> dict[str, Any]:
    payload = decode_token(token)

    if payload["type"] != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Wrong token: 'type' is not 'refresh'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload
