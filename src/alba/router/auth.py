from typing import Annotated, Any

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Form, HTTPException
from pydantic import BaseModel, ConfigDict

from alba.exceptions import InvalidPasswordError, NotFoundError
from alba.security import (
    create_access_token,
    create_refresh_token,
    jwt_refresh_required,
)
from alba.services import UserService


class LoginResponse(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
    )

    access_token: str
    refresh_token: str
    token_type: str


class RefreshResponse(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
    )

    access_token: str
    token_type: str


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
@inject
def login(
    username: Annotated[str, Form(max_length=255)],
    password: Annotated[str, Form(max_length=255)],
    user_service: Annotated[UserService, Depends(Provide["user_service"])],
) -> Any:
    try:
        user = user_service.login(username, password)

        access_token = create_access_token(user.username)
        refresh_token = create_refresh_token(user.username)

    except NotFoundError as ex:
        raise HTTPException(status_code=404, detail=ex.message) from ex
    except InvalidPasswordError as ex:
        raise HTTPException(status_code=401, detail=ex.message) from ex

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }


@router.get("/refresh", response_model=RefreshResponse)
@inject
def refresh(decoded_token: Annotated[dict[str, Any], Depends(jwt_refresh_required)]) -> Any:
    username = decoded_token["sub"]

    access_token = create_access_token(username)

    return {"access_token": access_token, "token_type": "bearer"}
