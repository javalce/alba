from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel

from alba.database import Database


class SuccessMessage(BaseModel):
    message: str


router = APIRouter(prefix="/documents", tags=["document"])


@router.post("", responses={500: {"description": "Internal server error"}})
@inject
def add_document(files: list[UploadFile], db: Database = Depends(Provide["db"])):
    try:
        db.add_documents([file.file for file in files])
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="There was an error while uploading the documents"
        ) from e

    return SuccessMessage(message=f"Succesfully added {[file.filename for file in files]}")


@router.post("/reset", responses={500: {"description": "Internal server error"}})
@inject
def reset_documents(db: Database = Depends(Provide["db"])):
    try:
        db.clear_database()
        db.initialize()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="There was an error while resetting the documents"
        ) from e

    return SuccessMessage(message="Succesfully reset the documents")
