from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from typing_extensions import Annotated

from alba.database import Database

router = APIRouter(prefix="/documents", tags=["document"])


@router.post("")
@inject
def add_document(files: list[UploadFile], db: Annotated[Database, Depends(Provide["db"])]):
    try:
        db.add_documents(files)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="There was an error while uploading the documents"
        ) from e

    return {"message": f"Succesfully added {[file.filename for file in files]}"}


@router.post("/reset")
@inject
def reset_documents(db: Annotated[Database, Depends(Provide["db"])]):
    try:
        db.clear_database()
        db.initialize()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="There was an error while resetting the documents"
        ) from e

    return {"message": "Documents resetted"}
