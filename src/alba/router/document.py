from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, UploadFile
from typing_extensions import Annotated

from alba.database import Database

router = APIRouter(prefix="/documents", tags=["document"])


@router.post("")
@inject
def add_document(files: list[UploadFile], db: Annotated[Database, Depends(Provide["db"])]):
    try:
        db.add_documents(files)
    except Exception:
        return {"message": "There was an error while uploading the documents"}

    return {"message": f"Succesfully added {[file.filename for file in files]}"}
