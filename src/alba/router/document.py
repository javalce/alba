from typing import Annotated

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from alba.database import Database


class ResponseMessage(BaseModel):
    message: str


router = APIRouter(prefix="/documents", tags=["document"])


async def add_documents_to_db(
    files: list[UploadFile],
    db: Database = Depends(Provide["db"]),
):
    # TODO: Send email to the user to notify if the documents are uploaded or there was an error
    db.add_documents([file.file for file in files])


@router.post("", responses={500: {"description": "Internal server error"}})
@inject
async def add_document(
    files: Annotated[list[UploadFile], File(description="List of files to upload")],
    backgound_tasks: BackgroundTasks,
):

    backgound_tasks.add_task(add_documents_to_db, files)

    return ResponseMessage(message=f"Adding {[file.filename for file in files]} documents")


@router.post("/reset", responses={500: {"description": "Internal server error"}})
@inject
async def reset_documents(db: Database = Depends(Provide["db"])):
    try:
        db.clear_database()
        db.initialize()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="There was an error while resetting the documents"
        ) from e

    return ResponseMessage(message="Succesfully reset the documents")
