import copy
from typing import Annotated

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from alba.milvus_database import MilvusDatabase
from alba.services import DocumentService


class ResponseMessage(BaseModel):
    message: str


router = APIRouter(prefix="/documents", tags=["document"])


@inject
def add_documents_to_db(
    files: list[UploadFile],
    document_service: DocumentService = Provide["document_service"],
    milvus_db: MilvusDatabase = Provide["milvus_db"],
):
    files = [file for file in files if not document_service.verify_document(file)]
    if files:
        # TODO: Send email to the user to notify if the documents are uploaded or there was an error

        document_service.add_documents(files)
        milvus_db.add_documents([(file.file, file.filename) for file in files])


@router.post("")
@inject
async def add_document(
    files: Annotated[list[UploadFile], File(description="List of files to upload")],
    backgound_tasks: BackgroundTasks,
):
    backgound_tasks.add_task(add_documents_to_db, files=copy.deepcopy(files))

    return ResponseMessage(
        message=f"Adding {[file.filename for file in files]} documents. You will be notified once the documents are uploaded"
    )


@router.post("/reset", responses={500: {"description": "Internal server error"}})
@inject
async def reset_documents(db: MilvusDatabase = Depends(Provide["milvus_db"])):
    try:
        db.clear_database()
        db.initialize()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="There was an error while resetting the documents"
        ) from e

    return ResponseMessage(message="Succesfully reset the documents")
