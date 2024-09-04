import copy
import uuid
from typing import Annotated, Any

from dependency_injector.wiring import Provide, inject
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Security,
    UploadFile,
)
from pydantic import BaseModel, ConfigDict

from alba.milvus_database import MilvusDatabase
from alba.security import jwt_required
from alba.services import DocumentService


class ResponseMessage(BaseModel):
    message: str


class DocumentResponse(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
    )

    id: uuid.UUID
    name: str
    total: int


router = APIRouter(
    prefix="/documents",
    tags=["document"],
    dependencies=[Security(jwt_required)],
)


@inject
def add_documents_to_db(
    file: UploadFile,
    document_service: DocumentService = Provide["document_service"],
    milvus_db: MilvusDatabase = Provide["milvus_db"],
):
    if not document_service.verify_document(file):
        # TODO: Send email to the user to notify if the documents are uploaded or there was an error

        milvus_db.add_documents(file)


@router.get("", response_model=list[DocumentResponse])
@inject
def get_documents(
    name: str | None = None,
    document_service: DocumentService = Depends(Provide["document_service"]),
) -> Any:
    if name:
        return document_service.find_all_by_name(name)

    return document_service.find_all_with_number_of_decrees()


@router.post("")
@inject
def add_document(
    file: Annotated[UploadFile, File(description="List of files to upload")],
    backgound_tasks: BackgroundTasks,
):
    backgound_tasks.add_task(add_documents_to_db, file=copy.deepcopy(file))

    return ResponseMessage(
        message=f"Adding {file.filename} documents. You will be notified once the documents are uploaded"
    )


@router.post("/reset", responses={500: {"description": "Internal server error"}})
@inject
def reset_documents(
    document_service: DocumentService = Depends(Provide["document_service"]),
    milvus_db: MilvusDatabase = Depends(Provide["milvus_db"]),
):
    try:
        document_service.delete_all()
        milvus_db.clear_database()
        milvus_db.initialize()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="There was an error while resetting the documents"
        ) from e

    return ResponseMessage(message="Succesfully reset the documents")
