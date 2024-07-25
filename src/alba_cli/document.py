from pathlib import Path
from typing import Annotated

from alba import models
from alba.milvus_database import MilvusDatabase
from alba.services import DocumentService
from dependency_injector.wiring import Provide, inject
from sqlalchemy import delete
from sqlalchemy_toolkit import DatabaseManager
from typer import Argument, Typer

app = Typer(rich_markup_mode="rich")


@inject
def __add_document(
    files: list[str],
    document_service: DocumentService = Provide["document_service"],
    milvus_db: MilvusDatabase = Provide["milvus_db"],
):
    files = [file for file in files if not document_service.verify_document(file)]
    document_service.add_documents(files)
    milvus_db.add_documents(files)


@inject
def __purge_documents(
    db: DatabaseManager = Provide["db"],
    milvus_db: MilvusDatabase = Provide["milvus_db"],
):
    with db.session_ctx() as session:
        session.execute(delete(models.Document))
        session.commit()

    milvus_db.initialize()
    milvus_db.clear_database()


@app.command("add")
def add_document(
    files: Annotated[
        list[Path],
        Argument(
            help="The file paths to the document to be added.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ]
):
    """
    Adds a document to the database.
    """
    files = [str(file) for file in files]
    __add_document(files)


@app.command("clear")
def remove_documents():
    """
    Removes all documents from the database.
    """
    __purge_documents()
