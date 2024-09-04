from pathlib import Path
from typing import Annotated

from dependency_injector.wiring import Provide, inject
from sqlalchemy import delete
from sqlalchemy_toolkit import DatabaseManager
from typer import Argument, Typer

from alba import models
from alba.milvus_database import MilvusDatabase
from alba.services import DocumentService

app = Typer(rich_markup_mode="rich")


@inject
def __add_document(
    file: str,
    db: DatabaseManager = Provide["db"],
    document_service: DocumentService = Provide["document_service"],
    milvus_db: MilvusDatabase = Provide["milvus_db"],
):
    with db.session_ctx():
        if not document_service.verify_document(file):
            milvus_db.add_documents(file)


@inject
def __purge_documents(
    db: DatabaseManager = Provide["db"],
    milvus_db: MilvusDatabase = Provide["milvus_db"],
):
    with db.session_ctx() as session:
        session.execute(delete(models.Document))
        session.execute(delete(models.Decree))
        session.commit()

    milvus_db.initialize()
    milvus_db.clear_database()


@app.command("add")
def add_document(
    file: Annotated[
        Path,
        Argument(
            help="The file path to the document to be added.",
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
    file = str(file)
    __add_document(file)


@app.command("clean")
def remove_documents():
    """
    Removes all documents from the database.
    """
    __purge_documents()
