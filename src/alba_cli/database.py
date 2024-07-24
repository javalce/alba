from pathlib import Path
from typing import Annotated

from alba.milvus_database import MilvusDatabase
from alba.services import DocumentService
from dependency_injector.wiring import Provide, inject
from sqlalchemy_toolkit import DatabaseManager, Entity
from typer import Argument, Typer

app = Typer(rich_markup_mode="rich")


db: DatabaseManager = Provide["db"]
document_service = Provide["document_service"]
milvus_db: MilvusDatabase = Provide["milvus_db"]


@inject
def __init_database(
    db: DatabaseManager = Provide["db"],
    milvus_db: MilvusDatabase = Provide["milvus_db"],
):
    Entity.metadata.create_all(db.engine)
    milvus_db.initialize()


@inject
def __clear_database(
    db: DatabaseManager = Provide["db"],
    milvus_db: MilvusDatabase = Provide["milvus_db"],
):
    Entity.metadata.drop_all(db.engine)
    milvus_db.clear_database()


@inject
def __add_document(
    files: list[str],
    document_service: DocumentService = Provide["document_service"],
    milvus_db: MilvusDatabase = Provide["milvus_db"],
):
    files = [file for file in files if not document_service.verify_document(file)]
    document_service.add_documents(files)
    milvus_db.add_documents(files)


@app.command("init")
def initialize_database():
    """
    Creates the milvus database schema.
    """
    __init_database()


@app.command("drop")
def clear_database():
    """
    Drops the milvus database schema.
    """
    __clear_database()


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


@app.command("reset")
def reset_database():
    """
    Resets the database.
    """
    __clear_database()
    __init_database()
