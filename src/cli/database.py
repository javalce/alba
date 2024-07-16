from pathlib import Path
from typing import Annotated

from typer import Argument, Typer

from src.database import Database

app = Typer(rich_markup_mode="rich")


@app.command("init")
def initialize_database_schema():
    """
    Creates the milvus database schema.
    """
    db = Database()
    db.initialize()


@app.command("drop")
def clear_database_schema():
    """
    Drops the milvus database schema.
    """
    db = Database()
    db.clear_database()


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

    db = Database()
    db.add_documents(files)
