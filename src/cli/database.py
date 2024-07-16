from typer import Typer

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
