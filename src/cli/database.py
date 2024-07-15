from typer import Typer

from src.database.database import Database

app = Typer(rich_markup_mode="rich")


@app.command("init")
def create_schema():
    """
    Creates the milvus database schema.
    """
    db = Database()

    db.initialize()
