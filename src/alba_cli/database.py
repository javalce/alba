from dependency_injector.wiring import Provide, inject
from sqlalchemy_toolkit import DatabaseManager, Entity
from typer import Typer

from alba.milvus_database import MilvusDatabase

app = Typer(rich_markup_mode="rich")


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


@app.command("init")
def initialize_database():
    """
    Creates the database schema.
    """
    __init_database()


@app.command("drop")
def clear_database():
    """
    Drops the database schema. [red]This will delete all data in the database.[/ red]
    """
    __clear_database()


@app.command("reset")
def reset_database():
    """
    Drops and recreates the database schema. [red]This will delete all data in the database.[/ red]
    """
    __clear_database()
    __init_database()
