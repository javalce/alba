from typer import Typer

from config.config import get_config

app = Typer(rich_markup_mode="rich")


@app.command("create")
def create_schema():
    """
    Creates the milvus database schema.
    """
    config = get_config()

    print(config)
