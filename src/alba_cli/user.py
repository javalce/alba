from typing import Annotated

from dependency_injector.wiring import Provide, inject
from rich import print
from sqlalchemy_toolkit import DatabaseManager
from typer import Option, Typer

from alba import models

app = Typer(rich_markup_mode="rich")


@inject
def __add_user(
    user: models.User,
    db: DatabaseManager = Provide["db"],
):
    with db.session_ctx() as session:
        session.add(user)
        session.commit()


@app.command("add")
def add_user(
    username: Annotated[str, Option(prompt=True)],
    password: Annotated[str, Option(prompt=True, confirmation_prompt=True, hide_input=True)],
):
    print(f"Username: {username}")
    print(f"Password: {password}")

    user = models.User(username=username, password=password)

    __add_user(user)
