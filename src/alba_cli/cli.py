import uvicorn
from alba.container import Container
from rich import print
from rich.padding import Padding
from rich.panel import Panel
from typer import Typer

from alba_cli.database import app as db_app

from .logging import setup_logging

setup_logging()

app = Typer(rich_markup_mode="rich")

app.add_typer(db_app, name="db", help="Database management commands.")


def _run(command: str, reload: bool = True):
    host = "0.0.0.0"
    port = 8000

    serving_str = f"[dim]Serving at:[/dim] [link]http://{host}:{port}[/link]\n\n[dim]API docs:[/dim] [link]http://{host}:{port}/docs[/link]"

    if command == "dev":
        panel = Panel(
            f"{serving_str}\n\n[dim]Running in development mode, for production use:[/dim] \n\n[b]alba run[/b]",
            title="Alba CLI - Development mode",
            expand=False,
            padding=(1, 2),
            style="black on yellow",
        )
    else:
        panel = Panel(
            f"{serving_str}\n\n[dim]Running in production mode, for development use:[/dim] \n\n[b]alba dev[/b]",
            title="Alba CLI - Production mode",
            expand=False,
            padding=(1, 2),
            style="green",
        )

    print(Padding(panel, 1))

    uvicorn.run(
        app="alba.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def dev():
    """
    Run the [bold]Alba[/bold] app in [yellow]development[/yellow] mode. 🧪

    This is equivalent to [bold]alba run[/bold] but with [bold]reload[/bold] enabled.
    """
    _run(command="dev")


@app.command()
def run():
    """
    Run the [bold]Alba[/bold] app in [green]production[/green] mode. 🚀

    This is equivalent to [bold]alba dev[/bold] but with [bold]reload[/bold] disabled.
    """
    _run(command="run", reload=False)


def main():
    container = Container()
    container.wire(modules=["alba_cli.database"])

    app()
