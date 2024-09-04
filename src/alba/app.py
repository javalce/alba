from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy_toolkit.ext.fastapi import SQLAlchemyMiddleware

from alba.container import Container
from alba.router import auth_router, chat_router, document_router


def create_app():
    container = Container()

    app = FastAPI(
        title="Alba - Asistente de Búsqueda Local y Privado",
    )
    app.state.container = container

    app.add_middleware(
        SQLAlchemyMiddleware,
        db=container.db(),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_router, prefix="/api")
    app.include_router(document_router, prefix="/api")
    app.include_router(auth_router, prefix="/api")

    return app
