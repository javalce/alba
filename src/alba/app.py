from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from alba.chat import chat_router
from alba.chatbot import Chatbot


def create_app():
    app = FastAPI(
        title="Alba - Asistente de Búsqueda Local y Privado",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.chatbot = Chatbot()

    app.include_router(chat_router, prefix="/api")

    return app
