from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.chatbot import Chatbot

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


class Message(BaseModel):
    role: str
    content: str


class Messages(BaseModel):
    messages: List[Message]


@app.get("/api/chat")
def chat(request: Messages):
    chatbot: Chatbot = app.state.chatbot

    message = request.messages[-1]

    return StreamingResponse(chatbot.respond_w_sources(message.content))
