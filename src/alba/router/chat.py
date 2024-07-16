from typing import List

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from alba.chatbot import Chatbot

router = APIRouter(prefix="/chat", tags=["chat"])


class Message(BaseModel):
    role: str
    content: str


class Messages(BaseModel):
    messages: List[Message]


@router.post("")
def chat(data: Messages, request: Request):
    chatbot: Chatbot = request.app.state.chatbot

    message = data.messages[-1]

    return StreamingResponse(chatbot.respond_w_sources(message.content))
