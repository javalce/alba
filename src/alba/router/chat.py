from typing import Annotated, List

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter
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
@inject
def chat(data: Messages, chatbot: Annotated[Chatbot, Provide["chatbot"]]):
    message = data.messages[-1]

    return StreamingResponse(chatbot.respond_w_sources(message.content))
