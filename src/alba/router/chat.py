from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from alba.chatbot import Chatbot

router = APIRouter(prefix="/chat", tags=["chat"])


class Message(BaseModel):
    role: str
    content: str


class Messages(BaseModel):
    messages: list[Message]


@router.post("")
@inject
def chat(data: Messages, chatbot: Chatbot = Depends(Provide["chatbot"])):
    messages = data.messages

    if len(messages) == 1:
        response_generator = chatbot.respond_w_sources(messages[-1].content)
    else:
        response_generator = chatbot.respond_w_messages([m.model_dump() for m in messages])

    return StreamingResponse(response_generator)
