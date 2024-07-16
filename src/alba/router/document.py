from fastapi import APIRouter, Request, UploadFile

from alba.chatbot import Chatbot

router = APIRouter(prefix="/documents", tags=["document"])


@router.post("")
def add_document(files: list[UploadFile], request: Request):
    chatbot: Chatbot = request.app.state.chatbot

    chatbot.long_term_mem.db.add_document(files)
