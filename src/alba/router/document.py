from fastapi import APIRouter, Request, UploadFile

from alba.chatbot import Chatbot

router = APIRouter(prefix="/documents", tags=["document"])


@router.post("")
def add_document(files: list[UploadFile], request: Request):
    chatbot: Chatbot = request.app.state.chatbot

    try:
        chatbot.long_term_mem.db.add_documents(files)
    except Exception:
        return {"message": "There was an error while uploading the documents"}

    return {"message": f"Succesfully added {[file.filename for file in files]}"}
