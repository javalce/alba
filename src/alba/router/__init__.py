from alba.router.auth import router as auth_router
from alba.router.chat import router as chat_router
from alba.router.document import router as document_router

__all__ = [
    "auth_router",
    "document_router",
    "chat_router",
]
