import uuid

from sqlalchemy import UUID, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy_toolkit import Entity


class Document(Entity):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        "id", UUID, primary_key=True, nullable=False, default=uuid.uuid4()
    )
    name: Mapped[str] = mapped_column("name", String(255), nullable=False, index=True)
    hash_value: Mapped[str] = mapped_column("hash_value", Text, nullable=False, index=True)
