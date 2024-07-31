import datetime
import uuid

from sqlalchemy import UUID, Date, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy_toolkit import Entity


class Document(Entity):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        "id", UUID, primary_key=True, nullable=False, default=uuid.uuid4()
    )
    name: Mapped[str] = mapped_column("name", String(255), nullable=False, index=True)
    hash_value: Mapped[str] = mapped_column("hash_value", Text, nullable=False, index=True)

    decrees: Mapped[list["Decree"]] = relationship()


class Decree(Entity):
    __tablename__ = "decrees"

    id: Mapped[int] = mapped_column(
        "id", Integer, primary_key=True, nullable=False, autoincrement=True
    )
    number: Mapped[int] = mapped_column("number", Integer, nullable=False)
    date: Mapped[datetime.date] = mapped_column("date", Date, nullable=False)
    document_id: Mapped[uuid.UUID] = mapped_column(
        "document_id", UUID, ForeignKey("documents.id"), nullable=False
    )
