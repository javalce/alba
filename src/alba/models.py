from __future__ import annotations

import datetime
import hashlib
import os
import uuid

from fastapi import UploadFile
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

    decrees: Mapped[list[Decree]] = relationship()

    @classmethod
    def generate_sha256(cls, text: bytes):
        return hashlib.sha256(text).hexdigest()

    @classmethod
    def from_filepath(cls, filepath: str):
        with open(filepath, "rb") as f:
            text = f.read()

        return cls(
            name=os.path.basename(filepath),
            hash_value=cls.generate_sha256(text),
        )

    @classmethod
    def from_upload_file(cls, file: UploadFile):
        f = file.file
        text = f.read()
        f.seek(0)

        return cls(
            name=file.filename,
            hash_value=cls.generate_sha256(text),
        )

    @classmethod
    def create_document(cls, file: UploadFile | str):
        if isinstance(file, str):
            return cls.from_filepath(file)

        return cls.from_upload_file(file)


class Decree(Entity):
    __tablename__ = "decrees"
    __table_args__ = (
        {
            "sqlite_autoincrement": True,
        },
    )

    id: Mapped[int] = mapped_column(
        "id", Integer, primary_key=True, nullable=False, autoincrement=True
    )
    number: Mapped[int] = mapped_column("number", Integer, nullable=False)
    date: Mapped[datetime.date] = mapped_column("date", Date, nullable=False)
    document_id: Mapped[uuid.UUID] = mapped_column(
        "document_id", UUID, ForeignKey("documents.id"), nullable=False
    )
