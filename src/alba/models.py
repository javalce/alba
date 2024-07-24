from typing import Optional

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy_toolkit import Entity


class Document(Entity):
    __tablename__ = "documents"

    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    hash_value: Mapped[str] = mapped_column(String(), nullable=False)
