from sqlalchemy import delete, select
from sqlalchemy_toolkit import SQLAlchemyRepository

from alba import models


class DocumentRepository(SQLAlchemyRepository[models.Document, int]):
    entity_class = models.Document

    def save_all(self, documents: list[models.Document]):
        self.session.add_all(documents)
        self.session.commit()

    def find_by_hash(self, hash_value: str):
        query = select(self.entity_class).where(self.entity_class.hash_value == hash_value)
        return self.session.execute(query).scalar_one_or_none()

    def delete_all(self):
        query = delete(self.entity_class)

        self.session.execute(query)
        self.session.commit()

    def find_all_by_name(self, name: str):
        query = select(self.entity_class).where(self.entity_class.name.ilike(f"%{name}%"))
        return self.session.execute(query).scalars().all()
