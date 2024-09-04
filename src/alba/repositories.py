import uuid

from sqlalchemy import delete, func, select
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
        query = (
            select(
                self.entity_class.id.label("id"),
                self.entity_class.name.label("name"),
                func.count(models.Decree.id).label("total"),
            )
            .where(self.entity_class.name.ilike(f"%{name}%"))
            .join_from(self.entity_class, models.Decree)
            .group_by(self.entity_class.id)
        )

        return self.session.execute(query).mappings().all()

    def find_all_with_number_of_decrees(self):
        query = (
            select(
                self.entity_class.id.label("id"),
                self.entity_class.name.label("name"),
                func.count(models.Decree.id).label("total"),
            )
            .join_from(self.entity_class, models.Decree)
            .group_by(self.entity_class.id)
        )

        return self.session.execute(query).mappings().all()


class DecreeRepository(SQLAlchemyRepository[models.Decree, int]):
    entity_class = models.Decree

    def save_all(self, decrees: list[models.Decree]):
        self.session.add_all(decrees)
        self.session.commit()


class UserRepository(SQLAlchemyRepository[models.User, uuid.UUID]):
    entity_class = models.User

    def find_by_username(self, username: str):
        query = select(models.User).where(models.User.username == username)

        return self.session.execute(query).scalar_one_or_none()
