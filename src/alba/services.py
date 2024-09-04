from fastapi import UploadFile

from alba import models, repositories
from alba.exceptions import InvalidPasswordError, NotFoundError
from alba.security import verify_password


class DocumentService:

    def __init__(self, repository: repositories.DocumentRepository):
        self.repository = repository

    def find_all(self):
        return self.repository.find_all()

    def find_all_by_name(self, name: str):
        return self.repository.find_all_by_name(name)

    def find_all_with_number_of_decrees(self):
        return self.repository.find_all_with_number_of_decrees()

    def add_document(self, file: UploadFile):
        document = models.Document.create_document(file)

        return self.repository.save(document)

    def add_documents(self, files: list[UploadFile | str]):
        documents = [models.Document.create_document(file) for file in files]
        self.repository.save_all(documents)

    def verify_document(self, file: UploadFile | str):
        sha256_returned = models.Document.create_document(file).hash_value
        document = self.repository.find_by_hash(sha256_returned)

        return document is not None

    def delete_all(self):
        self.repository.delete_all()


class DecreeService:
    def __init__(self, repository: repositories.DecreeRepository):
        self.repository = repository

    def add_decrees(self, decrees: list[models.Decree]):
        self.repository.save_all(decrees)


class UserService:
    def __init__(self, repository: repositories.UserRepository):
        self.repository = repository

    def login(self, username: str, password: str):
        user = self.repository.find_by_username(username)

        if user is None:
            raise NotFoundError("User not found")

        if not verify_password(password, user.password):
            raise InvalidPasswordError("Invalid password")

        return user
