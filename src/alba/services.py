from fastapi import UploadFile

from alba import models, repositories


class DocumentService:

    def __init__(self, repository: repositories.DocumentRepository):
        self.repository = repository

    def find_all(self):
        return self.repository.find_all()

    def find_all_by_name(self, name: str):
        return self.repository.find_all_by_name(name)

    def find_all_with_number_of_decrees(self):
        return self.repository.find_all_with_number_of_decrees()

    def add_documents(self, files: list[UploadFile | str]):
        documents = [models.Document.create_document(file) for file in files]
        self.repository.save_all(documents)

    def verify_document(self, file: UploadFile | str):
        sha256_returned = models.Document.create_document(file).hash_value
        document = self.repository.find_by_hash(sha256_returned)

        return document is not None

    def delete_all(self):
        self.repository.delete_all()
