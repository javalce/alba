import hashlib

from fastapi import UploadFile

from alba import models, repositories


class DocumentService:

    def __init__(self, repository: repositories.DocumentRepository):
        self.repository = repository

    def add_document(self, file: UploadFile):
        document = models.Document(name=file.filename, hash_value=self.create_sha256(file))
        self.repository.save(document)

    def add_documents(self, files: list[UploadFile]):
        documents = [
            models.Document(name=file.filename, hash_value=self.create_sha256(file))
            for file in files
        ]
        self.repository.save_all(documents)

    def create_sha256(self, file: UploadFile):
        f = file.file
        data = f.read()

        return hashlib.sha256(data).hexdigest()

    def verify_document(self, file: UploadFile):
        sha256_returned = self.create_sha256(file)
        document = self.repository.find_by_hash(sha256_returned)

        return document is not None
