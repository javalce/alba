import re
import uuid
import fitz  # PyMuPDF
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Document:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str = None
    text: str = ""
    metadata: dict = field(default_factory=dict)


class DocumentEngine:
    def _identify_decree_type(self, text: str) -> str:
        if "SEPEI" in text:
            return "SEPEI"
        else:
            return "standard"  # Default to standard if none of the specific markers are found

    def _parse_standard_decree(
        self, text: str, page_number: int, decree_number: str, decree_date: str
    ) -> Document:
        text_start = text.find("OBJETO:")
        header_start = text.find("Decreto Nº")
        text = (
            text[text_start:header_start].strip()
            if header_start != -1
            else text.strip()
        )  # Default to full text if pattern not found
        return Document(
            parent_id=None,
            text=text,
            metadata={
                "page": page_number,
                "number": decree_number,
                "date": decree_date,
                "type": "standard",
            },
        )

    def _parse_SEPEI_decree(
        self, text: str, page_number: int, decree_number: str, decree_date: str
    ) -> Document:
        # TODO: Implement specific parsing logic for SEPEI decrees here
        return Document(
            parent_id=None,
            text=text.strip(),
            metadata={
                "page": page_number,
                "number": decree_number,
                "date": decree_date,
                "type": "SEPEI",
            },
        )

    def docs_from_decree_files(self, files: List[str]) -> List[Document]:
        documents = []
        decree_number_n_date_pattern = re.compile(
            r"Decreto Nº(\d+) de (\d{2}/\d{2}/\d{4})", re.DOTALL
        )

        for file_path in files:
            pdf = fitz.open(file_path)

            for page_num in range(len(pdf)):
                text = pdf[page_num].get_text("text")

                number_n_date = decree_number_n_date_pattern.search(text)
                number = number_n_date.group(1) if number_n_date else None
                date = number_n_date.group(2) if number_n_date else None

                decree_type = self._identify_decree_type(text)
                if decree_type == "standard":
                    doc = self._parse_standard_decree(text, page_num + 1, number, date)
                elif decree_type == "SEPEI":
                    doc = self._parse_SEPEI_decree(text, page_num + 1, number, date)

                documents.append(doc)

            pdf.close()
        return documents

    def generate_documents(self, files, files_type) -> List[Document]:
        # Convert to list if necessary
        if not isinstance(files, list):
            files = [files]

        documents = []
        if files_type == "decrees":
            # Extract info from each decree
            decrees = self._decrees_from_files(files)

            # All files of all types to be converted to the standard Document format;
            # this will make it easy to ingest other types of documents in the future
            documents = self._docs_from_decrees(decrees)
        else:
            raise ValueError(f"Unsupported file type: {files_type}")

        return documents
