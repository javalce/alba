import re
import uuid
import fitz  # PyMuPDF
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import logging
from config.config import Config
from src.utils import setup_logging

setup_logging()


@dataclass
class Document:
    id: str
    text: str = ""
    metadata: dict = field(default_factory=dict)


class DocumentEngine:
    def _identify_decree_type(self, text: str) -> str:
        if "servicio provincial de extinción de incendios" in text.lower():
            return "SEPEI"
        else:
            return "standard"  # Default to standard if none of the specific markers are found

    def _parse_standard_decree(
        self,
        text: str,
        page_number: int,
        decree_number: str,
        decree_date: str,
        multi_page_decree: bool,
    ) -> Document:
        if multi_page_decree:
            text_start = 0  # Start from the beginning of the text
        else:
            text_start = text.find("OBJETO:")

        header_start = text.find("Decreto Nº")
        text = (
            text[text_start:header_start].strip()
            if header_start != -1
            else text.strip()
        )

        doc = Document(
            id=f"{decree_number}_{page_number}",
            text=text,
            metadata={
                "page": page_number,
                "date": decree_date,
                "type": "standard",
                "number": decree_number,
            },
        )
        return doc

    def _parse_SEPEI_decree(
        self, text: str, page_number: int, decree_number: str, decree_date: str
    ) -> Document:
        doc = Document(
            id=f"{decree_number}_{page_number}",
            text=text.strip(),
            metadata={
                "page": page_number,
                "date": decree_date,
                "type": "SEPEI",
                "number": decree_number,
            },
        )

        return doc

    def _docs_from_decree_files(self, files: List[str]) -> List[Document]:
        documents = []
        decree_number_n_date_pattern = re.compile(
            r"Decreto Nº(\d+) de (\d{2}/\d{2}/\d{4})", re.DOTALL
        )
        previous_decree_number = None  # Track the decree number across pages

        for file_path in files:
            pdf = fitz.open(file_path)
            for page_num in range(len(pdf)):
                raw_text = pdf[page_num].get_text("text")
                text = self._clean_text(raw_text)

                number_n_date = decree_number_n_date_pattern.search(text)
                number = number_n_date.group(1) if number_n_date else None
                date = number_n_date.group(2) if number_n_date else None

                # Determine if this page continues the previous decree
                multi_page_decree = number == previous_decree_number

                decree_type = self._identify_decree_type(text)
                if decree_type == "standard":
                    doc = self._parse_standard_decree(
                        text, page_num + 1, number, date, multi_page_decree
                    )
                elif decree_type == "SEPEI":
                    doc = self._parse_SEPEI_decree(text, page_num + 1, number, date)

                documents.append(doc)
                previous_decree_number = number  # Update the previous decree number

            pdf.close()
        return documents

    def generate_documents(self, files, files_type) -> List[Document]:
        # Convert to list if necessary
        if not isinstance(files, list):
            files = [files]

        documents = []
        if files_type == "decrees":
            # Generate documents from decree files
            documents = self._docs_from_decree_files(files)
        else:
            raise ValueError(f"Unsupported file type: {files_type}")

        return documents

    def _clean_text(self, text: str) -> str:
        # Use a temporary placeholder for "\n\n"
        placeholder = (
            "\ue000"  # Using a Private Use Area Unicode character as a placeholder
        )

        # First, replace "\n \n" with the placeholder
        text = text.replace("\n\n", placeholder)
        text = text.replace("\n \n", placeholder)

        # Then, replace remaining "\n" with a single space
        text = text.replace("\n", " ")

        # Finally, replace the placeholder with "\n\n"
        text = text.replace(placeholder, "\n\n")
        return text

    def _chunk_documents(self, documents):
        chunk_size = Config.get("chunk_size")
        overlap = Config.get("chunk_overlap")

        chunked_documents = []
        for doc in documents:
            text = doc.text
            current_pos = 0
            text_length = len(text)
            logging.info(f"Document ID {doc.id}: Length {text_length}")

            chunk_counter = 1
            while current_pos < text_length:
                if current_pos + chunk_size >= text_length:
                    chunk = text[current_pos:text_length]
                    current_pos = text_length
                else:
                    end_pos = current_pos + chunk_size
                    next_space = text.find(" ", end_pos - overlap)
                    if next_space == -1 or next_space >= end_pos + overlap:
                        next_space = end_pos
                    chunk = text[current_pos:next_space].strip()
                    current_pos = next_space

                if chunk:
                    logging.info(
                        f"Document ID {doc.id}: Chunk {chunk_counter} Length {len(chunk)}"
                    )
                    new_doc = Document(
                        id=f"{doc.id}_{chunk_counter}",
                        text=chunk,
                        metadata={"parent_id": doc.id},
                    )
                    chunked_documents.append(new_doc)
                    chunk_counter += 1

        return chunked_documents
