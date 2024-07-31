# Standard library imports
import re
from dataclasses import dataclass, field
from tempfile import SpooledTemporaryFile
from typing import List

# Third-party imports
import fitz  # PyMuPDF
from tqdm import tqdm

# Local application imports
from alba.config import Config
from alba.services import DocumentService
from alba.utils.utils import setup_logging

setup_logging()


@dataclass
class Document:
    """
    A dataclass representing a document.

    Attributes:
        id (str): The unique identifier of the document.
        text (str): The text content of the document.
        metadata (dict): Additional metadata associated with the document.
    """

    id: str
    text: str = ""
    metadata: dict = field(default_factory=dict)


class DocumentEngine:
    """
    A class for processing and generating documents from various file types.
    """

    def __init__(self, config: Config, document_service: DocumentService):
        """
        Initialize the DocumentEngine object with configuration settings.
        """
        self.config = config
        self.document_service = document_service

    def _identify_decree_type(self, text: str) -> str:
        """
        Identify the type of decree based on the text content.

        Args:
            text (str): The text content of the decree.

        Returns:
            str: The type of decree ('SEPEI' or 'standard').
        """
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
        """
        Parse a standard decree and create a Document object.

        Args:
            text (str): The text content of the decree.
            page_number (int): The page number of the decree.
            decree_number (str): The decree number.
            decree_date (str): The decree date.
            multi_page_decree (bool): Indicates if the decree spans multiple pages.

        Returns:
            Document: The parsed standard decree as a Document object.
        """
        if multi_page_decree:
            text_start = 0  # Start from the beginning of the text
        else:
            text_start = text.find("OBJETO:")

        header_start = text.find("Decreto Nº")
        text = text[text_start:header_start].strip() if header_start != -1 else text.strip()

        doc = Document(
            id=f"{decree_number}_{page_number}",
            text=f"Decreto Nº{decree_number}\n{text}",
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
        """
        Parse a SEPEI decree and create a Document object.

        Args:
            text (str): The text content of the decree.
            page_number (int): The page number of the decree.
            decree_number (str): The decree number.
            decree_date (str): The decree date.

        Returns:
            Document: The parsed SEPEI decree as a Document object.
        """
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

    def _docs_from_decree_files(
        self, files: List[str | tuple[SpooledTemporaryFile, str]]
    ) -> List[Document]:
        """
        Generate a list of Document objects from decree files.

        Args:
            files (List[str]): A list of file paths to decree files.

        Returns:
            List[Document]: A list of Document objects generated from the decree files.
        """
        documents = []
        decree_number_n_date_pattern = re.compile(
            r"Decreto Nº(\d+) de (\d{2}/\d{2}/\d{4})", re.DOTALL
        )
        previous_decree_number = None  # Track the decree number across pages

        for file in files:
            if isinstance(file, tuple):
                f, filename = file
                data = f.read()
                f.seek(0)
                pdf = fitz.open(stream=data, filename=filename)
            else:
                pdf = fitz.open(file)
                filename = file

            document = self.document_service.add_document(file)

            for page_num in tqdm(range(len(pdf)), desc=f"Processing {filename}"):
                raw_text = pdf[page_num].get_text("text")
                text = self._clean_text(raw_text)

                number_n_date = decree_number_n_date_pattern.search(text)
                number = number_n_date.group(1) if number_n_date else None
                date = number_n_date.group(2) if number_n_date else None
                multi_page_decree = (
                    number == previous_decree_number
                )  # Determine if this page continues the previous decree

                decree_type = self._identify_decree_type(text)
                if decree_type == "standard":
                    doc = self._parse_standard_decree(
                        text, page_num + 1, number, date, multi_page_decree
                    )
                elif decree_type == "SEPEI":
                    doc = self._parse_SEPEI_decree(text, page_num + 1, number, date)

                documents.append(doc)
                previous_decree_number = number

            pdf.close()
        return documents

    def generate_documents(self, files, files_type) -> List[Document]:
        """
        Generate a list of Document objects from the given files.

        Args:
            files: A single file path or a list of file paths.
            files_type: The type of files being processed.

        Returns:
            List[Document]: A list of Document objects generated from the files.

        Raises:
            ValueError: If the file type is unsupported.
        """
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
        """
        Clean the text by replacing newline characters and multiple spaces.

        Args:
            text (str): The input text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        # Use a temporary placeholder for "\n\n"
        placeholder = "\ue000"  # Using a Private Use Area Unicode character as a placeholder

        # First, replace "\n \n" with the placeholder
        text = text.replace("\n\n", placeholder)
        text = text.replace("\n \n", placeholder)

        # Then, replace remaining "\n" with a single space
        text = text.replace("\n", " ")

        # Finally, replace the placeholder with "\n\n"
        text = text.replace(placeholder, "\n\n")
        return text

    def _chunk_documents(self, documents):
        """
        Split the documents into chunks of a specified size with overlap.

        Args:
            documents: A list of Document objects to be chunked.

        Returns:
            A list of chunked Document objects.
        """
        chunk_size = self.config.CHUNK_SIZE
        overlap = self.config.CHUNK_OVERLAP

        chunked_documents = []
        for doc in documents:
            text = doc.text
            current_pos = 0
            text_length = len(text)

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
                    new_doc = Document(
                        id=f"{doc.id}_{chunk_counter}",
                        text=chunk,
                        metadata={"parent_id": doc.id},
                    )
                    chunked_documents.append(new_doc)
                    chunk_counter += 1

        return chunked_documents
