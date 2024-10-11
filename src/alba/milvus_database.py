import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import nltk
from fastapi import UploadFile
from milvus_model.hybrid import BGEM3EmbeddingFunction
from milvus_model.sparse import BM25EmbeddingFunction
from milvus_model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus import Collection, DataType, FieldSchema, MilvusClient, connections
from pymilvus.orm.schema import CollectionSchema
from tqdm import tqdm

from alba.config import Config
from alba.document_engine import Document, DocumentEngine
from alba.utils.ner_extraction import EntityExtractor
from alba.utils.utils import setup_logging

# Setup logging
setup_logging()

DOCS_COLLECTION_NAME = "docs"
CHUNKS_COLLECTION_NAME = "chunks"


class MilvusDatabase:
    def __init__(
        self,
        config: Config,
        document_engine: DocumentEngine,
        ner_extractor: EntityExtractor,
    ):
        self.config = config
        connections.connect(uri=self.config.MILVUS_URI)
        self.__client = MilvusClient(uri=self.config.MILVUS_URI)
        self.__doc_engine = document_engine

        self.ner_extractor = ner_extractor

        self.__dense_ef = self.__load_dense_embedding()
        self.__sparse_ef = self.__load_sparse_embedding()

        self.load_collections()

    def load_collections(self):
        if self.__client.has_collection(DOCS_COLLECTION_NAME):
            self.__client.load_collection(DOCS_COLLECTION_NAME)
            self.docs = Collection(DOCS_COLLECTION_NAME)

        if self.__client.has_collection(CHUNKS_COLLECTION_NAME):
            self.__client.load_collection(CHUNKS_COLLECTION_NAME)
            self.chunks = Collection(CHUNKS_COLLECTION_NAME)

    def __load_sparse_embedding(self, corpus=None):
        # More info here: https://milvus.io/docs/embed-with-bm25.md
        # If a bm25 model already exists, load it. Otherwise, create a new one.
        model_path = self.config.SPARSE_EMBED_FUNC_PATH
        if os.path.exists(model_path):
            # Load the existing model
            with open(model_path, "rb") as file:
                bm25_ef = pickle.load(file)
        else:
            # Check if the 'stopwords' dataset is not already loaded in NLTK
            if "stopwords" not in nltk.corpus.util.lazy_imports:
                # Download the 'stopwords' dataset using NLTK's download utility
                nltk.download("stopwords")

            # Create an analyzer for processing documents, here specifying Spanish language
            analyzer = build_default_analyzer(language="sp")
            # Initialize a BM25 embedding function with the previously created analyzer
            bm25_ef = BM25EmbeddingFunction(analyzer)

            # Check if a corpus is provided to fit the model
            if corpus:
                # Fit the BM25 embedding function to the provided corpus
                bm25_ef.fit(corpus)
                # Serialize the BM25 model into a file for persistence
                with open(model_path, "wb") as file:
                    pickle.dump(bm25_ef, file)  # Use Python's pickle module for serialization

        return bm25_ef

    def __load_dense_embedding(self):
        # Más información aquí: https://milvus.io/docs/embed-with-bgm-m3.md
        bgeM3_ef = BGEM3EmbeddingFunction(
            model_name="BAAI/bge-m3",  # Especifica el nombre del modelo
            device="cpu",
            use_fp16=False,  # Usa FP16 si está en CUDA, de lo contrario False
            return_colbert_vecs=False,  # No se necesitan vectores de salida de COLBERT
            return_dense=True,  # Vectores densos para búsqueda semántica
            return_sparse=False,  # Los dispersos los tomaremos de bm25
        )
        return bgeM3_ef

    def __get_docs_schema(self) -> CollectionSchema:
        """
        Returns a schema for the long-term memory database of documents.
        """
        schema = CollectionSchema(
            [
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    max_length=32,
                    is_primary=True,
                    auto_id=False,
                ),
                FieldSchema(name="page", dtype=DataType.INT64),
                FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="number", dtype=DataType.INT64),
                FieldSchema(
                    name="text",
                    dtype=DataType.VARCHAR,
                    max_length=self.config.MAX_DOC_SIZE,
                ),
                # Add a docs vector; Milvus requires at least one vector field in the schema
                # This is not used, just a workaround to satisfy the schema requirements
                FieldSchema(name="docs_vector", dtype=DataType.FLOAT_VECTOR, dim=2),
            ],
            description="Collection for storing text and metadata of each document",
            enable_auto_id=False,
        )

        return schema

    def __get_chunks_schema(self) -> CollectionSchema:
        """
        Returns a schema for the long-term memory database of chunks.
        """
        schema = CollectionSchema(
            [
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    max_length=32,
                    is_primary=True,
                    auto_id=False,
                ),
                FieldSchema(
                    name="dense_vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.__dense_ef.dim["dense"],
                ),
                FieldSchema(
                    name="sparse_vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.__sparse_ef.dim,
                ),
                FieldSchema(
                    name="parent_id",
                    dtype=DataType.VARCHAR,
                    max_length=32,
                ),
                FieldSchema(
                    name="text",
                    dtype=DataType.VARCHAR,
                    max_length=self.config.CHUNK_TEXT_SIZE,
                ),
                FieldSchema(name="entities", dtype=DataType.JSON),
            ],
            description="Collection for storing chunk embeddings",
            enable_auto_id=False,
        )

        return schema

    def __create_docs_schema(self) -> CollectionSchema:
        schema = self.__get_docs_schema()
        self.__client.create_collection(collection_name=DOCS_COLLECTION_NAME, schema=schema)

        # Adjusted index_params structure to be a list containing one dictionary
        index_params = [
            {
                "field_name": "docs_vector",
                "params": {
                    "metric_type": "L2",
                    # Assuming you want to use an IVF_FLAT index as an example
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128},  # Example specific parameter for IVF_FLAT
                },
            }
        ]

        self.__client.create_index(
            collection_name=DOCS_COLLECTION_NAME,
            index_params=index_params,  # Now passing a list of dictionaries
        )

        logging.info("Created documents schema and index.")

    def __create_chunks_schema(self) -> CollectionSchema:
        schema = self.__get_chunks_schema()
        self.__client.create_collection(collection_name=CHUNKS_COLLECTION_NAME, schema=schema)

        # Define index parameters for both fields as a list of dictionaries
        index_params = [
            {
                "field_name": "dense_vector",
                "params": {
                    "metric_type": "IP",  # Inner Product, assuming normalized vectors for cosine similarity
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128},
                },
            },
            {
                "field_name": "sparse_vector",
                "params": {
                    "metric_type": "IP",  # Inner Product
                    # Assuming default index_type and other parameters if needed
                },
            },
        ]

        # Create indexes on the specified fields
        self.__client.create_index(
            collection_name=CHUNKS_COLLECTION_NAME,
            index_params=index_params,  # Pass the list of dictionaries as index_params
        )

        logging.info("Created chunks schema and indexes.")

    def initialize(self):
        self.__create_docs_schema()
        self.__create_chunks_schema()

    def clear_database(self) -> None:
        """
        Drops all collections in the database.
        """
        collections = self.__client.list_collections()
        for collection_name in collections:
            self.__client.drop_collection(collection_name)
            logging.info(f"Deleted documents from collection: {collection_name}")

    def __insert_chunk_records(self, chunk_records):
        """
        Batch load chunk records, including embeddings, into the chunks collection.
        """
        self.chunks.insert(chunk_records)
        logging.info(f"Inserted {len(chunk_records)} chunk records.")

    def add_documents(
        self, files: str | UploadFile | List[str | UploadFile], type: str = "decrees"
    ) -> None:
        logging.info(f"Adding documents of type {type} to the database.")
        # Generate documents, format them into database records, and insert them
        documents = self.__doc_engine.generate_documents(files, type)
        doc_records = self.__generate_doc_records(documents)
        self.__client.insert(collection_name=DOCS_COLLECTION_NAME, data=doc_records)

        # Generate chunks, format them into database records, and insert them
        logging.info("Processing and inserting chunk records.")
        self.__process_and_insert_chunks(documents)

    def __process_and_insert_chunks(self, documents, batch_size: int = 10, start_from: int = 0):
        """
        Process documents in batches, create chunk records, insert them into the database,
        and log the process with a progress bar. Updated to include a starting index.

        Args:
            documents (List[Document]): List of Document objects to process.
            batch_size (int): Number of documents to process in each batch.
            start_from (int): Index to start processing from.
        """

        # Adjust total batches based on the new start_from parameter
        adjusted_docs = documents[start_from:]
        num_batches = len(adjusted_docs) // batch_size + (
            1 if len(adjusted_docs) % batch_size > 0 else 0
        )

        # Process documents in batches starting from start_from
        for i in tqdm(range(num_batches), desc="Processing and inserting document chunks"):
            # Adjust batch slice indices based on start_from
            start_idx = start_from + i * batch_size
            end_idx = min(start_from + (i + 1) * batch_size, len(documents))

            # Select batch documents
            batch_documents = documents[start_idx:end_idx]

            try:
                # Generate and insert chunk records for the current batch
                self.__process_and_insert_chunk_batch(batch_documents)
            except Exception as e:
                # Log the next start_from value before raising the error
                next_start_from = start_idx + batch_size  # Or end_idx for more precision
                logging.error(
                    f"Error processing documents. Next start_from should be: {next_start_from}. Error: {e}"
                )
                raise  # Re-raise the error to stop the process

        logging.info("Completed processing and inserting chunks.")

    def __process_and_insert_chunk_batch(self, batch_documents: List[Document]):
        """
        Generate chunk records for a batch of documents and insert them into the database.

        Args:
            batch_documents (List[Document]): List of Document objects to process.
        """

        # Generate chunks for the documents
        chunks = self.__doc_engine._chunk_documents(batch_documents)

        # Generate chunk records, including both dense and sparse embeddings
        chunk_records = self.__generate_chunk_records(chunks)

        # Insert chunk records into the database
        self.__insert_chunk_records(chunk_records)

    def __generate_doc_records(self, documents: List[Document]) -> List[Dict[str, Any]]:
        records = []
        for doc in documents:
            record = {
                "id": doc.id,
                "page": doc.metadata.get("page", 0),
                "date": doc.metadata.get("date", ""),
                "type": doc.metadata.get("type", ""),
                "number": (
                    int(doc.metadata.get("number", 0))
                    if doc.metadata.get("number") is not None
                    else 0
                ),  # Ensure number is not None
                "text": doc.text[: self.config.DOC_SIZE],  # Truncate text if necessary
                "docs_vector": [0.0, 0.0],  # Dummy vector for the docs_vector field
            }
            records.append(record)

        return records

    def __generate_chunk_records(self, chunks: List[Document]) -> List[Dict[str, Any]]:
        """
        Prepare the records for inserting into the _chunks collection,
        including both dense and sparse embeddings, formatted as dictionaries.
        """
        records = []

        # Extract chunk texts to generate embeddings
        chunk_texts = [chunk.text for chunk in chunks]
        # Generate dense embeddings using the BGE-M3 function
        dense_embeddings = self.__dense_ef.encode_documents(chunk_texts)["dense"]

        # Generate sparse embeddings using the BM25 function
        raw_sparse_embeddings = self.__sparse_ef.encode_documents(chunk_texts)

        # Convert sparse embeddings from csr_array format to a list for insertion
        sparse_embeddings = [
            sparse_embedding.toarray().tolist()  # Ensure it's a list of lists, not a list of arrays
            for sparse_embedding in raw_sparse_embeddings
        ]

        # Prepare records with both embeddings for each chunk
        for i, chunk in tqdm(enumerate(chunks), desc="Generating chunk records"):
            entities = self.ner_extractor.extract_entities(chunk.text)
            entities_list = [{"type": ent[0], "value": ent[1]} for ent in entities]
            # Extract the document ID from the chunk ID
            parent_document_id = chunk.id.split("_")[0]

            # Check if LAW entity already exists
            law_entity_exists = any(
                ent
                for ent in entities_list
                if ent["type"] == "LAW" and ent["value"] == parent_document_id
            )

            # Add the LAW entity with the parent document ID if it doesn't already exist
            if not law_entity_exists:
                entities_list.append({"type": "LAW", "value": parent_document_id})

            record = {
                "id": chunk.id,
                "dense_vector": dense_embeddings[i].tolist(),
                "sparse_vector": sparse_embeddings[i],
                "parent_id": chunk.metadata.get("parent_id", 0),
                "text": chunk.text[: self.config.CHUNK_SIZE],
                "entities": entities_list,
            }
            records.append(record)

        return records

    def get_document_by_id(self, data_id: str) -> Optional[Document]:
        """
        Gets a document from the Milvus 'docs' collection by its ID.

        Args:
            data_id (str): Unique ID associated with the document.

        Returns:
            Optional[Document]: Document associated with the given ID, or None if not found.
        """
        try:
            res = self.__client.get(
                collection_name=DOCS_COLLECTION_NAME,
                ids=[data_id],  # Milvus expects a list of IDs
            )

            if not res:
                logging.info(f"Document with ID {data_id} not found")
                return None

            fields = res[0]  # Extract the fields of the result
            # Create a new metadata dictionary that includes everything except 'id' and 'text'
            metadata = {
                key: value
                for key, value in fields.items()
                if key not in ["id", "text", "docs_vector"]
            }

            document = Document(
                id=fields["id"],
                text=fields.get("text", ""),
                metadata=metadata,  # Pass the new metadata dictionary
            )

            return document

        except Exception as e:
            logging.error(f"An error occurred while retrieving document ID {data_id}: {e}")
            return None

    def encode_dense_query(self, queries: List[str]):
        return self.__dense_ef.encode_queries(queries)

    def encode_sparse_query(self, queries: List[str]):
        return self.__sparse_ef.encode_queries(queries)
