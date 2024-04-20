import os
import torch
import shutil
from tqdm import tqdm
import fitz
import pickle
import nltk
import uuid
import json
from pathlib import Path
import logging
from pathlib import Path
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.config import Config
from src.document_engine import Document, DocumentEngine
from pymilvus import connections
from pymilvus import MilvusClient, DataType, Collection
from pymilvus.client.abstract import AnnSearchRequest, WeightedRanker, SearchResult
from milvus_model.hybrid import BGEM3EmbeddingFunction
from milvus_model.sparse import BM25EmbeddingFunction
from milvus_model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus.orm.schema import CollectionSchema, FieldSchema
from src.utils import setup_logging

setup_logging()


class LongTermMemory:
    def __init__(self):
        self.run_mode = Config.get("run_mode")

        # Define the weights for the hybrid search
        self.DENSE_SEARCH_WEIGHT = 0.25
        self.SPARSE_SEARCH_WEIGHT = 0.75

        # Establish a connection to the Milvus server
        self._client = self._connect()

        self._doc_engine = DocumentEngine()
        self._dense_ef = self._load_embedding_function("dense")
        self._sparse_ef = self._load_embedding_function("sparse")

        self._docs, self._chunks = None, None
        self._load_collections()

    def _connect(self):
        host = Config.get("MILVUS_HOST")
        port = Config.get("MILVUS_PORT")
        connections.connect(host=host, port=port)
        client = MilvusClient(host=host, port=port)
        return client

    def _load_embedding_function(self, ef_type, corpus=None):
        if ef_type == "sparse":
            # More info here: https://milvus.io/docs/embed-with-bm25.md
            # If a bm25 model already exists, load it. Otherwise, create a new one.
            model_path = Config.get("sparse_embed_func_path")
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
                        pickle.dump(
                            bm25_ef, file
                        )  # Use Python's pickle module for serialization

            return bm25_ef
        elif ef_type == "dense":
            # Determina el dispositivo en función de la disponibilidad de CUDA
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

            # Configura la función de embedding para usar GPU si está disponible, y FP16 si es apropiado
            use_fp16 = True if device.startswith("cuda") else False

            # Más información aquí: https://milvus.io/docs/embed-with-bgm-m3.md
            bgeM3_ef = BGEM3EmbeddingFunction(
                model_name="BAAI/bge-m3",  # Especifica el nombre del modelo
                device=device,  # Usa el dispositivo determinado (CPU o GPU)
                use_fp16=use_fp16,  # Usa FP16 si está en CUDA, de lo contrario False
            )
            return bgeM3_ef
        else:
            raise ValueError(f"Unsupported embedding function type: {ef_type}")

    def _load_collections(self):
        # RES_LOAD (reset and load): delete all documents and load new ones
        # RES_LOAD_FILES (reset database and load files): load chunks from files
        # NO_RES_NO_LOAD (no reset, no load): do not delete or load documents
        if self.run_mode == "RES_LOAD":
            self.delete_documents("all")

        # Load collections for all run modes
        self._docs = self._load_docs_collection()
        self._chunks = self._load_chunks_collection()

        # Load documents and chunks from the raw data folder
        self._ingest_folder()

    def _ingest_folder(self):
        # Define paths using configuration settings
        raw_folder = Path(Config.get("raw_data_folder"))
        processed_folder = Path(Config.get("processed_data_folder"))
        staged_folder = Path(Config.get("stage_data_folder"))
        staged_folder.mkdir(exist_ok=True)  # Ensure the staging folder exists

        if self.run_mode == "RES_LOAD":

            # Read and generate documents from PDF files in the raw data folder
            files = [str(file) for file in raw_folder.glob("*.pdf")]
            documents = self._doc_engine.generate_documents(files, "decrees")

            # Insert the generated document records into the database
            doc_records = self._generate_doc_records(documents)
            self._client.insert(collection_name="docs", data=doc_records)
            logging.info(f"Finished inserting {len(doc_records)} document records.")

            # Generate chunk records from the documents
            chunks = self._doc_engine._chunk_documents(documents)
            chunk_records = self._generate_chunk_records(chunks)

            # When loading a full folder, save chunk records to JSON files in the staging folder
            # to avoid memory issues when inserting a large number of records;
            # this also allows for partial recovery from errors
            record_buffer = []
            file_count = 0  # To keep track of file naming

            # Batch and save chunk records to JSON files in the staging folder
            for i, record in enumerate(chunk_records):
                record_buffer.append(record)
                # Write to file either when the buffer size reaches 100 or at the end of the list
                if len(record_buffer) >= 100 or i == len(chunk_records) - 1:
                    file_path = staged_folder / f"chunks_{file_count}.json"
                    with open(file_path, "w") as f:
                        json.dump(record_buffer, f)
                    record_buffer = []  # Reset buffer for the next batch
                    file_count += 1  # Increment file counter

        if self.run_mode in ["RES_LOAD", "RES_LOAD_FILES"]:
            # Dynamically set max_workers based on system capabilities and task type
            max_workers = min(16, (os.cpu_count() or 1) * 4)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                tasks = []
                file_paths = list(sorted(staged_folder.iterdir()))

                # Submit tasks
                for file_path in file_paths:
                    task = executor.submit(
                        self._process_and_move_file, file_path, processed_folder
                    )
                    tasks.append(task)

                # Process results as they complete
                for task in as_completed(tasks):
                    try:
                        task.result()  # This will re-raise any exception that occurred in the task
                    except Exception as e:
                        logging.error(f"Error during file processing: {e}")
                        # Additional error handling can be added here if needed

    def _process_and_move_file(self, file_path, processed_folder):
        """Process a file and then move it to a processed folder."""
        try:
            with open(file_path, "r") as file:
                chunks = json.load(file)
            self._insert_chunk_records(chunks)
            shutil.move(str(file_path), processed_folder)
        except Exception as e:
            logging.error(f"Failed to process and move file {file_path}: {e}")
            raise  # Re-raise the exception to be caught by the ThreadPoolExecutor handling

    def _load_docs_collection(self):
        if not self._client.has_collection("docs"):
            schema = self._create_docs_schema()
            self._client.create_collection(collection_name="docs", schema=schema)

            # Adjusted index_params structure to be a list containing one dictionary
            index_params = [
                {
                    "field_name": "docs_vector",
                    "params": {
                        "metric_type": "L2",
                        # Assuming you want to use an IVF_FLAT index as an example
                        "index_type": "IVF_FLAT",
                        "params": {
                            "nlist": 128
                        },  # Example specific parameter for IVF_FLAT
                    },
                }
            ]

            self._client.create_index(
                collection_name="docs",
                index_params=index_params,  # Now passing a list of dictionaries
            )

        self._client.load_collection("docs")
        return Collection(name="docs")

    def _load_chunks_collection(self):
        if not self._client.has_collection("chunks"):
            schema = self._create_chunks_schema()
            self._client.create_collection(collection_name="chunks", schema=schema)

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
        self._client.create_index(
            collection_name="chunks",
            index_params=index_params,  # Pass the list of dictionaries as index_params
        )

        self._client.load_collection("chunks")
        return Collection(name="chunks")

    def _create_docs_schema(self) -> CollectionSchema:
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
                    max_length=Config.get("max_doc_size"),
                ),
                # Add a docs vector; Milvus requires at least one vector field in the schema
                # This is not used, just a workaround to satisfy the schema requirements
                FieldSchema(name="docs_vector", dtype=DataType.FLOAT_VECTOR, dim=2),
            ],
            description="Collection for storing text and metadata of each document",
            enable_auto_id=False,
        )

        return schema

    def _create_chunks_schema(self) -> CollectionSchema:
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
                    dim=self._dense_ef.dim["colbert_vecs"],
                ),
                FieldSchema(
                    name="sparse_vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self._sparse_ef.dim,
                ),
                FieldSchema(
                    name="parent_id",
                    dtype=DataType.VARCHAR,
                    max_length=32,
                ),
                FieldSchema(
                    name="text",
                    dtype=DataType.VARCHAR,
                    max_length=Config.get("chunk_size") + Config.get("chunk_overlap"),
                ),
            ],
            description="Collection for storing chunk embeddings",
            enable_auto_id=False,
        )

        return schema

    def delete_documents(self, collection: str) -> None:
        """
        Deletes documents from database.
        """
        if collection == "all":
            collections = self._client.list_collections()
            for collection_name in collections:
                self._client.drop_collection(collection_name)
        else:
            self._client.drop_collection(collection)
        logging.info(f"Deleted documents from collection: {collection}")

    def add_documents(self, files: List[str], type: str = "decrees") -> None:
        logging.info(f"Adding documents of type {type} to the database.")
        # Generate documents, format them into database records, and insert them
        documents = self._doc_engine.generate_documents(files, type)
        doc_records = self._generate_doc_records(documents)
        self._client.insert(collection_name="docs", data=doc_records)

        # Generate chunks, format them into database records, and insert them
        logging.info("Processing and inserting chunk records.")
        self._process_and_insert_chunks(documents)

    def _process_and_insert_chunks(
        self, documents, batch_size: int = 10, start_from: int = 0
    ):
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
        total_batches = len(adjusted_docs) // batch_size + (
            1 if len(adjusted_docs) % batch_size > 0 else 0
        )

        # Process documents in batches starting from start_from
        for i in tqdm(
            range(total_batches), desc="Processing and inserting document chunks"
        ):
            # Adjust batch slice indices based on start_from
            start_idx = start_from + i * batch_size
            end_idx = min(start_from + (i + 1) * batch_size, len(documents))

            # Select batch documents
            batch_documents = documents[start_idx:end_idx]

            try:
                # Generate and insert chunk records for the current batch
                self._process_and_insert_chunk_batch(batch_documents)
            except Exception as e:
                # Log the next start_from value before raising the error
                next_start_from = (
                    start_idx + batch_size
                )  # Or end_idx for more precision
                logging.error(
                    f"Error processing documents. Next start_from should be: {next_start_from}. Error: {e}"
                )
                raise  # Rethrow the exception or handle it as you see fit

        logging.info("Completed processing and inserting chunks.")

    def _process_and_insert_chunk_batch(self, batch_documents: List[Document]):
        """
        Generate chunk records for a batch of documents and insert them into the database.

        Args:
            batch_documents (List[Document]): List of Document objects to process.
        """

        # Generate chunks for the documents
        chunks = self._doc_engine._chunk_documents(batch_documents)

        # Generate chunk records, including both dense and sparse embeddings
        chunk_records = self._generate_chunk_records(chunks)

        # Insert chunk records into the database
        self._insert_chunk_records(chunk_records)
        logging.info(f"Processed and inserted {len(chunk_records[0])} chunk records.")

    def _generate_doc_records(self, documents: List[Document]) -> List[Dict[str, any]]:
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
                "text": doc.text[
                    : Config.get("doc_size")
                ],  # Truncate text if necessary
                "docs_vector": [0.0, 0.0],  # Dummy vector for the docs_vector field
            }
            records.append(record)

        return records

    def _generate_chunk_records(self, chunks: List[Document]) -> List[Dict[str, any]]:
        """
        Prepare the records for inserting into the _chunks collection,
        including both dense and sparse embeddings, formatted as dictionaries.
        """
        records = []

        # Extract chunk texts to generate embeddings
        chunk_texts = [chunk.text for chunk in chunks]
        # Generate dense embeddings using the BGE-M3 function
        dense_embeddings = self._dense_ef.encode_documents(chunk_texts)["dense"]

        # Generate sparse embeddings using the BM25 function
        raw_sparse_embeddings = self._sparse_ef.encode_documents(chunk_texts)

        # Convert sparse embeddings from csr_array format to a list for insertion
        sparse_embeddings = [
            sparse_embedding.toarray().tolist()[
                0
            ]  # Ensure it's a list of lists, not a list of arrays
            for sparse_embedding in raw_sparse_embeddings
        ]

        # Prepare records with both embeddings for each chunk
        for i, chunk in enumerate(chunks):
            record = {
                "id": chunk.id,
                "dense_vector": dense_embeddings[i].tolist(),  # Dense embedding
                "sparse_vector": sparse_embeddings[i],  # Sparse embedding
                "parent_id": chunk.metadata.get("parent_id", 0),  # Parent document ID
                "text": chunk.text[: Config.get("chunk_size")],  # Truncated text
            }
            records.append(record)

        return records

    def _insert_doc_records(self, doc_records: List[List[any]]) -> None:
        """
        Batch load document records into the _docs collection.

        Args:
            doc_records: A list of document records to insert.
        """
        # Transpose the doc_records list to match the structure expected by Milvus insert method.
        field_values = list(zip(*doc_records))

        # Construct a dictionary where keys are field names and values are lists of field values.
        data_to_insert = {
            "id": field_values[0],
            "page": field_values[1],
            "date": field_values[2],
            "type": field_values[3],
            "number": field_values[4],
            "text": field_values[5],
        }

        # Insert the formatted records into the _docs collection.
        insert_result = self._docs.insert(data_to_insert)
        logging.info(f"Inserted {insert_result.insert_count} document records.")

    def _insert_chunk_records(self, chunk_records):
        """
        Batch load chunk records, including embeddings, into the _chunks collection.
        """
        insert_result = self._chunks.insert(chunk_records)
        logging.info(f"Inserted {insert_result.insert_count} chunk records.")

    def get_context(self, query: str, n_docs=2) -> List[Document]:
        """
        Retrieves relevant documents from the database based on a query.
        Args:
            query (str): Input query.

        Returns:
            List[Document]: List of relevant documents.
        """
        n_results = 10
        results = self._find_relevant_chunks(query, n_results)
        documents = self._retrieve_parent_documents(results, n_docs)
        context = self._create_context(documents)

        return context

    def _find_relevant_chunks(
        self, query: str, n_results: int
    ) -> List[AnnSearchRequest]:
        """
        Retrieves relevant context from the database based on a query.

        Args:
            query (str): Input query string.
            n_docs (int): Number of documents to retrieve context for.

        Returns:
            The context as a string, aggregated from relevant documents.
        """

        # Generate dense embedding for the query
        raw_query_dense_embeddings = self._dense_ef.encode_queries([query])
        dense_query_embedding = [raw_query_dense_embeddings["dense"][0].tolist()]

        # Generate sparse embedding for the query
        raw_query_sparse_embeddings = self._sparse_ef.encode_queries(
            [query]
        )  # Returns a csr_matrix

        # Convert sparse embedding from csr_matrix format to a list for insertion
        sparse_query_embedding = raw_query_sparse_embeddings.toarray().tolist()

        # AnnSearchRequest for dense embeddings
        dense_search_request = AnnSearchRequest(
            data=dense_query_embedding,
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 100}},
            limit=n_results,
        )

        # AnnSearchRequest for sparse embeddings
        sparse_search_request = AnnSearchRequest(
            data=sparse_query_embedding,
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"nprobe": 100}},
            limit=n_results,
        )

        # Step 3: Perform Hybrid Search
        # Execute hybrid_search using the prepared search requests and a reranking strategy
        response = self._chunks.hybrid_search(
            reqs=[
                dense_search_request,
                sparse_search_request,
            ],  # List of search requests
            rerank=WeightedRanker(
                self.DENSE_SEARCH_WEIGHT, self.SPARSE_SEARCH_WEIGHT
            ),  # Reranking strategy
            output_fields=["parent_id"],
            limit=n_results,
        )

        return response[0]

    def _retrieve_parent_documents(
        self, response: SearchResult, n_docs: int
    ) -> List[Document]:
        # Retrieve n_docs unique parent IDs from the response
        unique_parent_ids = []
        for hit in response:
            parent_id = hit.entity.get("parent_id")
            if parent_id not in unique_parent_ids:
                unique_parent_ids.append(parent_id)
                if len(unique_parent_ids) == n_docs:
                    break

        documents = [
            self._get_document_by_id(parent_id) for parent_id in unique_parent_ids
        ]

        return documents

    def _create_context(self, documents: List[Document]) -> (str, str):
        context = ""
        sources_list = []

        for doc in documents:
            # Create header for each document
            header = f"###DECRETO {doc.metadata['number']}###\n"
            # Add the text of the document to the context
            context += f"{header}{doc.text}\n\n"
            # Record the source of the document, which is its number and page
            sources_list.append(
                f"- Decreto {doc.metadata['number']} (página {doc.metadata['page']})"
            )

        # Combine the sources into a single string prefixed with "SOURCES:"
        sources = "Fuentes consultadas:\n" + "\n".join(sources_list)

        return context, sources

    def _get_document_by_id(self, data_id: str) -> Optional[Document]:
        """
        Gets a document from the Milvus 'docs' collection by its ID.

        Args:
            data_id (str): Unique ID associated with the document.

        Returns:
            Optional[Document]: Document associated with the given ID, or None if not found.
        """
        try:
            res = self._client.get(
                collection_name="docs",
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
            logging.error(
                f"An error occurred while retrieving document ID {data_id}: {e}"
            )
            return None
