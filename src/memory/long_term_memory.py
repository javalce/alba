import os
import json
import pickle
import nltk
import logging
from pathlib import Path
from typing import List, Optional
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
                if "stopwords" not in nltk.corpus.util.lazy_imports:
                    nltk.download("stopwords")

                analyzer = build_default_analyzer(language="sp")
                bm25_ef = BM25EmbeddingFunction(analyzer)

                # If a corpus is provided, fit the model to the corpus
                if corpus:
                    bm25_ef.fit(corpus)
                    # Serialize and save the model to a file
                    with open(model_path, "wb") as file:
                        pickle.dump(bm25_ef, file)

            return bm25_ef
        elif ef_type == "dense":
            # More info here: https://milvus.io/docs/embed-with-bgm-m3.md
            bgeM3_ef = BGEM3EmbeddingFunction(
                model_name="BAAI/bge-m3",  # Specify the model name
                device="cpu",  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
                use_fp16=False,  # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
            )
            return bgeM3_ef
        else:
            raise ValueError(f"Unsupported embedding function type: {ef_type}")

    def _load_collections(self):
        data_folder = Path(Config.get("raw_data_folder"))

        # For RES_LOAD (reset and load), delete all documents before proceeding
        run_mode = Config.get("run_mode")
        if run_mode == "RES_LOAD":
            self.delete_documents("all")

        # Collect file paths using list comprehension
        initial_files = [
            str(file_path)
            for file_path in data_folder.rglob("*")
            if file_path.is_file()
        ]

        # Load collections for all run modes
        self._docs = self._load_docs_collection()
        self._chunks = self._load_chunks_collection()

        # Add documents if not in NO_RES_NO_LOAD mode (no reset, no load)
        if run_mode != "NO_RES_NO_LOAD":
            self.add_documents(initial_files)

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
                    max_length=Config.get("doc_size"),
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

    def _count_docs(self) -> int:
        """
        Counts the number of documents in the database.
        """
        pass

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

    def add_documents(self, files: List[str], type: str = "decrees") -> None:
        # Generate documents, format them into database records, and insert them
        documents = self._doc_engine.generate_documents(files, type)
        doc_records = self._generate_doc_records(documents)
        # self._client.insert(collection_name="docs", data=doc_records)
        self._docs.insert(doc_records)

        # Generate chunks, format them into database records, and insert them
        chunks = self._doc_engine._chunk_documents(documents)
        chunk_records = self._generate_chunk_records(chunks)
        self._chunks.insert(chunk_records)

    def _generate_doc_records(self, documents: List[Document]) -> List[List[any]]:
        """
        Prepare the records for inserting into the _docs collection.
        """
        records = []
        for doc in documents:
            record = [
                doc.id,  # Assuming the ID is an integer or can be represented as one
                doc.metadata.get("page", 0),
                doc.metadata.get("date", ""),
                doc.metadata.get("type", ""),
                int(
                    doc.metadata.get("number", 0)
                ),  # Assuming number can be represented as int
                doc.text[: Config.get("doc_size")],  # Truncate text if necessary
                [0.0, 0.0],  # Dummy vector for the docs_vector field
            ]
            records.append(record)

        # Transpose doc_records to match Milvus' expected input format for insert
        transposed_records = list(map(list, zip(*records)))
        return transposed_records

    def _generate_chunk_records(self, chunks: List[Document]) -> List[List[any]]:
        """
        Prepare the records for inserting into the _chunks collection,
        including both dense and sparse embeddings.
        """
        records = []

        # Extract chunk texts to generate embeddings
        chunk_texts = [chunk.text for chunk in chunks]
        for chunk in chunks:
            if len(chunk.text) == 0:
                print(f"Empty chunk text for document ID: {chunk.id}")
            elif len(chunk.text) > Config.get("chunk_size"):
                print(f"Chunk text exceeds maximum size for document ID: {chunk.id}")

        # Generate dense embeddings using the BGE-M3 function
        dense_embeddings = self._dense_ef.encode_documents(chunk_texts)["dense"]

        # Generate sparse embeddings using the BM25 function
        raw_sparse_embeddings = self._sparse_ef.encode_documents(chunk_texts)

        # Convert sparse embeddings from csr_array format to a list for insertion
        sparse_embeddings = [
            sparse_embedding.toarray().tolist()
            for sparse_embedding in raw_sparse_embeddings
        ]

        # Prepare records with both embeddings for each chunk
        for i, chunk in enumerate(chunks):
            # Prepare the record for the current chunk with the necessary fields
            record = [
                chunk.id,
                dense_embeddings[
                    i
                ].tolist(),  # Use the dense embedding and convert to list
                sparse_embeddings[i][0],  # Use the converted sparse embedding
                chunk.metadata.get("parent_id", 0),  # Extract parent_id from metadata
                chunk.text[
                    : Config.get("chunk_size")
                ],  # Truncate text to specified chunk size if necessary
            ]

            # Append the prepared record to the list of records
            records.append(record)

        transposed_records = list(map(list, zip(*records)))
        return transposed_records

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
        print(f"Inserted {insert_result.insert_count} document records.")

    def _insert_chunk_records(self, chunk_records):
        """
        Batch load chunk records, including embeddings, into the _chunks collection.
        """
        field_values = list(zip(*chunk_records))
        data_to_insert = {
            "id": field_values[0],
            "dense_vector": field_values[1],
            "sparse_vector": field_values[2],
            "parent_id": field_values[3],
            "text": field_values[4],
        }

        insert_result = self._chunks.insert(data_to_insert)
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
        results = self._search_documents(query, n_results)
        documents = self._retrieve_documents(results, n_docs)
        context = self._create_context(documents)

        return context

    def _search_documents(self, query: str, n_results: int) -> List[AnnSearchRequest]:
        """
        Retrieves relevant context from the database based on a query.

        Args:
            query (str): Input query string.
            n_docs (int): Number of documents to retrieve context for.

        Returns:
            The context as a string, aggregated from relevant documents.
        """
        # Step 1: Generate Query Embeddings
        # Generate dense embedding for the query
        raw_query_dense_embeddings = self._dense_ef.encode_queries([query])
        dense_query_embedding = [raw_query_dense_embeddings["dense"][0].tolist()]

        # Generate sparse embedding for the query
        raw_query_sparse_embeddings = self._sparse_ef.encode_queries(
            [query]
        )  # Assume returns a list of embeddings
        sparse_query_embedding = raw_query_sparse_embeddings.toarray().tolist()

        # Step 2: Create Search Requests
        # AnnSearchRequest for dense embeddings
        dense_search_request = AnnSearchRequest(
            data=dense_query_embedding,
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=n_results,
        )

        # AnnSearchRequest for sparse embeddings
        sparse_search_request = AnnSearchRequest(
            data=sparse_query_embedding,
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=n_results,
        )

        # Step 3: Perform Hybrid Search
        # Execute hybrid_search using the prepared search requests and a reranking strategy
        response = self._chunks.hybrid_search(
            reqs=[
                dense_search_request,
                sparse_search_request,
            ],  # List of search requests
            rerank=WeightedRanker(0.5, 0.5),  # Reranking strategy
            output_fields=["parent_id"],
            limit=n_results,
        )

        return response[0]

    def _retrieve_documents(
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
            # Assumes 'id' field is named 'id' in the Milvus collection schema
            # Replace 'collection_name="quick_setup"' with your actual collection name, e.g., 'docs'
            res = self._client.get(
                collection_name="docs",  # Use the name of your collection here
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
