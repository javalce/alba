import os
import json
import pickle
import nltk
from typing import List, Tuple, Optional
from config.config import Config
from src.chatbot.document_engine import Document
from pymilvus import connections
from pymilvus import MilvusClient, DataType, Collection
from pymilvus.client.abstract import AnnSearchRequest, WeightedRanker
from milvus_model.hybrid import BGEM3EmbeddingFunction
from milvus_model.sparse import BM25EmbeddingFunction
from milvus_model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus.orm.schema import CollectionSchema, FieldSchema


class LongTermMemory:
    def __init__(self):
        # Establish a connection to the Milvus server
        self.connect()
        self._dense_ef = self._load_embedding_function("dense")
        self._sparse_ef = self._load_embedding_function("sparse")
        self._docs = self._create_docs_collection()
        self._chunks = self._create_chunks_collection()

    def connect(self) -> MilvusClient:
        """
        Connects to the Milvus server and returns the client object.
        """
        # Establish a connection to the Milvus server
        port = Config.get("MILVUS_PORT")
        connections.connect("default", host=Config.get("MILVUS_HOST"), port=port)

        client = MilvusClient(uri=f"http://localhost:{port}")
        collections = client.list_collections()
        for collection_name in collections:
            collection = Collection(name=collection_name)
            collection.drop()
            print(f"Dropped collection: {collection_name}")

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

    def _create_docs_collection(self):

        schema = self._create_docs_schema()
        docs_coll = Collection(name="docs", schema=schema)

        return docs_coll

    def _create_chunks_collection(self):

        schema = self._create_chunks_schema()
        chunks_coll = Collection(name="chunks", schema=schema)

        bgeM3_index_params = {
            "metric_type": "IP",  # Inner Product, assuming normalized vectors for cosine similarity
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        bm25_index_params = {
            "metric_type": "IP",  # Inner Product
        }

        chunks_coll.create_index(
            field_name="dense_vector", index_params=bgeM3_index_params
        )
        chunks_coll.create_index(
            field_name="sparse_vector", index_params=bm25_index_params
        )
        return chunks_coll

    def _create_docs_schema(self) -> CollectionSchema:
        """
        Returns a schema for the long-term memory database of documents.
        """
        schema = CollectionSchema(
            [
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
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
                # Add a dummy vector; Milvus requires at least one vector field in the schema
                # This is not used, just a workaround to satisfy the schema requirements
                FieldSchema(name="dummy_vector", dtype=DataType.FLOAT_VECTOR, dim=2),
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
                    dtype=DataType.INT64,
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
                FieldSchema(name="parent_id", dtype=DataType.INT64),
                FieldSchema(
                    name="text",
                    dtype=DataType.VARCHAR,
                    max_length=Config.get("chunk_size"),
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

    def delete_documents(self, criteria: str) -> None:
        """
        Deletes documents from database.
        """
        pass

    def add_documents(
        self,
        documents: List[Document],
    ):
        # Generate and insert document records
        doc_records = self._generate_doc_records(documents)
        # Transpose doc_records to match Milvus' expected input format for insert
        transposed_doc_records = list(map(list, zip(*doc_records)))
        self._docs.insert(transposed_doc_records)

        # Generate chunks, their embeddings, and insert chunk records
        chunks = self._chunk_documents(documents)
        chunk_records = self._generate_chunk_records(chunks)
        transposed_chunk_records = list(map(list, zip(*chunk_records)))
        self._chunks.insert(transposed_chunk_records)

    def _generate_doc_records(self, documents: List[Document]) -> List[List[any]]:
        """
        Prepare the records for inserting into the _docs collection.
        """
        records = []
        for doc in documents:
            record = [
                int(
                    doc.id
                ),  # Assuming the ID is an integer or can be represented as one
                doc.metadata.get("page", 0),
                doc.metadata.get("date", ""),
                doc.metadata.get("type", ""),
                int(
                    doc.metadata.get("number", 0)
                ),  # Assuming number can be represented as int
                doc.text[: Config.get("doc_size")],  # Truncate text if necessary
                [0.0, 0.0],  # Dummy vector for the dummy_vector field
            ]
            records.append(record)
        return records

    def _generate_chunk_records(self, chunks: List[Document]) -> List[List[any]]:
        """
        Prepare the records for inserting into the _chunks collection,
        including both dense and sparse embeddings.
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
            sparse_embedding.toarray().tolist()
            for sparse_embedding in raw_sparse_embeddings
        ]

        # Prepare records with both embeddings for each chunk
        for i, chunk in enumerate(chunks):
            # Prepare the record for the current chunk with the necessary fields
            record = [
                int(chunk.id),  # Convert chunk ID to integer if necessary
                dense_embeddings[
                    i
                ].tolist(),  # Use the dense embedding and convert to list
                sparse_embeddings[i],  # Use the converted sparse embedding
                int(
                    chunk.metadata.get("parent_id", 0)
                ),  # Extract parent_id from metadata, convert to int
                chunk.text[
                    : Config.get("chunk_size")
                ],  # Truncate text to specified chunk size if necessary
            ]

            # Append the prepared record to the list of records
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
        print(f"Inserted {insert_result.insert_count} document records.")

    def _insert_chunk_records(self, chunk_records: List[List[any]]) -> None:
        """
        Batch load chunk records, including embeddings, into the _chunks collection.

        Args:
            chunk_records: A list of chunk records to insert, including embeddings.
        """
        # Transpose the chunk_records list to match the structure expected by Milvus insert method.
        field_values = list(zip(*chunk_records))

        # Construct a dictionary where keys are field names and values are lists of field values.
        # Note: The sparse vector is expected to be already in a list format suitable for insertion.
        data_to_insert = {
            "id": field_values[0],
            "dense_vector": field_values[1],
            "sparse_vector": field_values[2],
            "parent_id": field_values[3],
            "text": field_values[4],
        }

        # Insert the formatted records into the _chunks collection.
        insert_result = self._chunks.insert(data_to_insert)
        print(f"Inserted {insert_result.insert_count} chunk records.")

    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        # Define chunking parameters: size and overlap.
        chunk_size = Config.get("chunk_size")
        overlap = Config.get("chunk_overlap")

        chunked_documents = []
        for doc in documents:
            text = doc.text
            current_pos = 0  # Start position for chunking.

            text_length = len(text)
            chunk_counter = 1  # Initialize chunk counter for each document.
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
                    metadata = {
                        "parent_id": doc.id,
                        "chunk_id": f"{doc.id}_chunk_{chunk_counter}",  # Unique chunk identifier.
                    }
                    chunked_documents.append(
                        Document(id=metadata["chunk_id"], text=chunk, metadata=metadata)
                    )
                    chunk_counter += 1

        return chunked_documents

    def get_context(self, query: str, n_docs=2) -> List[Document]:
        """
        Retrieves relevant documents from the database based on a query.
        Args:
            query (str): Input query.

        Returns:
            List[Document]: List of relevant documents.
        """
        # Step 1: Generate embeddings for the query
        dense_query_embedding = self.bge_m3_ef.encode_queries([query])["dense"]
        raw_sparse_embeddings = self.bm25_ef.encode_queries([query])

        # Convert sparse embeddings from csr_matrix to a list for easier manipulation
        sparse_embeddings = raw_sparse_embeddings.toarray().tolist()

        # Step 2: Create AnnSearchRequest for dense embeddings
        dense_search_request = AnnSearchRequest(
            data=dense_query_embedding,
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=10,
        )

        # Step 3: Create AnnSearchRequest for sparse embeddings
        sparse_search_request = AnnSearchRequest(
            data=sparse_embeddings,
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=10,
        )

        # Step 4: Perform a hybrid search
        response = self._chunks.hybrid_search(
            reqs=[dense_search_request, sparse_search_request],
            rerank=WeightedRanker(0.5, 0.5),  # Adjust weights as needed
            limit=10,
        )
        # The M3 colbert scoring function needs the query length to normalize the score to the range 0 to 1.
        # This helps when combining the score with the other scoring functions.
        query_embeddings = self._dense_ef.encode_queries([query])

        # TODO: Implement the query to retrieve relevant documents

        # Retrieve n_docs unique parent IDs from the response
        all_parent_ids = [hit["fields"]["parent_id"] for hit in response.hits]
        unique_parent_ids = []
        for id in all_parent_ids:
            if id not in unique_parent_ids:
                unique_parent_ids.append(id)
                if len(unique_parent_ids) == n_docs:
                    break

        parent_documents = [
            self._get_document_by_id(parent_id) for parent_id in unique_parent_ids
        ]

        context = ""
        for doc in parent_documents:
            # Extract decree number from metadata
            metadata = json.loads(doc.metadata)
            doc_number = metadata["number"]
            # Crear header for each document
            header = f"###DECRETO {doc_number}###\n"
            # Add the text of the document to the context
            context += f"{header}{doc.text}\n\n"
        return context

    def _get_document_by_id(self, data_id: str) -> Optional[Document]:
        """
        Gets a document from the database by its ID.

        Args:
            data_id (str): Unique ID associated with the document.

        Returns:
            Optional[Document]: Document associated with the given ID, or None if not found.
        """
        response = self._app.get_data(schema="RAGdb2", data_id=data_id)

        if not response.is_successful():
            print(f"Document with ID {data_id} not found")
            return None

        fields = response.json["fields"]
        document = Document(
            id=data_id,
            parent_id="",
            text=fields.get("text", ""),
            metadata=fields.get("metadata", {}),
        )

        return document
