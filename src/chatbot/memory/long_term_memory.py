import os
import json
import pickle
from typing import List, Dict, Optional
from config.config import Config
from FlagEmbedding import BGEM3FlagModel
from src.chatbot.document_engine import Document
from vespa.application import Vespa
from vespa.package import (
    ApplicationPackage,
    Field,
    FieldSet,
    Document as VespaDocument,
    Schema,
    FirstPhaseRanking,
)
from vespa.deployment import VespaDocker
from vespa.io import VespaResponse, VespaQueryResponse
from vespa.package import RankProfile, Function, FirstPhaseRanking


class LongTermMemory:
    def __init__(self):
        self._db_name = Config.get("db_name")
        self._schema = Config.get("schema_name")
        self._app = self._create_db_app()
        self._model = BGEM3FlagModel(Config.get("embedding_model"), use_fp16=False)

    def _create_db_app(self):
        """
        Creates and deploys the Vespa application package for the database
        """

        # Check if the Vespa application is already running
        # If it is, return the existing app
        app = Vespa(url="http://localhost:8080")
        if isinstance(app, Vespa):
            return app

        # Create Vespa schema and rank profile
        db_schema = self._create_schema()
        rank_profile = self._create_rank_profile()
        db_schema.add_rank_profile(rank_profile)

        # Create and deploy Vespa application package
        db_app_package = ApplicationPackage(name=self._schema, schema=[db_schema])
        db_container = VespaDocker()

        app = db_container.deploy(application_package=db_app_package)
        return app

    def _create_schema(self):
        """
        Returns a Vespa schema for the long-term memory database.
        """
        bge_m3_schema = Schema(
            name=self._schema,
            document=VespaDocument(
                fields=[
                    Field(name="id", type="string", indexing=["summary"]),
                    Field(name="parent_id", type="string", indexing=["summary"]),
                    Field(
                        name="text",
                        type="string",
                        indexing=["summary", "index"],
                        index="enable-bm25",
                    ),
                    Field(
                        name="metadata",
                        type="string",
                        indexing=["summary", "attribute"],  # Adjust indexing as needed
                    ),
                    Field(
                        name="lexical_rep",
                        type="tensor<bfloat16>(t{})",
                        indexing=["summary", "attribute"],
                    ),
                    Field(
                        name="dense_rep",
                        type="tensor<bfloat16>(x[1024])",
                        indexing=["summary", "attribute"],
                        attribute=["distance-metric: angular"],
                    ),
                    Field(
                        name="colbert_rep",
                        type="tensor<bfloat16>(t{}, x[1024])",
                        indexing=["summary", "attribute"],
                    ),
                ],
            ),
            fieldsets=[FieldSet(name="default", fields=["text"])],
        )
        return bge_m3_schema

    def _create_rank_profile(self):
        """
        Creates a default rank profile (used to rank search results semantically)
        """
        rank_profile = RankProfile(
            name="m3hybrid",
            inputs=[
                ("query(q_dense)", "tensor<bfloat16>(x[1024])"),
                ("query(q_lexical)", "tensor<bfloat16>(t{})"),
                ("query(q_colbert)", "tensor<bfloat16>(qt{}, x[1024])"),
                ("query(q_len_colbert)", "float"),
            ],
            functions=[
                Function(
                    name="dense",
                    expression="cosine_similarity(query(q_dense), attribute(dense_rep),x)",
                ),
                Function(
                    name="lexical",
                    expression="sum(query(q_lexical) * attribute(lexical_rep))",
                ),
                Function(
                    name="max_sim",
                    expression="sum(reduce(sum(query(q_colbert) * attribute(colbert_rep) , x),max, t),qt)/query(q_len_colbert)",
                ),
            ],
            first_phase=FirstPhaseRanking(
                expression="0.4*dense + 0.2*lexical +  0.4*max_sim",
                rank_score_drop_limit=0.0,
            ),
            match_features=["dense", "lexical", "max_sim", "bm25(text)"],
        )

        return rank_profile

    def _count_docs(self) -> int:
        """
        Counts the number of documents in the database.
        """
        # The YQL query to retrieve all document IDs from a specific schema
        yql_query = f"select id from {self._schema} where true;"

        # Perform the query and count the number of items returned
        response = self._app.query(
            yql=yql_query, query_model=None, type="all", body={"hits": 0}
        )
        number_of_documents = response.number_documents_indexed

        print(f"Number of documents in the schema: {number_of_documents}")

    def delete_documents(self, criteria: str) -> None:
        """
        Deletes documents from database.
        """
        # Execute the delete operation
        if criteria == "all":
            response = self._app.delete_all_docs(
                content_cluster_name=self._db_name, schema=self._schema
            )
            # TODO: Implement logging system
            print(response)
        else:
            if not isinstance(criteria, list):
                criteria = [criteria]
            for c in criteria:
                self._app.delete_data(schema=self._schema, data_id=c)

    def add_documents(self, documents: List[Document]) -> None:
        """
        Adds documents to the database, optionally generating embeddings for each document.

        Args:
            documents (List[Document]): A list of documents to be processed and stored.
            generate_embeddings (bool): A flag indicating whether embeddings should be generated for the documents.
        """

        # We first load the parent records, which contains all the text in each document
        # The idea now in the data ingestion state is to first feed the parent records and
        # then feed the chunk records. In the retrieval phase, we'll find the most relevant
        # chunks and then retrieve their parent records, which are ultimately what will be
        # used as context for the model to generate a response.
        # Notice we don't need embeddings for the parent records
        db_document_records = self._generate_records(
            documents, generate_embeddings=False
        )
        self._load_records(db_document_records)

        chunks = self._chunk_documents(documents)
        db_chunk_records = self._generate_records(chunks, generate_embeddings=True)
        self._load_records(db_chunk_records)

    def _generate_records(
        self, documents: List[Document], generate_embeddings: bool
    ) -> List[Dict]:
        db_records = []

        if generate_embeddings:
            # Extract texts from documents for embedding generation
            # TODO: REMOVE THIS
            # Check if the embeddings file exists
            embeddings_file = "embeddings.pkl"
            if os.path.exists(embeddings_file):
                # Load embeddings from the file
                with open(embeddings_file, "rb") as f:
                    embeddings = pickle.load(f)
            else:
                # Extract texts from documents for embedding generation
                texts = [doc.text for doc in documents]

                # Assume self._model is your embeddings model instance and it's already set up
                embeddings = self._model.encode(
                    texts,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=True,
                )

                # Save the generated embeddings to a file for future use
                with open(embeddings_file, "wb") as f:
                    pickle.dump(embeddings, f)

        for i, document in enumerate(documents):
            # Initialize the fields dictionary, including embeddings if generated
            fields = {
                "text": document.text,
                "parent_id": document.parent_id if document.parent_id else "",
                "metadata": json.dumps(document.metadata),
            }

            if generate_embeddings:
                # Convert numpy arrays to lists before assigning them
                fields["lexical_rep"] = {
                    k: float(v) for k, v in embeddings["lexical_weights"][i].items()
                }
                fields["dense_rep"] = embeddings["dense_vecs"][i].tolist()
                fields["colbert_rep"] = {
                    str(j): vec.tolist()
                    for j, vec in enumerate(embeddings["colbert_vecs"][i])
                }

            # Construct the record in the format expected by the database
            record = {
                "id": document.id,
                "fields": fields,
            }

            db_records.append(record)

        return db_records

    def _load_records(self, records: List[Dict]) -> None:
        # Define a callback function to handle the feed response for each document
        def feed_callback(response: VespaResponse, id: str):
            if response.status_code != 200:
                print(f"Failed to feed document {id}: {response.json}")
            else:
                print(f"Document {id} successfully fed")

        # Feed the iterable of processed embeddings to Vespa
        self._app.feed_iterable(
            iter=records,  # The iterable of documents to be fed
            schema=Config.get("db_name"),  # The schema to feed the data into
            callback=feed_callback,  # Optional: handle the outcome of each operation
        )

    def _chunk_documents(
        self, documents: List[Document], chunk_size: int = 250, overlap: int = 50
    ) -> List[Document]:
        chunked_documents = []

        for doc in documents:
            text = doc.text
            current_pos = 0
            text_length = len(text)

            while current_pos < text_length:
                if current_pos + chunk_size >= text_length:
                    # Last chunk may be smaller than chunk_size
                    chunk = text[current_pos:text_length]
                    current_pos = text_length
                else:
                    end_pos = current_pos + chunk_size
                    next_space = text.find(" ", end_pos - overlap)
                    if next_space == -1 or next_space >= end_pos + overlap:
                        # If there's no space in the overlap zone, or it's too far ahead, just cut directly
                        next_space = end_pos

                    chunk = text[current_pos:next_space].strip()
                    current_pos = next_space

                if chunk:
                    chunked_documents.append(
                        Document(parent_id=doc.id, text=chunk, metadata=doc.metadata)
                    )

        return chunked_documents

    def get_documents(self, query: str, n_docs=5) -> List[Document]:
        """
        Retrieves documents relevant to a given query.
        The nearestNeighbor(embedding,q) function finds the documents in the
        RAGdb database that are nearest to the query embedding vector q. The
        targetHits:n instructs the database to find the n closest documents
        according to the embedding distance and then rank them using the specified
        rank-profile fusion. After the nearest neighbor search is complete, the rank
        function further processes these documents according to the fusion ranking profile,
        which could include additional relevancy computations like BM25 scores or
        other rank features.

        Finally, limit 5 tells Vespa that of these ranked results, only the top 5
        should be returned in the response.

        Args:
            query (str): Input query.

        Returns:
            List[Document]: List of relevant documents.
        """

        # The M3 colbert scoring function needs the query length to normalize the score to the range 0 to 1.
        # This helps when combining the score with the other scoring functions.
        query_embeddings = self._model.encode(
            [query], return_dense=True, return_sparse=True, return_colbert_vecs=True
        )
        query_length = query_embeddings["colbert_vecs"][0].shape[0]
        # TODO: Create a method to generate these fields
        query_fields = {
            "input.query(q_lexical)": {
                key: float(value)
                for key, value in query_embeddings["lexical_weights"][0].items()
            },
            "input.query(q_dense)": query_embeddings["dense_vecs"][0].tolist(),
            "input.query(q_colbert)": str(
                {
                    index: query_embeddings["colbert_vecs"][0][index].tolist()
                    for index in range(query_embeddings["colbert_vecs"][0].shape[0])
                }
            ),
            "input.query(q_len_colbert)": query_length,
        }

        response: VespaQueryResponse = self._app.query(
            yql="select id, parent_id, text from RAGdb2 where userQuery() or ({targetHits:100}nearestNeighbor(dense_rep,q_dense))",
            ranking="m3hybrid",
            query=query[0],
            body={**query_fields},
        )
        assert response.is_successful()

        parent_ids = [hit["fields"]["parent_id"] for hit in response.hits[:n_docs]]
        parent_documents = [
            self._get_document_by_id(parent_id) for parent_id in parent_ids
        ]

        return parent_documents

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
