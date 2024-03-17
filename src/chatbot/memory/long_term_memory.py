import re
import json
from typing import List, Dict
from config.config import Config
from FlagEmbedding import BGEM3FlagModel
from vespa.package import (
    ApplicationPackage,
    Field,
    FieldSet,
    Document,
    Schema,
    RankProfile,
    FirstPhaseRanking,
)
from vespa.deployment import VespaDocker
from vespa.io import VespaResponse, VespaQueryResponse
from vespa.package import RankProfile, Function, FirstPhaseRanking


class LongTermMemory:
    def __init__(self):
        self.app = self._create_db_app()
        self._model = BGEM3FlagModel(Config.get("embedding_model"), use_fp16=False)

    def _create_db_app(self):
        """
        Creates and deploys the Vespa application package for the database
        """

        # Create Vespa schema and rank profile
        db_schema = self._create_schema()
        rank_profile = self._create_rank_profile()
        db_schema.add_rank_profile(rank_profile)

        # Create and deploy Vespa application package
        db_name = Config.get("db_name")
        db_app_package = ApplicationPackage(name=db_name, schema=[db_schema])
        db_container = VespaDocker()

        app = db_container.deploy(application_package=db_app_package)
        return app

    def _create_schema(self):
        """
        Returns a Vespa schema for the long-term memory database.
        """
        bge_m3_schema = Schema(
            name=Config.get("db_name"),
            document=Document(
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
            texts = [doc.text for doc in documents]
            # Generate embeddings for the texts
            embeddings = self.model.encode(
                texts, return_dense=True, return_sparse=True, return_colbert_vecs=True
            )

        for i, document in enumerate(documents):
            # Initialize the fields dictionary, including embeddings if generated
            fields = {
                "text": document.text,
                "parent_id": document.parent_id if document.parent_id else "",
            }

            if generate_embeddings:
                # Directly assign the tensor embeddings to their respective fields
                fields["lexical_rep"] = embeddings["lexical_weights"][i]
                fields["dense_rep"] = embeddings["dense_vecs"][i]
                fields["colbert_rep"] = embeddings["colbert_vecs"][i]

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

        # Feed the iterable of processed embeddings to Vespa
        self.app.feed_iterable(
            iter=records,  # The iterable of documents to be fed
            schema=Config.get("db_name"),  # The schema to feed the data into
            callback=feed_callback,  # Optional: handle the outcome of each operation
        )

    def _chunk_documents(
        text: str, chunk_size: int = 250, overlap: int = 50
    ) -> List[str]:
        paragraphs = text.split("\n\n")
        chunks = []

        for paragraph in paragraphs:
            if len(paragraph) <= chunk_size:
                chunks.append(paragraph)
            else:
                sentences = re.split(r"(?<=[.!?]) +", paragraph)
                sentence_chunk = ""

                for sentence in sentences:
                    if len(sentence_chunk) + len(sentence) <= chunk_size:
                        sentence_chunk += sentence + " "
                    else:
                        chunks.append(sentence_chunk.strip())
                        # Start new chunk with an overlap from the previous chunk if it's long enough
                        sentence_chunk = sentence_chunk[-overlap:] + sentence + " "

                if sentence_chunk:
                    chunks.append(sentence_chunk.strip())

        # If even sentences are too long, split by chunk size directly (last resort), including overlap
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= chunk_size:
                final_chunks.append(chunk)
            else:
                # Adjust the loop to reduce the step size by the overlap amount to ensure the overlap
                for i in range(0, len(chunk), chunk_size - overlap):
                    end_index = i + chunk_size
                    # Ensure the chunk does not exceed the text length
                    chunk_to_add = (
                        chunk[i:end_index] if end_index <= len(chunk) else chunk[i:]
                    )
                    final_chunks.append(chunk_to_add)

        return final_chunks


# class LongTermMemory:
#     def __init__(self):
#         self.app = self._create_db_app()
#         self._model = BGEM3FlagModel(Config.get("embedding_model"), use_fp16=False)

#     def _create_db_app(self):
#         """
#         Creates and deploys the Vespa application package for the database
#         """

#         # Create Vespa schema and rank profile
#         db_schema = self._create_schema()
#         rank_profile = self._create_rank_profile()
#         db_schema.add_rank_profile(rank_profile)

#         # Create and deploy Vespa application package
#         db_name = Config.get("db_name")
#         db_app_package = ApplicationPackage(name=db_name, schema=[db_schema])
#         db_container = VespaDocker()

#         app = db_container.deploy(application_package=db_app_package)
#         return app

#     def _create_schema(self):
#         """
#         Returns a Vespa schema for the long-term memory database.
#         """
#         bge_m3_schema = Schema(
#             name=Config.get("db_name"),
#             document=Document(
#                 fields=[
#                     Field(name="id", type="string", indexing=["summary"]),
#                     Field(
#                         name="text",
#                         type="string",
#                         indexing=["summary", "index"],
#                         index="enable-bm25",
#                     ),
#                     Field(
#                         name="lexical_rep",
#                         type="tensor<bfloat16>(t{})",
#                         indexing=["summary", "attribute"],
#                     ),
#                     Field(
#                         name="dense_rep",
#                         type="tensor<bfloat16>(x[1024])",
#                         indexing=["summary", "attribute"],
#                         attribute=["distance-metric: angular"],
#                     ),
#                     Field(
#                         name="colbert_rep",
#                         type="tensor<bfloat16>(t{}, x[1024])",
#                         indexing=["summary", "attribute"],
#                     ),
#                 ],
#             ),
#             fieldsets=[FieldSet(name="default", fields=["text"])],
#         )
#         return bge_m3_schema

#     def _create_rank_profile(self):
#         """
#         Creates a default rank profile (used to rank search results semantically)
#         """
#         rank_profile = RankProfile(
#             name="m3hybrid",
#             inputs=[
#                 ("query(q_dense)", "tensor<bfloat16>(x[1024])"),
#                 ("query(q_lexical)", "tensor<bfloat16>(t{})"),
#                 ("query(q_colbert)", "tensor<bfloat16>(qt{}, x[1024])"),
#                 ("query(q_len_colbert)", "float"),
#             ],
#             functions=[
#                 Function(
#                     name="dense",
#                     expression="cosine_similarity(query(q_dense), attribute(dense_rep),x)",
#                 ),
#                 Function(
#                     name="lexical",
#                     expression="sum(query(q_lexical) * attribute(lexical_rep))",
#                 ),
#                 Function(
#                     name="max_sim",
#                     expression="sum(reduce(sum(query(q_colbert) * attribute(colbert_rep) , x),max, t),qt)/query(q_len_colbert)",
#                 ),
#             ],
#             first_phase=FirstPhaseRanking(
#                 expression="0.4*dense + 0.2*lexical +  0.4*max_sim",
#                 rank_score_drop_limit=0.0,
#             ),
#             match_features=["dense", "lexical", "max_sim", "bm25(text)"],
#         )

#         return rank_profile

#     def add_documents(self, documents):
#         """
#         Upserts the documents into the database using feed_iterable method for bulk insertion

#         Args:
#             documents (list): A list of Document objects
#         """
#         # Process each document for chunking if applicable
#         chunks = []
#         for doc in documents:
#             # Here, you could apply different chunking strategies based on doc.metadata or other criteria
#             chunked_docs = self._chunk_document(doc)
#             chunks.extend(chunked_docs)  # Add chunked documents to the final list

#         db_documents = self._format_documents(chunks)


#     def add_chunks(self, records):
#         """
#         Upserts the records into the database using feed_iterable method for bulk insertion

#         Args:
#             records (dict): A dictionary containing 'ids', 'dense_vecs', 'lexical_weights', 'colbert_vecs', and 'metadata'
#         """
#         db_records = self._format_records(records)

#         # Define a callback function to handle the feed response for each document
#         # TODO: Log errors and send them to the front end
#         def _feed_callback(response: VespaResponse, id: str):
#             if response.status_code != 200:
#                 print(f"Failed to feed document {id}: {response.json}")
#             else:
#                 print(f"Successfully fed document {id}")

#         # Feed the iterable of processed embeddings to Vespa
#         self.app.feed_iterable(
#             iter=db_records,  # The iterable of documents to be fed
#             schema=Config.get("db_name"),  # The schema to feed the data into
#             callback=_feed_callback,  # Optional: handle the outcome of each operation
#         )

#     def recall_docs(self, query):
#         """
#         Searches the long-term memory for facts relevant to the query.

#         Parameters:
#         - query (str): The search query to find relevant facts.

#         Returns:
#         - VespaQueryResponse: The response from the Vespa database.
#         """
#         # Generate embeddings for the query (lexical/sparse, dense, and colbert representations)
#         query_embeddings = self._model.encode(
#             query, return_dense=True, return_sparse=True, return_colbert_vecs=True
#         )

#         # Colbert scoring function needs query length to normalize to 0-1 scores;
#         # it helps when combining the score with other scoring functions
#         # More details: https://github.com/vespa-engine/pyvespa/blob/master/docs/sphinx/source/examples/mother-of-all-embedding-models-cloud.ipynb
#         query_length = query_embeddings["colbert_vecs"][0].shape[0]
#         query_fields = {
#             "input.query(q_lexical)": {
#                 key: float(value)
#                 for key, value in query_embeddings["lexical_weights"][0].items()
#             },
#             "input.query(q_dense)": query_embeddings["dense_vecs"][0].tolist(),
#             "input.query(q_colbert)": str(
#                 {
#                     index: query_embeddings["colbert_vecs"][0][index].tolist()
#                     for index in range(query_embeddings["colbert_vecs"][0].shape[0])
#                 }
#             ),
#             "input.query(q_len_colbert)": query_length,
#         }

#         # Query the database
#         response: VespaQueryResponse = self.app.query(
#             yql="select id, text from m where userQuery() or ({targetHits:10}nearestNeighbor(dense_rep,q_dense))",
#             ranking="m3hybrid",
#             query=query[0],
#             body={**query_fields},
#         )
#         assert response.is_successful()

#         json_response = json.dumps(response.hits[0], indent=2)
#         return json_response

#     def _format_records(self, records):
#         db_records = []  # To hold the processed data for upsertion

#         # Converts the records into the format expected by the database
#         for i in range(len(records["ids"])):
#             # Retrieve the text from the metadata
#             text = records["metadata"][i]["text"]

#             # Prepare the Vespa field data for this embedding
#             # TODO: Add the metadata fields to the Vespa fields
#             db_record = {
#                 "id": str(records["ids"][i]),  # Ensure each document has a unique ID
#                 "fields": {
#                     "text": text,
#                     "lexical_rep": {
#                         key: float(value)
#                         for key, value in records["lexical_weights"][i].items()
#                     },
#                     "dense_rep": records["dense_vecs"][i].tolist(),
#                     "colbert_rep": {
#                         i: vec.tolist()
#                         for i, vec in enumerate(records["colbert_vecs"][i])
#                     },
#                 },
#             }

#             db_records.append(db_record)

#         return db_records
