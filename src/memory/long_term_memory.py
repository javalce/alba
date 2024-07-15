import json
from typing import List, Tuple

from pymilvus.client.abstract import AnnSearchRequest, SearchResult, WeightedRanker

from config.config import get_config
from src.database import Database
from src.document_engine import Document
from src.utils.ner_extraction import EntityExtractor
from src.utils.utils import setup_logging

# Setup logging
setup_logging()


class LongTermMemory:
    def __init__(self):
        self.config = get_config()

        self.ner_extractor = EntityExtractor(self.config.LOCATIONS_FILE)

        # Define the weights for the hybrid search
        self.DENSE_SEARCH_WEIGHT = 0.1
        self.SPARSE_SEARCH_WEIGHT = 0.9

        # Create a connection to the database
        self.db = Database()

    def get_context(self, query: str, n_docs=2) -> List[Document]:
        """
        Retrieves relevant documents from the database based on a query.
        Args:
            query (str): Input query.

        Returns:
            List[Document]: List of relevant documents.
        """
        n_results = 1000
        results = self._find_relevant_chunks(query, n_results)
        documents = self._retrieve_parent_documents(results, n_docs)
        context = self._create_context(documents)

        return context

    def _find_relevant_chunks(self, query: str, n_results: int) -> List[AnnSearchRequest]:
        """
        Retrieves relevant context from the database based on a query.

        Args:
            query (str): Input query string.
            n_docs (int): Number of documents to retrieve context for.

        Returns:
            The context as a string, aggregated from relevant documents.
        """

        # Generate dense embedding for the query
        raw_query_dense_embeddings = self.db.encode_dense_query([query])
        dense_query_embedding = [raw_query_dense_embeddings["dense"][0].tolist()]

        # Generate sparse embedding for the query
        raw_query_sparse_embeddings = self.db.encode_sparse_query([query])  # Returns a csr_matrix

        # Convert sparse embedding from csr_matrix format to a list for insertion
        sparse_query_embedding = raw_query_sparse_embeddings.toarray().tolist()

        # AnnSearchRequest for dense embeddings
        dense_search_request = AnnSearchRequest(
            data=dense_query_embedding,
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 1000}},
            limit=n_results,
        )

        # AnnSearchRequest for sparse embeddings
        sparse_search_request = AnnSearchRequest(
            data=sparse_query_embedding,
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"nprobe": 1000}},
            limit=n_results,
        )

        extracted_entities = self.ner_extractor.extract_entities(query)
        if extracted_entities:
            # Construct the JSON array for JSON_CONTAINS_ANY
            entity_list = [
                {"type": entity_type, "value": entity_value.replace('"', '\\"')}
                for entity_type, entity_value in extracted_entities
            ]
            entity_list_json = json.dumps(entity_list)  # Convert to JSON string

            # Construct the filter using JSON_CONTAINS_ANY
            dynamic_filter = f"JSON_CONTAINS_ANY(entities, {entity_list_json})"

        response = None
        # Perform Hybrid Search
        if extracted_entities:
            response = self.db.chunks.hybrid_search(
                reqs=[
                    dense_search_request,
                    sparse_search_request,
                ],
                rerank=WeightedRanker(self.DENSE_SEARCH_WEIGHT, self.SPARSE_SEARCH_WEIGHT),
                output_fields=["parent_id", "entities"],
                limit=n_results,
                filter=dynamic_filter,
            )

            # Filter the response to only include chunks with the extracted entities
            hits = []
            for hit in response[0]:
                entities = hit.entity.get("entities")
                if any(ent in entities for ent in entity_list):
                    hits.append(hit)
            if hits:
                response[0] = hits

        # If no entities were extracted or no relevant chunks were found, perform a general search
        if not extracted_entities or not len(response):
            response = self.db.chunks.hybrid_search(
                reqs=[
                    dense_search_request,
                    sparse_search_request,
                ],
                rerank=WeightedRanker(self.DENSE_SEARCH_WEIGHT, self.SPARSE_SEARCH_WEIGHT),
                output_fields=["parent_id"],
                limit=n_results,
            )
        return response[0]

    def _retrieve_parent_documents(self, response: SearchResult, n_docs: int) -> List[Document]:
        # Retrieve n_docs unique parent IDs from the response
        unique_parent_ids = []
        for hit in response:
            parent_id = hit.entity.get("parent_id")
            if parent_id not in unique_parent_ids:
                unique_parent_ids.append(parent_id)
                if len(unique_parent_ids) == n_docs:
                    break

        documents = [self.db.get_document_by_id(parent_id) for parent_id in unique_parent_ids]

        return documents

    def _create_context(self, documents: List[Document]) -> Tuple[str, str]:
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
