from dependency_injector import containers, providers
from sqlalchemy_toolkit import DatabaseManager

from alba.chatbot import Chatbot
from alba.config import Config
from alba.document_engine import DocumentEngine
from alba.memory.long_term_memory import LongTermMemory
from alba.milvus_database import MilvusDatabase
from alba.repositories import DocumentRepository
from alba.services import DocumentService
from alba.utils.ner_extraction import EntityExtractor


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(packages=["alba.router"])

    config = providers.Singleton(Config)

    ner_extractor = providers.Singleton(
        EntityExtractor,
        config=config,
        locations_file=config.provided.LOCATIONS_FILE,
    )

    db = providers.Singleton(DatabaseManager, db_url=config.provided.DB_URL)

    document_repository = providers.Singleton(DocumentRepository, db=db)

    document_service = providers.Singleton(DocumentService, repository=document_repository)

    document_engine = providers.Singleton(DocumentEngine, config=config)

    milvus_db = providers.Singleton(
        MilvusDatabase,
        config=config,
        document_engine=document_engine,
        ner_extractor=ner_extractor,
    )

    long_term_memory = providers.Singleton(
        LongTermMemory,
        config=config,
        ner_extractor=ner_extractor,
        db=milvus_db,
    )

    chatbot = providers.Singleton(Chatbot, config=config, long_term_mem=long_term_memory)
