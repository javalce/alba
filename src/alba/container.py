from dependency_injector import containers, providers
from sqlalchemy_toolkit import DatabaseManager

from alba.chatbot import Chatbot
from alba.config import Config
from alba.document_engine import DocumentEngine
from alba.memory.long_term_memory import LongTermMemory
from alba.milvus_database import MilvusDatabase
from alba.repositories import DecreeRepository, DocumentRepository, UserRepository
from alba.services import DecreeService, DocumentService, UserService
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

    decree_repository = providers.Singleton(DecreeRepository)

    decree_service = providers.Singleton(DecreeService, repository=decree_repository)

    document_repository = providers.Singleton(DocumentRepository)

    document_service = providers.Singleton(DocumentService, repository=document_repository)

    user_repository = providers.Singleton(UserRepository)

    user_service = providers.Singleton(UserService, repository=user_repository)

    document_engine = providers.Singleton(
        DocumentEngine,
        config=config,
        document_service=document_service,
        decree_service=decree_service,
    )

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
