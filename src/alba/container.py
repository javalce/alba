from dependency_injector import containers, providers

from alba.chatbot import Chatbot
from alba.config import Config
from alba.database import Database
from alba.document_engine import DocumentEngine
from alba.memory.long_term_memory import LongTermMemory
from alba.utils.ner_extraction import EntityExtractor


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(packages=["alba.router"])

    config = providers.Configuration(pydantic_settings=[Config])

    ner_extractor = providers.Singleton(EntityExtractor, config=config)

    document_engine = providers.Singleton(DocumentEngine, config=config)

    db = providers.Singleton(
        Database,
        config=config,
        document_engine=document_engine,
        ner_extractor=ner_extractor,
    )

    long_term_memory = providers.Singleton(
        LongTermMemory,
        config=config,
        ner_extractor=ner_extractor,
        db=db,
    )

    chatbot = providers.Singleton(Chatbot, config=config, long_term_mem=long_term_memory)
