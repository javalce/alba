from functools import lru_cache
from pathlib import Path
from typing import Union

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Retrieve the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Config(BaseSettings):
    """
    A class for managing configuration settings.

    This class loads the configuration from a JSON file and provides a method
    to retrieve configuration values by key.
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    DB_URL: str = Field(default=f"sqlite:///{BASE_DIR}/db.sqlite")
    MODEL_API_URL: str = Field(default="http://localhost:11434/api/generate")
    TEMPLATES_PATH: str = Field(default="src/alba/templates/template.json")
    DEFAULT_INFERENCE_MODEL: str = Field(default="llama3")
    INFERENCE_MODEL: Union[str, None] = Field(default=None)
    DENSE_EMBED_FUNC_DIM: int = Field(default=1024)
    SPARSE_EMBED_FUNC_PATH: str = Field(default="models/bm25_model.pkl")
    DB_COLLECTION: str = Field(default="rag")
    DB_PATH: str = Field(default="chatbot/db")
    MILVUS_HOST: str = Field(default="127.0.0.1")
    MILVUS_PORT: int = Field(default=19530)
    LOCATIONS_FILE: str = Field(default="config/locations.json")
    BATCH_SIZE: int = Field(default=100)
    DOC_SIZE: Union[int, None] = Field(default=None)
    MAX_DOC_SIZE: int = Field(default=8192)
    CHUNK_SIZE: int = Field(default=500)
    CHUNK_OVERLAP: int = Field(default=100)
    CHUNK_TEXT_SIZE: int = Field(default=1000)
    LOG_PATH: str = Field(default="logs/log.log")
    FALSE_POSITIVES_FILE: str = Field(default="config/false_positives.txt")


@lru_cache
def get_config():
    return Config()
