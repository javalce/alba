from pymilvus import CollectionSchema, DataType, FieldSchema

from config.config import get_config


def get_document_schema():
    return CollectionSchema(
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
                max_length=get_config().MAX_DOC_SIZE,
            ),
            # Add a docs vector; Milvus requires at least one vector field in the schema
            # This is not used, just a workaround to satisfy the schema requirements
            FieldSchema(name="docs_vector", dtype=DataType.FLOAT_VECTOR, dim=2),
        ],
        description="Collection for storing text and metadata of each document",
        enable_auto_id=False,
    )
