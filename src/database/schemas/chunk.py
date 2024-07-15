from pymilvus import CollectionSchema, DataType, FieldSchema

from config.config import get_config


def get_chunk_schema(dense_dim: int, sparse_dim: int):
    return CollectionSchema(
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
                dim=dense_dim,
            ),
            FieldSchema(
                name="sparse_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=sparse_dim,
            ),
            FieldSchema(
                name="parent_id",
                dtype=DataType.VARCHAR,
                max_length=32,
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=get_config().CHUNK_TEXT_SIZE,
            ),
            FieldSchema(name="entities", dtype=DataType.JSON),
        ],
        description="Collection for storing chunk embeddings",
        enable_auto_id=False,
    )
