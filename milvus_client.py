from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from config import MILVUS_URI, MILVUS_TOKEN, MILVUS_COLLECTION, EMBEDDING_DIM

_collection: Collection | None = None


def connect_milvus():
    connections.connect(
        alias="default",
        uri=MILVUS_URI,
        token=MILVUS_TOKEN,
    )


def get_collection() -> Collection:
    global _collection
    if _collection is None:
        _collection = Collection(MILVUS_COLLECTION)
    return _collection


def init_milvus_collection():
    if utility.has_collection(MILVUS_COLLECTION):
        return

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="kb_id", dtype=DataType.INT64),
        FieldSchema(name="doc_id", dtype=DataType.INT64),
        FieldSchema(name="chunk_id", dtype=DataType.INT64),
    ]
    schema = CollectionSchema(fields, description="Document vectors collection")
    col = Collection(MILVUS_COLLECTION, schema)

    dense_index = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 1024},
    }
    col.create_index("dense_vector", dense_index)

    sparse_index = {
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "IP",
    }
    col.create_index("sparse_vector", sparse_index)

    col.load()


def reinit_collection():
    """Drop and recreate the Milvus collection. All existing vectors will be lost."""
    global _collection
    if utility.has_collection(MILVUS_COLLECTION):
        utility.drop_collection(MILVUS_COLLECTION)
    _collection = None
    init_milvus_collection()
    return get_collection()


def disconnect_milvus():
    connections.disconnect("default")
