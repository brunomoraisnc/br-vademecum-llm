from pathlib import Path


# Project root path
PATH = Path(".")

# Storage path
PATH_STORAGE = PATH / "storage"

# Vector database parameters
VECTOR_DATABASE_NAME = "vector_db"
VECTOR_DATABASE_PATH = PATH_STORAGE / "interim"

# OpenAI LLM model parameters
LLM_MODEL_NAME = "gpt-3.5-turbo"
LLM_MODEL_TEMPERATURE = 0

# HuggingFace embeddings model parameters
EMBEDDINGS_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDINGS_MODEL_KWARGS = {'device': 'cpu'}
EMBEDDINGS_ENCODE_KWARGS = {'normalize_embeddings': False}
