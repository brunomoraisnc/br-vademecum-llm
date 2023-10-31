from pathlib import Path


# Project root path
PATH = Path(".")

# Storage path
PATH_STORAGE = PATH / "storage"

VECTOR_DATABASE_NAME = "vector_db"
VECTOR_DATABASE_PATH = PATH_STORAGE / "interim"

LLM_MODEL_NAME = "gpt-3.5-turbo"
LLM_MODEL_TEMPERATURE = 0