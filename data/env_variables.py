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

# OpenAI settings
OPENAI_API_KEY = ""

# Azure OpenAI deployment settings (if needed)
AZURE_OPENAI_DEPLOYMENT_NAME = "azure"
AZURE_OPENAI_MODEL_NAME = "gpt-35-turbo-16k"
AZURE_OPENAI_API_ENDPOINT = "https://<insert-your-domain-here>.openai.azure.com/"
OPENAI_API_VERSION = "2022-12-01"
AZURE_OPENAI_API_KEY = "<insert-your-api-key-here>"

# HuggingFace Endpoint parameters
HUGGINGFACE_API_TOKEN = "<insert-your-hf-api-key-here>"

# LLama endpoint parameters
LLAMA2_API_TOKEN = "<insert-your-llama-api-key-here>"