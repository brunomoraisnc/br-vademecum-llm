from data.env_variables import EMBEDDINGS_ENCODE_KWARGS, EMBEDDINGS_MODEL_KWARGS, EMBEDDINGS_MODEL_NAME
from langchain.embeddings import HuggingFaceBgeEmbeddings, SentenceTransformerEmbeddings


def get_embeddings_model():
    if EMBEDDINGS_MODEL_NAME == "BAAI/bge-base-en-v1.5":
        embedding_model = HuggingFaceBgeEmbeddings(
            model_name=EMBEDDINGS_MODEL_NAME,
            model_kwargs=EMBEDDINGS_MODEL_KWARGS,
            encode_kwargs=EMBEDDINGS_ENCODE_KWARGS
        )
    elif EMBEDDINGS_MODEL_NAME == "intfloat/multilingual-e5-large":
        embedding_model = SentenceTransformerEmbeddings(
            model_name=EMBEDDINGS_MODEL_NAME,
            model_kwargs = EMBEDDINGS_MODEL_KWARGS
        )

    return embedding_model
