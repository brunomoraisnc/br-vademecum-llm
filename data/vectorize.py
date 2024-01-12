from data.env_variables import EMBEDDINGS_ENCODE_KWARGS, EMBEDDINGS_MODEL_KWARGS, EMBEDDINGS_MODEL_NAME
from langchain.embeddings import HuggingFaceBgeEmbeddings


def get_embeddings_model():
    hf_embedding_model = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDINGS_MODEL_NAME,
        model_kwargs=EMBEDDINGS_MODEL_KWARGS,
        encode_kwargs=EMBEDDINGS_ENCODE_KWARGS
    )

    return hf_embedding_model
