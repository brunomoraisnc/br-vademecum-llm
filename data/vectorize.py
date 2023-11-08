from langchain.embeddings import HuggingFaceBgeEmbeddings


# Set HuggingFace embeddings model parameters
EMBEDDINGS_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDINGS_MODEL_KWARGS = {'device': 'cpu'}
EMBEDDINGS_ENCODE_KWARGS = {'normalize_embeddings': False}


def get_embeddings_model():
    hf_embedding_model = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDINGS_MODEL_NAME,
        model_kwargs=EMBEDDINGS_MODEL_KWARGS,
        encode_kwargs=EMBEDDINGS_ENCODE_KWARGS
    )

    return hf_embedding_model
