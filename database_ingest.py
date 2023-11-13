"""This is the logic for ingesting Notion data into LangChain."""
import faiss
from langchain.vectorstores import FAISS

from data.vectorize import get_embeddings_model
from data.env_variables import PATH_STORAGE, VECTOR_DATABASE_NAME
from data import store

# from langchain.embeddings import OpenAIEmbeddings
# from data.transform import DataObject


def main():
    # Load docs
    docs_filepath = PATH_STORAGE / "raw"
    sources, contents = store.load_docs(filepath=docs_filepath)

    # Initialize embeddings model
    embeddings_model = get_embeddings_model()

    # Store data on vector database
    database = store.store_on_faiss(
        texts=contents,
        embeddings_model=embeddings_model,
        metadatas=sources
    )

    # Persist vector database
    store.persist(
        database,
        index_path=VECTOR_DATABASE_NAME,
        vector_model_path=VECTOR_DATABASE_NAME
    )


if __name__ == "__main__":
    main()
