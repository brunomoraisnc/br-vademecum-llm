import json
import pickle

import faiss
from langchain.vectorstores import FAISS


def load_json(filepath, encoding: str = "utf-8"):
    try:
        with open(filepath, encoding=encoding) as f:
            data = f.load()
    except UnicodeDecodeError as ude:
        print(f"File load error: {filepath}\n{ude}\n\n")
    
    data = json.dump(data)
    return data


def load_docs(filepath):
    docs = load_json(filepath)

    sources, contents = [], []

    for doc in docs:
        source, content = doc["url"], doc["content"]

        source.extend(source)
        contents.extend(content)

    return sources, contents


def store_on_faiss(texts: list, embeddings_model, metadatas: list):
    # Initialize FAISS database and insert data on it along with its embeddings and metadata
    store = FAISS.from_texts(texts, embeddings_model, metadatas=metadatas)

    return store


def persist(store, index_path: str, vector_model_path: str):
    # Write index
    faiss.write_index(store.index, index_path)
    
    vector_model_path = "".join([vector_model_path, ".pkl"])

    try:
        with open(vector_model_path, "wb") as f:
            pickle.dump(store, f)
    except Exception as e:
        print("Database not saved")


class VectorDatabaseRetriever:
    def __init__(self, database_filepath, database_name):
        self.store = self.load_store(database_filepath, database_name)
        self.index = self.load_index(database_filepath, database_name)
    
    
    def load_store(database_filepath, database_name):
        store_filepath = database_filepath / (database_name + ".pkl")
        store = pickle.load(store_filepath)

        return store
    

    def load_index(database_filepath, database_name):
        index_filepath = database_filepath / (database_name + ".index")
        index = faiss.read_index(index_filepath)

        return index
