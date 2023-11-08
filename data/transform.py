from langchain.text_splitter import CharacterTextSplitter
from data import DataObject


def split_text(data, chunk_size: str = 1500, sep: str = "\n"):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, separator=sep)
    splits = text_splitter.split_text(data)

    return splits
