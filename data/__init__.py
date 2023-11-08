from data import env_variables
from data.transform import split_text


class DataObject:
    def __init__(self, metadatas, contents, sources):
        self.metadatas = metadatas
        self.contents = contents
        self.sources = sources
    
    
    def split_contents(self, chunk_size: int = 1500, sep: str = "\n"):
        docs, metadatas = [], []

        for i, data in enumerate(self.content):
            splits = split_text(data,chunk_size=chunk_size, sep=sep)
            docs.extend(splits)
            metadatas.extend([{"source": self.sources[i]}] * len(splits))
        
        return splits, docs, metadatas
    
    def vectorize():
        pass
    

    def store():
        pass
    

