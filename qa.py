from data.env_variables import VECTOR_DATABASE_NAME, VECTOR_DATABASE_PATH, LLM_MODEL_NAME, LLM_MODEL_TEMPERATURE
from data.store import VectorDatabaseRetriever

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

import argparse


parser = argparse.ArgumentParser(description='Faça uma pergunta')
parser.add_argument('question', type=str, help='Pergunta a ser feita')
args = parser.parse_args()

vector_db = VectorDatabaseRetriever(
    database_filepath=VECTOR_DATABASE_PATH,
    database_name=VECTOR_DATABASE_NAME
)

chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=LLM_MODEL_TEMPERATURE),
    retriever=vector_db.store.as_retriever()
)

result = chain({"question": args.question})

print(f"Resposta: {result['answer']}")
print(f"Referências: {result['sources']}")
