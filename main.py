from data.env_variables import VECTOR_DATABASE_NAME, VECTOR_DATABASE_PATH, LLM_MODEL_NAME, LLM_MODEL_TEMPERATURE
from data.store import VectorDatabaseRetriever

import pickle

import streamlit as st
from streamlit_chat import message

from langchain.llms.openai import OpenAI
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
# from langchain.chains import VectorDBQAWithSourcesChain


def get_text():
    input_text = st.text_input("Você: ", "Olá, como você está?", key="input")
    return input_text


def main():

    # Load the LangChain.
    vector_db = VectorDatabaseRetriever(
        database_filepath=VECTOR_DATABASE_PATH,
        database_name=VECTOR_DATABASE_NAME
    )

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=OpenAI(temperature=0),
        vectorstore=vector_db.store
    )

    st.set_page_config(page_title="BR Vademecum GPT", page_icon=":judge:")
    st.header("BR Vademecum GPT")

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []


    user_input = get_text()

    if user_input:
        # result = chain({"question": user_input})
        # output = f"Answer: {result['answer']}\nSources: {result['sources']}"
        result = chain({"question": user_input})
        output = f"Resposta: {result['answer']}\nReferências: {result['sources']}"

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")


if __name__ == "__main__":
    main()
