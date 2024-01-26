# import environment variables
from data.env_variables import AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_MODEL_NAME, \
    AZURE_OPENAI_API_ENDPOINT, OPENAI_API_VERSION, AZURE_OPENAI_API_KEY, \
    HUGGINGFACE_API_TOKEN, LLAMA2_API_TOKEN, OPENAI_API_KEY, NVIDIANGC_API_KEY
from dotenv import load_dotenv

# import software general purpose libs
import os
import psutil
import logging as log

# import langchain debug mode
from langchain.globals import set_debug

# import langchain document loader
from langchain.document_loaders import PyPDFLoader

# import message handlers
from streamlit_chat import message
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# import embedding processing objects
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings

# import vector database
from langchain.vectorstores.chroma import Chroma

# import data retrieval chain
from langchain.chains import RetrievalQAWithSourcesChain

# import langchain models from huggingface
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

# import langchain models
from langchain.llms.gpt4all import GPT4All
from langchain.chat_models import AzureChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# import hugging face transformers lib - only for quantized models
# import transformers
# from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline

# import streamlit web framework
import streamlit as st


# start debugging
set_debug(True)

# start logging
log.basicConfig(filename="logs/app.log", level=log.DEBUG)


N_THREADS = psutil.cpu_count()


def load_vector_database():
    log.info("Initializing Vector DB")
    sentence_transformer_ef = SentenceTransformerEmbeddings(model_name="intfloat/multilingual-e5-large", model_kwargs = {'device': 'cpu'})
    
    st.session_state.vectordb = Chroma(persist_directory="./documents_cache/qa_retrieval", embedding_function=sentence_transformer_ef)

def get_local_gpt4all_models():
    local_models = {}
    local_models["ggml-gpt4all-j-v1.3-groovy"] = "./model_cache/ggml-gpt4all-j-v1.3-groovy.bin"
    local_models["mistral-7b-openorca.Q4_0"] = "./model_cache/mistral-7b-openorca.Q4_0.gguf"
    # local_models["ggml-mpt-7b-instruct"] = "./model_cache/ggml-mpt-7b-instruct.bin"
    # local_models["ggml-gpt4all-l13b-snoozy"] = "./model_cache/ggml-gpt4all-l13b-snoozy.bin"
    # local_models["ggml-v3-13b-hermes-q5_1"] = "./model_cache/ggml-v3-13b-hermes-q5_1.bin"
    # local_models["ggml-vicuna-13b-1.1-q4_2"] = "./model_cache/ggml-vicuna-13b-1.1-q4_2.bin"
    
    return local_models


def get_llm_instance(model_interface: str):
    if model_interface == "azure":
        llm_instance = AzureChatOpenAI(
            deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
            model_name=AZURE_OPENAI_MODEL_NAME,
            azure_endpoint=AZURE_OPENAI_API_ENDPOINT,
            openai_api_version=OPENAI_API_VERSION,
            openai_api_key=AZURE_OPENAI_API_KEY,
            openai_api_type="azure"
        )
    elif model_interface == "openai":
        llm_instance = ChatOpenAI(
            temperature=0.1,
            openai_api_key=""
        )
    elif model_interface == "gpt4all":
        local_models = get_local_gpt4all_models()
        callbacks = [StreamingStdOutCallbackHandler()]

        llm_instance = GPT4All(
            # model=local_models["mistral-7b-openorca.Q4_0"],
            model="model_cache/zephyr-7b-beta.Q3_K_S.gguf",
            # allow_download=True,
            callbacks=callbacks,
            verbose=True,
            # device="gpu",
            device="nvidia",
            # n_threads=16,
            # n_threads=N_THREADS,
        )
    elif model_interface == "huggingface-falcon":
        llm_instance = HuggingFaceHub(
            verbose=True,
            task="text-generation",
            repo_id="tiiuae/falcon-40b-instruct"
        )
    elif model_interface == "huggingface-mistral-7b":
        llm_instance = HuggingFacePipeline.from_model_id(
            # model_id="mistralai/Mistral-7B-Instruct-v0.1",
            model_id="Open-Orca/Mistral-7B-OpenOrca",
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 10},
            device=0
        )
    elif model_interface == "huggingface-endpoint-zephyr-7b":
        endpoint_url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
        headers = {"Authorization": "Bearer "}
        llm_instance = HuggingFaceEndpoint(
            endpoint_url=endpoint_url,
            task="text-generation",
            huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
        )
    elif model_interface == "zephyr-7b-beta":
        llm_instance = HuggingFacePipeline.from_model_id(
            model_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            # pipeline_kwargs={"max_new_tokens": 10},
            device=0
        )
    elif model_interface == "huggingface-api-llama2":
        llm_instance = HuggingFacePipeline.from_model_id(
            model_id="meta-llama/Llama-2-7b-chat-hf",
            task="text-generation",
            device="cuda",
            pipeline_kwargs={
                "token": LLAMA2_API_TOKEN
            }
        )
    elif model_interface == "nvidia-mixtral":
        callbacks = [StreamingStdOutCallbackHandler()]
        llm_instance = ChatNVIDIA(
            model="mixtral_8x7b",
            nvidia_api_key=NVIDIANGC_API_KEY,
            callbacks=callbacks,
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            seed=42
        )
    
    return llm_instance


def initialize_conversation_chain():
    vectordb = st.session_state.vectordb
    callbacks = [StreamingStdOutCallbackHandler()]
    local_models = get_local_gpt4all_models()
    
    retriever_instance = vectordb.as_retriever(search_kwargs={'k':4})

    # llm_instance = get_llm_instance("huggingface-endpoint-zephyr-7b")
    llm_instance = get_llm_instance("nvidia-mixtral")
    # llm_instance = get_llm_instance("gpt4all")

    log.info("Inicializando")
    st.session_state.qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm_instance,
        chain_type="stuff",
        retriever=retriever_instance
    )


def handle_user_input(user_question, response_container):
    if user_question is None:
        return
    
    qa_chain:RetrievalQAWithSourcesChain = st.session_state.qa_chain
    
    response_container.empty()
    # Handle user Queries
    with response_container.container():
        with st.spinner("Gerando resposta..."):
            log.info(f"Gerando resposta para consulta do cliente: {user_question}")
            response = qa_chain({"question":user_question}, return_only_outputs=True)
        
            # st.write(response)                
            st.write(response["answer"])
            
            with st.expander(label="Sources", expanded=False):
                for source in response["sources"]:
                    st.write(source)


def process_new_uploads(pdf_docs):
    vectordb:Chroma = st.session_state.vectordb
    for doc in pdf_docs:
        log.info(f"Processa arquivo: {doc.name}")
        
        with open(os.path.join("tmp_documents",doc.name),"wb") as f:
            f.write(doc.getbuffer())
        
        loader = PyPDFLoader(file_path=f"./tmp_documents/{doc.name}")        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=640, chunk_overlap=128)
        
        log.info("Particiona texto")
        text_chunks = text_splitter.split_documents(loader.load())
        # log.info("Chunks: %s", text_chunks)
        
        log.info("Processa embeddings e adiciona documento ao Vector DB")
        vectordb.add_documents(documents=text_chunks)
        vectordb.persist()

        os.remove(f"./tmp_documents/{doc.name}")
        log.info(f"Arquivo processado com sucesso: {doc.name}")
    

def main():
    load_dotenv()
    st.set_page_config(page_title="Converse com seus documentos", page_icon=":books:")
    
    st.header("Converse com seus documentos :books:")
    
    if "vectordb" not in st.session_state:
        with st.spinner("Inicializando Vector DB..."):
            load_vector_database()
    
    if "qa_chain" not in st.session_state:
        with st.spinner("Inicializando AI Model..."):
            initialize_conversation_chain()
    
    user_question = st.text_input("Faça sua pergunta aqui")
    
    response_container = st.empty()
    
    if user_question:
        handle_user_input(user_question, response_container)
        user_question = None

    with st.sidebar:
        st.subheader("Seus documentos")
        pdf_docs = st.file_uploader(
            "Insira seus PDFs aqui e clique em 'Processar'",
            accept_multiple_files=True
        )
        if st.button("Processar"):
            with st.spinner("Processando..."):
                process_new_uploads(pdf_docs)


if __name__ == "__main__":
    main()
