import docx2txt  
import streamlit as st  
from langchain_huggingface import HuggingFaceEmbeddings  
from llama_index.llms.ollama import Ollama  
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 

st.set_page_config(page_title = "IA Contratos", page_icon = ":100:", layout = "centered")  
st.title("Leitor de Contratos")  
st.title("App Assistente de Contratos") 

if "messages" not in st.session_state.keys():  
    st.session_state.messages = [
        {"role": "assistant", "content": "Digite sua pergunta"}  
    ]

llm = Ollama(model = "deepseek-r1", request_timeout = 600.0)  
embed_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")  

@st.cache_resource(show_spinner = False)
def rmta_modulo_rag():
    with st.spinner(text = "Carregando e indexando os documentos Streamlit – aguarde! Isso deve levar de 1 a 2 minutos."):  
        reader = SimpleDirectoryReader(input_dir = "./documentos", recursive = True)  
        docs = reader.load_data()  
        Settings.llm = llm  
        Settings.embed_model = embed_model  
        index = VectorStoreIndex.from_documents(docs)  
        return index  

index = rmta_modulo_rag()  

if "chat_engine" not in st.session_state.keys():  
    st.session_state.chat_engine = index.as_chat_engine(chat_mode = "condense_question", verbose = True)  

if prompt := st.chat_input("Sua pergunta"):  
    st.session_state.messages.append({"role": "user", "content": prompt})  

for message in st.session_state.messages: 
    with st.chat_message(message["role"]):  
        st.write(message["content"])  

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Consultando..."):
            user_message = st.session_state.messages[-1]["content"]
            contextual_prompt = f"Você é um assistente jurídico especializado. O usuário fez a seguinte pergunta: '{user_message}'. Considere todos os documentos jurídicos disponíveis e forneça uma resposta detalhada e precisa."
            response = st.session_state.chat_engine.chat(contextual_prompt)
            st.write(response.response)
            st.session_state.messages.append({"role": "assistant", "content": response.response})
