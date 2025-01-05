import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
#from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama, ChatOpenAI #Definir o modelo ex: ChatOllama, ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings, OpenAIEmbeddings #Definir o modelo de Embedding ex: OllamaEmbeddings, OpenAIEmbeddings
from openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from pathlib import Path
import os

# Carrega variáveis de ambiente e chaves de acesso.
_ = load_dotenv(find_dotenv())

# Modelo de Linguagem
llm = ChatOpenAI()

#Caso OLLAMA seja utilizado
# É necessário ter o Ollama instalado na sua máquina local, ou no servidor que for utilizar.
# No meu caso, estou usando o servidor da Asimov.
#ollama_server_url = "http://192.168.1.5:11434" 
#model_local = ChatOllama(model="llama3.1:8b-instruct-q4_K_S")


@st.cache_resource #avaliar o uso do st.cache_data, pois não rodou.. mesmo sendo a indicação da Asimov
def load_csv_data():
    try:
        st.write("Carregando dados CSV...")  # Log para depuração
        file_path = "/mount/src/oraculo-sienge/knowledge_base_sienge.csv"
        absolute_path = os.path.abspath(file_path)
        st.write(f"Caminho absoluto do arquivo CSV: {absolute_path}")
        
        if not Path(file_path).is_file():
            st.error(f"Arquivo {file_path} não encontrado.")
            return None
        
        loader = CSVLoader(file_path=file_path)  # Documento csv com os dados específicos, exemplo FAQ da empresa.
        embeddings = OpenAIEmbeddings()  # O Embedding permite transformar textos em números (vetores) e facilitar a localização de documentos semelhantes.
        documents = loader.load()
        vectorstore = FAISS.from_documents(documents, embeddings)  # FAISS é um 'banco de dados vetoriais' que permite guardar os documentos já convertidos em vetores.
        retriever = vectorstore.as_retriever()  # retriever permite que o banco de dados seja utilizado como busca para puxar as informações.
        return retriever
    except Exception as e:
        st.error(f"Erro ao carregar dados CSV: {e}")
        return None

retriever = load_csv_data()
if retriever is None:
    st.stop()

st.title("Oráculo - Sienge Plataforma")

# Configuração do prompt e do modelo - RAG
rag_template = """
Você é um atendente de uma empresa.
Seu trabalho é conversar com os clientes, consultando a base de 
conhecimentos da empresa, e dar 
uma resposta simples e precisa para ele, baseada na 
base de dados da empresa fornecida como 
contexto.

Contexto: {context} 

Pergunta do cliente: {question}
"""
human = "{text}"
prompt = ChatPromptTemplate.from_template(rag_template)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm #Substituir pelo modelo de linguagem que deseja utilizar se for Ollama utilizar a variável model_local, se for ChatGPT, utilizar a variável llm.
)

#Testanto uma conversa no terminal:
#while TRue:
#    user_input = input("Você (Digite sua pergunta):")
#    response = chain.invoke(user_input)
#    print(response.content)

#Utilização no Streamlit

if "messages" not in st.session_state:
    #st.session_state["messages"] = []
    st.session_state.messages = []

# Exibe mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Caixa de entrada para o usuário
if user_input := st.chat_input("Você:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Adiciona um container para a resposta do modelo
    response = chain.invoke(user_input)  # Passe o texto diretamente
    full_response = response.content if hasattr(response, "content") else str(response)
    
    response_container = st.chat_message("assistant")
    response_container.markdown(full_response)

    # Salva a resposta completa no histórico
    st.session_state.messages.append({"role": "assistant", "content": full_response})
