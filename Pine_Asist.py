import streamlit as st
import pinecone
import os
from utils import *
#from langchain.vectorstores import Pinecone
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from pinecone import ServerlessSpec
from pinecone import Pinecone, ServerlessSpec
#from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
import getpass

    






FILE_LIST = "archivos.txt"
#OPENAI_API_KEY = "Añadir OpenAI API Key"

spec = ServerlessSpec(cloud='aws', region='us-east-1')

st.write("Ingresa API k pinecone")
ke = st.text_input('Ingresa tu Clave',key=1)
st.write("Ingresa API k OpenAI")
ke2 = st.text_input('Ingresa tu Clave',key=2)
os.environ["OPENAI_API_KEY"] =ke2

if ke and ke2:
    #st.set_page_config('preguntaDOC')
    pc = Pinecone(api_key=ke)
    index = pc.Index('langchain-test-index')
    st.header("Pregunta a tu PDF")
    
    with st.sidebar:
        archivos = load_name_files(FILE_LIST)
        files_uploaded = st.file_uploader(
            "Carga tu archivo",
            type="pdf",
            accept_multiple_files=True
            )
        
        if st.button('Procesar'):
            for pdf in files_uploaded:
                if pdf is not None and pdf.name not in archivos:
                    archivos.append(pdf.name)
                    text_to_pinecone(pdf)
    
            archivos = save_name_files(FILE_LIST, archivos)
    
        if len(archivos)>0:
            st.write('Archivos Cargados:')
            lista_documentos = st.empty()
            with lista_documentos.container():
                for arch in archivos:
                    st.write(arch)
                if st.button('Borrar Documentos'):
                    archivos = []
                    clean_files(FILE_LIST)
                    lista_documentos.empty()
    
    
    if len(archivos)>0:
        user_question = st.text_input("Pregunta: ")
        if user_question:
            os.environ["OPENAI_API_KEY"] = ke2
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            
            vstore = Pinecone.from_existing_index("langchain-test-index", embeddings)
    
            docs = vstore.similarity_search(user_question, 3)
            llm = ChatOpenAI(model_name='gpt-4o-mini')
            chain = load_qa_chain(llm, chain_type="stuff")
            respuesta = chain.run(input_documents=docs, question=user_question)
    
            st.write(respuesta)
