from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pickle,os
import streamlit as st

def embed_doc():

    if len(os.listdir("data"))>0:
        ##load data
        loader = DirectoryLoader('data',glob="**/*.*")
        raw_documents= loader.load()
        print("Total number of documents read :",len(raw_documents))


        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                       chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)

        # Load Data to vectorstore
       
        embeddings=HuggingFaceEmbeddings(model_name=model_name)
        vectorstore = FAISS.from_documents(documents, embeddings)


        # Save vectorstore
        with open("vectorstore.pkl", "wb") as f:
            pickle.dump(vectorstore, f)
        
