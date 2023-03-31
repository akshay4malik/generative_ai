from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
#from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pickle,os
import streamlit as st
from langchain.embeddings import (
    SelfHostedEmbeddings,
    SelfHostedHuggingFaceEmbeddings,
    SelfHostedHuggingFaceInstructEmbeddings,
)
import logging

def get_pipeline():
   # Must be inside the function in notebooks
    model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return pipeline("feature-extraction", model=model, tokenizer=tokenizer)


def inference_fn(pipeline, prompt):
    # Return last hidden state of the model
    if isinstance(prompt, list):
        return [emb[0][-1] for emb in pipeline(prompt)]
    return pipeline(prompt)[0][-1]

def embed_doc():

    if len(os.listdir("data"))>0:
        ##load data
        logging.info("Loading the document ...")
        loader = DirectoryLoader('data',glob="**/*.*")
        raw_documents= loader.load()
        print("Total number of documents read :",len(raw_documents))

        logging.info("Loading document Complete... Starting text splitter")
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                       chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)

        logging.info("Text splitting complete... Loading embedding module...")
        # Load Data to vectorstore
        #model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        #embeddings = model.encode(documents)
        # embeddings = SelfHostedEmbeddings(
        #                         model_load_fn=get_pipeline,
        #                             hardware="cpu",
        #                         model_reqs=["./", "torch", "transformers"],
        #                         inference_fn=inference_fn,
        #                     )
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)
        logging.info("Embeddings object created... Start saving vectors to vectorstore")
       # embeddings=HuggingFaceEmbeddings(model_name=st.secrets["model_name"])
        vectorstore = FAISS.from_documents(documents, embeddings)
        logging.info("Vector Store part is now complete... Saving vectors to pickle file")


        # Save vectorstore
        with open("vectorstore.pkl", "wb") as f:
            pickle.dump(vectorstore, f)
        logging.info("Saved vector to FAISS DB successfully...")
        
