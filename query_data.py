from langchain.prompts.prompt import PromptTemplate
# from langchain.llms import HuggingFaceHub
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
import streamlit as st

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI assistant for answering questions about the document that you have uploaded.
You are given the following extracted parts of a long document and a question. Provide a conversational answer in below Answer.
Also, classify the context into either one of the categories such as "Finance" or "HR" or "General" in below Topic.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.

Question: {question}
=========
{context}
=========
Answer in Markdown:
Topic in Markdown:
"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])


def get_chain(vectorstore):
    
    # llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0, "max_length":256},
    #                      huggingfacehub_api_token=hf_api_key)
    
    llm = OpenAI(temperature=0,openai_api_key=st.secrets["open_api_key"])  #https://platform.openai.com/account/api-keys

   
    qa_chain = ChatVectorDBChain.from_llm(
        llm,
        vectorstore,
        qa_prompt=QA_PROMPT,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )
    return qa_chain
