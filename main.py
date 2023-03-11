import streamlit as st
from app import app_1
from agents_tools import app_2
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():

        ##Page settings and header
    st.set_page_config(page_title="Generative AI Use Cases",layout="wide",page_icon=':robot:')

    tab1, tab2= st.tabs(["File Chat", "General Trivia"])

    with tab1:
        
        app_1()

    with tab2:
        app_2()

if __name__=="__main__":
    main()


   
