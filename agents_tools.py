from langchain.agents import load_tools #https://langchain.readthedocs.io/en/latest/reference/integrations.html
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
import streamlit as st

def app_2():

    html_text="""<div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">General Question Trivia</h2></div>   
    """
    st.markdown(html_text,unsafe_allow_html=True)

        # llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0, "max_length":256},
    #                      huggingfacehub_api_token=hf_api_key)

    llm = OpenAI(temperature=0,openai_api_key=open_api_key) # https://huggingface.co/settings/tokens 

    tools=load_tools(['serpapi','llm-math'],llm=llm,serpapi_api_key=serpapi_api_key)

    agent=initialize_agent(tools,llm,agent='zero-shot-react-description',verbose=True)

    # agent.run("How many countries are there in the world?")

    input_from_text = st.text_area('Please input your question for general trivia here',value= "")

    entered=st.button("Enter")

    if input_from_text and entered:

        qa=agent.run(input_from_text)
       
        st.write("** Fetched results is listed below :pencil: ** \n\n " + qa)

            



