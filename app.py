import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.llms import HuggingFaceHub
from query_data import _template,CONDENSE_QUESTION_PROMPT,QA_PROMPT,get_chain
from ingest_data import embed_doc
import pickle
import os
import logging



def app_1():


    html_text="""<div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">File to chat Q/A demo</h2></div>   
    """
    st.markdown(html_text,unsafe_allow_html=True)

    st.subheader('Explore documents using :blue[chatbot] :fire:.Simply upload your documents and start asking questions to :robot_face:')



    ##File operations
    with st.form("my-form", clear_on_submit=True):
            uploaded_files=st.file_uploader("Please upload your document",accept_multiple_files=True,label_visibility="visible")
            submitted = st.form_submit_button("submit")


            if len([uploaded_file for uploaded_file in uploaded_files if uploaded_file.name not in os.listdir("data") ])>0:
                    print("length of list uploaded_files : ",len(uploaded_files))
                        
                    for i in [uploaded_file for uploaded_file in uploaded_files if uploaded_file.name not in os.listdir("data") ]:
                        print("Name of file : ",i.name)
                        with open("data/" + i.name,"wb") as f:
                            f.write(i.getbuffer())
                            st.write("Uploaded File " + i.name + " successfully")

                    
                    with st.spinner("Document is being vectorized.Please be patient..."):

                        embed_doc()
                

    # Loading vector store
    if "vectorstore.pkl" in os.listdir("."):
        with open("vectorstore.pkl",'rb') as f:
            vectorstore=pickle.load(f)
            st.success('Knowledge Base loaded! :white_check_mark:')
        logging.info("Loading chain....")
        chain=get_chain(vectorstore)
        logging.info("Chain Loaded...")

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []
    logging.info("Ready to load user inpput")
    placeholder=st.empty()
    def get_text():
        input_text = st.text_input("You: ", value="", key="input")
        return input_text

    user_input = get_text()
    logging.info("User input loaded...")
    print(st.session_state.input)

    if user_input:
        docs=vectorstore.similarity_search_with_score(user_input)

        

        print("Matched docs percentage",[x[1] for x in docs[:2]])

        output=chain.run(input=user_input,
                        vectorstore=vectorstore,
                        context=docs[:2],
                        chat_history=[],
                        question=user_input,
                        QA_PROMPT=QA_PROMPT,
                        CONDENSE_QUESTION_PROMPT=CONDENSE_QUESTION_PROMPT,
                        template=_template)
        st.session_state.past.append(user_input)

        st.session_state.generated.append(output)

    if st.session_state["generated"]:

        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            #bot output
            message(st.session_state["generated"][i], key=str(i))
            #user output
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")



