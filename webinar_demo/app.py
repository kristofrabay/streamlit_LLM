import streamlit as st

#import credentials
import os
import openai
#os.environ["OPENAI_API_KEY"] = credentials.openai_api

with st.sidebar:
    openai.api_key = st.text_input('OpenAI API Key', type = 'password', key = 'openai_key')


from copy import deepcopy
import numpy as np

import plotly.express as px

import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")

from utils import (
    load_pdf,
    load_docx,
    QA_st,
    Instruction_st,
    InformationExtraction_st,
    Summarization_st,
    concat_docs_count_tokens,
    MODEL_COST,
    MODEL_INPUT_TOKEN_SUMM_LIMIT
)



st.title("🦜🔗 Integrate your Enterprise Knowledge with OpenAI")
st.write("Let's interact with our documents📄")
st.image(
            #"https://uploads-ssl.webflow.com/6373bd91f9c91ce6e5c834bd/641dc6fb789c517f713b1f8b_shutterstock_2253228203_.jpg",
            "https://media.licdn.com/dms/image/D5622AQHQNzqXBdLfMw/feedshare-shrink_1280/0/1686842979161?e=1689811200&v=beta&t=sqQ1FCtHwHWit2BOXBnk9GzZr20FfptFYRAY3F8iyQc",
            width=500, # Manually Adjust the width of the image as per requirement
        )

st.sidebar.title("Instructions")
st.sidebar.markdown(
    """
    This is a web application that allows your documents to interact with OpenAI's [`gpt`](https://platform.openai.com/docs/models/gpt-3-5) models.\n
    You may select 
    1. QA, 
    2. Instruction, 
    3. Information Extraction and 
    4. Summarization.\n
    """
)

MODEL = st.sidebar.selectbox('Choose your model:', options = ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4'])
TEMPERATURE = st.sidebar.select_slider('Set your temperature', [str(round(i, 2)) for i in np.linspace(0.0, 1.0, 101)], ) 
TEMPERATURE = float(TEMPERATURE)

st.write("Author: [Kristof Rabay](https://github.com/kristofrabay)")

st.header('⛱️Playground')

#### UPLOAD DOCS #####

docs = []

uploaded_file = st.file_uploader(
    """
    Upload your file! You may include PDF or docx😎

    Be aware that only .pdf supports page citation.
    """, 
                     type = ['pdf', 'docx'], accept_multiple_files=False)
    
if uploaded_file:
    
    filename = uploaded_file.name

    if uploaded_file.name.endswith(".pdf"):
        
        pdf_doc_chunks = load_pdf(uploaded_file, filename = filename)
        docs.extend(pdf_doc_chunks)

    elif uploaded_file.name.endswith('.docx'):

        docx_doc_chunks = load_docx(uploaded_file, filename = filename)
        docs.extend(docx_doc_chunks)


WHOLE_DOC, input_tokens = concat_docs_count_tokens(docs, encoding)


#### STORE DOCS IN VECTOR DATABASE
#embeddings, db = create_db(docs)

#### END OF UPLOAD PART ####

if len(docs) == 0:
    st.write('Upload your document!')

else:

    #### CHOOSE USE CASE
    if len(input_tokens) <= MODEL_INPUT_TOKEN_SUMM_LIMIT[MODEL]:

        st.write('Number of input tokens (' + str(len(input_tokens)) + ') can fit into prompt!')
        st.write('💰💰💰 Approximate cost of processing, not including prompt message and completion:', str(round(MODEL_COST[MODEL] * len(input_tokens) / 1000, 5)), 'USD')

        USECASES = ['QA', 'Instruction', 'Information-extraction', 'Summarization']
        USE_CASE = st.radio('Select your use-case', USECASES, horizontal=True)


        if USE_CASE == 'QA':
            QA_st(context = WHOLE_DOC, model = MODEL, temperature = TEMPERATURE)
        
        elif USE_CASE == 'Instruction':
            Instruction_st(context = WHOLE_DOC, model = MODEL, temperature = TEMPERATURE)
        
        elif USE_CASE == 'Information-extraction':
            InformationExtraction_st(context = WHOLE_DOC, model = MODEL, temperature = TEMPERATURE)
        
        else:        
            Summarization_st(context = WHOLE_DOC, model = MODEL, temperature = TEMPERATURE)
        

    else:
        st.write('Number of input tokens is large (' + str(len(input_tokens)) + ') choose a model with a larger context')  
    
    
    