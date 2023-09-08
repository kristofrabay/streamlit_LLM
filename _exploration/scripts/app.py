import streamlit as st

import credentials
import os
os.environ["OPENAI_API_KEY"] = credentials.openai_api

from copy import deepcopy
import numpy as np

import plotly.express as px

import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")

from utils import (
    load_pdf,
    load_HTML,
    load_docx,
    load_txt,
    QA,
    Summarization, 
    SimpleChat,
    create_db
)



st.title("🦜🔗 Integrate your Enterprise Knowledge with OpenAI")
st.image(
            "https://uploads-ssl.webflow.com/6373bd91f9c91ce6e5c834bd/641dc6fb789c517f713b1f8b_shutterstock_2253228203_.jpg",
            width=600, # Manually Adjust the width of the image as per requirement
        )
st.sidebar.title("Instructions")
st.sidebar.markdown(
    """
    This is a web application that allows your documents to interact with OpenAI's [`gpt`](https://platform.openai.com/docs/models/gpt-3-5) models.\n
    You may select (1) Summarization and (2) Question-Answering.\n
    Upon Summarization you have to select a file, which will be summarized by ChatGPT.
    Upon QA, you enter a **query** in the **text box** and **press enter** to receive a **response** from the ChatGPT
    """
)

MODEL = st.sidebar.selectbox('Choose your model:', options = ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4'])
TEMPERATURE = st.sidebar.select_slider('Set your temperature', [str(round(i, 2)) for i in np.linspace(0.0, 1.0, 101)], ) 
TEMPERATURE = float(TEMPERATURE)

st.write("Author: [Kristof Rabay](https://github.com/kristofrabay)")

st.header('⛱️Playground')

#### UPLOAD DOCS #####

DOCUMENTS_TO_CHOOSE_FROM = []
docs = []

uploaded_files = st.file_uploader(
    """
    Upload your files! You may include PDFs, txt, docx and HTML files. Additional extension-support to come later 😎

    Be aware that only .pdf supports page citation, so docx, HTML, etc... will not be able to cite where the information is included in the original file.
    """, 
                     type = ['pdf', 'html', 'txt', 'docx'], accept_multiple_files=True)
    
if uploaded_files:
    
    for uploaded_file in uploaded_files:

        filename = uploaded_file.name
        DOCUMENTS_TO_CHOOSE_FROM.append(filename)

        if uploaded_file.name.endswith(".pdf"):
            
            pdf_doc_chunks = load_pdf(uploaded_file, filename = filename)
            docs.extend(pdf_doc_chunks)
        
        elif uploaded_file.name.endswith('.txt'):

            txt_doc_chunks = load_txt(uploaded_file, filename = filename)
            docs.extend(txt_doc_chunks)

        elif uploaded_file.name.endswith('.docx'):

            docx_doc_chunks = load_docx(uploaded_file, filename = filename)
            docs.extend(docx_doc_chunks)

        elif uploaded_file.name.endswith('.html'):

            html_doc_chunks = load_HTML(uploaded_file, filename = filename)
            docs.extend(html_doc_chunks)


    docs_original = deepcopy(docs)



    #### STORE DOCS IN VECTOR DATABASE
    embeddings, db = create_db(docs)

#### END OF UPLOAD PART ####

SIMPLE_CHATTING_OPTIONS = ['I simply want to chat with OpenAI 😊', 'I want to interact with my documents 📄']
SIMPLE_CHATTING_OPTION = st.radio('What do you want to do?', SIMPLE_CHATTING_OPTIONS, horizontal=True)


if SIMPLE_CHATTING_OPTION == 'I simply want to chat with OpenAI 😊':
    SimpleChat(model = MODEL, temperature = TEMPERATURE)

else:

    if len(DOCUMENTS_TO_CHOOSE_FROM) == 0:
        st.write('Upload your documents!')

    else:

        #### CHOOSE USE CASE

        USECASES = ['Question-Answering', 'Summarization']
        USE_CASE = st.radio('Select your use-case', USECASES, horizontal=True)


        if USE_CASE == 'Question-Answering':
            QA(db = db, embeddings = embeddings, model = MODEL, temperature = TEMPERATURE)
        else:
            DOCUMENT = st.selectbox('Select the document to summarize', DOCUMENTS_TO_CHOOSE_FROM )
            Summarization(input_doc = DOCUMENT, model = MODEL, temperature = TEMPERATURE, 
                          docs_original=docs_original, tiktoken_encoding=encoding)
            

### PRICING CHART ###

st.header('')
st.header('💲Pricing calculator')

PRICING = {'gpt-3.5-turbo' : {'input_tokens' : 0.0015, 'output_tokens' : 0.002},
           'gpt-3.5-turbo-16k' : {'input_tokens' : 0.003, 'output_tokens' : 0.004},
           'gpt-4' : {'input_tokens' : 0.03, 'output_tokens' : 0.06},}

my_expander = st.expander(label='Set your parameters (file count, word count, etc...)')
with my_expander:
    cols=st.columns(3)
    with cols[0]:
        NUM_OF_FILES = st.slider('Number of files', min_value = 1, max_value = 250, value = 100)
        PAGES_PER_FILE = st.slider('Number of pages / file', min_value = 1, max_value = 100, value = 10)
    with cols[1]:
        WORDS_PER_PAGE = st.slider('Number of word / page', min_value = 250, max_value = 750, value = 500)
        TOKEN_TO_WORDS = st.slider('Words / 100 tokens', min_value = 50, max_value = 100, value = 75)
        #TOKEN_TO_WORDS = st.number_input('Words / token ratio', min_value = 0.0, max_value = 1.0, value = 0.75)

    with cols[2]:
        AVG_PROMPT_MESSAGE = st.slider('# of prompt system message tokens', min_value = 10, max_value = 500, value = 250)
        AVG_COMPLETION_LENGTH = st.slider('# of expected completion tokens', min_value = 10, max_value = 1000, value = 300)


TOTAL_INPUT_TOKENS = NUM_OF_FILES * PAGES_PER_FILE * (WORDS_PER_PAGE / (TOKEN_TO_WORDS/100)) + AVG_PROMPT_MESSAGE
TOTAL_OUTPUT_TOKENS = AVG_COMPLETION_LENGTH

plot_dict = {'Models': PRICING.keys(), 
             'Input price ($)': [round(i['input_tokens'] * TOTAL_INPUT_TOKENS / 1000, 1) for i in PRICING.values()], 
             'Output price ($)': [round(i['output_tokens'] * TOTAL_OUTPUT_TOKENS / 1000, 1) for i in PRICING.values()]}

st.write(f'Total number input words: {NUM_OF_FILES * PAGES_PER_FILE * WORDS_PER_PAGE:,d}')


fig = px.bar(plot_dict, y="Models", 
             x=["Input price ($)", "Output price ($)"], 
             title="Expected cost ($) associated with `OpenAI` models", 
             text_auto=True, log_x=True,# template = 'plotly_white',
             height = 400, width = 800, labels = {'value' : 'Price in USD'})
fig.update_layout(legend_title_text='Token category', 
                  legend = {'orientation' : 'h', 'xanchor' : 'right', 'yanchor' : 'bottom', 'y' : -0.5, 'x' : 1})


st.plotly_chart(fig, use_container_width=True, theme = 'plotly_white')