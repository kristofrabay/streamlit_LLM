# packages
import streamlit as st
from audio_recorder_streamlit import audio_recorder

from gtts import gTTS
from io import BytesIO

import numpy as np
from copy import deepcopy

#from credentials import openai_api
import os
import openai
import assemblyai as aai

st.sidebar.image("https://workable-application-form.s3.amazonaws.com/advanced/production/60e736cd0d17a4bf36cee848/2042faad-2e15-4eae-bac0-78a91c23c4c3", use_column_width=True)
st.sidebar.image("https://ww1.freelogovectors.net/wp-content/uploads/2023/01/openai-logo-freelogovectors.net_.png", use_column_width=True)

with st.sidebar:

    openai_api = st.text_input('OpenAI API Key', type = 'password', key = 'openai_key')
    openai.api_key = openai_api
    os.environ["OPENAI_API_KEY"] = openai_api

    assemblyai_api = st.text_input('AssemblyAI API Key', type = 'password', key = 'assemblyai_api')
    aai.settings.api_key = assemblyai_api


import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")

from utils import (
    load_pdf,
    load_HTML,
    load_docx,
    load_txt,
    create_db,
    concat_docs_count_tokens
)

# openai models, settings
embedder = 'text-embedding-ada-002'

MODEL_RELEVANT_DOC_NUMBER = {'gpt-3.5-turbo' : 3,
                            'gpt-3.5-turbo-16k' : 5,
                            'gpt-4' : 5}

MODEL_INPUT_TOKEN_SUMM_LIMIT = {'gpt-3.5-turbo' : 3200,
                                'gpt-3.5-turbo-16k' : 14200,
                                'gpt-4' : 7200}

MODEL_COST = {'gpt-3.5-turbo' : 0.0015,
              'gpt-3.5-turbo-16k' : 0.003,
              'gpt-4' : 0.03}


MAX_CONTEXT_QUESTIONS = {'gpt-3.5-turbo' : 10,
                        'gpt-3.5-turbo-16k' : 40,
                        'gpt-4' : 20}


# functions, prompts
def generate_embeddings(text):
    response = openai.Embedding.create(input=text, model = embedder)
    embeddings = response['data'][0]['embedding']
    return embeddings

def generate_response(messages, MODEL, TEMPERATURE, MAX_TOKENS):
    completion = openai.ChatCompletion.create(
        model=MODEL, 
        messages=messages, 
        temperature=TEMPERATURE, 
        max_tokens=MAX_TOKENS)
    return completion.choices[0]['message']['content']

def retrieve_relevant_chunks(user_input, db, model):

    query_embedded = generate_embeddings(user_input)

    sim_docs = db.max_marginal_relevance_search_by_vector(query_embedded, k = MODEL_RELEVANT_DOC_NUMBER[model])
    results = [doc.metadata['source'].split("\\")[-1] + "-page-" + str(doc.metadata['page'] )+ ": " + doc.page_content.replace("\n", "").replace("\r", "") for doc in sim_docs]
    sources = "\n".join(results)

    return sources

def TTS(text):

    sound_file = BytesIO()
    tts = gTTS(text, lang='en')
    tts.write_to_fp(sound_file)

    return sound_file

#### TEMPORARY DUPLICATIONS
# documents are out of scope for now, simply chat with ChatGPT

default_system_prompt = """Act as an assistant that helps people with their questions relating to a wide variety of documents. 
Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question. 
Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brakets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].
If you did not use the information below to answer the question, do not include the source name or any square brackets."""

default_system_prompt = """Act as an assistant that helps people with their questions.
If you don't know the answer, do not make one up, ask clarifying questions."""

system_message = """{system_prompt}

Sources:
{sources}

"""

system_message = """{system_prompt}"""

question_message = """
Question: {question}

Answer: 
"""

question_message = """{question}"""


# streamlit app
st.title("Voice-Powered OpenAI Assistant")
st.header("Talk (literally) to Generative AI")
st.write("Author: [Kristof Rabay](https://github.com/kristofrabay)")
#st.sidebar.image("https://hiflylabs.com/_next/static/media/greenOnDark.35e68199.svg", use_column_width=True)

st.sidebar.title("Instructions")
st.sidebar.markdown(
    """
    You may set the following settings\n

    1. OpenAI model selection
        - gpt-3.5-turbo
        - gpt-3.5-turbo-16k
        - gpt-4

    1. Prompt parameters
        - System message
        - max_tokens
        - temperature"""
)

MODEL = st.radio('Select the OpenAI model you want to use', 
                 ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4'], horizontal=True)

prompt_expander = st.expander(label='Set your Prompt settings')
with prompt_expander:
    cols=st.columns(2)
    with cols[0]:
        SYSTEM_MESSAGE = st.text_area('Set a system message', value = default_system_prompt, height = 400)
    with cols[1]:
        TEMPERATURE = float(st.select_slider('Set your temperature', [str(round(i, 2)) for i in np.linspace(0.0, 2, 101)], value = '1.0')) 
        MAX_TOKENS = st.slider('Number of max output tokens', min_value = 1, max_value = 1500, value = 1024)



#### UPLOAD DOCS #####

#DOCUMENTS_TO_CHOOSE_FROM = []
#docs = []

#uploaded_files = st.file_uploader("Upload your files! You may include PDFs, txt, docx and HTML files 😎 \n Be aware that only .pdf supports page citation, so docx, HTML, etc... will not be able to cite where the information is included in the original file.", 
#                     type = ['pdf', 'html', 'txt', 'docx'], accept_multiple_files=True)

#if uploaded_files:
#
#    if not openai_api:
#        st.warning('🔑🔒 Paste your OpenAI API key on the sidebar 🔑🔒')
#
#    else:
#    
#        for uploaded_file in uploaded_files:
#
#            filename = uploaded_file.name
#            DOCUMENTS_TO_CHOOSE_FROM.append(filename)
#
#            if uploaded_file.name.endswith(".pdf"):
#                
#                pdf_doc_chunks = load_pdf(uploaded_file, filename = filename)
#                docs.extend(pdf_doc_chunks)
#            
#            elif uploaded_file.name.endswith('.txt'):
#
#                txt_doc_chunks = load_txt(uploaded_file, filename = filename)
#                docs.extend(txt_doc_chunks)
#
#            elif uploaded_file.name.endswith('.docx'):
#
#                docx_doc_chunks = load_docx(uploaded_file, filename = filename)
#                docs.extend(docx_doc_chunks)
#
#            elif uploaded_file.name.endswith('.html'):
#
#                html_doc_chunks = load_HTML(uploaded_file, filename = filename)
#                docs.extend(html_doc_chunks)
#
#
#        docs_original = deepcopy(docs)
#
#
#        #### STORE DOCS IN VECTOR DATABASE
#        embeddings, db = create_db(docs)

#### END OF UPLOAD PART ####


if not openai_api:
        st.warning('🔑🔒 Paste your OpenAI API key on the sidebar 🔑🔒')

if not assemblyai_api:
        st.warning('🔑🔒 Paste your AssemblyAI API key on the sidebar 🔑🔒')

#### Clear cache ####

col1, col2 = st.columns([2, 1])

with col1:
    st.caption("""To get rid of chat history and start a new session, please clear cache memory. 
            This is suggested in case of document deletion or addition as well.""")
with col2:
    if st.button("Clear cache"):
        st.cache_data.clear()
        for key in st.session_state.keys():
            del st.session_state[key]


#### end of clear cache

if False:#len(DOCUMENTS_TO_CHOOSE_FROM) == 0:
        st.write('Upload your documents!')

else:
    
    #WHOLE_DOC, input_tokens = concat_docs_count_tokens(docs, encoding)
    #st.write('Number of input tokens: ' + str(len(input_tokens)))
    #st.write('💰 Approx. cost of processing, not including completion:', str(round(MODEL_COST[MODEL] * (len(input_tokens) + 500) / 1000, 5)), 'USD')


    msg = st.chat_message('assistant')
    msg.write("Hello 👋") #Ask me questions about your uploaded documents!

    ### chat elements integration

    # Set a default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = MODEL

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if QUERY := st.chat_input("Enter your question here"):


        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(QUERY)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            #full_response = ""


            #if len(input_tokens) <= MODEL_INPUT_TOKEN_SUMM_LIMIT[MODEL]: # maybe we can fit everything into the prompt, why not
            #    results = [doc.metadata['source'].split("\\")[-1] + "-page-" + str(doc.metadata['page'] )+ ": " + doc.page_content.replace("\n", "").replace("\r", "") for doc in docs]
            #    sources = "\n".join(results)   
            #else:
            #    sources = retrieve_relevant_chunks(QUERY, db, MODEL)

            messages =[
                        {"role": "system", "content" : "You are a helpful assistant helping people answer their questions."}, # related to documents.
                        {"role": "user", "content": system_message.format(system_prompt = SYSTEM_MESSAGE)}, #, sources=sources
                        *st.session_state.messages,
                        {"role": "user", "content": question_message.format(question=QUERY)}
                        ]
                        
            full_response = generate_response(messages, MODEL, TEMPERATURE, MAX_TOKENS)

            #st.write(f"{response}")
            #msg.write(response)
            message_placeholder.markdown(full_response)

        # Add user and AI message to chat history
        st.session_state.messages.append({"role": "user", "content": QUERY})
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # to always fit in context, either limit historic messages, or count tokens
        # current solution: if we reach model-specific max msg number, remove first q-a pair

        if len(st.session_state.messages) >= MAX_CONTEXT_QUESTIONS[MODEL] * 2:
            st.session_state.messages = st.session_state.messages[2:]

        #if len(st.session_state.messages) > 0:
             
            #sources_expander = st.expander(label='Check sources identified as relevant')
            #with sources_expander:
            #    #st.write('\n')
            #    if len(input_tokens) <= MODEL_INPUT_TOKEN_SUMM_LIMIT[MODEL]:
            #        st.write('All sources were used within the prompt')
            #    else:
            #        #st.write("Below are the sources that have been identified as relevant:")
            #        st.text(sources)


        col3, col4 = st.columns([2, 1])

        with col3:
            st.write("Play AI's message")
            if full_response:
                sound_file = TTS(full_response)
                st.audio(sound_file, format="audio/wav")
            else:
                sound_file = TTS('Hello')
                st.audio(sound_file, format="audio/wav")

        with col4:
            st.write("Record Human's message")
            audio_bytes = audio_recorder(text = 'Click & record', pause_threshold=2.0, icon_size = "2x",)
            #if audio_bytes:
            #    st.audio(audio_bytes, format="audio/wav")