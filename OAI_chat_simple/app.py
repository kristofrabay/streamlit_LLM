# packages
import streamlit as st
import numpy as np
from copy import deepcopy

#from credentials import openai_api
import os
import openai

st.sidebar.image("https://ww1.freelogovectors.net/wp-content/uploads/2023/01/openai-logo-freelogovectors.net_.png", use_column_width=True)

with st.sidebar:

    openai_api = st.text_input('OpenAI API Key', type = 'password', key = 'openai_key')
    openai.api_key = openai_api
    os.environ["OPENAI_API_KEY"] = openai_api


import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")


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




default_system_prompt = """Act as an assistant that helps people with their questions.
If you don't know the answer, do not make one up, ask clarifying questions."""

system_message = """{system_prompt}"""

question_message = """{question}"""


# streamlit app
st.title("OpenAI Assistant Integrated with Streamlit")
st.header("Talk to your selected Generative AI")
st.write("Author: [Kristof Rabay](https://github.com/kristofrabay)")

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


if not openai_api:
        st.warning('🔑🔒 Paste your OpenAI API key on the sidebar 🔑🔒')

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


msg = st.chat_message('assistant')
msg.write("Hello 👋") 

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

    if not openai_api:
        st.warning('🔑🔒 Paste your OpenAI API key on the sidebar 🔑🔒')

    else:

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(QUERY)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            

            messages =[
                        {"role": "system", "content" : "You are a helpful assistant helping people answer their questions."}, 
                        {"role": "user", "content": system_message.format(system_prompt = SYSTEM_MESSAGE)}, 
                        *st.session_state.messages,
                        {"role": "user", "content": question_message.format(question=QUERY)}
                        ]
                        
            full_response = generate_response(messages, MODEL, TEMPERATURE, MAX_TOKENS)

            message_placeholder.markdown(full_response)

        # Add user and AI message to chat history
        st.session_state.messages.append({"role": "user", "content": QUERY})
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # to always fit in context, either limit historic messages, or count tokens
        # current solution: if we reach model-specific max msg number, remove first q-a pair

        if len(st.session_state.messages) >= MAX_CONTEXT_QUESTIONS[MODEL] * 2:
            st.session_state.messages = st.session_state.messages[2:]