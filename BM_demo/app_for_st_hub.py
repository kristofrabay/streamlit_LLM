#### PASSWORD
import hmac
import streamlit as st
import toml

#secrets = toml.load('./BM_demo/secrets.toml')
st_pw = st.secrets['password']
openai_api = st.secrets['openai_api']

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st_pw):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the passward is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Kérem adja meg a jelszót", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 A jelszó nem helyes")
    return False


if not check_password():
    st.stop() 




#### END OF PASSWORD

# packages
import numpy as np
from copy import deepcopy

#from credentials import openai_api
import os
import openai

openai.api_key = openai_api
os.environ["OPENAI_API_KEY"] = openai_api

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/Insignia_Hungary_Police.svg/1200px-Insignia_Hungary_Police.svg.png", use_column_width=False, width = 100)
#with st.sidebar:
#    openai_api = st.text_input('OpenAI API Key', type = 'password', key = 'openai_key')
#    openai.api_key = openai_api
#    os.environ["OPENAI_API_KEY"] = openai_api


import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")


# openai models, settings
embedder = 'text-embedding-ada-002'

MODEL_RELEVANT_DOC_NUMBER = {'gpt-3.5-turbo' : 5,
                            'gpt-4' : 10,
                            'gpt-4-1106-preview' : 20}

MODEL_INPUT_TOKEN_SUMM_LIMIT = {'gpt-3.5-turbo' : 14500,
                                'gpt-4' : 7200,
                                'gpt-4-1106-preview' : 125000}

MODEL_MAX_TOKEN_LIMIT = {'gpt-3.5-turbo' : 16384,
                        'gpt-4' : 8192,
                        'gpt-4-1106-preview' : 128000}

MODEL_COST = {'gpt-3.5-turbo' : 0.0015,
              'gpt-4' : 0.04,
              'gpt-4-1106-preview' : 0.015}


MAX_CONTEXT_QUESTIONS = {'gpt-3.5-turbo' : 10,
                        'gpt-4' : 20,
                        'gpt-4-1106-preview' : 120}


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




### load full text
with open('FULL_TEXT.txt',  encoding='utf-8') as f:
    full_text = f.read()


#default_system_prompt = """Act as an assistant that helps people with their questions relating to a wide variety of documents. 
#Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question. 
#Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brakets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].
#If you did not use the information below to answer the question, do not include the source name or any square brackets."""

default_system_prompt = """Act as an assistant that helps people with their questions relating to a wide variety of documents. 
Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question. 
Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brakets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].
ONLY include the source references at the end of your answer.
If you did not use the information below to answer the question, do not include the source name or any square brackets."""

system_message = """{system_prompt}

Sources:
{sources}

"""

question_message = """
{question}

Assistant: 
"""


# streamlit app
st.title("MI-vezérelt nyomozó-asszisztens")
#st.header("Integrate Generative AI with Your Knowledge")
st.write("A felületet készítette: Hiflylabs Zrt.")

st.sidebar.title("Információ")
st.sidebar.markdown(
    """
    Jelen környezet anonimizált, minta rendőrségi dokumentumokkal való MI-vezérelt kommunikációt demonstrál.\n
    A dokumentumokat előzetesen megkaptuk, feldolgoztuk.\n
    A háttérben futó modellek természetes nyelvfeldolgozó készséggel bírnak, így szövegek értelmezéséhez, generálásához is alkalmazhatók.\n
    """
)

#MODEL = st.radio('Select the OpenAI model you want to use', 
#                 ['gpt-4-1106-preview'], horizontal=True) #'gpt-3.5-turbo', 'gpt-4',

MODEL = 'gpt-4-1106-preview'

prompt_expander = st.expander(label='Állítható MI paraméterek')
with prompt_expander:
    cols=st.columns(2)
    with cols[0]:
        SYSTEM_MESSAGE = st.text_area('Alapüzenet', value = default_system_prompt, height = 400)
    with cols[1]:
        TEMPERATURE = float(st.select_slider('Kreativitás', [str(round(i, 2)) for i in np.linspace(0.0, 2, 101)], value = '0.0')) 
        MAX_TOKENS = st.slider('Generált szöveghossz max.', min_value = 1, max_value = MODEL_INPUT_TOKEN_SUMM_LIMIT[MODEL], value = 1024)



#### UPLOAD DOCS #####



#if not openai_api:
#    st.warning('🔑🔒 Paste your OpenAI API key on the sidebar 🔑🔒')

#### END OF UPLOAD PART ####


#### Clear cache ####

col1, col2 = st.columns([2, 1])

with col1:
    st.caption("""Eddigi chat-memória törlése, új beszélgetés kezdeményezése""")
with col2:
    if st.button("Cache-memória törlése"):
        st.cache_data.clear()
        for key in st.session_state.keys():
            del st.session_state[key]

#### end of clear cache

    
input_tokens = encoding.encode(full_text)
st.write('💰 Bemeneti tokenek száma: ' + str(len(input_tokens)) + '. Becsült feldolgozási költség: ' + str(round(MODEL_COST[MODEL] * (len(input_tokens) + 500) / 1000, 3)), 'USD')
#st.write('💰 Becsült feldolgozási költség:', str(round(MODEL_COST[MODEL] * (len(input_tokens) + 500) / 1000, 5)), 'USD')


msg = st.chat_message('assistant')
msg.write("Üdvözlöm 👋 Miben segíthetem a nyomozói munkáját?")

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

        if len(input_tokens) <= MODEL_INPUT_TOKEN_SUMM_LIMIT[MODEL]: # maybe we can fit everything into the prompt, why not
            #print('include all documents')
            #results = [doc.metadata['source'].split("\\")[-1] + "-page-" + str(doc.metadata['page'] )+ ": " + doc.page_content.replace("\n", "").replace("\r", "") for doc in docs]
            #sources = "\n".join(results)   
            sources = full_text
        else:
            pass
            #sources = retrieve_relevant_chunks(QUERY, db, MODEL)


        messages =[
                    {"role": "system", "content" : "You are a helpful assistant helping people answer their questions related to documents."},
                    {"role": "user", "content": system_message.format(system_prompt = SYSTEM_MESSAGE, sources=sources)},
                    *st.session_state.messages,
                    {"role": "user", "content": question_message.format(question=QUERY)}
                    ]
        
        # to always fit in context, either limit historic messages, or count tokens
        # current solution: if we reach model-specific max msg number or token count, remove q-a pairs from beginning until conditions are met
        
        current_token_count = len(encoding.encode(' '.join([i['content'] for i in messages])))

        while (len(messages)-3 > MAX_CONTEXT_QUESTIONS[MODEL] * 2) or (current_token_count >= MODEL_INPUT_TOKEN_SUMM_LIMIT[MODEL]):

            messages.pop(3)            
            current_token_count = len(encoding.encode(' '.join([i['content'] for i in messages])))

        full_response = generate_response(messages, MODEL, TEMPERATURE, MAX_TOKENS)

        message_placeholder.markdown(full_response)

    # Add user and AI message to chat history
    st.session_state.messages.append({"role": "user", "content": QUERY})
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    #if len(st.session_state.messages) > 0:
    #
    #    sources_expander = st.expander(label='Check sources identified as relevant')
    #    with sources_expander:
    #        #st.write('\n')
    #        if len(input_tokens) <= MODEL_INPUT_TOKEN_SUMM_LIMIT[MODEL]:
    #            st.write('All sources were used within the prompt')
    #        else:
    #            #st.write("Below are the sources that have been identified as relevant:")
    #            st.text(sources)
