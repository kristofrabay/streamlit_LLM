import streamlit as st

from io import BytesIO
from pypdf import PdfReader
import docx2txt
from bs4 import BeautifulSoup
import re

import openai
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document

MODEL_COST = {'gpt-3.5-turbo' : 0.00175,
              'gpt-3.5-turbo-16k' : 0.0035,
              'gpt-4' : 0.035}

MODEL_RELEVANT_DOC_NUMBER = {'gpt-3.5-turbo' : 5,
                            'gpt-3.5-turbo-16k' : 10,
                            'gpt-4' : 15}

MODEL_INPUT_TOKEN_SUMM_LIMIT = {'gpt-3.5-turbo' : 2700,
                                'gpt-3.5-turbo-16k' : 5400,
                                'gpt-4' : 10800}


### Splitters for different data sources ###
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 200)
html_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN, chunk_size=1500, chunk_overlap=200)


def add_context_to_doc_chunks(_docs):

    # adding the filename to each chunk my help the relevany search

    for i in _docs:
        i.page_content = i.metadata['source'].split("\\")[-1].split('.')[0] + ' --- ' + i.page_content

    return _docs


def clean_HTML(html):

    soup = BeautifulSoup(html, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text


@st.experimental_memo()
def load_pdf(pdf_as_bytes, splitter = text_splitter, filename = 'pdf'):

    pdf_as_bytes = PdfReader(pdf_as_bytes)

    #text = ''
    DOCS = []

    for pagenum, page in enumerate(pdf_as_bytes.pages):

        page_text = page.extract_text()

        #text += page_text

        text_splitted = splitter.split_text(page_text)
        docs = [Document(page_content=t, metadata={'source' : filename, 'page' : str(pagenum+1)}) for t in text_splitted]
        docs = add_context_to_doc_chunks(docs)
        DOCS.append(docs)

    DOCS = [item for sublist in DOCS for item in sublist]

    return DOCS#, text



@st.experimental_memo()
def load_txt(file, splitter = text_splitter, filename = 'txt'):

    DOCS = []

    text = file.read().decode("utf-8")
    text = re.sub(r"\n\s*\n", "\n\n", text)

    text_splitted = splitter.split_text(text)
    docs = [Document(page_content=t, metadata={'source' : filename, 'page' : 'all'}) for t in text_splitted]
    docs = add_context_to_doc_chunks(docs)
    DOCS.append(docs)

    DOCS = [item for sublist in DOCS for item in sublist]

    return DOCS


@st.experimental_memo()
def load_docx(file, splitter = text_splitter, filename = 'docx'):

    DOCS = []

    text = docx2txt.process(file) 
    text = re.sub(r"\n\s*\n", "\n\n", text)

    text_splitted = splitter.split_text(text)
    docs = [Document(page_content=t, metadata={'source' : filename, 'page' : 'all'}) for t in text_splitted]
    docs = add_context_to_doc_chunks(docs)
    DOCS.append(docs)

    DOCS = [item for sublist in DOCS for item in sublist]

    return DOCS

@st.experimental_memo()
def load_HTML(file, splitter = html_splitter, filename = 'html'):

    DOCS = []

    text = file.read().decode("utf-8")
    text = clean_HTML(text)
    text = re.sub(r"\n\s*\n", "\n\n", text)

    text_splitted = splitter.split_text(text)
    docs = [Document(page_content=t, metadata={'source' : filename, 'page' : 'all'}) for t in text_splitted]
    docs = add_context_to_doc_chunks(docs)
    DOCS.append(docs)

    DOCS = [item for sublist in DOCS for item in sublist]

    return DOCS







def QA_ChatCompletion(user_input, model, db, embeddings, temperature):
    
    #### PROMPT PARTS ####

    PROMPT = """
    You are an intelligent assistant helping humans with their questions related to a wide variety of documents. 
    Answer ONLY with the facts listed in the list of sources below. 
    If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. 
    If asking a clarifying question to the user would help, ask the question. 
    Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. 
    Use square brackets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].

    Sources:
    {sources}

    Question: {question}

    Answer:
    
    """

    #### PROCESS QUERY; RUN SIMILARITY SEARCH ####

    query_embedded = embeddings.embed_query(user_input)


    sim_docs = db.max_marginal_relevance_search_by_vector(query_embedded, k = MODEL_RELEVANT_DOC_NUMBER[model])
    results = [doc.metadata['source'].split("\\")[-1] + "-page-" + str(doc.metadata['page'] )+ ": " + doc.page_content.replace("\n", "").replace("\r", "") for doc in sim_docs]
    content = "\n".join(results)


    #### CRAFT MESSAGE TO CHATCOMPLETION ####

    message = [{'role' : 'user', 'content' : PROMPT.format(sources = content, question = user_input)}]
    

    #### CALL CHATGPT API ####

    completion = openai.ChatCompletion.create(
        model=model, 
        messages=message, 
        temperature=temperature, 
        max_tokens=512,)

    return completion.choices[0]['message']['content']


def QA(model, db,  embeddings, temperature):
    '''
    This function gets the user input, pass it to ChatGPT function and 
    displays the response
    '''
    # Get user input
    user_query = st.text_input("Enter your question here")
    if user_query:
        # Pass the query to the ChatGPT function
        response = QA_ChatCompletion(user_query, model, db,  embeddings, temperature)
        return st.write(f"{response}")
    






    

def Summarize_Short_Doc(input_doc, model, temperature):

    #### PROMPT PARTS ####

    PROMPT = """
    Given the below document, your job is to create a summary of the text.
    Do not add any of your internal knowledge to the context.
    Feel free to create longer summaries, including all relevant information found in the original text.

    Document: {document}

    Summary:
    
    """

    message = [{"role": "system", "content": "You are an intelligent assistant helping humans understand contents of long documents."},
               {'role' : 'user', 'content' : PROMPT.format(document = input_doc)}]
    

    #### CALL CHATGPT API ####

    completion = openai.ChatCompletion.create(
        model=model, 
        messages=message, 
        temperature=temperature, 
        max_tokens=1024,)
    
    return completion.choices[0]['message']['content']
    
    
def Summarization(input_doc,  model, temperature, docs_original, tiktoken_encoding):
    
    llm_summ = ChatOpenAI(model_name=model, temperature=temperature)
    chain = load_summarize_chain(llm_summ, chain_type="map_reduce")

    WHOLE_DOC = ' '.join([i.page_content for i in docs_original if i.metadata['source'] == input_doc])
    input_tokens = tiktoken_encoding.encode(WHOLE_DOC)

    st.write('Given', str(len(input_tokens)), 'number of input tokens we should expect a summary in', str(round(len(input_tokens) / 1000 * 4.2 / 60, 2)), 'minutes')

    if len(input_tokens) <= MODEL_INPUT_TOKEN_SUMM_LIMIT[model]:
        st.write('Number of input tokens is small (' + str(len(input_tokens)) + ') so we can fit the whole document into one prompt!')
    else:
        st.write('Number of input tokens is large (' + str(len(input_tokens)) + ') so we will use a chain and call `OpenAI` multiple times.')

    st.write('💰💰💰 Approximate cost:', str(round(MODEL_COST[model] * len(input_tokens) / 1000, 5)), 'USD')

    want_to_continue = st.button('Continue with expected cost')

    if want_to_continue:

        st.write('Constructing summary...')

        if len(input_tokens) <= MODEL_INPUT_TOKEN_SUMM_LIMIT[model]:
            summary = Summarize_Short_Doc(input_doc = WHOLE_DOC, model = model, temperature = temperature)

        else:
            text_splitter_for_summ = RecursiveCharacterTextSplitter(chunk_size = 7500, chunk_overlap = 200)
            texts_for_summ = text_splitter_for_summ.split_text(WHOLE_DOC)
            docs_for_summ = [Document(page_content=t) for t in texts_for_summ]
            summary = chain.run(docs_for_summ)

        return st.write(summary)
    



def SimpleChat(model, temperature):
    '''
    This function gets the user input, pass it to ChatGPT function and 
    displays the response
    '''
    # Get user input
    user_query = st.text_input("You're using " + model + ' with ' + str(temperature) + " temperature")
    if user_query:
        # Pass the query to the ChatGPT function
        response = ChatGPT(user_query, model, temperature)
        return st.write(f"{response}")
    

def ChatGPT(user_query, model, temperature):
    ''' 
    This function uses the OpenAI API to generate a response to the given 
    user_query using the ChatGPT model
    '''
    # Use the OpenAI API to generate a response

    message = [{'role' : 'user', 'content' : user_query}]

    completion = openai.ChatCompletion.create(
        model=model, 
        messages=message, 
        temperature=temperature, 
        max_tokens=512,)
    
    response = completion.choices[0]['message']['content']
    return response


@st.cache(allow_output_mutation=True)
def create_db(docs):

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    return embeddings, db