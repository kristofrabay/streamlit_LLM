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
        DOCS.append(docs)

    DOCS = [item for sublist in DOCS for item in sublist]

    return DOCS#, text



@st.experimental_memo()
def load_docx(file, splitter = text_splitter, filename = 'docx'):

    DOCS = []

    text = docx2txt.process(file) 
    text = re.sub(r"\n\s*\n", "\n\n", text)

    text_splitted = splitter.split_text(text)
    docs = [Document(page_content=t, metadata={'source' : filename, 'page' : 'all'}) for t in text_splitted]
    DOCS.append(docs)

    DOCS = [item for sublist in DOCS for item in sublist]

    return DOCS




def QA(question, context, model, temperature):

    prompt = f"""
    You are an intelligent assistant helping humans with their questions related to a wide variety of documents. 
    Use the following pieces of context to answer the users question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer

    {context}

    Question: 
    {question}

    Answer: """

    message = [{"role": "user", "content": prompt}]

    result = openai.ChatCompletion.create(model=model,  
                                          messages = message, 
                                          temperature=temperature, 
                                          top_p=1)
    
    completion = result['choices'][0]['message']['content']
    #usage_info = result['usage'].to_dict()

    return completion#, usage_info

def QA_st(context, model, temperature):
    '''
    This function gets the user input, pass it to ChatGPT function and 
    displays the response
    '''
    # Get user input
    user_query = st.text_area("Enter your question here")
    if user_query:
        # Pass the query to the ChatGPT function
        response = QA(user_query, context, model, temperature)
        return st.write(response)

   


def Instruction(instruction, context, model, temperature):

    prompt = f"""Use the following pieces of context to complete the given task. 
    You do not need to modify the original text, just come up with the suggested modifications.

    {context}

    Instruction: 
    {instruction}

    Completion: """

    message = [{"role": "user", "content": prompt}]

    result = openai.ChatCompletion.create(model=model,  
                                          messages = message, 
                                          temperature=temperature, 
                                          top_p=1)
    
    completion = result['choices'][0]['message']['content']
    #usage_info = result['usage'].to_dict()

    return completion#, usage_info

def Instruction_st(context, model, temperature):
    '''
    This function gets the user input, pass it to ChatGPT function and 
    displays the response
    '''
    # Get user input
    user_query = st.text_area("Enter your instruction here")
    if user_query:
        # Pass the query to the ChatGPT function
        response = Instruction(user_query, context, model, temperature)
        return st.write(response)
    




def InformationExtraction(context, extract_elements, model, temperature):

    prompt = f"""Use the following pieces of context to complete the task at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Task:
    Extract all {extract_elements} from the text.
    List them as key-value pairs, organize them into categories.

    Extracted information: """

    message = [{"role": "user", "content": prompt}]

    result = openai.ChatCompletion.create(model=model,  
                                          messages = message, 
                                          temperature=temperature, 
                                          top_p=1)
    
    completion = result['choices'][0]['message']['content']
    #usage_info = result['usage'].to_dict()

    return completion#, usage_info

def InformationExtraction_st(context, model, temperature):
    '''
    This function gets the user input, pass it to ChatGPT function and 
    displays the response
    '''
    # Get user input
    user_query = st.text_input("List elements you want to extract (separate them with commas)", 
                               value = "keywords, technical aspects, product attributes")
    user_button = st.button("Run keyword-extraction")
    if user_query and user_button:
        # Pass the query to the ChatGPT function
        response = InformationExtraction(context, user_query, model, temperature)
        return st.write(response)

    

def Summarization(context, model, temperature):

    prompt = f"""Write a concise summary of the following. 
    When someone reads your summary, they should have a clear idea and overview of what the original text was about.

    {context}

    CONCISE SUMMARY: """

    message = [{"role": "user", "content": prompt}]

    result = openai.ChatCompletion.create(model=model,  
                                          messages = message, 
                                          temperature=temperature, 
                                          top_p=1)
    
    completion = result['choices'][0]['message']['content']
    #usage_info = result['usage'].to_dict()

    return completion#, usage_info

def Summarization_st(context, model, temperature):
    '''
    This function gets the user input, pass it to ChatGPT function and 
    displays the response
    '''
    # Get user input
    user_query = st.button("Run summarization")
    if user_query:
        # Pass the query to the ChatGPT function
        response = Summarization(context, model, temperature)
        return st.write(response)



def concat_docs_count_tokens(docs, tiktoken_encoding):

    WHOLE_DOC = ' '.join([i.page_content for i in docs])
    input_tokens = tiktoken_encoding.encode(WHOLE_DOC)

    return WHOLE_DOC, input_tokens

