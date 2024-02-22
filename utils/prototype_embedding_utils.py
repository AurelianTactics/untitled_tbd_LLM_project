'''
Prototype embedding utilities for getting text to chunks to simple embeddings

Embeddings can be used for LLM RAG

Simple example and few functions. Wishlist has the expansion

To do:
- TEST load the data
- TEST split the data
- TEST make the retriever
- TEST make the vectorstore
- the retriver works with the chain (chain is outside of this file)


Wishlist
    Ideally all these things interface nicely with each other:
        - storage for the raw data: text, voice, pictures, video, etc
        - storage for the embeddings
            - possibly storage for the intermediate steps like the splitting and what not
        - can query dynanmically. Examples:
            - all raw data from this entity
            - all text data from this entity for the past n days
            - the "core" / base / some "key" data that is periodically updated (this may be raw or embedded)
        - can retreive the raw
        - can retrieve the embeddings
- researched and tested and ablated all the intermediate and final steps for raw data to embeddings

Later
- how the data should be broken upa nd structured
- what the core data is
- waht the rolling base data is

'''

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


def run_orchestrator_prototype(file_path: str):
    data = load_local_data_prototype(file_path)
    all_splits = split_data_prototype(data)
    vectorstore = get_vectorstore_prototype(all_splits)
    retriever = get_retriever_prototype(vectorstore)

    return retriever


def load_local_data_prototype(file_path: str):
    '''
    TBD: doing a simple loacl text file for now, have to test if langchain has a specific way it wants it
    '''
    loader = TextLoader(file_path)
    data = loader.load()
    
    return data


def split_data_prototype(data: str):
    '''
    Split the data into chunks
    '''
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    return all_splits


def get_vectorstore_prototype(all_splits):
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
    return vectorstore


def get_retriever_prototype(vectorstore):
    # k is the number of chunks to retrieve
    retriever = vectorstore.as_retriever(k=4)

    return retriever