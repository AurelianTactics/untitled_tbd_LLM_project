'''
App uses LLM at different points
Simple stand in utility script to return an LLM for the langchain chain



Later
research different LLM services (OpenAI, Gemini, etc.)
research the args and options within an LLM

Wishlist
works with multiple LLM companies
offline LLM usage


'''

from langchain_openai import ChatOpenAI

def get_llm(model_name='gpt-3.5-turbo'):
    '''
    Obvious to dos here
    '''
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    return llm
