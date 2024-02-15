'''
Base Prompt Utils

To do
Work out your chat prompt testing process
Integrate with logging
as you use more chats, come up with the prompts
Likely will be based on tools and action used
    For example can imagine a common RAG chat so will have a prompt for that
'''

from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

def get_basic_chat_template(system_content: str, human_message: str):
    basic_chat = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "{system_content}"
                )
            ),
            HumanMessagePromptTemplate.from_template("{human_message}"),
        ]
    )

    return basic_chat
