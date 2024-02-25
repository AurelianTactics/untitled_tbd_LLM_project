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
from langchain import hub

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


def get_hub_prompt(hub_pull_argument='rlm/rag-prompt'):
    prompt = hub.pull(hub_pull_argument)
    example_messages = prompt.invoke(
        {"context": "filler context", "question": "filler question"}
        ).to_messages()

    return example_messages
