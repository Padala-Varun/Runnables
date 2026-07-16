# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
load_dotenv()
# pyrefly: ignore [missing-import]
from langchain_mistralai import ChatMistralAI
# pyrefly: ignore [missing-import]
from langchain.tools import tool
# pyrefly: ignore [missing-import]
from rich import print
# pyrefly: ignore [missing-import]
from langchain_core.messages import HumanMessage


#1. Creating a tool
@tool
def get_text_length(text:str) ->int:
    """Return the number of character in a given text"""
    return len(text)

llm=ChatMistralAI(model="mistral-small-2506")

#tool binding
llm_with_tools=llm.bind_tools([get_text_length])

message = []

query=HumanMessage("Return the number of charcters in the given text :'Hello how are you'")

message.append(query)

result = llm_with_tools.invoke(message)

message.append(result)

print(message)