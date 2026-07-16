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

tools = {
    "get_text_length":get_text_length
}

llm=ChatMistralAI(model="mistral-small-2506")

#tool binding
llm_with_tools=llm.bind_tools([get_text_length])

message = []

prompt = input("You : ") 

query=HumanMessage(prompt)

message.append(query)

result = llm_with_tools.invoke(message)

message.append(result)

#print(message)

if result.tool_calls:
    tool_name=result.tool_calls[0]["name"]
    tool_message=tools[tool_name].invoke(result.tool_calls[0])
    #message.append(tool_message)

    
    #print(message)

    #result = llm_with_tools.invoke(message)
print(result.content)