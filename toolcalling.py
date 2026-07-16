# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
load_dotenv()
# pyrefly: ignore [missing-import]
from langchain_mistralai import ChatMistralAI
# pyrefly: ignore [missing-import]
from langchain.tools import tool
# pyrefly: ignore [missing-import]
from rich import print

#1. Creating a tool
@tool
def get_text_length(text:str) ->int:
    """Return the number of character in a given text"""
    return len(text)

llm=ChatMistralAI(model="mistral-small-2506")

#tool binding
llm_with_tools=llm.bind_tools([get_text_length])

#result=llm.invoke("Hello")

#print(result)

result = llm_with_tools.invoke("Use this get_text_length tool to find the length of this sentence : Hello,how are you ?")

if result.tool_calls:
    tool_call=result.tool_calls[0]
    tool_result=get_text_length.invoke(tool_call["args"])
    

    final_response=llm.invoke(
        f"The length of the text is {tool_result}"
    )

    print(final_response.content)

