#1) first step = loading all the libraries 

# pyrefly: ignore [missing-import]
from toolcalling1 import llm_with_tools
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
load_dotenv()

import os
import requests #This is used to send HTTP requests to websites and APIs.

# pyrefly: ignore [missing-import]
from langchain_mistralai import ChatMistralAI
# pyrefly: ignore [missing-import]
from langchain.tools import tool
# pyrefly: ignore [missing-import]
from langchain_core.messages import HumanMessage,ToolMessage
# pyrefly: ignore [missing-import]
from tavily import TavilyClient
# pyrefly: ignore [missing-import]
from rich import print


#2)Now lets create the tools
  #3)Weather tool
@tool
def get_weather(city:str)->str:
    """Get current weather of the city"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"
    response = requests.get(url)
    data=response.json()

    print("DEBUG: ",data)

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    
    return f"Weather in {city} : {desc} and {temp}°C"


  #4) Tavily news tool

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""
    
    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )
    
    results = response.get("results", [])
    
    if not results:
        return f"No news found for {city}"
    
    news_list = []
    
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        
        news_list.append(
            f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}..."
        )
    
    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)
    
#5) Setup LLM

llm = ChatMistralAI(model="mistral-small-2506")


tools = {
    "get_weather":get_weather,
    "get_news":get_news
}

llm_with_tools=llm.bind_tools([get_weather,get_news])

#Agent Loop - very important

messages = []

print("City Intelligence System")
print("Type 'Exit' to exit")

while True:
    user_input=input("YOU : ")
    if user_input.lower()=="exit":
        break
    messages.append(HumanMessage(content=user_input))

    while True:
        result=llm_with_tools.invoke(messages)

        messages.append(result)

        #if tool is required

        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name=tool_call['name']
                
                #Human in the loop
                confirm = input(f"Agent wants to call the {tool_name} Approve (yes/no)")

                if confirm.lower() == "no":
                    print("Tool call denied and I cannot bring the latest information")
                    break

                #execute tool

                tool_result=tools[tool_name].invoke(tool_call)
                messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call['id']
                ))
            continue

        else:
            print(result.content)
            break


    



    
