# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()


# pyrefly: ignore [missing-import]
from langchain_tavily import TavilySearch

# pyrefly: ignore [missing-import]
from langchain_mistralai import ChatMistralAI
# pyrefly: ignore [missing-import]
from langchain_core.prompts import ChatPromptTemplate
# pyrefly: ignore [missing-import]
from langchain_core.output_parsers import StrOutputParser


Search_tool = TavilySearch(
    max_results=5,   
)

model = ChatMistralAI(
    model="mistral-small-2506"
)

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant

    Summarize the following news into clear bullet points.

    {news}
    """
)

runnables=prompt|model|StrOutputParser()


news_result=Search_tool.run("Latest Ai news of 2026")
print(runnables.invoke({"news":news_result}))
