# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
load_dotenv()
# pyrefly: ignore [missing-import]
from langchain_mistralai import ChatMistralAI
# pyrefly: ignore [missing-import]
from langchain_core.prompts import ChatPromptTemplate
# pyrefly: ignore [missing-import]
from langchain_core.output_parsers import StrOutputParser
# pyrefly: ignore [missing-import]
from langchain_core.runnables import RunnableParallel,RunnableLambda

model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()


short_prompt = ChatPromptTemplate.from_template("Explain {topic} in 2-3 lines")
long_prompt = ChatPromptTemplate.from_template("Explain {topic} in detail")

topic = "machine learning"

#using manual setup


formatted_short=short_prompt.format_messages(topic=topic)
response_short=model.invoke(formatted_short)
str_out=parser.parse(response_short.content)
print("Short: ",str_out)
print("------------------------------------------------------------------")
formatted_long=long_prompt.format_messages(topic=topic)
response_long=model.invoke(formatted_long)
str_out_long=parser.parse(response_long.content)
print("Long: ",str_out_long)
print("----------------------------------------------------")

#using runnables

runnable=RunnableParallel({
    "short":RunnableLambda(lambda x :x['short']) |short_prompt | model | parser,
    "long":RunnableLambda(lambda x :x['long']) |long_prompt | model | parser
})

result=runnable.invoke({
    "short":{"topic":"machine Learning"},
    "long":{"topic":"deep learning"}
})

print(result['short'])
print("-----------------------")
print(result['long'])



#runnable lambda used when we send multiple inputs to parallel runnable(different inputs)