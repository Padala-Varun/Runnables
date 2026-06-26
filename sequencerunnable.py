# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
load_dotenv()
# pyrefly: ignore [missing-import]
from langchain_mistralai import ChatMistralAI
# pyrefly: ignore [missing-import]
from langchain_core.prompts import ChatPromptTemplate
# pyrefly: ignore [missing-import]
from langchain_core.output_parsers import StrOutputParser




# 1. Prompt Template
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)

# 2. Model
model = ChatMistralAI(model="mistral-small-2506")

# 3. Output Parser
parser = StrOutputParser()

#set up manual flow

formatted_prompt=prompt.format_messages(topic="Machine Learning")
response = model.invoke(formatted_prompt)
final_output=parser.parse(response)

print(final_output)

print("-------------------------------------------------------------------------------------------------------------")
#using Runnable(pipe | pipe | pipe) to setup flow

runnable = prompt | model | parser

result=runnable.invoke("Machine Learning")

print(result)
