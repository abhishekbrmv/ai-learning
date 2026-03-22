# practical_5_langchain.py

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer this question clearly in 2 sentences: {question}"
)

chain = prompt | llm

response = chain.invoke({"question": "What is a vector database?"})
print(response.content)