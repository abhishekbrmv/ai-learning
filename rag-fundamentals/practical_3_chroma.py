__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()
chroma = chromadb.Client()
collection = chroma.create_collection("my_first_store")

docs = [
    "LangChain is a framework for building LLM applications",
    "Chroma is an open-source vector database",
    "RAG stands for Retrieval Augmented Generation",
    "Embeddings convert text into numerical vectors",
    "The Eiffel Tower is located in Paris, France",
]

embeddings = [
    client.embeddings.create(input=d, model="text-embedding-3-small").data[0].embedding
    for d in docs
]

collection.add(
    documents=docs,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(docs))]
)

query = "What is a vector database?"
q_embedding = client.embeddings.create(
    input=query, model="text-embedding-3-small"
).data[0].embedding

results = collection.query(query_embeddings=[q_embedding], n_results=2)

print("Query:", query)
print("Top matches:")
for doc in results["documents"][0]:
    print(" -", doc)
