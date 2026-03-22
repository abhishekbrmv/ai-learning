__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# practical_6_full_rag.py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

load_dotenv()
# ── Step 1: Your document ──
text = """
Artificial intelligence is transforming how we build software.
Machine learning models can now understand text, images, and audio.

Large language models like GPT and Claude are trained on massive datasets.
They learn patterns in language and can generate human-like responses.

RAG stands for Retrieval Augmented Generation. It combines search with
generation. First you retrieve relevant documents, then pass them to an
LLM to generate a grounded answer. This prevents hallucination because
the LLM is forced to use only the provided context.

Vector databases store embeddings and allow similarity search. Chroma,
Pinecone, and Qdrant are popular options. Each chunk of text gets
converted into a vector and stored for later retrieval.

LangChain is a framework that makes it easy to build LLM applications.
It provides tools for chunking, embedding, retrieval, and chaining
prompts with language models.
"""

# ── Step 2: Chunk it ──
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = splitter.split_text(text)
print(f"Total chunks: {len(chunks)}")

# ── Step 3: Embed and store in Chroma ──
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_texts(chunks, embeddings)
print("Stored in Chroma successfully")

# ── Step 4: Retrieve relevant chunks ──
query = "How does RAG prevent hallucination?"
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
retrieved_docs = retriever.invoke(query)
context = "\n".join([doc.page_content for doc in retrieved_docs])
print(f"\nRetrieved context:\n{context}")

# ── Step 5: Build prompt and get answer ──
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, say "I don't know".

Context: {context}

Question: {question}
"""
)

chain = prompt | llm
response = chain.invoke({"context": context, "question": query})
print(f"\nQuestion: {query}")
print(f"Answer: {response.content}")
