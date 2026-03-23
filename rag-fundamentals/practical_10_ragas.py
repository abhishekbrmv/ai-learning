# practical_10_ragas.py

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from ragas import evaluate, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Step 1: RAG pipeline ──
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

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = splitter.split_text(text)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_texts(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. Answer using only the context below.
If the answer is not in the context, say "I don't know".

Context: {context}
Question: {question}
"""
)
chain = prompt | llm

# ── Step 2: Test cases ──
test_cases = [
    {
        "question": "What is RAG?",
        "reference": "RAG stands for Retrieval Augmented Generation. It combines search with generation."
    },
    {
        "question": "What are popular vector databases?",
        "reference": "Chroma, Pinecone, and Qdrant are popular vector database options."
    },
    {
        "question": "How does RAG prevent hallucination?",
        "reference": "RAG prevents hallucination by forcing the LLM to use only the provided context."
    },
]

# ── Step 3: Run pipeline and collect results ──
samples = []

for tc in test_cases:
    docs = retriever.invoke(tc["question"])
    context = "\n".join([doc.page_content for doc in docs])
    answer = chain.invoke({"context": context, "question": tc["question"]})

    samples.append({
        "user_input":         tc["question"],
        "retrieved_contexts": [doc.page_content for doc in docs],
        "response":           answer.content,
        "reference":          tc["reference"]
    })
    print(f"Q: {tc['question']}")
    print(f"A: {answer.content}\n")

# ── Step 4: Evaluate with RAGAS ──
print("Running RAGAS evaluation...")

openai_client = OpenAI()
ragas_llm = llm_factory("gpt-3.5-turbo", client=openai_client)
ragas_embeddings = RagasOpenAIEmbeddings(
    model="text-embedding-3-small",
    client=openai_client
)

dataset = EvaluationDataset.from_list(samples)

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=ragas_llm,
    embeddings=ragas_embeddings
)

print("\nRAGAS Scores:")
print(results)