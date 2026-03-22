# practical_4_chunking.py
from langchain_text_splitters  import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)

sample_text = """
Artificial intelligence is transforming how we build software. 
Machine learning models can now understand text, images, and audio.

Large language models like GPT and Claude are trained on massive datasets.
They learn patterns in language and can generate human-like responses.

RAG stands for Retrieval Augmented Generation. It combines search with 
generation. First you retrieve relevant documents, then pass them to an 
LLM to generate a grounded answer.

Vector databases store embeddings and allow similarity search. Chroma, 
Pinecone, and Qdrant are popular options. Each chunk of text gets 
converted into a vector and stored for later retrieval.
"""

# Strategy 1 — Fixed size chunking
fixed_splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separator=""
)
fixed_chunks = fixed_splitter.split_text(sample_text)

# Strategy 2 — Recursive chunking (recommended default)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
recursive_chunks = recursive_splitter.split_text(sample_text)

# Compare outputs
print("=" * 50)
print(f"FIXED SIZE: {len(fixed_chunks)} chunks")
print("=" * 50)
for i, chunk in enumerate(fixed_chunks):
    print(f"\nChunk {i+1} ({len(chunk)} chars):")
    print(chunk)

print("\n" + "=" * 50)
print(f"RECURSIVE: {len(recursive_chunks)} chunks")
print("=" * 50)
for i, chunk in enumerate(recursive_chunks):
    print(f"\nChunk {i+1} ({len(chunk)} chars):")
    print(chunk)
