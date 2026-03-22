# rag-fundamentals

I built this to understand how RAG actually works under the hood — not just use a library that does it for me.

Six practicals, each one adding a piece to the puzzle. By the end they all connect into a working RAG pipeline.

## What I built

- `practical_1` — turn text into vectors using OpenAI embeddings
- `practical_2` — measure how similar two vectors are (cosine similarity)
- `practical_3` — store vectors in Chroma and retrieve by meaning, not keyword
- `practical_4` — split documents into chunks, compare fixed vs recursive strategies
- `practical_5` — build reusable prompt templates and chain them to an LLM
- `practical_6` — full RAG pipeline: load text → chunk → embed → retrieve → answer

## Run it

```bash
pip install openai langchain langchain-openai langchain-community langchain-text-splitters chromadb python-dotenv

# add your key to .env
OPENAI_API_KEY=your-key-here

python3 practical_6_full_rag.py
```

## What I learned

The core idea behind RAG is simple — instead of trusting the LLM's memory, you fetch the relevant information yourself and hand it to the model. The LLM just reads and answers. No guessing.

The tricky parts are chunking (how you split documents affects what gets retrieved) and prompt design (how you frame the context determines answer quality).
