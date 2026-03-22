# practical_1_embeddings.py
# Goal: see what an embedding actually looks like

from dotenv import load_dotenv
from openai import OpenAI
import os
load_dotenv()

client = OpenAI()

texts = [
    "The cat sat on the mat",
    "A feline rested on the rug",   # semantically similar
    "Python is a programming language"  # unrelated
]

embeddings = [
    client.embeddings.create(input=t, model="text-embedding-3-small").data[0].embedding
    for t in texts
]

print(f"Vector size: {len(embeddings[0])}")  # 1536 dimensions
print(f"First 5 values: {embeddings[0][:5]}")

