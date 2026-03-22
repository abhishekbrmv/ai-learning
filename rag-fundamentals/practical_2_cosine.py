# practical_2_cosine.py
# Goal: see similarity scores between sentences

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()

def get_embedding(text):
    return client.embeddings.create(
        input=text, model="text-embedding-3-small"
    ).data[0].embedding

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

base = "The cat sat on the mat"
similar = "A feline rested on the rug"
unrelated = "Python is a programming language"

e_base = get_embedding(base)
e_sim   = get_embedding(similar)
e_unrel = get_embedding(unrelated)

print(f"Base vs Similar:   {cosine_similarity(e_base, e_sim):.4f}")   # ~0.88+
print(f"Base vs Unrelated: {cosine_similarity(e_base, e_unrel):.4f}") # ~0.20