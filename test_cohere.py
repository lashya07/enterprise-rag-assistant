import os

from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings


load_dotenv()

api_key = os.getenv("COHERE_API_KEY")

if not api_key:
    raise ValueError("COHERE_API_KEY not found in .env")


embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=api_key
)


text = "What should I do if there is leakage through the valve gland?"

vector = embeddings.embed_query(text)


print("Cohere embedding works!")
print("Vector dimensions:", len(vector))
print("First 5 values:", vector[:5])