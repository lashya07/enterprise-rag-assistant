import os

from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings


load_dotenv()


def get_embedding_model():

    api_key = os.getenv("COHERE_API_KEY")

    if not api_key:
        raise ValueError(
            "COHERE_API_KEY not found in .env"
        )

    return CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=api_key
    )