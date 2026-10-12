import os

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS

from app.ingestion.embeddings import get_embedding_model


load_dotenv()


VECTORSTORE_PATH = "data/vectorstore"


def main():

    print("=" * 70)
    print("RAG RETRIEVAL TEST")
    print("=" * 70)


    # --------------------------------------------------
    # Load embeddings
    # --------------------------------------------------

    print("\nLoading embedding model...")

    embedding_model = (
        get_embedding_model()
    )

    print("✓ Embedding model ready")


    # --------------------------------------------------
    # Load FAISS
    # --------------------------------------------------

    print("\nLoading vector database...")

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    print("✓ Vector database loaded")


    # --------------------------------------------------
    # Test questions
    # --------------------------------------------------

    questions = [

        "Explain the first example discussed in the document.",

        "What is augmented intelligence?",

        "What is artificial intelligence?",

        "What are the benefits mentioned in the document?"

    ]


    # --------------------------------------------------
    # Search
    # --------------------------------------------------

    for question in questions:

        print("\n")
        print("=" * 70)

        print("QUESTION:")
        print(question)

        print("=" * 70)


        results = (
            vectorstore
            .similarity_search_with_score(
                question,
                k=5
            )
        )


        for index, (
            document,
            distance
        ) in enumerate(
            results,
            start=1
        ):

            print(
                f"\nRESULT {index}"
            )

            print(
                f"Source: "
                f"{document.metadata.get('source')}"
            )

            print(
                f"Page: "
                f"{document.metadata.get('page')}"
            )

            print(
                f"Distance: "
                f"{distance:.4f}"
            )

            print(
                "\nText:"
            )

            print(
                document.page_content[:500]
            )


if __name__ == "__main__":

    main()