import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from langchain_community.vectorstores import FAISS
from app.ingestion.embeddings import get_embedding_model


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

VECTORSTORE_PATH = "data/vectorstore"

# Retrieve more candidates, then filter weak matches
RETRIEVAL_K = 8

# FAISS distance:
# lower = more similar
#
# You can adjust this after testing.
DISTANCE_THRESHOLD = 1.5


# ============================================================
# ENVIRONMENT CHECK
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found.\n"
        "Please add GEMINI_API_KEY=your_key to your .env file."
    )


# ============================================================
# INITIALIZE GEMINI
# ============================================================

print("\nLoading Gemini...")

client = genai.Client(
    api_key=GEMINI_API_KEY
)

print("✓ Gemini ready")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = get_embedding_model()

print("✓ Embedding model ready")


# ============================================================
# LOAD FAISS VECTOR DATABASE
# ============================================================

if not os.path.exists(VECTORSTORE_PATH):
    raise FileNotFoundError(
        f"\nVector database not found at: {VECTORSTORE_PATH}\n\n"
        "Run this command first:\n"
        "python -m app.retrieval.vectorstore"
    )


print("Loading FAISS vector database...")

vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

print("✓ FAISS vector database loaded")


# ============================================================
# HELPER: GET DOCUMENT NAME
# ============================================================

def get_source_name(metadata):
    """
    Safely get the source document name.
    """

    source = metadata.get("source", "Unknown document")

    return Path(source).name


# ============================================================
# RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve_documents(question):

    print("\nSearching documents...")

    results = vectorstore.similarity_search_with_score(
        question,
        k=RETRIEVAL_K
    )

    relevant_results = []

    for document, distance in results:

        if distance <= DISTANCE_THRESHOLD:

            relevant_results.append(
                (document, distance)
            )

    return results, relevant_results


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(relevant_results):

    context_parts = []

    for index, (document, distance) in enumerate(
        relevant_results,
        start=1
    ):

        metadata = document.metadata

        source = get_source_name(metadata)

        page = metadata.get(
            "page",
            metadata.get("section", "Unknown")
        )

        context_parts.append(
            f"""
--- DOCUMENT {index} ---

Source: {source}
Page/Section: {page}
Retrieval Distance: {distance:.4f}

Content:
{document.page_content}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, context):

    prompt = f"""
You are an enterprise document question-answering assistant.

Your job is to answer the user's question using ONLY the
information contained in the retrieved document context.

IMPORTANT RULES:

1. Do NOT use outside knowledge.
2. Do NOT invent facts.
3. Do NOT assume information that is not explicitly present.
4. If the context does not contain enough information to answer
   the question, say:

   "I could not find enough information in the uploaded documents
   to answer this question."

5. Prefer the most relevant document sections.
6. If multiple documents contain useful information, combine them.
7. Do not mix unrelated documents.
8. Keep the answer clear and professional.
9. When possible, explain the answer using numbered steps.
10. Include the document name and page number for important claims.
11. If the document contains safety warnings, preserve them.
12. Never create a citation for information that is not in the context.

USER QUESTION:
{question}

RETRIEVED DOCUMENT CONTEXT:
{context}

Now answer the user's question.
"""

    print("\nGenerating answer...")

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text


# ============================================================
# DISPLAY RETRIEVAL INFORMATION
# ============================================================

def display_retrieval_results(
    all_results,
    relevant_results
):

    print("\n" + "-" * 70)

    print("RETRIEVAL QUALITY")

    print("-" * 70)

    print(
        f"Retrieved candidates : {len(all_results)}"
    )

    print(
        f"Relevant chunks used : {len(relevant_results)}"
    )

    print(
        f"Distance threshold   : {DISTANCE_THRESHOLD}"
    )

    print("\nRETRIEVAL RESULTS")

    print("-" * 70)

    for index, (document, distance) in enumerate(
        all_results,
        start=1
    ):

        metadata = document.metadata

        source = get_source_name(metadata)

        page = metadata.get(
            "page",
            metadata.get("section", "Unknown")
        )

        status = (
            "✓ USED"
            if distance <= DISTANCE_THRESHOLD
            else "✗ FILTERED"
        )

        print(
            f"{index}. {source} | "
            f"Page: {page} | "
            f"Distance: {distance:.4f} | "
            f"{status}"
        )


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(relevant_results):

    print("\n" + "-" * 70)

    print("SOURCES USED")

    print("-" * 70)

    seen = set()

    for document, distance in relevant_results:

        metadata = document.metadata

        source = get_source_name(metadata)

        page = metadata.get(
            "page",
            metadata.get("section", "Unknown")
        )

        key = (source, page)

        if key not in seen:

            print(
                f"• {source} — Page/Section {page}"
            )

            seen.add(key)


# ============================================================
# MAIN CHAT LOOP
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("ENTERPRISE DOCUMENT RAG ASSISTANT")
    print("=" * 70)

    print(
        "Ask questions about any documents stored in your "
        "document knowledge base."
    )

    print(
        "The assistant retrieves relevant information and "
        "generates grounded answers."
    )

    print("\nCommands:")
    print("  exit / quit / q  → Exit")
    print("  sources          → Explain retrieval")
    print("=" * 70)


    while True:

        try:

            question = input("\nYou: ").strip()

        except KeyboardInterrupt:

            print("\n\nGoodbye!")

            break


        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if question.lower() in [
            "exit",
            "quit",
            "q"
        ]:

            print("\nGoodbye!")

            break


        # ----------------------------------------------------
        # EMPTY QUESTION
        # ----------------------------------------------------

        if not question:

            continue


        # ----------------------------------------------------
        # RETRIEVE
        # ----------------------------------------------------

        all_results, relevant_results = retrieve_documents(
            question
        )


        # ----------------------------------------------------
        # NO RELEVANT DOCUMENT
        # ----------------------------------------------------

        if not relevant_results:

            print("\n" + "=" * 70)

            print("AI ANSWER")

            print("=" * 70)

            print(
                "\nI could not find enough relevant information "
                "in the uploaded documents to answer this question."
            )

            display_retrieval_results(
                all_results,
                relevant_results
            )

            continue


        # ----------------------------------------------------
        # BUILD CONTEXT
        # ----------------------------------------------------

        context = build_context(
            relevant_results
        )


        # ----------------------------------------------------
        # GENERATE ANSWER
        # ----------------------------------------------------

        try:

            answer = generate_answer(
                question,
                context
            )

        except Exception as error:

            print("\nGemini generation failed.")

            print(f"Error: {error}")

            continue


        # ----------------------------------------------------
        # DISPLAY ANSWER
        # ----------------------------------------------------

        print("\n" + "=" * 70)

        print("AI ANSWER")

        print("=" * 70)

        print(answer)


        # ----------------------------------------------------
        # RETRIEVAL DETAILS
        # ----------------------------------------------------

        display_retrieval_results(
            all_results,
            relevant_results
        )


        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        display_sources(
            relevant_results
        )

        print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()