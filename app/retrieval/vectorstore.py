import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings

from app.ingestion.chunker import (
    load_document,
    create_chunks
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# PATHS
# ============================================================

DOCUMENTS_DIR = Path("data/documents")
VECTORSTORE_DIR = Path("data/vectorstore")


# ============================================================
# SUPPORTED FILE TYPES
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".csv",
    ".xlsx",
    ".xls"
}


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

def create_vectorstore():

    print("\n========================================")
    print("ENTERPRISE DOCUMENT RAG")
    print("========================================")


    # --------------------------------------------------------
    # Check Cohere API key
    # --------------------------------------------------------

    cohere_api_key = os.getenv("COHERE_API_KEY")

    if not cohere_api_key:

        raise ValueError(
            "\nCOHERE_API_KEY was not found.\n\n"
            "Please create a .env file in the project root "
            "and add:\n\n"
            "COHERE_API_KEY=your_actual_cohere_api_key\n"
        )

    print("✓ Cohere API key found")


    # --------------------------------------------------------
    # Check documents directory
    # --------------------------------------------------------

    if not DOCUMENTS_DIR.exists():

        print(
            f"\nDocuments directory does not exist:\n"
            f"{DOCUMENTS_DIR}"
        )

        print("\nCreating directory...")

        DOCUMENTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            "\nPlease put your PDF, DOCX, TXT, CSV, "
            "or XLSX files inside:"
        )

        print(DOCUMENTS_DIR)

        return


    # --------------------------------------------------------
    # Find supported documents
    # --------------------------------------------------------

    files = [
        file
        for file in DOCUMENTS_DIR.iterdir()
        if (
            file.is_file()
            and file.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ]


    if not files:

        print("\nNo supported documents found.")

        print(
            "\nSupported file types:"
        )

        print("  PDF")
        print("  DOCX")
        print("  TXT")
        print("  CSV")
        print("  XLSX")
        print("  XLS")

        print(
            f"\nPut your documents inside:\n"
            f"{DOCUMENTS_DIR}"
        )

        return


    # --------------------------------------------------------
    # Display documents
    # --------------------------------------------------------

    print(
        f"\nFound {len(files)} document(s):"
    )

    for file in files:

        print(f"  - {file.name}")


    # --------------------------------------------------------
    # Load documents
    # --------------------------------------------------------

    print("\nLoading documents...")

    all_documents = []

    successful_files = 0

    failed_files = 0


    for file in files:

        try:

            documents = load_document(file)

            all_documents.extend(documents)

            successful_files += 1

            print(
                f"✓ {file.name}: "
                f"{len(documents)} section(s)"
            )

        except Exception as error:

            failed_files += 1

            print(
                f"✗ Failed to load "
                f"{file.name}:"
            )

            print(
                f"  Error: {error}"
            )


    # --------------------------------------------------------
    # Check loaded documents
    # --------------------------------------------------------

    if not all_documents:

        print(
            "\nNo documents could be loaded."
        )

        return


    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    print("\nCreating chunks...")

    chunks = create_chunks(
        all_documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )


    if not chunks:

        print(
            "\nNo chunks were created."
        )

        return


    # --------------------------------------------------------
    # Prepare texts
    # --------------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]


    # --------------------------------------------------------
    # Prepare metadata
    # --------------------------------------------------------

    metadatas = []

    for chunk in chunks:

        metadata = {
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"]
        }

        metadatas.append(metadata)


    # --------------------------------------------------------
    # Load Cohere embedding model
    # --------------------------------------------------------

    print(
        "\nLoading Cohere embedding model..."
    )

    embedding_model = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=cohere_api_key
    )

    print(
        "✓ Cohere embedding model ready"
    )


    # --------------------------------------------------------
    # Create FAISS vector database
    # --------------------------------------------------------

    print(
        "\nCreating FAISS vector database..."
    )

    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas
    )


    # --------------------------------------------------------
    # Create vectorstore directory
    # --------------------------------------------------------

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Save FAISS database
    # --------------------------------------------------------

    vectorstore.save_local(
        str(VECTORSTORE_DIR)
    )


    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "VECTOR DATABASE CREATED SUCCESSFULLY"
    )

    print(
        "========================================"
    )

    print(
        f"Documents found      : {len(files)}"
    )

    print(
        f"Documents processed  : {successful_files}"
    )

    print(
        f"Documents failed     : {failed_files}"
    )

    print(
        f"Sections loaded      : {len(all_documents)}"
    )

    print(
        f"Chunks created       : {len(chunks)}"
    )

    print(
        f"Vectorstore location : {VECTORSTORE_DIR}"
    )

    print(
        "\nYour documents are now indexed and "
        "ready for semantic search."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    create_vectorstore()