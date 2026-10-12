import os
import json
import shutil
import re

import streamlit as st

from dotenv import load_dotenv
from google import genai

from langchain_community.vectorstores import FAISS

from rank_bm25 import BM25Okapi

from sentence_transformers import CrossEncoder

from app.ingestion.chunker import load_pdf, create_chunks
from app.ingestion.embeddings import get_embedding_model


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in .env")
    st.stop()


if not COHERE_API_KEY:
    st.error("COHERE_API_KEY not found in .env")
    st.stop()


client = genai.Client(
    api_key=GEMINI_API_KEY
)


DOCUMENTS_PATH = "data/documents"
VECTORSTORE_PATH = "data/vectorstore"

MANIFEST_PATH = os.path.join(
    VECTORSTORE_PATH,
    "manifest.json"
)

BM25_PATH = os.path.join(
    VECTORSTORE_PATH,
    "bm25.json"
)


os.makedirs(
    DOCUMENTS_PATH,
    exist_ok=True
)

os.makedirs(
    VECTORSTORE_PATH,
    exist_ok=True
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Enterprise Document AI Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if "selected_document" not in st.session_state:
    st.session_state.selected_document = "📚 All Documents"


if "last_results" not in st.session_state:
    st.session_state.last_results = []


if "last_search_query" not in st.session_state:
    st.session_state.last_search_query = ""


if "last_hybrid_results" not in st.session_state:
    st.session_state.last_hybrid_results = []


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_pdf_files():
    """
    Return all PDF files currently stored.
    """

    if not os.path.exists(DOCUMENTS_PATH):
        return []

    return sorted(
        [
            filename
            for filename in os.listdir(DOCUMENTS_PATH)
            if filename.lower().endswith(".pdf")
        ]
    )


def save_manifest(sources):
    """
    Save documents included in the current knowledge base.
    """

    data = {
        "sources": sorted(sources)
    }

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )


def load_manifest():
    """
    Load documents included in the current knowledge base.
    """

    if not os.path.exists(MANIFEST_PATH):
        return []

    try:

        with open(
            MANIFEST_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        return data.get(
            "sources",
            []
        )

    except Exception:

        return []


def vectorstore_exists():
    """
    Check whether FAISS exists.
    """

    return os.path.exists(
        os.path.join(
            VECTORSTORE_PATH,
            "index.faiss"
        )
    )


def bm25_exists():
    """
    Check whether BM25 database exists.
    """

    return os.path.exists(
        BM25_PATH
    )


def hybrid_database_exists():
    """
    Check whether both FAISS and BM25 exist.
    """

    return (
        vectorstore_exists()
        and
        bm25_exists()
    )


def clear_vectorstore():
    """
    Delete the existing FAISS and BM25 databases.
    """

    if os.path.exists(VECTORSTORE_PATH):

        for filename in os.listdir(
            VECTORSTORE_PATH
        ):

            path = os.path.join(
                VECTORSTORE_PATH,
                filename
            )

            if os.path.isfile(path):

                os.remove(path)

            elif os.path.isdir(path):

                shutil.rmtree(path)


def delete_document(filename):
    """
    Delete a PDF document.
    """

    file_path = os.path.join(
        DOCUMENTS_PATH,
        filename
    )

    if os.path.exists(file_path):

        os.remove(file_path)

        return True

    return False


# ============================================================
# TEXT TOKENIZATION
# ============================================================

def tokenize_text(text):
    """
    Simple tokenizer for BM25.
    """

    if not text:
        return []

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


# ============================================================
# BUILD KNOWLEDGE BASE
# ============================================================

def build_knowledge_base(source_files):

    if not source_files:

        raise ValueError(
            "No documents selected."
        )


    all_documents = []
    processed_documents = []


    # --------------------------------------------------------
    # Load PDFs
    # --------------------------------------------------------

    for filename in source_files:

        file_path = os.path.join(
            DOCUMENTS_PATH,
            filename
        )


        if not os.path.exists(file_path):

            continue


        documents = load_pdf(
            file_path
        )


        if documents:

            all_documents.extend(
                documents
            )

            processed_documents.append(
                filename
            )


    if not all_documents:

        raise ValueError(
            "No text could be extracted from the selected documents."
        )


    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    chunks = create_chunks(
        all_documents
    )


    if not chunks:

        raise ValueError(
            "No chunks were created."
        )


    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    embedding_model = (
        get_embedding_model()
    )


    # --------------------------------------------------------
    # Prepare FAISS data
    # --------------------------------------------------------

    texts = []
    metadatas = []


    for chunk in chunks:

        text = chunk.get(
            "text",
            ""
        )


        if not text.strip():

            continue


        texts.append(
            text
        )


        metadatas.append(
            {
                "source": chunk.get(
                    "source",
                    "Unknown"
                ),

                "page": chunk.get(
                    "page",
                    "Unknown"
                ),

                "chunk_id": chunk.get(
                    "chunk_id",
                    "Unknown"
                )
            }
        )


    if not texts:

        raise ValueError(
            "No valid text chunks were available."
        )


    # --------------------------------------------------------
    # Clear old databases
    # --------------------------------------------------------

    clear_vectorstore()


    # --------------------------------------------------------
    # Create FAISS
    # --------------------------------------------------------

    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas
    )


    # --------------------------------------------------------
    # Save FAISS
    # --------------------------------------------------------

    vectorstore.save_local(
        VECTORSTORE_PATH
    )


    # --------------------------------------------------------
    # Create BM25
    # --------------------------------------------------------

    tokenized_documents = [
        tokenize_text(text)
        for text in texts
    ]


    bm25_data = {
        "texts": texts,
        "metadatas": metadatas,
        "tokenized_documents": tokenized_documents
    }


    with open(
        BM25_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            bm25_data,
            file,
            ensure_ascii=False
        )


    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    save_manifest(
        processed_documents
    )


    return {
        "documents": len(processed_documents),
        "chunks": len(texts),
        "sources": processed_documents
    }


# ============================================================
# LOAD VECTORSTORE
# ============================================================

@st.cache_resource
def load_vectorstore():

    embedding_model = (
        get_embedding_model()
    )


    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )


    return vectorstore


# ============================================================
# LOAD BM25
# ============================================================

@st.cache_resource
def load_bm25():

    if not bm25_exists():

        raise FileNotFoundError(
            "BM25 database not found. "
            "Please rebuild the knowledge base."
        )


    with open(
        BM25_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)


    tokenized_documents = data.get(
        "tokenized_documents",
        []
    )


    if not tokenized_documents:

        raise ValueError(
            "BM25 database is empty. "
            "Please rebuild the knowledge base."
        )


    bm25 = BM25Okapi(
        tokenized_documents
    )


    return bm25, data


# ============================================================
# LOAD RERANKER
# ============================================================

@st.cache_resource
def load_reranker():

    """
    CrossEncoder reranker.
    """

    model = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return model


# ============================================================
# FAISS RETRIEVAL
# ============================================================

def retrieve_faiss(
    vectorstore,
    question,
    selected_document,
    k=10
):

    if selected_document == "📚 All Documents":

        return vectorstore.similarity_search_with_score(
            question,
            k=k
        )


    def source_filter(metadata):

        return (
            metadata.get("source")
            == selected_document
        )


    return vectorstore.similarity_search_with_score(
        question,
        k=k,
        filter=source_filter
    )


# ============================================================
# BM25 RETRIEVAL
# ============================================================

def retrieve_bm25(
    question,
    selected_document,
    k=10
):

    bm25, data = load_bm25()


    texts = data.get(
        "texts",
        []
    )


    metadatas = data.get(
        "metadatas",
        []
    )


    if not texts or not metadatas:

        return []


    query_tokens = tokenize_text(
        question
    )


    # --------------------------------------------------------
    # Empty query protection
    # --------------------------------------------------------

    if not query_tokens:

        return []


    scores = bm25.get_scores(
        query_tokens
    )


    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )


    results = []


    for index in ranked_indices:

        if len(results) >= k:

            break


        metadata = metadatas[index]


        if (
            selected_document
            !=
            "📚 All Documents"
        ):

            if (
                metadata.get("source")
                != selected_document
            ):

                continue


        results.append(
            {
                "text": texts[index],

                "metadata": metadata,

                "bm25_score": float(
                    scores[index]
                )
            }
        )


    return results


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def hybrid_retrieve(
    vectorstore,
    question,
    selected_document,
    semantic_k=10,
    keyword_k=10
):

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    faiss_results = retrieve_faiss(
        vectorstore,
        question,
        selected_document,
        k=semantic_k
    )


    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    bm25_results = retrieve_bm25(
        question,
        selected_document,
        k=keyword_k
    )


    candidates = {}


    # --------------------------------------------------------
    # Add FAISS candidates
    # --------------------------------------------------------

    for document, distance in faiss_results:

        metadata = document.metadata


        key = (
            metadata.get("source"),
            metadata.get("page"),
            metadata.get("chunk_id")
        )


        candidates[key] = {
            "text": document.page_content,

            "metadata": metadata,

            "faiss_distance": float(
                distance
            ),

            "bm25_score": 0.0
        }


    # --------------------------------------------------------
    # Add BM25 candidates
    # --------------------------------------------------------

    for item in bm25_results:

        metadata = item["metadata"]


        key = (
            metadata.get("source"),
            metadata.get("page"),
            metadata.get("chunk_id")
        )


        if key in candidates:

            candidates[key]["bm25_score"] = (
                item["bm25_score"]
            )

        else:

            candidates[key] = {
                "text": item["text"],

                "metadata": metadata,

                "faiss_distance": None,

                "bm25_score": item["bm25_score"]
            }


    return list(
        candidates.values()
    )


# ============================================================
# RERANK CANDIDATES
# ============================================================

def rerank_candidates(
    question,
    candidates,
    top_k=6
):

    if not candidates:

        return []


    reranker = load_reranker()


    pairs = [
        (
            question,
            candidate["text"]
        )

        for candidate in candidates
    ]


    scores = reranker.predict(
        pairs
    )


    for candidate, score in zip(
        candidates,
        scores
    ):

        candidate["rerank_score"] = float(
            score
        )


    candidates.sort(
        key=lambda item:
        item["rerank_score"],
        reverse=True
    )


    return candidates[:top_k]


# ============================================================
# COMPLETE RETRIEVAL PIPELINE
# ============================================================

def retrieve_documents(
    vectorstore,
    question,
    selected_document,
    k=6
):

    candidates = hybrid_retrieve(
        vectorstore,
        question,
        selected_document,
        semantic_k=10,
        keyword_k=10
    )


    reranked = rerank_candidates(
        question,
        candidates,
        top_k=k
    )


    return reranked


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    context_parts = []


    for index, result in enumerate(
        results,
        start=1
    ):

        metadata = result.get(
            "metadata",
            {}
        )


        source = metadata.get(
            "source",
            "Unknown"
        )


        page = metadata.get(
            "page",
            "Unknown"
        )


        chunk_id = metadata.get(
            "chunk_id",
            "Unknown"
        )


        rerank_score = result.get(
            "rerank_score",
            0.0
        )


        context_parts.append(
            f"""
============================================================
RETRIEVED DOCUMENT {index}
============================================================

SOURCE:
{source}

PAGE:
{page}

CHUNK:
{chunk_id}

RERANK SCORE:
{rerank_score:.4f}

CONTENT:
{result["text"]}
"""
        )


    return "\n".join(
        context_parts
    )


# ============================================================
# CONVERSATIONAL QUERY REWRITE
# ============================================================

def create_search_query(
    question,
    chat_history
):

    if not chat_history:

        return question


    recent_history = chat_history[-6:]


    history_text = []


    for message in recent_history:

        role = message.get(
            "role",
            ""
        )


        content = message.get(
            "content",
            ""
        )


        if role in [
            "user",
            "assistant"
        ]:

            history_text.append(
                f"{role.upper()}: {content}"
            )


    history_text = "\n".join(
        history_text
    )


    prompt = f"""
You are a query rewriting component for an enterprise RAG system.

Convert the latest user question into a standalone search query.

Use conversation history ONLY to resolve references such as:

- it
- this
- that
- they
- the first example
- the second example
- the previous step
- what about this
- why
- how about the second one

Do NOT answer the question.

Do NOT add facts.

Do NOT invent information.

Return ONLY the rewritten search query.

============================================================
CONVERSATION HISTORY
============================================================

{history_text}

============================================================
LATEST USER QUESTION
============================================================

{question}

============================================================

Standalone search query:
"""


    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )


    rewritten = (
        response.text.strip()
        if response.text
        else ""
    )


    if not rewritten:

        return question


    return rewritten


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    search_query,
    context,
    chat_history,
    selected_document
):

    history_text = ""


    if chat_history:

        recent_history = chat_history[-8:]


        history_parts = []


        for message in recent_history:

            role = message.get(
                "role",
                ""
            )


            content = message.get(
                "content",
                ""
            )


            if role in [
                "user",
                "assistant"
            ]:

                history_parts.append(
                    f"{role.upper()}: {content}"
                )


        history_text = "\n".join(
            history_parts
        )


    prompt = f"""
You are an enterprise document-grounded AI assistant.

Answer the user's question using ONLY the retrieved
enterprise document context.

============================================================
SELECTED DOCUMENT SCOPE
============================================================

{selected_document}

============================================================
STRICT GROUNDING RULES
============================================================

1. Use ONLY the supplied document context for factual claims.

2. Do NOT use outside knowledge.

3. Do NOT invent facts.

4. Do NOT guess.

5. Conversation history may ONLY be used to understand
   references and conversational context.

6. Conversation history is NOT a factual source.

7. If the answer is not supported by the retrieved context,
   say exactly:

"I could not find this information in the provided documents."

8. Do not mix unrelated documents.

9. Cite the actual document name and page number when
   available.

10. Never invent a source.

11. Never invent a page number.

12. Preserve important safety warnings and precautions.

13. If several documents contain relevant information,
    distinguish them clearly.

14. Answer follow-up questions naturally.

15. Keep the answer concise but complete.

============================================================
CONVERSATION HISTORY
============================================================

{history_text}

============================================================
CURRENT USER QUESTION
============================================================

{question}

============================================================
SEARCH QUERY
============================================================

{search_query}

============================================================
RETRIEVED DOCUMENT CONTEXT
============================================================

{context}

============================================================

Now provide the grounded answer.
"""


    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )


    return (
        response.text.strip()
        if response.text
        else
        "I could not generate an answer."
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🤖 Enterprise Document AI Assistant"
)


st.markdown(
    """
**Multi-document Hybrid Conversational RAG**

Semantic retrieval + BM25 keyword retrieval +
CrossEncoder reranking + Gemini grounded answers.
"""
)


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "📄 Document Management"
)


# ============================================================
# UPLOAD DOCUMENTS
# ============================================================

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    st.sidebar.subheader(
        "Selected Files"
    )


    for uploaded_file in uploaded_files:

        st.sidebar.write(
            f"📄 {uploaded_file.name}"
        )


    if st.sidebar.button(
        "📥 Add Documents",
        type="primary"
    ):

        saved_count = 0


        for uploaded_file in uploaded_files:

            file_path = os.path.join(
                DOCUMENTS_PATH,
                uploaded_file.name
            )


            with open(
                file_path,
                "wb"
            ) as file:

                file.write(
                    uploaded_file.getbuffer()
                )


            saved_count += 1


        st.sidebar.success(
            f"{saved_count} document(s) added."
        )


        st.rerun()


# ============================================================
# STORED DOCUMENTS
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "📚 Stored Documents"
)


stored_documents = get_pdf_files()


if stored_documents:

    for filename in stored_documents:

        st.sidebar.write(
            f"📄 {filename}"
        )

else:

    st.sidebar.caption(
        "No PDF documents uploaded."
    )


# ============================================================
# DOCUMENT DROPDOWN
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "🎯 Select Document"
)


if stored_documents:

    document_options = [
        "📚 All Documents"
    ] + stored_documents


    current_selection = (
        st.session_state.selected_document
    )


    if current_selection not in document_options:

        current_selection = (
            "📚 All Documents"
        )


    selected_document = st.sidebar.selectbox(
        "Choose the document to search:",
        document_options,
        index=document_options.index(
            current_selection
        )
    )


    st.session_state.selected_document = (
        selected_document
    )


else:

    selected_document = (
        "📚 All Documents"
    )


    st.sidebar.info(
        "Upload documents to enable document selection."
    )


# ============================================================
# DELETE DOCUMENT
# ============================================================

if stored_documents:

    st.sidebar.divider()

    st.sidebar.subheader(
        "🗑️ Delete Document"
    )


    document_to_delete = st.sidebar.selectbox(
        "Select document to delete:",
        stored_documents,
        key="delete_document_select"
    )


    if st.sidebar.button(
        "🗑️ Delete Selected Document"
    ):

        try:

            deleted = delete_document(
                document_to_delete
            )


            if not deleted:

                st.sidebar.error(
                    "Document was not found."
                )

            else:

                if (
                    st.session_state.selected_document
                    == document_to_delete
                ):

                    st.session_state.selected_document = (
                        "📚 All Documents"
                    )


                st.session_state.chat_history = []

                st.session_state.last_results = []

                st.session_state.last_search_query = ""


                clear_vectorstore()

                st.cache_resource.clear()


                st.sidebar.success(
                    f"{document_to_delete} deleted."
                )


                st.sidebar.warning(
                    "Knowledge base cleared. "
                    "Rebuild it after deleting a document."
                )


                st.rerun()


        except Exception as error:

            st.sidebar.error(
                f"Could not delete document: {error}"
            )


# ============================================================
# KNOWLEDGE BASE
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "🧠 Knowledge Base"
)


if st.sidebar.button(
    "🔨 Build / Rebuild Knowledge Base",
    type="primary"
):

    current_documents = get_pdf_files()


    if not current_documents:

        st.sidebar.error(
            "No PDF documents found."
        )

    else:

        try:

            with st.spinner(
                "Building hybrid knowledge base..."
            ):

                stats = build_knowledge_base(
                    current_documents
                )


                st.cache_resource.clear()


            st.sidebar.success(
                "Hybrid knowledge base created!"
            )


            st.sidebar.write(
                f"Documents indexed: "
                f"{stats['documents']}"
            )


            st.sidebar.write(
                f"Chunks created: "
                f"{stats['chunks']}"
            )


            st.rerun()


        except Exception as error:

            st.sidebar.error(
                f"Failed to build knowledge base:\n{error}"
            )


# ============================================================
# KNOWLEDGE BASE STATUS
# ============================================================

if hybrid_database_exists():

    indexed_sources = load_manifest()


    st.sidebar.success(
        "🟢 Hybrid Knowledge Base Ready"
    )


    if indexed_sources:

        st.sidebar.caption(
            "Indexed documents:"
        )


        for source in indexed_sources:

            st.sidebar.caption(
                f"• {source}"
            )

else:

    st.sidebar.warning(
        "🔴 Hybrid Knowledge Base Not Built"
    )


# ============================================================
# NEW CHAT
# ============================================================

st.sidebar.divider()


if st.sidebar.button(
    "🆕 New Chat"
):

    st.session_state.chat_history = []

    st.session_state.last_results = []

    # FIXED: must be a string, not a list
    st.session_state.last_search_query = ""

    st.session_state.last_hybrid_results = []

    st.rerun()


# ============================================================
# MAIN AREA
# ============================================================

st.subheader(
    "💬 Conversational Hybrid RAG"
)


st.info(
    f"Searching: **{selected_document}**"
)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.chat_history:

    role = message.get(
        "role"
    )


    content = message.get(
        "content",
        ""
    )


    if role == "user":

        with st.chat_message("user"):

            st.write(
                content
            )


    elif role == "assistant":

        with st.chat_message("assistant"):

            st.write(
                content
            )


            sources = message.get(
                "sources",
                []
            )


            if sources:

                with st.expander(
                    "📚 Sources"
                ):

                    for source in sources:

                        st.markdown(
                            f"📄 **{source['source']}** "
                            f"— Page {source['page']}"
                        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    # --------------------------------------------------------
    # Validate documents
    # --------------------------------------------------------

    if not stored_documents:

        st.error(
            "Please upload at least one PDF document."
        )

        st.stop()


    # --------------------------------------------------------
    # Validate hybrid database
    # --------------------------------------------------------

    if not hybrid_database_exists():

        st.error(
            """
            Hybrid knowledge base has not been built yet.

            Please click **Build / Rebuild Knowledge Base**
            from the sidebar.
            """
        )

        st.stop()


    # --------------------------------------------------------
    # Validate selected document
    # --------------------------------------------------------

    indexed_sources = load_manifest()


    if (
        selected_document != "📚 All Documents"
        and selected_document not in indexed_sources
    ):

        st.error(
            f"'{selected_document}' is not currently indexed. "
            "Please rebuild the knowledge base."
        )

        st.stop()


    # --------------------------------------------------------
    # Add user message
    # --------------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )


    with st.chat_message("user"):

        st.write(
            question
        )


    # ========================================================
    # QUERY REWRITE
    # ========================================================

    with st.spinner(
        "Understanding conversation..."
    ):

        try:

            search_query = create_search_query(
                question,
                st.session_state.chat_history[:-1]
            )

        except Exception:

            search_query = question


    st.session_state.last_search_query = (
        search_query
    )


    # ========================================================
    # LOAD DATABASES
    # ========================================================

    try:

        vectorstore = load_vectorstore()

        load_bm25()

    except Exception as error:

        st.error(
            f"Could not load knowledge base: {error}"
        )

        st.stop()


    # ========================================================
    # HYBRID RETRIEVAL + RERANKING
    # ========================================================

    with st.spinner(
        "Searching with hybrid retrieval and reranking..."
    ):

        try:

            results = retrieve_documents(
                vectorstore,
                search_query,
                selected_document,
                k=6
            )

        except Exception as error:

            st.error(
                f"Retrieval failed: {error}"
            )

            st.stop()


    st.session_state.last_results = results

    st.session_state.last_hybrid_results = results


    # ========================================================
    # NO RESULTS
    # ========================================================

    if not results:

        answer = (
            "I could not find this information "
            "in the provided documents."
        )

        sources = []


    else:

        # ----------------------------------------------------
        # Build context
        # ----------------------------------------------------

        context = build_context(
            results
        )


        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        with st.spinner(
            "Generating grounded answer..."
        ):

            try:

                answer = generate_answer(
                    question,
                    search_query,
                    context,
                    st.session_state.chat_history[:-1],
                    selected_document
                )

            except Exception as error:

                st.error(
                    f"Gemini generation failed: {error}"
                )

                st.stop()


        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        sources = []

        seen_sources = set()


        for result in results:

            metadata = result.get(
                "metadata",
                {}
            )


            source = metadata.get(
                "source",
                "Unknown"
            )


            page = metadata.get(
                "page",
                "Unknown"
            )


            key = (
                source,
                page
            )


            if key not in seen_sources:

                sources.append(
                    {
                        "source": source,
                        "page": page,
                        "rerank_score":
                            result.get(
                                "rerank_score",
                                0.0
                            )
                    }
                )


                seen_sources.add(
                    key
                )


    # ========================================================
    # SAVE ASSISTANT RESPONSE
    # ========================================================

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )


    # ========================================================
    # DISPLAY ANSWER
    # ========================================================

    with st.chat_message("assistant"):

        st.write(
            answer
        )


        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        if sources:

            with st.expander(
                "📚 Sources"
            ):

                for source in sources:

                    st.markdown(
                        f"📄 **{source['source']}** "
                        f"— Page {source['page']}"
                    )


        # ----------------------------------------------------
        # Search query
        # ----------------------------------------------------

        with st.expander(
            "🧠 Conversational Search Query"
        ):

            st.write(
                search_query
            )


        # ----------------------------------------------------
        # Retrieval details
        # ----------------------------------------------------

        if results:

            with st.expander(
                "🔎 Hybrid Retrieval & Reranking Details"
            ):

                for index, result in enumerate(
                    results,
                    start=1
                ):

                    metadata = result.get(
                        "metadata",
                        {}
                    )


                    source = metadata.get(
                        "source",
                        "Unknown"
                    )


                    page = metadata.get(
                        "page",
                        "Unknown"
                    )


                    chunk_id = metadata.get(
                        "chunk_id",
                        "Unknown"
                    )


                    faiss_distance = result.get(
                        "faiss_distance"
                    )


                    bm25_score = result.get(
                        "bm25_score",
                        0.0
                    )


                    rerank_score = result.get(
                        "rerank_score",
                        0.0
                    )


                    st.markdown(
                        f"### Result {index}"
                    )


                    st.write(
                        f"**Source:** {source}"
                    )


                    st.write(
                        f"**Page:** {page}"
                    )


                    st.write(
                        f"**Chunk:** {chunk_id}"
                    )


                    st.write(
                        f"**FAISS Distance:** "
                        f"{faiss_distance}"
                    )


                    st.write(
                        f"**BM25 Score:** "
                        f"{bm25_score:.4f}"
                    )


                    st.write(
                        f"**Rerank Score:** "
                        f"{rerank_score:.4f}"
                    )


                    st.write(
                        result["text"]
                    )


                    st.divider()