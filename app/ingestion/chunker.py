from pathlib import Path

from pypdf import PdfReader
from docx import Document
import pandas as pd

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# PDF LOADER
# ============================================================

def load_pdf(file_path):

    reader = PdfReader(file_path)

    documents = []

    for page_number, page in enumerate(reader.pages):

        text = page.extract_text()

        if text and text.strip():

            documents.append({
                "text": text,
                "page": page_number + 1,
                "source": Path(file_path).name
            })

    return documents


# ============================================================
# DOCX LOADER
# ============================================================

def load_docx(file_path):

    document = Document(file_path)

    documents = []

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    full_text = "\n".join(paragraphs)

    if full_text.strip():

        documents.append({
            "text": full_text,
            "page": 1,
            "source": Path(file_path).name
        })

    return documents


# ============================================================
# TXT LOADER
# ============================================================

def load_txt(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        text = file.read()

    documents = []

    if text.strip():

        documents.append({
            "text": text,
            "page": 1,
            "source": Path(file_path).name
        })

    return documents


# ============================================================
# CSV LOADER
# ============================================================

def load_csv(file_path):

    dataframe = pd.read_csv(file_path)

    text = dataframe.to_string(index=False)

    documents = []

    if text.strip():

        documents.append({
            "text": text,
            "page": 1,
            "source": Path(file_path).name
        })

    return documents


# ============================================================
# EXCEL LOADER
# ============================================================

def load_excel(file_path):

    excel_file = pd.ExcelFile(file_path)

    documents = []

    for sheet_name in excel_file.sheet_names:

        dataframe = pd.read_excel(
            file_path,
            sheet_name=sheet_name
        )

        text = dataframe.to_string(index=False)

        if text.strip():

            documents.append({
                "text": text,
                "page": 1,
                "source": f"{Path(file_path).name} - {sheet_name}"
            })

    return documents


# ============================================================
# GENERIC DOCUMENT LOADER
# ============================================================

def load_document(file_path):

    file_path = Path(file_path)

    extension = file_path.suffix.lower()

    print(f"Loading: {file_path.name}")
    print(f"File type: {extension}")

    if extension == ".pdf":

        return load_pdf(file_path)

    elif extension == ".docx":

        return load_docx(file_path)

    elif extension == ".txt":

        return load_txt(file_path)

    elif extension == ".csv":

        return load_csv(file_path)

    elif extension in [".xlsx", ".xls"]:

        return load_excel(file_path)

    else:

        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: PDF, DOCX, TXT, CSV, XLSX, XLS."
        )


# ============================================================
# CHUNK CREATION
# ============================================================

def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = []

    for document in documents:

        text_chunks = splitter.split_text(
            document["text"]
        )

        for chunk_number, chunk_text in enumerate(text_chunks):

            chunks.append({
                "text": chunk_text,
                "page": document["page"],
                "source": document["source"],
                "chunk_id": chunk_number
            })

    return chunks


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_file = (
        "data/documents/"
        "lt-valves-gate-globe-check-valves-asme-b1634.pdf"
    )

    documents = load_document(test_file)

    print("\nPages/sections loaded:", len(documents))

    chunks = create_chunks(documents)

    print("Total chunks:", len(chunks))

    if chunks:

        print("\n========== FIRST CHUNK ==========\n")

        print(chunks[0]["text"])

        print("\nMetadata:")

        print("Source:", chunks[0]["source"])
        print("Page:", chunks[0]["page"])
        print("Chunk ID:", chunks[0]["chunk_id"])