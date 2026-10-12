# Enterprise Document AI Assistant (RAG)

An AI-powered document question-answering application that allows users to upload multiple PDF documents and ask questions using natural language. The system retrieves relevant information from uploaded documents and generates contextual answers using Google's Gemini models.

## 🚀 Features

- **Multi-PDF Upload:** Upload multiple PDF documents for question answering.
- **Retrieval-Augmented Generation (RAG):** Generates answers grounded in retrieved document content.
- **Hybrid Retrieval:** Combines semantic search using FAISS with keyword-based search using BM25.
- **Cross-Encoder Reranking:** Reorders retrieved chunks to improve relevance.
- **Conversational Query Rewriting:** Uses conversation context to help interpret follow-up questions.
- **Document Selection:** Query all indexed documents or select a specific document.
- **Source Citations:** View the source document associated with retrieved information.
- **Retrieval Inspection:** Inspect retrieved chunks to understand how the system finds relevant content.
- **Knowledge Base Management:** Build or rebuild the index and remove documents.
- **Interactive UI:** Built with Streamlit.

## 🏗️ Architecture

```text
                PDF Documents
                      |
                      v
              PDF Text Extraction
                      |
                      v
                Text Chunking
                      |
                      v
              Cohere Embeddings
                      |
                      v
                FAISS Index
                      |
             +--------+--------+
             |                 |
             v                 v
       Semantic Search    BM25 Search
             |                 |
             +--------+--------+
                      |
                      v
              Candidate Chunks
                      |
                      v
              Cross-Encoder Reranking
                      |
                      v
             Relevant Document Context
                      |
                      v
               Google Gemini
                      |
                      v
             Answer with Sources
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| User Interface | Streamlit |
| Large Language Model | Google Gemini |
| Embeddings | Cohere `embed-english-v3.0` |
| Vector Database | FAISS |
| Keyword Retrieval | BM25 |
| Reranking | Sentence Transformers Cross-Encoder |
| PDF Processing | Python PDF ingestion pipeline |
| Language | Python |

## 📁 Project Structure

```text
Enterprise-Document-AI-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
│
├── app/
│   └── ingestion/
│       ├── chunker.py
│       └── embeddings.py
│
└── data/
    └── vectorstore/
        ├── index.faiss
        ├── index.pkl
        ├── bm25.json
        └── manifest.json
```

*Note: `.env` should remain local and must not be committed to GitHub. The vector store files are generated data and may not exist until you build the knowledge base. Adjust this structure if your actual project differs.*

## ⚙️ Prerequisites

Install the following before running the project:

- Python 3.10 or a compatible version supported by your installed dependencies
- Git
- A Google Gemini API key
- A Cohere API key

## 🔑 API Key Setup

Create API keys using these services:

- Google AI Studio: https://aistudio.google.com/
- Cohere: https://dashboard.cohere.com/

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
COHERE_API_KEY=your_cohere_api_key
```

Replace the placeholder values with your actual API keys.

**Security:** Never upload your `.env` file, API keys, or other secrets to GitHub.

## 💻 Installation and Setup

### 1. Clone the repository

Replace `YOUR_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_USERNAME/Enterprise-Document-AI-Assistant.git
cd Enterprise-Document-AI-Assistant
```

If you already have the project on your computer, open a terminal in the project root instead.

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```cmd
venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt` file yet, create one containing the dependencies used by your actual project. The following is a starting point; confirm it against your imports and PDF processing implementation before using it:

```text
streamlit
python-dotenv
google-genai
langchain-community
langchain-cohere
faiss-cpu
rank-bm25
sentence-transformers
pypdf
```

Save the file as `requirements.txt`, then run:

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Ensure your `.env` file is in the same directory as `app.py` and contains valid API keys.

```env
GEMINI_API_KEY=your_gemini_api_key
COHERE_API_KEY=your_cohere_api_key
```

### 6. Run the application

From the project root, with your virtual environment activated:

```bash
streamlit run app.py
```

Streamlit will display a local URL, usually:

```text
http://localhost:8501
```

Open that URL in your browser.

## 📖 How to Use

1. Start the application using `streamlit run app.py`.
2. Upload one or more PDF documents.
3. Build or rebuild the knowledge base when prompted or when needed.
4. Select **All Documents** or choose a particular indexed document.
5. Ask questions in natural language.
6. Review the answer and its source citations.
7. Inspect retrieved chunks when you want to understand the retrieval process.
8. Upload additional documents or rebuild the knowledge base when your document collection changes.

## 🔍 Retrieval Pipeline

### 1. PDF Processing

Uploaded PDFs are processed to extract text. The extracted text is split into smaller chunks so that relevant passages can be retrieved efficiently.

### 2. Embedding Generation

Cohere's `embed-english-v3.0` model converts text chunks into vector embeddings.

### 3. Semantic Search

FAISS retrieves document chunks based on semantic similarity to the user's query.

### 4. Keyword Search

BM25 retrieves candidate chunks based on keyword overlap with the query.

### 5. Reranking

A Sentence Transformers cross-encoder, `cross-encoder/ms-marco-MiniLM-L-6-v2`, scores candidate query-and-document pairs and reorders them by relevance.

### 6. Answer Generation

The selected document context is passed to Google Gemini to generate an answer grounded in the retrieved content.

## 🧪 Testing Checklist

- [ ] Launch the Streamlit application successfully.
- [ ] Upload a valid PDF document.
- [ ] Build the knowledge base without errors.
- [ ] Ask a question whose answer is present in the document.
- [ ] Check whether the response includes the expected source.
- [ ] Test keyword-heavy and concept-based questions.
- [ ] Test follow-up questions that depend on conversation context.
- [ ] Select an individual document and compare the results with all-document search.
- [ ] Test the retrieval inspection feature.
- [ ] Rebuild the knowledge base after changing the document collection.

## ⚠️ Limitations

- Answer quality depends on PDF text extraction, chunking, retrieval, and the quality of the source documents.
- Scanned PDFs may require OCR if they do not contain extractable text.
- Gemini and Cohere usage may be subject to API quotas, rate limits, and pricing.
- Conversation history may be lost when the application session ends or the app restarts.
- Removing a document may require rebuilding the index, depending on the current implementation.
- Retrieval-augmented generation can still produce inaccurate answers. Verify important information against the original documents.
- The application should not be considered a secure multi-user production service without additional authentication, authorization, and data protection.

## 🔒 Security Notes

- Keep `.env` out of version control.
- Never hardcode API keys in Python files.
- Avoid committing confidential PDFs or sensitive indexed document content.
- Treat uploaded PDFs as untrusted input.
- Protect FAISS index files because they may contain document-derived content.
- Only load locally stored FAISS indexes from trusted sources.

## 🚧 Future Improvements

- Add OCR support for scanned PDFs.
- Implement persistent conversation history.
- Add metadata filtering and improved hybrid score fusion.
- Add document-level index updates without rebuilding everything.
- Add evaluation metrics for retrieval quality and answer faithfulness.
- Add user authentication and access controls.
- Add automated tests and a CI workflow.
- Deploy the application to a suitable hosting platform.

## 👨‍💻 Running Locally — Quick Reference

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Enterprise-Document-AI-Assistant.git

# Enter the project directory
cd Enterprise-Document-AI-Assistant

# Create a virtual environment
python -m venv venv
```

Activate the environment on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Then install and run:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Create your `.env` file before launching the application.

## 📄 License

Choose a license before distributing this project publicly. If you intend to use the MIT License, add a `LICENSE` file containing the MIT license text.

---

**Built with Python, Streamlit, FAISS, Cohere, Sentence Transformers, and Google Gemini.**
