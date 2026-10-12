from langchain_community.vectorstores import FAISS
from app.ingestion.embeddings import get_embedding_model


embedding_model = get_embedding_model()

vectorstore = FAISS.load_local(
    "vectorstore",
    embedding_model,
    allow_dangerous_deserialization=True
)

question = "What should I do if there is leakage through the gland?"

results = vectorstore.similarity_search(
    question,
    k=4
)

print("\nQUESTION:")
print(question)

print("\nRELEVANT DOCUMENT SECTIONS:")

for i, document in enumerate(results):

    print(f"\n========== RESULT {i + 1} ==========")

    print("Source:", document.metadata.get("source"))
    print("Page:", document.metadata.get("page"))

    print("\nText:")
    print(document.page_content[:1000])