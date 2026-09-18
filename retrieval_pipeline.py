import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# Chroma database location
PERSIST_DIRECTORY = "db/chroma_db"

# Use the SAME embedding model used during ingestion
embedding_model = OllamaEmbeddings(
        model="nomic-embed-text"
    )

# Load existing Chroma vector store
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# User query
query = "What is MATERIALS"

# Retrieve top 5 relevant chunks
retriever = db.as_retriever(search_kwargs={"k": 5})
relevant_docs = retriever.invoke(query)

print("=" * 60)
print("User Query:", query)
print("=" * 60)

if not relevant_docs:
    print("No relevant documents found.")
else:
    print(f"\nRetrieved {len(relevant_docs)} document chunks.\n")

    for i, doc in enumerate(relevant_docs, start=1):
        print(f"📄 Document {i}")
        print("Source :", doc.metadata.get("source"))
        print("Page   :", doc.metadata.get("page", "N/A"))
        print("Content:")
        print(doc.page_content)
        print("-" * 60)