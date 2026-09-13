import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

# Check Google API Key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Chroma database path
PERSIST_DIRECTORY = "db/chroma_db"

# Same embedding model used during ingestion
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load existing Chroma vector store
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Gemini LLM
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0
)

# User query
query = "What type  of certification is required for make a new flag?"

# Retrieve relevant document chunks
retriever = db.as_retriever(search_kwargs={"k": 5})
relevant_docs = retriever.invoke(query)

print("=" * 60)
print("User Query:", query)
print("=" * 60)

# Check if any documents were retrieved
if not relevant_docs:
    print("No relevant documents found in the vector database.")
    exit()

# Build context from retrieved documents
context = ""

print("\n--- Retrieved Context ---")
for i, doc in enumerate(relevant_docs, start=1):
    print(f"\nDocument {i}")
    print(f"Source: {doc.metadata.get('source')}")
    print(f"Page: {doc.metadata.get('page', 'N/A')}")
    print(doc.page_content[:300], "...\n")

    context += doc.page_content + "\n\n"

# Prompt Gemini
messages = [
    SystemMessage(
        content=(
            "You are a helpful RAG assistant. "
            "Answer ONLY using the provided context. "
            "Do not use outside knowledge. "
            "If the answer is not present in the context, reply exactly:\n"
            "'I don't have enough information from the provided documents.'"
        )
    ),
    HumanMessage(
        content=f"""
Context:
{context}

Question:
{query}

Answer in a clear and concise way.
"""
    ),
]

# Generate answer
result = model.invoke(messages)

print("\n" + "=" * 60)
print("Final Answer")
print("=" * 60)
print(result.content)