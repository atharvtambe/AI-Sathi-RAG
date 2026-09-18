import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import OllamaEmbeddings

# =====================================================
# Load API Key
# =====================================================
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file.")

# =====================================================
# Chroma Database Location
# =====================================================
PERSIST_DIRECTORY = "db/chroma_db"

# =====================================================
# Gemini Embedding Model (Must match ingestion pipeline)
# =====================================================
embedding_model = OllamaEmbeddings(
        model="nomic-embed-text"
    )

# =====================================================
# Load Existing Chroma Database
# =====================================================
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model,
)

print("✅ ChromaDB Loaded Successfully.")

# =====================================================
# Gemini 3.6 Flash Model
# =====================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
    temperature=0,
)

# =====================================================
# User Query
# =====================================================
query = input("\n🔍 Ask your BIS question:\n> ")

# =====================================================
# Retriever (MMR gives better diversity)
# =====================================================
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.7,
    },
)

relevant_docs = retriever.invoke(query)

print("\n" + "=" * 70)
print("USER QUERY")
print("=" * 70)
print(query)

if len(relevant_docs) == 0:
    print("\n❌ No relevant documents found.")
    exit()

# =====================================================
# Build Context
# =====================================================
context = ""

print("\n" + "=" * 70)
print("RETRIEVED DOCUMENTS")
print("=" * 70)

for i, doc in enumerate(relevant_docs, start=1):

    source = doc.metadata.get("source", "Unknown Source")
    page = doc.metadata.get("page", "N/A")

    print(f"\n📄 Document {i}")
    print(f"Source : {source}")
    print(f"Page   : {page}")
    print("-" * 50)
    print(doc.page_content[:300])
    print("...")

    context += f"""
SOURCE: {source}
PAGE: {page}

CONTENT:
{doc.page_content}

---------------------------------------
"""

# =====================================================
# Prompt for Gemini
# =====================================================
messages = [
    SystemMessage(
        content="""
You are BIS AI Assistant built for Smart India Hackathon 2026.

Your job is to answer questions ONLY from the retrieved BIS documents.

Rules:
1. Do NOT use outside knowledge.
2. If information is missing, reply:
   "I don't have enough information from the provided BIS documents."
3. If multiple documents contain relevant information, combine them.
4. Mention the document source and page number.
5. Keep answers concise and factual.
6. If the user asks about certification, clearly mention the certification scheme if present.
"""
    ),
    HumanMessage(
        content=f"""
Retrieved BIS Context

{context}

Question:
{query}

Return your answer in this format:

Answer:
<answer>

Source(s):
- <source name> (Page <page>)
"""
    ),
]

# =====================================================
# Generate Answer
# =====================================================
print("\n🤖 Generating answer with Gemini...\n")

try:
    response = llm.invoke(messages)

    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(response.content)

except Exception as e:
    print("❌ Gemini Error:")
    print(e)