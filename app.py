import os
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Chroma + Ollama Embeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Gemini LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

# ==========================================================
# Load Environment Variables
# ==========================================================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

# ==========================================================
# Paths
# ==========================================================
PERSIST_DIRECTORY = "db/chroma_db"

# ==========================================================
# FastAPI App
# ==========================================================
app = FastAPI(
    title="BIS AI Assistant API",
    description="Smart India Hackathon 2026 - BIS RAG Assistant",
    version="1.0.0",
)

# ==========================================================
# Ollama Embeddings (LOCAL)
# ==========================================================
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text:latest",
    base_url="http://127.0.0.1:11434",
)

# ==========================================================
# Load Chroma Vector Database
# ==========================================================
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.7,
    },
)

print("✅ ChromaDB loaded successfully.")

# ==========================================================
# Gemini 3.6 Flash
# ==========================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

print("✅ Gemini model loaded successfully.")

# ==========================================================
# Conversation History
# ==========================================================
chat_history = []

# ==========================================================
# Request Model
# ==========================================================
class QueryRequest(BaseModel):
    question: str

# ==========================================================
# Health Check
# ==========================================================
@app.get("/")
def health_check():
    return {
        "status": "running",
        "message": "🤖 BIS AI Assistant API is running successfully."
    }

# ==========================================================
# Ask Question Endpoint
# ==========================================================
@app.post("/chat")
def ask_question(request: QueryRequest):
    try:

        user_question = request.question.strip()

        # -----------------------------------------------
        # Rewrite follow-up question using history
        # -----------------------------------------------
        if chat_history:

            rewrite_prompt = [
                SystemMessage(
                    content=(
                        "Rewrite the latest user question into a standalone question "
                        "using previous conversation history. "
                        "Return ONLY the rewritten question."
                    )
                )
            ] + chat_history + [
                HumanMessage(content=user_question)
            ]

            rewritten_question = llm.invoke(rewrite_prompt).content.strip()

        else:
            rewritten_question = user_question

        # -----------------------------------------------
        # Retrieve Relevant Context
        # -----------------------------------------------
        docs = retriever.invoke(rewritten_question)

        if not docs:
            return {
                "question": user_question,
                "answer": "I don't have enough information from the provided BIS documents.",
                "sources": [],
            }

        context = ""
        sources = []

        for doc in docs:

            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")

            sources.append({
                "source": source,
                "page": page,
            })

            context += f"""
SOURCE: {source}
PAGE: {page}

CONTENT:
{doc.page_content}

-----------------------------------------
"""

        # -----------------------------------------------
        # Generate Answer
        # -----------------------------------------------
        messages = [
            SystemMessage(
                content=(
                    "You are BIS AI Assistant built for Smart India Hackathon 2026.\n\n"
                    "Rules:\n"
                    "1. Answer ONLY using the retrieved context.\n"
                    "2. Do NOT use outside knowledge.\n"
                    "3. If answer not found, say exactly:\n"
                    "'I don't have enough information from the provided BIS documents.'\n"
                    "4. Mention document source and page number whenever possible."
                )
            ),
            HumanMessage(
                content=f"""
Retrieved Context:

{context}

Question:
{user_question}

Provide answer in this format:

Answer:
...

References:
- Source (Page Number)
"""
            ),
        ]

        response = llm.invoke(messages)

        # Remove Gemini "extras" and keep only text
        if isinstance(response.content, list):
            answer = "\n".join(
                item["text"]
                for item in response.content
                if isinstance(item, dict)
                and item.get("type") == "text"
                and "text" in item
            )
        else:
            answer = str(response.content)

        # -----------------------------------------------
        # Save Conversation History
        # -----------------------------------------------
        chat_history.append(HumanMessage(content=user_question))
        chat_history.append(AIMessage(content=answer))

        # Keep last 5 conversations only
        if len(chat_history) > 10:
            chat_history[:] = chat_history[-10:]

        # -----------------------------------------------
        # Response
        # -----------------------------------------------
        return {
            "question": user_question,
            "search_question": rewritten_question,
            "answer": answer,
            "sources": sources,
            "history_length": len(chat_history) // 2,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================================
# View Conversation History
# ==========================================================
@app.get("/history")
def history():

    history_data = []

    for msg in chat_history:

        role = (
            "user"
            if isinstance(msg, HumanMessage)
            else "assistant"
        )

        history_data.append({
            "role": role,
            "content": msg.content,
        })

    return {
        "total_messages": len(history_data),
        "history": history_data,
    }

# ==========================================================
# Clear Conversation History
# ==========================================================
@app.post("/clear-history")
def clear_history():
    chat_history.clear()
    return {
        "message": "Conversation history cleared successfully."
    }

# ==========================================================
# Run API
# ==========================================================
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )

