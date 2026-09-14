import os
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

PERSIST_DIRECTORY = "db/chroma_db"

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="AI-Sathi-RAG API",
    description="History-Aware RAG Chatbot using Hugging Face + ChromaDB + Gemini",
    version="2.0.0",
)

# -----------------------------
# Embeddings
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Load ChromaDB
# -----------------------------
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# Gemini LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

# -----------------------------
# Conversation History
# -----------------------------
chat_history = []

# -----------------------------
# Request Model
# -----------------------------
class QueryRequest(BaseModel):
    question: str


# -----------------------------
# Home Endpoint
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "🤖 AI-Sathi-RAG API is running successfully!"
    }


# -----------------------------
# Ask Endpoint (History-Aware RAG)
# -----------------------------
@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        user_question = request.question

        # -------------------------------------
        # Step 1 : Rewrite follow-up question
        # -------------------------------------
        if chat_history:

            rewrite_messages = [
                SystemMessage(
                    content=(
                        "Rewrite the user's latest question into a standalone question "
                        "using the conversation history. "
                        "Return ONLY the rewritten question."
                    )
                )
            ] + chat_history + [
                HumanMessage(content=user_question)
            ]

            rewritten_question = llm.invoke(rewrite_messages).content.strip()

        else:
            rewritten_question = user_question

        # -------------------------------------
        # Step 2 : Retrieve Context
        # -------------------------------------
        docs = retriever.invoke(rewritten_question)

        if not docs:
            return {
                "question": user_question,
                "search_question": rewritten_question,
                "answer": "I don't have enough information from the provided documents.",
                "sources": [],
            }

        context = "\n\n".join(doc.page_content for doc in docs)

        # -------------------------------------
        # Step 3 : Generate Final Answer
        # -------------------------------------
        messages = [
            SystemMessage(
                content=(
                    "You are AI-Sathi, a helpful RAG assistant.\n"
                    "Answer ONLY from the provided context.\n"
                    "If the answer is not available, say: "
                    "'I don't have enough information from the provided documents.'"
                )
            )
        ] + chat_history + [
            HumanMessage(
                content=f"""
Context:
{context}

Question:
{user_question}
"""
            )
        ]

        response = llm.invoke(messages)

        answer = response.content

        # -------------------------------------
        # Step 4 : Save Conversation
        # -------------------------------------
        chat_history.append(HumanMessage(content=user_question))
        chat_history.append(AIMessage(content=answer))

        # -------------------------------------
        # Step 5 : Return JSON
        # -------------------------------------
        return {
            "question": user_question,
            "search_question": rewritten_question,
            "answer": answer,
            "sources": [
                {
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                }
                for doc in docs
            ],
            "history_length": len(chat_history) // 2,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Clear Chat History Endpoint
# -----------------------------
@app.post("/clear-history")
def clear_history():
    chat_history.clear()
    return {
        "message": "Conversation history cleared successfully."
    }


# -----------------------------
# View Chat History Endpoint
# -----------------------------
@app.get("/history")
def history():
    history_data = []

    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = "system"

        history_data.append(
            {
                "role": role,
                "content": msg.content,
            }
        )

    return {
        "total_messages": len(history_data),
        "history": history_data,
    }