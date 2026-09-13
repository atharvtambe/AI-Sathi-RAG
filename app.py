import os
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env")

PERSIST_DIRECTORY = "db/chroma_db"

app = FastAPI(
    title="RAG Chatbot API",
    description="PDF Question Answering using Chroma + Gemini",
    version="1.0"
)


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

retriever = db.as_retriever(search_kwargs={"k": 5})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0
)

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {
        "message": "RAG API is running 🚀"
    }


@app.post("/ask")
def ask_question(request: QueryRequest):

    try:
        docs = retriever.invoke(request.question)

        if not docs:
            return {
                "question": request.question,
                "answer": "I don't have enough information from the provided documents.",
                "sources": []
            }

        context = "\n\n".join([doc.page_content for doc in docs])

        messages = [
            SystemMessage(
                content=(
                    "You are a helpful RAG assistant. "
                    "Answer ONLY from the provided context. "
                    "If the answer isn't available, say you don't have enough information."
                )
            ),
            HumanMessage(
                content=f"""
Context:
{context}

Question:
{request.question}
"""
            )
        ]

        response = llm.invoke(messages)

        return {
            "question": request.question,
            "answer": response.content,
            "sources": [
                {
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page")
                }
                for doc in docs
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))