import os
from dotenv import load_dotenv
import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"

from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

#
PERSIST_DIRECTORY = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)



model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)



chat_history = []


def ask_question(user_question: str):
    """History-aware RAG Question Answering"""

    print(f"\n🧑 You asked: {user_question}")



    if chat_history:
        rewrite_messages = [
            SystemMessage(
                content=(
                    "Rewrite the user's latest question into a standalone question "
                    "using the previous conversation history. "
                    "Return ONLY the rewritten question."
                )
            )
        ] + chat_history + [
            HumanMessage(content=user_question)
        ]

        rewritten = model.invoke(rewrite_messages)
        search_question = rewritten.content.strip()

        print(f"🔍 Standalone Question: {search_question}")

    else:
        search_question = user_question


    retriever = db.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(search_question)

    print(f"\n📄 Retrieved {len(relevant_docs)} document(s):")

    context = ""

    for i, doc in enumerate(relevant_docs, 1):
        preview = doc.page_content[:150].replace("\n", " ")
        page = doc.metadata.get("page", "Unknown")

        print(f"{i}. Page {page}: {preview}...")

        context += (
            f"\n--- Document {i} (Page {page}) ---\n"
            f"{doc.page_content}\n"
        )


    # Step 3: Generate Answer

    final_messages = [
        SystemMessage(
            content=(
                "You are an AI assistant that answers ONLY from the provided context. "
                "If the answer is not found, reply: "
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

Instructions:
- Answer only from the context.
- Keep the answer clear and concise.
- Mention the page number if possible.
"""
        )
    ]

    result = model.invoke(final_messages)
    answer = result.content


    # Step 4: Save Conversation History

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print("\n🤖 AI Answer:\n")
    print(answer)

    return answer


# Chat Loop

def start_chat():
    print("=" * 50)
    print("🤖 AI-Sathi-RAG Chatbot")
    print("Ask questions about your PDF documents.")
    print("Type 'quit' to exit.")
    print("=" * 50)

    while True:
        question = input("\nYour Question: ")

        if question.lower() in ["quit", "exit"]:
            print("\n👋 Goodbye!")
            break

        ask_question(question)


if __name__ == "__main__":
    start_chat()