import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader,PyMuPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()


def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")

    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    # Keep only pages with actual text
    documents = [
        doc for doc in documents
        if doc.page_content and doc.page_content.strip()
    ]

    print(f"Loaded {len(documents)} pages with text.")

    if len(documents) == 0:
        raise ValueError(
            "No readable text found in the PDF. The PDF may be scanned or image-based."
        )

    return documents



def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks"""
    print("\nSplitting documents...")

    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create ChromaDB vector store using Hugging Face embeddings"""
    print("\nCreating embeddings...")

    embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    print("Creating ChromaDB...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("Vector store created successfully!")
    return vectorstore


def main():
    print("=== RAG Document Ingestion Pipeline ===\n")

    docs_path = "docs"
    persist_directory = "db/chroma_db"

    # Remove old DB (important if switching from Mistral/Google embeddings)
    if os.path.exists(persist_directory):
        print("Removing old vector database...")
        shutil.rmtree(persist_directory)

    # Load PDFs
    documents = load_documents(docs_path)

    # Split into chunks
    chunks = split_documents(documents)

    # Create vector store
    create_vector_store(chunks, persist_directory)

    print("\nIngestion completed successfully!")


if __name__ == "__main__":
    main()