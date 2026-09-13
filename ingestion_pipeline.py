import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()


def load_documents(docs_path="docs"):
    """Load all PDF files from the docs directory"""
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory '{docs_path}' not found.")

    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    if not documents:
        raise FileNotFoundError(f"No PDF files found inside '{docs_path}'.")

    print(f"Loaded {len(documents)} pages.")

    # Preview first 2 pages
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}")
        print("Source:", doc.metadata["source"])
        print("Length:", len(doc.page_content))
        print("Preview:", doc.page_content[:120], "...")

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

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
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