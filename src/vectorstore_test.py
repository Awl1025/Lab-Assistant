from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import get_provider, require_env

def make_embeddings():
    provider = get_provider()

    # Option A: Google embeddings (good if you're already using Gemini)
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        require_env("GOOGLE_API_KEY")
        return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # Option B: Local embeddings (no API key; slower; bigger install)
    # Switch PROVIDER to "local" if you want to force this path
    if provider in ("local", "huggingface"):
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # If you picked Groq for LLM, you can still choose local embeddings:
    if provider == "groq":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    raise ValueError(f"Unknown PROVIDER: {provider}")

if __name__ == "__main__":
    embeddings = make_embeddings()

    test_docs = [
        Document(page_content="Python is a programming language often used for data science.", metadata={"source": "test"}),
        Document(page_content="Paris is the capital city of France.", metadata={"source": "test"}),
        Document(page_content="Machine learning is a subset of artificial intelligence.", metadata={"source": "test"}),
        Document(page_content="ChromaDB is a vector database used for similarity search.", metadata={"source": "test"}),
    ]

    vectorstore = Chroma.from_documents(
        test_docs,
        embeddings,
        collection_name="infra_test",
        persist_directory="chroma_db",
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    query = "What is a vector database used for?"
    results = retriever.invoke(query)

    print(f"Query: {query}\nTop results:")
    for i, doc in enumerate(results, start=1):
        print(f"{i}. {doc.page_content} (source={doc.metadata.get('source')})")
