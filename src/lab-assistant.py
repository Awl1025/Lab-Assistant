import os
from itertools import islice
from dotenv import load_dotenv
from datasets import load_dataset

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

PERSIST_DIR = "./chroma_db_pes2ox"

# Chunking settings (token-approx)
CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 120 
N_PAPERS = 100


def build_splitter():
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

def load_pes2ox_docs_streaming(n_papers: int, splitter: RecursiveCharacterTextSplitter):
    ds = load_dataset("laion/Pes2oX-fulltext", split="train", streaming=True)

    docs = []
    for row in islice(ds, n_papers):
        title = row.get("title") or ""
        text = row.get("text") or ""
        paper_id = row.get("id")
        source = row.get("source")
        version = row.get("version")
        created = row.get("created")
        added = row.get("added")

        if not text.strip():
            continue

        # Split full text into structure-aware chunks
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "paper_id": paper_id,
                        "title": title,
                        "source": source,
                        "version": version,
                        "created": created,
                        "added": added,
                        "chunk_index": i,
                    },
                )
            )

    return docs


def check_llm():
    print("\n--- 1. Testing Gemini Connection ---")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    res = llm.invoke("Infrastructure check: Are you online?")
    print(f"Gemini Response: {res.content}")

def run_vector_proof():
    print("\n--- 2. Testing Vector Database (Chroma) using Pes2oX + chunking ---")
    splitter = build_splitter()
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    print(f"Streaming first {N_PAPERS} papers from Pes2oX...")
    docs = load_pes2ox_docs_streaming(N_PAPERS, splitter)
    print(f"Built {len(docs)} chunk-documents.")
    if not docs:
        raise RuntimeError("No documents created—dataset rows may be empty or filtered out.")
    db = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIR)

    query = "What assumptions are made about Lipschitzness / smoothness of the loss?"
    results = db.similarity_search(query, k=3)

    print(f"\nQuery: {query}\nTop Results:")
    for r in results:
        title = (r.metadata.get("title") or "")[:120]
        pid = r.metadata.get("paper_id")
        ci = r.metadata.get("chunk_index")
        snippet = r.page_content[:250].replace("\n", " ")
        print(f"- paper_id={pid} chunk={ci} title={title!r}")
        print(f"  snippet: {snippet}...\n")

def image_description_proof(image_path):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    try:
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
        response = llm.invoke(HumanMessage(content="Describe this image:", attachments=[("image.jpg", img_data)]))
        print(f"Image Description: {response.content}")
    except Exception as e:
        print(f"❌ Image Proof Error: {e}")


if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("❌ Error: GOOGLE_API_KEY not found! Make sure your .env is set.")
    print("✅ API Key found.")
    check_llm()
    run_vector_proof()