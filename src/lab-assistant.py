import os
from itertools import islice
from dotenv import load_dotenv
from datasets import load_dataset

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "./chroma_db_pes2ox"
CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 120 
N_PER_DATASET = 1000


def build_splitter():
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

def check_llm():
    print("\nTesting Gemini Connection")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    res = llm.invoke("Infrastructure check: Are you online?")
    print(f"Gemini Response: {res.content}")

def run_vector_proof():
    print("\n Building Vector Store")
    splitter = build_splitter()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma( persist_directory=PERSIST_DIR, embedding_function=embeddings)

    # Pes2oX dataset is smaller and more consistent, so we can stream and add in one pass
    print(f"Streaming {N_PER_DATASET} Pes2oX papers")
    ds1 = load_dataset("laion/Pes2oX-fulltext", split="train", streaming=True)
    for row in islice(ds1, N_PER_DATASET):
        text = row.get("text") or ""
        if not text.strip():
            continue
        chunks = splitter.split_text(text)
        documents = [
            Document(
                page_content=c,
                metadata={
                    "dataset": "Pes2oX",
                    "paper_id": row.get("id"),
                }
            )
            for c in chunks
        ]
        db.add_documents(documents)

    # PubMed dataset is larger and more variable, so we stream and add incrementally to avoid memory issues
    print(f"Streaming {N_PER_DATASET} PubMed papers")
    ds2 = load_dataset("common-pile/pubmed", split="train", streaming=True)
    for row in islice(ds2, N_PER_DATASET):
        text = row.get("text") or ""
        if not text.strip():
            continue
        chunks = splitter.split_text(text)
        documents = [
            Document(
                page_content=c,
                metadata={
                    "dataset": "PubMed",
                    "pmid": row.get("id"),
                }
            )
            for c in chunks
        ]
        db.add_documents(documents)

    # -------- Query --------
    query = "What assumptions are made about Lipschitz continuity in optimization?"
    results = db.similarity_search(query, k=3)

    print(f"Query: {query}\n")
    for r in results:
        print(f"[{r.metadata.get('dataset')}]")
        print(r.page_content[:250].replace("\n", " "))
        print()

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
    else:
        print("✅ API Key found.")
    check_llm()
    run_vector_proof()