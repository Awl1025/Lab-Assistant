import os
from pathlib import Path
from itertools import islice
from dotenv import load_dotenv
from datasets import load_dataset
import re
import gradio as gr

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from deepagents import create_deep_agent
from langchain_core.tools import tool

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# MVP config
CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 120 
N_PER_DATASET = 500
CHUNKS_PER_DATASET = 25  # retrieve this many chunks per dataset

# Persistent storage paths
BASE_DIR = Path("./lab_assistant_store")
PAPERS_DB_PATH = BASE_DIR / "papers_chroma_db"
FEEDBACK_DB_PATH = BASE_DIR / "feedback_chroma_db"
FEEDBACK_K = 4

APP = {
    "db": None,
    "feedback_db": None,
    "agent": None,
    "sources": [],
    "knowledge_state": {"papers_seen": set(), "topics": set()},
    "paper_scores": {},
    "last_query": "",
    "last_answer": "",
    "print_sources_tool": None,
}

def build_splitter():
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

def ensure_store_dirs():
    BASE_DIR.mkdir(parents=True, exist_ok=True)

def has_paper_store(db):
    try:
        return db._collection.count() > 0
    except Exception:
        return False

def init_chroma_databases(embeddings):
    papers_db = Chroma(
        collection_name="papers",
        persist_directory=str(PAPERS_DB_PATH.resolve()),
        embedding_function=embeddings,
    )
    feedback_db = Chroma(
        collection_name="feedback",
        persist_directory=str(FEEDBACK_DB_PATH.resolve()),
        embedding_function=embeddings,
    )
    return papers_db, feedback_db

def check_llm():
    print("\nTesting Gemini Connection")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    res = llm.invoke("Infrastructure check: Are you online?")
    print(f"Gemini Response: {res.content}")

def initialize_app():
    ensure_store_dirs()

    print("\nInitializing persistent Chroma databases")
    splitter = build_splitter()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db, feedback_db = init_chroma_databases(embeddings)

    if has_paper_store(db):
        print(f"Loaded existing paper Chroma database from {PAPERS_DB_PATH}")
    else:
        print(f"No existing paper Chroma database found. Building and saving to {PAPERS_DB_PATH}")

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
                        "title": row.get("title"),
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
                        "title": row.get("title"),
                    }
                )
                for c in chunks
            ]
            db.add_documents(documents)

        print(f"Streaming {N_PER_DATASET} COREX-18 papers...")
        ds3 = load_dataset("laion/COREX-18text", split="train", streaming=True)
        for row in islice(ds3, N_PER_DATASET):
            text = row.get("text") or ""
            if not text.strip():
                continue
            chunks = splitter.split_text(text)
            documents = [
                Document(
                    page_content=c,
                    metadata={
                        "dataset": "COREX-18",
                        "paper_id": row.get("id"),
                        "title": row.get("title"),
                    }
                )
                for c in chunks
            ]
            db.add_documents(documents)

    APP["db"] = db
    APP["feedback_db"] = feedback_db
    APP["sources"] = []
    APP["knowledge_state"] = {"papers_seen": set(), "topics": set()}
    APP["paper_scores"] = {}

    @tool
    def search_paper(query: str):
        """Search scientific papers and return a unique list of paper sources."""
        results = []
        for ds_name in ("Pes2oX", "PubMed", "COREX-18"):
            ds_hits = APP["db"].max_marginal_relevance_search(
                query,
                k=CHUNKS_PER_DATASET,
                fetch_k=CHUNKS_PER_DATASET * 3,
                lambda_mult=0.7,
                filter={"dataset": ds_name}
            )
            results.extend(ds_hits)
        paper_chunks = {}
        for r in results:
            dataset = r.metadata.get("dataset")
            title = r.metadata.get("title")
            paper_id = r.metadata.get("paper_id") or r.metadata.get("pmid")
            key = (dataset, paper_id, title)
            if key not in paper_chunks:
                paper_chunks[key] = []
            paper_chunks[key].append(r.page_content)
        contexts = []
        seen = set()
        for (dataset, paper_id, title), chunks in paper_chunks.items():
            current_key = (dataset, paper_id, title)
            if current_key in seen:
                continue
            seen.add(current_key)
            APP["sources"].append(f"[{dataset}] {title} (ID: {paper_id})")
            best_chunk = chunks[0][:800]
            contexts.append(
                f"SOURCE (ID: {paper_id}) {title} | {dataset}\n{best_chunk}"
            )
            if len(contexts) >= 15:
                break
        return "\n\n".join(contexts)
    
    @tool
    def print_sources(response_text: str):
        """Print only the sources that were used in the agent response."""
        cited_ids = set(re.findall(r"ID:\s*(\d+)", response_text))
        filtered = []
        for s in APP["sources"]:
            for cid in cited_ids:
                if cid in s:
                    filtered.append(s)
                    APP["paper_scores"][cid] = APP["paper_scores"].get(cid, 0) + 1
                    break
        unique = list(dict.fromkeys(filtered))
        if not unique:
            return "No cited sources found."
        return "\n".join(f"{i}. {s}" for i, s in enumerate(unique, start=1))
    APP["print_sources_tool"] = print_sources

    @tool
    def search_feedback(query: str):
        """Retrieve prior yes/no feedback results relevant to the current query."""
        try:
            hits = APP["feedback_db"].similarity_search(query, k=FEEDBACK_K)
        except Exception:
            hits = []
        if not hits:
            return ""

        feedback_contexts = []
        for doc in hits:
            feedback = doc.metadata.get("feedback", "unknown")
            original_query = doc.metadata.get("query", "")
            feedback_contexts.append(
                f"FEEDBACK feedback={feedback} | prior_query={original_query}\n{doc.page_content}"
            )
        return "\n\n".join(feedback_contexts)

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    deep_agent_system_prompt = """
    You are an expert laboratory research assistant with access to tools.

    Your workflow:
    1. Use search_paper to retrieve relevant paper context for scientific questions.
    2. Use search_feedback to retrieve prior yes/no feedback memories when they may improve answer quality.
    3. Answer clearly and cite sources using (ID: ####) when using retrieved paper context.
    4. After writing the final answer, use print_sources exactly once to list only the cited sources.

    Rules:
    - Prefer retrieved paper evidence over unsupported general claims.
    - If the question is a follow-up, use the current context and prior retrieved context to stay consistent.
    - Treat feedback=yes memories as preferred answer patterns.
    - Treat feedback=no memories as patterns to avoid.
    - Do not call print_sources repeatedly.
    - Do not use Markdown formatting. Output plain terminal-friendly text only.
    - Do not use bold, italics, headings, or bullet symbols that rely on Markdown rendering.
    """

    agent = create_deep_agent(
        model=model,
        tools=[search_paper, search_feedback, print_sources],
        system_prompt=deep_agent_system_prompt,
    )
    APP["agent"] = agent

def ask_agent(user_query: str):
    APP["sources"].clear()

    if APP["knowledge_state"]["topics"]:
        last_topic = list(APP["knowledge_state"]["topics"])[-1]
        if not any(word in user_query.lower() for word in last_topic.lower().split()):
            APP["knowledge_state"]["topics"].clear()
            APP["knowledge_state"]["papers_seen"].clear()

    if APP["knowledge_state"]["topics"]:
        past_topics = " ".join(list(APP["knowledge_state"]["topics"])[:3])
        search_query = past_topics + " " + user_query
    else:
        search_query = user_query

    response = [
        {
            "role": "system",
            "content": (
                f"Current retrieval hint: {search_query}\n"
                "Decide for yourself when to call search_paper, search_feedback, and print_sources. "
                "Call search_paper before answering substantive scientific questions. "
                "Call print_sources exactly once after the final answer if the answer cites sources."
            ),
        },
        {"role": "user", "content": user_query},
    ]
    result = APP["agent"].invoke({"messages": response})

    for s in APP["sources"]:
        APP["knowledge_state"]["papers_seen"].add(s)
    APP["knowledge_state"]["topics"].add(user_query)

    final_message = ""
    for msg in reversed(result["messages"]):
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip() and content.strip() != "FINAL_ANSWER":
            final_message = content
            break
        if isinstance(content, list):
            joined = "".join(block.get("text", "") for block in content if isinstance(block, dict))
            if joined.strip() and joined.strip() != "FINAL_ANSWER":
                final_message = joined
                break

    if APP["print_sources_tool"] is not None:
        cited_sources = APP["print_sources_tool"].invoke(final_message)
    else:
        cited_sources = "No cited sources found."
    APP["last_query"] = user_query
    APP["last_answer"] = final_message
    return final_message, cited_sources, ""


def save_feedback(feedback_value: str):
    if not APP["last_query"] or not APP["last_answer"]:
        return "No answer available to rate yet."

    memory_text = f"User query: {APP['last_query']}\nAssistant response: {APP['last_answer']}"
    memory_doc = Document(
        page_content=memory_text,
        metadata={
            "query": APP["last_query"],
            "feedback": feedback_value,
        },
    )
    APP["feedback_db"].add_documents([memory_doc])
    return f"Stored {feedback_value} feedback in persistent memory."

def launch_gradio():
    with gr.Blocks(title="Lab Assistant") as demo:
        gr.Markdown("# Lab Assistant")
        gr.Markdown("Ask a scientific or technical question. The app uses a persistent Chroma paper database and persistent yes/no feedback memory.")

        query_input = gr.Textbox(label="Query", placeholder="Ask a question...", lines=3)
        submit_btn = gr.Button("Submit")
        answer_output = gr.Textbox(label="Agent Response", lines=14)
        sources_output = gr.Textbox(label="Cited Sources", lines=8)
        status_output = gr.Textbox(label="Status", lines=2)

        with gr.Row():
            yes_btn = gr.Button("Yes")
            no_btn = gr.Button("No")

        submit_btn.click(
            fn=ask_agent,
            inputs=[query_input],
            outputs=[answer_output, sources_output, status_output],
        )
        query_input.submit(
            fn=ask_agent,
            inputs=[query_input],
            outputs=[answer_output, sources_output, status_output],
        )
        yes_btn.click(fn=lambda: save_feedback("yes"), inputs=None, outputs=[status_output])
        no_btn.click(fn=lambda: save_feedback("no"), inputs=None, outputs=[status_output])

    demo.launch()

if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("❌ Error: GOOGLE_API_KEY not found!")
    else:
        print("✅ API Key found.")
    check_llm()
    initialize_app()
    launch_gradio()