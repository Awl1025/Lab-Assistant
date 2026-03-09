import os
from itertools import islice
from dotenv import load_dotenv
from datasets import load_dataset
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.agents import create_agent
from langchain_core.tools import tool

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# MVP config
CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 120 
N_PER_DATASET = 500
CHUNKS_PER_DATASET = 25  # retrieve this many chunks per dataset

def build_splitter():
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

def check_llm():
    print("\nTesting Gemini Connection")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    res = llm.invoke("Infrastructure check: Are you online?")
    print(f"Gemini Response: {res.content}")

def run_vector_proof():
    print("\nBuilding Vector Store")
    splitter = build_splitter()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(embedding_function=embeddings)

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

    # sources collected during RAG
    sources = []

    # learning state
    knowledge_state = {
        "papers_seen": set(),
        "topics": set()
    }

    # track how often papers are used in answers
    paper_scores = {}

    @tool
    def search_paper(query: str):
        """Search scientific papers and return a unique list of paper sources."""
        results = []
        for ds_name in ("Pes2oX", "PubMed", "COREX-18"):
            ds_hits = db.max_marginal_relevance_search(
                query,
                k=CHUNKS_PER_DATASET,
                fetch_k=CHUNKS_PER_DATASET * 3,
                lambda_mult=0.7,
                filter={"dataset": ds_name}
            )
            results.extend(ds_hits)
        paper_chunks = {}
        paper_scores_local = {}
        for r in results:
            dataset = r.metadata.get("dataset")
            title = r.metadata.get("title")
            paper_id = r.metadata.get("paper_id") or r.metadata.get("pmid")
            key = (dataset, paper_id, title)
            if key not in paper_chunks:
                paper_chunks[key] = []
                paper_scores_local[key] = []
            paper_chunks[key].append(r.page_content)
            paper_scores_local[key].append(1.0)

        # initial ranking by similarity score
        candidate_papers = sorted(
            paper_chunks.items(),
            key=lambda x: sum(paper_scores_local[x[0]]) / len(paper_scores_local[x[0]]),
        )

        # MMR diversification
        selected = []
        selected_keys = set()
        for key, chunks in candidate_papers:
            if len(selected) >= 15:
                break
            dataset, paper_id, title = key
            # avoid selecting the same paper twice
            if key in selected_keys:
                continue
            # simple diversity penalty: avoid too many papers from same dataset early
            if len(selected) < 5:
                selected.append((key, chunks))
                selected_keys.add(key)
            else:
                datasets_used = [k[0][0] for k in selected]
                if datasets_used.count(dataset) < 5:
                    selected.append((key, chunks))
                    selected_keys.add(key)

        ranked_papers = selected
        contexts = []
        for (dataset, paper_id, title), chunks in ranked_papers:
            sources.append(f"[{dataset}] {title} (ID: {paper_id})")
            # Keep only the best chunk per paper
            best_chunk = chunks[0][:800]
            contexts.append(
                f"SOURCE (ID: {paper_id}) {title} | {dataset}\n{best_chunk}"
            )
        return "\n\n".join(contexts)

    @tool
    def print_sources(response_text: str):
        """Print only the sources that were used in the agent response."""
        cited_ids = set()
        for match in re.findall(r"ID:\s*(\d+)", response_text):
            cited_ids.add(match)
        filtered = []
        for s in sources:
            for cid in cited_ids:
                if cid in s:
                    filtered.append(s)
                    # update paper score
                    paper_scores[cid] = paper_scores.get(cid, 0) + 1

                    break
        unique = list(dict.fromkeys(filtered))
        for i, s in enumerate(unique, start=1):
            print(f"{i}. {s}")
        return "Sources printed."

    # create agent with Gemini LLM
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    # create agent with access to tools
    agent = create_agent(
        model=model,
        tools=[search_paper, print_sources]
    )

    # persistent memory for the session
    memory = []
    print("\nLab Assistant Ready (or 'q' to quit).")
    while True:
        user_query = input("\nQuery> ")
        if user_query.lower() == 'q':
            print("Exiting Lab Assistant")
            break
        try:
            # reset sources for this query
            sources.clear()

            # detect topic shift and reset learning state if query is unrelated
            if knowledge_state["topics"]:
                last_topic = list(knowledge_state["topics"])[-1]

                # simple heuristic: if the new query shares no keywords with the last topic
                if not any(word in user_query.lower() for word in last_topic.lower().split()):
                    knowledge_state["topics"].clear()
                    knowledge_state["papers_seen"].clear()

            # build retrieval query using learned topic state
            if knowledge_state["topics"]:
                past_topics = " ".join(list(knowledge_state["topics"])[:3])
                search_query = past_topics + " " + user_query
            else:
                search_query = user_query
            context = search_paper.invoke(search_query)

            # Update learning state from retrieved sources
            for s in sources:
                knowledge_state["papers_seen"].add(s)

            knowledge_state["topics"].add(user_query)
            response = [
                {"role": "system", "content": 
                    """
                    You are an expert laboratory research assistant. Use the provided paper context to answer
                    the question and cite sources using (ID: ####).

                    The system maintains an internal knowledge state containing previously discussed topics
                    and papers that have already been retrieved. Use this prior knowledge to better
                    interpret follow-up questions and maintain topic continuity when relevant.

                    If a question builds on a previous topic, use the prior discussion and previously
                    seen papers to help refine your reasoning and citations.

                    If the question clearly introduces a new topic, focus primarily on the newly
                    retrieved paper context.
                    """
                },
            ]
            response.append({"role": "system", "content": context})

            # include previous memory as context
            for m in memory:
                response.append({"role": "user", "content": m["query"]})
                response.append({"role": "assistant", "content": m["response"]})

            # add current query
            response.append({"role": "user", "content": user_query})
            result = agent.invoke({"messages": response})

            # print agent response
            print("\nAgent Response:")
            message = result["messages"][-1].content
            if isinstance(message, list):
                message = "".join(block.get("text", "") for block in message)
            print(message)

            # print sources used in the response
            print("\nSources Used:")
            print_sources.invoke(message)

            # update state memory with the latest query and response
            memory.append({
                "query": user_query,
                "response": str(message)
            })
        except Exception as e:
            print(f"Error {e}")

if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("❌ Error: GOOGLE_API_KEY not found!")
    else:
        print("✅ API Key found.")
    check_llm()
    run_vector_proof()
