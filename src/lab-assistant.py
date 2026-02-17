import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# 1. Load the hidden .env file
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    print("❌ Error: GOOGLE_API_KEY not found!")
    exit()
else:
    print("✅ API Key found.")

def check_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    try:
        res = llm.invoke("Infrastructure check: Are you online?")
        print(f"Gemini Response: {res.content}")
    except Exception as e:
        print(f"❌ LLM Error: {e}")

def run_vector_proof():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        test_docs = [
            Document(page_content="The Pes2oX dataset contains cleaned full-text academic papers for LLM training.", metadata={"id": 1}),
        ]
        db = Chroma.from_documents(test_docs, embeddings, persist_directory="./chroma_db")
        query = "What is the Pes2oX dataset used for?"
        results = db.similarity_search(query, k=1)
        
        print(f"Query: {query}")
        print(f"Top Result: {results[0].page_content}")
    except Exception as e:
        print(f"❌ Vector DB Error: {e}")

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
    check_llm() 
    run_vector_proof()