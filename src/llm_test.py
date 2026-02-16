from dotenv import load_dotenv
load_dotenv()  # harmless in Codespaces; useful locally

from langchain_google_genai import ChatGoogleGenerativeAI
from config import require_env

if __name__ == "__main__":
    require_env("GOOGLE_API_KEY")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
    )

    resp = llm.invoke("Say hello in one sentence. Then list 3 things you can do.")
    print(resp.content)
