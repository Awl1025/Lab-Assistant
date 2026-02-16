from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-3.0-pro", temperature=0)
response = llm.invoke("Say hello")
print(response.content)
