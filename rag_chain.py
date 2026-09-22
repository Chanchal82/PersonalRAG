from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from vector_store import retriever


prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context.

Context:
{context}

Question:
{question}
""")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash"
)

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

response = rag_chain.invoke("What is an document scoring?")

print(response.text)