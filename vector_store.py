from langchain_chroma import Chroma
from document_loader import load_and_split_documents
from langchain_google_genai import GoogleGenerativeAIEmbeddings

chunks = load_and_split_documents()

embeddings=GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

results = vector_store.similarity_search(
    "what is the inverted index",
    k=2
)

for doc in results:
    print(doc.page_content)
    print(doc.metadata)