from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter




def load_and_split_documents():
    loader = PyPDFLoader("data/rag.pdf")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)       
    return chunks




# print(f"Loaded {len(documents)} documents from the PDF file.")
# print(documents[0].page_content) 
#  print(f"Split into {len(chunks)} chunks.")
# print(chunks[0].page_content) 
# print(chunks[0].metadata) 
# print(len(vector)) 
# print(vector[:5])