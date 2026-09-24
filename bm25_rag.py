from document_loader import load_and_split_documents
from rank_bm25 import BM25Okapi


documents = load_and_split_documents()

texts = [doc.page_content for doc in documents]

tokenized_documents = [
    text.lower().split()
    for text in texts
]

bm25 = BM25Okapi(tokenized_documents)


question = input("Ask a question: ")

tokenized_question = question.lower().split()

results = bm25.get_top_n(
    tokenized_question,
    texts,
    n=3
)


print("\nTOP 3 RESULTS:\n")

for i, result in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print(result)