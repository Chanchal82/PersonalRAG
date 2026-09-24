from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch

from vector_store import retriever


load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash"
)


web_search = TavilySearch(
    max_results=3
)


# -------------------------
# 1. Document Relevance Grader
# -------------------------

grader_prompt = ChatPromptTemplate.from_template("""
You are a document relevance grader.

Question:
{question}

Document:
{document}

Is this document relevant to answering the question?

Return only YES or NO.
""")


def grade_document(question, document):

    response = (grader_prompt | llm).invoke({
        "question": question,
        "document": document.page_content
    })

    return response.text.strip().upper() == "YES"


# -------------------------
# 2. Query Rewriter
# -------------------------

rewrite_prompt = ChatPromptTemplate.from_template("""
Rewrite the user's question into a better search query.

Original question:
{question}

Return only the rewritten search query.
""")


def rewrite_question(question):

    response = (rewrite_prompt | llm).invoke({
        "question": question
    })

    return response.text.strip()


# -------------------------
# 3. Generate Answer
# -------------------------

def generate_answer(question, context):

    response = llm.invoke(f"""
Answer the question using only the context below.

Question:
{question}

Context:
{context}

If the answer is not present in the context,
say that the information is not available.
""")

    return response.text


# -------------------------
# 4. CRAG
# -------------------------

question = input("Ask a question: ")


# Retrieve from our PDF
docs = retriever.invoke(question)


# Grade retrieved documents
relevant_docs = []

for doc in docs:

    if grade_document(question, doc):
        relevant_docs.append(doc)


# -------------------------
# PATH 1: Relevant documents found
# -------------------------

if relevant_docs:

    print("\nRelevant documents found.")

    context = "\n\n".join(
        doc.page_content
        for doc in relevant_docs
    )

    answer = generate_answer(
        question,
        context
    )

    print("\nANSWER:")
    print(answer)


# -------------------------
# PATH 2: No relevant documents
# -------------------------

else:

    print("\nNo relevant documents found.")

    # Rewrite query
    new_question = rewrite_question(question)

    print("\nRewritten query:")
    print(new_question)


    # Search the web
    search_results = web_search.invoke(new_question)


    # Extract web content
    web_context = "\n\n".join(
        result["content"]
        for result in search_results["results"]
    )

    answer = generate_answer(
        question,
        web_context
    )

    print("\nANSWER:")
    print(answer)