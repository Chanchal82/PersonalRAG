import json
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def load_pages(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()

        pages.append({
            "page": i + 1,
            "text": text
        })

    return pages


def build_document_text(pages):
    document_text = ""

    for page in pages:
        document_text += (
            f"\n\nPAGE {page['page']}\n"
            f"{page['text']}"
        )

    return document_text


def build_tree(llm, document_text):

    prompt = f"""
You are a document structure analyzer.

Analyze the document and create a hierarchical tree.

Each node must contain:
- title
- pages
- children

Pages should contain the page numbers belonging to that section.

Return ONLY valid JSON.

Example:

{{
    "title": "Document",
    "pages": [],
    "children": [
        {{
            "title": "Introduction",
            "pages": [1, 2],
            "children": []
        }},
        {{
            "title": "Information Retrieval",
            "pages": [3, 4, 5],
            "children": [
                {{
                    "title": "Inverted Index",
                    "pages": [4],
                    "children": []
                }}
            ]
        }}
    ]
}}

Document:

{document_text}
"""

    response = llm.invoke(prompt)

    print("\nGEMINI RESPONSE:")
    print(response.text)

    return json.loads(response.text)


def get_all_nodes(node):

    nodes = [node]

    for child in node.get("children", []):
        nodes.extend(get_all_nodes(child))

    return nodes


def select_section(llm, question, nodes):

    sections = []

    for node in nodes:
        sections.append({
            "title": node["title"],
            "pages": node.get("pages", [])
        })

    prompt = f"""
You are navigating a document tree.

Question:
{question}

Available sections:

{json.dumps(sections, indent=2)}

Choose the section most relevant to the question.

Return ONLY the exact section title.
"""

    response = llm.invoke(prompt)

    return response.text.strip()


def find_node(node, section_name):

    if node["title"].lower() == section_name.lower():
        return node

    for child in node.get("children", []):

        result = find_node(child, section_name)

        if result:
            return result

    return None


def get_context(pages, relevant_pages):

    context = ""

    for page in pages:

        if page["page"] in relevant_pages:

            context += f"\n\nPAGE {page['page']}\n"
            context += page["text"]

    return context


def generate_answer(llm, question, context):

    prompt = f"""
Answer the question using only the provided context.

Question:
{question}

Context:
{context}

If the answer is not present in the context,
say that the information is not available in the document.
"""

    response = llm.invoke(prompt)

    return response.text


def main():

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash"
    )

    pages = load_pages("data/rag.pdf")

    document_text = build_document_text(pages)

    print("Number of pages:", len(pages))
    print("Characters:", len(document_text))

    tree = build_tree(llm, document_text)

    print("\nDOCUMENT TREE:")
    print(json.dumps(tree, indent=2))

    nodes = get_all_nodes(tree)

    question = input("\nAsk a question: ")

    selected_section = select_section(
        llm,
        question,
        nodes
    )

    print("\nSelected section:", selected_section)

    selected_node = find_node(
        tree,
        selected_section
    )

    if selected_node is None:

        print("Could not find the relevant section.")

        return

    relevant_pages = selected_node.get("pages", [])

    print("Relevant pages:", relevant_pages)

    context = get_context(
        pages,
        relevant_pages
    )

    answer = generate_answer(
        llm,
        question,
        context
    )

    print("\nANSWER:")
    print(answer)


if __name__ == "__main__":
    main()