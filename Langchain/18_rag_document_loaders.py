# =============================================================================
# FILE: 18_rag_document_loaders.py
# PART: 5 - RAG  |  LEVEL: Intermediate
# =============================================================================
#
# THE STORY:
#   Your knowledge base doesn't have to be one text file.
#   It can be a folder full of PDFs, a spreadsheet of customer data,
#   a webpage, and a JSON export — all combined into one searchable index.
#
#   Real knowledge bases are messy. Multiple formats. Multiple sources.
#   This file shows you how to load ANY format and combine them
#   into a single RAG system with source-aware retrieval.
#
# WHAT YOU WILL LEARN:
#   1. How to load multiple document formats for RAG
#   2. All major document loaders with working examples
#   3. Combining docs from multiple sources into one vector store
#   4. Source attribution: which file did the answer come from?
#   5. load_any_file() — auto-detect loader by file extension
#
# HOW THIS CONNECTS:
#   Previous: 17_rag_basic.py — the 8-step RAG pipeline basics
#   Next:     19_rag_advanced_retrievers.py — MMR, multi-query, compression
# =============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# All available document loaders
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    DirectoryLoader,
)

load_dotenv()

print("=" * 60)
print("  CHAPTER 18: RAG with Multiple Document Loaders")
print("=" * 60)
print("""
  Real knowledge bases have multiple sources and formats.
  This file shows how to combine them all into one RAG system.
""")

# =============================================================================
# SECTION 1: THE FULL LOADER MENU (Choose what you need)
# =============================================================================
# Below is a reference for ALL available loaders.
# Uncomment the ones you need. The rest are shown for learning.

print("--- Section 1: Document Loader Reference ---")
print("""
  Available loaders (uncomment to use):

  # TEXT FILES (.txt)
  from langchain_community.document_loaders import TextLoader
  loader = TextLoader("path/to/file.txt", encoding="utf-8")

  # PDF FILES (.pdf)  |  pip install pypdf
  from langchain_community.document_loaders import PyPDFLoader
  loader = PyPDFLoader("path/to/file.pdf")
  # Each page = one Document with metadata["page"] = page_number

  # WORD DOCUMENTS (.docx)  |  pip install docx2txt
  from langchain_community.document_loaders import Docx2txtLoader
  loader = Docx2txtLoader("path/to/file.docx")

  # CSV FILES (.csv)
  from langchain_community.document_loaders import CSVLoader
  loader = CSVLoader(file_path="path/to/file.csv")
  # Each row = one Document

  # JSON FILES (.json)  |  pip install jq
  from langchain_community.document_loaders import JSONLoader
  loader = JSONLoader(file_path="data.json", jq_schema=".[]", text_content=False)

  # HTML FILES (.html)  |  pip install unstructured
  from langchain_community.document_loaders import UnstructuredHTMLLoader
  loader = UnstructuredHTMLLoader("path/to/file.html")

  # MARKDOWN FILES (.md)  |  pip install unstructured
  from langchain_community.document_loaders import UnstructuredMarkdownLoader
  loader = UnstructuredMarkdownLoader("path/to/file.md")

  # WEB PAGES  |  pip install beautifulsoup4
  from langchain_community.document_loaders import WebBaseLoader
  loader = WebBaseLoader("https://example.com/page")

  # ENTIRE DIRECTORY (all files of one type)
  from langchain_community.document_loaders import DirectoryLoader
  loader = DirectoryLoader("folder/", glob="**/*.pdf", loader_cls=PyPDFLoader)
""")

# =============================================================================
# SECTION 2: LOADING MULTIPLE SOURCES
# =============================================================================
# We'll build a multi-source knowledge base using the files in our data/ folder.

print("\n--- Section 2: Building a Multi-Source Knowledge Base ---")

all_docs = []

# Source 1: Text file (knowledge.txt)
try:
    txt_loader = TextLoader("Langchain/data/knowledge.txt", encoding="utf-8")
    txt_docs = txt_loader.load()
    all_docs.extend(txt_docs)
    print(f"  [TEXT] Loaded {len(txt_docs)} doc(s) from knowledge.txt")
except FileNotFoundError:
    print("  [TEXT] knowledge.txt not found — using fallback")
    txt_docs = [Document(
        page_content="Dinesh is learning LangChain to build AI agents. He works with n8n and automation.",
        metadata={"source": "Langchain/data/knowledge.txt"}
    )]
    all_docs.extend(txt_docs)

# Source 2: CSV file (sample.csv created in chapter 13)
try:
    csv_loader = CSVLoader(file_path="Langchain/data/sample.csv")
    csv_docs = csv_loader.load()
    all_docs.extend(csv_docs)
    print(f"  [CSV ] Loaded {len(csv_docs)} doc(s) from sample.csv")
except FileNotFoundError:
    print("  [CSV ] sample.csv not found — using fallback")
    csv_docs = [Document(
        page_content="name: Dinesh, role: AI Engineer, company: Freelance, skill: LangChain",
        metadata={"source": "Langchain/data/sample.csv", "row": 0}
    )]
    all_docs.extend(csv_docs)

# Source 3: Inline Documents (programmatically created)
# In production, these might come from a database query or API call
inline_docs = [
    Document(
        page_content="LangChain Agents use ReAct reasoning: Thought → Action → Observation → repeat.",
        metadata={"source": "inline", "topic": "agents", "author": "curriculum"}
    ),
    Document(
        page_content="LangGraph is used for complex multi-step workflows with cycles and state management.",
        metadata={"source": "inline", "topic": "langgraph", "author": "curriculum"}
    ),
    Document(
        page_content="FAISS is a local vector store that requires no external server or database setup.",
        metadata={"source": "inline", "topic": "vector_stores", "author": "curriculum"}
    ),
]
all_docs.extend(inline_docs)
print(f"  [INLINE] Added {len(inline_docs)} programmatic doc(s)")

print(f"\n  Total documents: {len(all_docs)}")
print(f"  Sources: {set(doc.metadata.get('source', 'unknown') for doc in all_docs)}")

# =============================================================================
# SECTION 3: THE auto-detect loader UTILITY FUNCTION
# =============================================================================
# A handy function that picks the right loader based on file extension.
# In production, this saves you from writing if/elif chains everywhere.

def load_any_file(file_path: str) -> list:
    """
    Auto-detect the right loader based on file extension.
    Returns a list of Documents.

    Supports: .txt, .pdf, .docx, .csv, .json, .html, .md
    """
    extension = os.path.splitext(file_path)[1].lower()

    loader_map = {
        ".txt": lambda p: TextLoader(p, encoding="utf-8"),
        ".csv": lambda p: CSVLoader(file_path=p),
    }

    # Try to import optional loaders
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader_map[".pdf"] = lambda p: PyPDFLoader(p)
    except ImportError:
        pass

    try:
        from langchain_community.document_loaders import Docx2txtLoader
        loader_map[".docx"] = lambda p: Docx2txtLoader(p)
    except ImportError:
        pass

    try:
        from langchain_community.document_loaders import UnstructuredHTMLLoader, UnstructuredMarkdownLoader
        loader_map[".html"] = lambda p: UnstructuredHTMLLoader(p)
        loader_map[".md"] = lambda p: UnstructuredMarkdownLoader(p)
    except ImportError:
        pass

    if extension not in loader_map:
        raise ValueError(f"Unsupported file type: {extension}")

    loader = loader_map[extension](file_path)
    return loader.load()

print("\n--- Section 3: load_any_file() Utility ---")
print("  This function auto-detects the right loader by file extension.")

# Test it with our CSV file
try:
    auto_loaded = load_any_file("Langchain/data/sample.csv")
    print(f"  Auto-loaded sample.csv: {len(auto_loaded)} documents")
except Exception as e:
    print(f"  Auto-load test: {e}")

# =============================================================================
# SECTION 4: BUILD THE RAG SYSTEM ON COMBINED SOURCES
# =============================================================================
print("\n--- Section 4: Building Multi-Source RAG System ---")

# Split all documents
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
split_docs = splitter.split_documents(all_docs)
print(f"\nAfter splitting: {len(split_docs)} chunks")

# Embeddings
embeddings = OpenAIEmbeddings(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/text-embedding-3-small",
)

# Vector store
print("Creating FAISS vector store (embedding all chunks)...")
vectorstore = FAISS.from_documents(split_docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
)

# RAG Prompt (with source attribution)
prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have this information."

Context:
{context}

Question: {question}

Answer:""")

# =============================================================================
# SECTION 5: RAG WITH SOURCE ATTRIBUTION
# =============================================================================
# Show WHICH document each answer came from — critical for transparency.

def rag_with_sources(question: str) -> dict:
    """
    RAG that returns the answer AND the source documents.
    Returns: {"answer": str, "sources": list[str]}
    """
    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(question)

    # Build context — preserve which chunks came from where
    context_parts = []
    sources = []
    for doc in retrieved_docs:
        context_parts.append(doc.page_content)
        source = doc.metadata.get("source", "unknown")
        if source not in sources:
            sources.append(source)

    context = "\n\n".join(context_parts)

    # Get answer from LLM
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    return {
        "answer": response.content,
        "sources": sources,
        "chunks_retrieved": len(retrieved_docs),
    }

print("\n--- Section 5: RAG with Source Attribution ---")

test_questions = [
    "What is Dinesh learning and why?",
    "What is the role of FAISS in LangChain?",
    "How do LangChain agents make decisions?",
    "What is Dinesh's favorite color?",  # Out of context
]

for question in test_questions:
    result = rag_with_sources(question)
    print(f"\nQ: {question}")
    print(f"A: {result['answer'].strip()}")
    print(f"   Sources ({result['chunks_retrieved']} chunks): {result['sources']}")

# =============================================================================
# INTERACTIVE LOOP
# =============================================================================
print()
print("=" * 50)
print("  Interactive Multi-Source RAG Chat")
print("=" * 50)
print("Type 'exit' to stop.\n")

while True:
    user_input = input("Ask a question: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    result = rag_with_sources(user_input)
    print(f"\nAI: {result['answer'].strip()}")
    print(f"   (Sources: {result['sources']})\n")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 18")
print("=" * 60)
print("""
  1. Different loaders produce the same Document format — mix and match freely
  2. Combine multi-source docs with list concatenation: all_docs = docs1 + docs2
  3. Each Document's metadata["source"] tracks which file it came from
  4. load_any_file() helper auto-picks the right loader by extension
  5. Source attribution: show WHICH document grounded each answer
  6. DirectoryLoader loads entire folders — great for batch ingestion

  Supported formats:
    .txt (TextLoader), .pdf (PyPDFLoader), .docx (Docx2txtLoader)
    .csv (CSVLoader), .json (JSONLoader), .html/.md (Unstructured)
    Web pages (WebBaseLoader), Directories (DirectoryLoader)

  Next up: 19_rag_advanced_retrievers.py
  MMR, multi-query, compression, and ensemble retrieval strategies.
""")
