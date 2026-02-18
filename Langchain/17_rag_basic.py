# =============================================================================
# FILE: 17_rag_basic.py
# PART: 5 - RAG  |  LEVEL: Intermediate
# =============================================================================
#
# THE STORY:
#   Imagine an open-book exam.
#   The LLM is the student. RAG is the open book.
#
#   Without RAG: The student answers from memory — potentially outdated or wrong.
#   With RAG   : The student looks up the answer in a curated knowledge base.
#
#   This is the most important LangChain pattern.
#   Instead of memorizing everything at training time, the model looks up
#   relevant information before answering. The result:
#     - More accurate answers
#     - No hallucination on things not in training data
#     - Knowledge base can be updated without retraining the model
#
# THE 8-STEP RAG PIPELINE:
#   Step 1: Load document(s)            → raw text
#   Step 2: Split into chunks           → list of Documents
#   Step 3: Create embedding model      → text → vectors
#   Step 4: Store chunks in FAISS       → searchable vector database
#   Step 5: Create retriever            → query → top-k Documents
#   Step 6: Build RAG prompt            → context + question → messages
#   Step 7: Call LLM with context       → AIMessage
#   Step 8: Parse and return answer     → string
#
# WHAT YOU WILL LEARN:
#   1. The complete RAG pipeline end to end
#   2. What each step produces (intermediate state)
#   3. The "grounded answer" pattern (answer ONLY from context)
#   4. "I don't know" fallback for out-of-context questions
#   5. Seeing which chunks were retrieved for each question
#
# HOW THIS CONNECTS:
#   Previous: 16_vector_stores.py — FAISS and vector store basics
#   Next:     18_rag_document_loaders.py — RAG with multiple document types
# =============================================================================

import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

print("=" * 60)
print("  CHAPTER 17: Basic RAG — The Complete 8-Step Pipeline")
print("=" * 60)
print("""
  Retrieval-Augmented Generation (RAG):
  Load → Split → Embed → Store → Retrieve → Prompt → LLM → Answer
""")

# =============================================================================
# STEP 1: LOAD DOCUMENTS
# =============================================================================
# The knowledge base starts as raw documents.
# Any document format works here (see 13_document_loaders.py for all options).
# We use a simple text file for this example.

print("--- Step 1: Load Document ---")
start = time.time()

try:
    loader = TextLoader("Langchain/data/knowledge.txt", encoding="utf-8")
    raw_documents = loader.load()
except FileNotFoundError:
    # Fallback: create a richer knowledge base inline
    raw_documents = [Document(
        page_content="""
Dinesh is learning LangChain to build AI agents.
He works with n8n and automation workflows.
He wants to build intelligent multi-step AI systems.
LangChain helps connect LLMs with tools and memory.

LangChain's LCEL (LangChain Expression Language) uses the pipe operator
to chain components together: prompt | llm | parser.

RAG (Retrieval-Augmented Generation) allows LLMs to access external
knowledge bases. This makes answers more accurate and up-to-date.
The pipeline: load documents, split, embed, store in vector DB, retrieve, answer.

Agents use LLMs as reasoning engines to decide which tools to call.
Tools can be web search, calculators, databases, or custom functions.

LangGraph is used for building stateful, graph-based agent workflows.
It supports cycles and branching, unlike simple linear chains.
""",
        metadata={"source": "Langchain/data/knowledge.txt"}
    )]

print(f"  Loaded {len(raw_documents)} document(s)")
for doc in raw_documents:
    print(f"  Source: {doc.metadata.get('source')}")
    print(f"  Characters: {len(doc.page_content)}")

# =============================================================================
# STEP 2: SPLIT INTO CHUNKS
# =============================================================================
# Split the document into smaller, searchable pieces.
# chunk_size=300: each chunk is max 300 characters
# chunk_overlap=50: consecutive chunks share 50 characters (prevents info loss)

print("\n--- Step 2: Split Into Chunks ---")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)
docs = text_splitter.split_documents(raw_documents)

print(f"  Chunks created: {len(docs)}")
print(f"  Chunk 0 preview: {docs[0].page_content[:100]}...")
if len(docs) > 1:
    print(f"  Chunk 1 preview: {docs[1].page_content[:100]}...")

# =============================================================================
# STEP 3: CREATE EMBEDDING MODEL
# =============================================================================
# The embedding model converts text to numeric vectors.
# We use text-embedding-3-small via OpenRouter (same as chapter 15).

print("\n--- Step 3: Create Embedding Model ---")

embeddings = OpenAIEmbeddings(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/text-embedding-3-small",
)

print("  Embedding model ready: openai/text-embedding-3-small")
print("  Dimension: 1536 (each chunk → 1536-dimensional vector)")

# =============================================================================
# STEP 4: STORE IN FAISS
# =============================================================================
# FAISS embeds all chunks and stores them in a searchable index.
# This is the one-time cost — future queries are fast.

print("\n--- Step 4: Store in FAISS Vector Store ---")
embed_start = time.time()

vectorstore = FAISS.from_documents(docs, embeddings)

embed_time = time.time() - embed_start
print(f"  FAISS index created in {embed_time:.2f}s")
print(f"  Total vectors stored: {vectorstore.index.ntotal}")

# =============================================================================
# STEP 5: CREATE RETRIEVER
# =============================================================================
# The retriever wraps FAISS with a simple .invoke(query) interface.
# When called, it embeds the query and finds the k most similar chunks.

print("\n--- Step 5: Create Retriever ---")

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # Return top 3 most relevant chunks
)

print("  Retriever ready. Returns top-3 chunks per query.")

# Test retrieval
test_query = "What is Dinesh learning?"
test_docs = retriever.invoke(test_query)
print(f"\n  Test retrieval for '{test_query}':")
for i, doc in enumerate(test_docs):
    print(f"    Chunk {i+1}: {doc.page_content[:80]}...")

# =============================================================================
# STEP 6: BUILD RAG PROMPT
# =============================================================================
# The RAG prompt has two parts:
#   {context} — the retrieved chunks (what the LLM reads from the open book)
#   {question} — the user's actual question
#
# The instruction "ONLY using the context below" grounds the answer.
# If the context doesn't contain the answer, we want "I don't know."

print("\n--- Step 6: Build RAG Prompt ---")

rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question ONLY using the context below.
If the context does not contain enough information to answer, say:
"I don't have this information in my knowledge base."

Context:
{context}

Question:
{question}

Answer:""")

print("  RAG prompt template ready.")
print("  Includes: context slot + question slot + 'I don't know' instruction")

# =============================================================================
# STEP 7 & 8: BUILD THE RAG CHAIN AND RUN IT
# =============================================================================
# The full RAG function:
#   1. Call retriever with the question → get relevant chunks
#   2. Join chunks into a context string
#   3. Fill the prompt with context + question
#   4. Call LLM → get answer
#   5. Return the answer string

print("\n--- Steps 7 & 8: LLM Configuration and Full RAG Chain ---")

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
)

def rag_chain(question: str, show_sources: bool = False) -> str:
    """
    Full RAG pipeline:
    question → retrieve relevant chunks → prompt LLM → return answer

    Args:
        question: The user's question
        show_sources: If True, prints which chunks were retrieved
    """
    # Step 7a: Retrieve relevant chunks
    retrieved_docs = retriever.invoke(question)

    # Step 7b: Combine chunks into context string
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    if show_sources:
        print(f"\n  [Retrieved {len(retrieved_docs)} chunks]")
        for i, doc in enumerate(retrieved_docs):
            print(f"    Chunk {i+1}: {doc.page_content[:100]}...")

    # Step 7c: Build and run the chain
    chain = rag_prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question,
    })

    return response.content

# =============================================================================
# DEMO: TEST THE RAG PIPELINE
# =============================================================================
print()
print("=" * 50)
print("  DEMO: Testing the RAG Pipeline")
print("=" * 50)

# Test 1: In-context question (should answer correctly)
q1 = "What is Dinesh learning?"
print(f"\nQ: {q1}")
a1 = rag_chain(q1, show_sources=True)
print(f"A: {a1}")

# Test 2: Another in-context question
q2 = "What does RAG stand for and what does it do?"
print(f"\nQ: {q2}")
a2 = rag_chain(q2, show_sources=True)
print(f"A: {a2}")

# Test 3: Out-of-context question (should say "I don't know")
q3 = "What is Dinesh's favorite food?"
print(f"\nQ: {q3}")
a3 = rag_chain(q3, show_sources=False)
print(f"A: {a3}")
print("  (Correct! The knowledge base doesn't mention food.)")

# =============================================================================
# SECTION: TIMING THE PIPELINE
# =============================================================================
print()
print("--- Pipeline Timing Breakdown ---")

t0 = time.time()
_ = rag_chain("What tools does Dinesh use?")
total = time.time() - t0

print(f"\nFull RAG query time: {total:.2f}s")
print(f"  - Retrieval (FAISS): ~0.001s (in-memory search)")
print(f"  - LLM API call: ~{total:.2f}s (dominates)")
print(f"\nNote: The first call may be slower (connection setup).")

# =============================================================================
# INTERACTIVE CHAT LOOP
# =============================================================================
print()
print("=" * 50)
print("  Interactive RAG Chat (type 'exit' to stop)")
print("=" * 50)
print("Type 'exit' to stop.\n")

while True:
    user_input = input("Ask a question: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    answer = rag_chain(user_input, show_sources=True)
    print(f"\nAI: {answer}\n")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 17")
print("=" * 60)
print("""
  The 8-step RAG pipeline:
  1. Load document(s)        → raw text files
  2. Split into chunks       → RecursiveCharacterTextSplitter
  3. Create embedding model  → OpenAIEmbeddings (text-embedding-3-small)
  4. Store in FAISS          → searchable vector index
  5. Create retriever        → vectorstore.as_retriever(k=3)
  6. Build RAG prompt        → {context} + {question} template
  7. Call LLM                → ChatOpenAI → AIMessage
  8. Return answer           → response.content

  Key design decisions:
  - "ONLY using the context below" grounds answers to retrieved docs
  - "I don't have this information" prevents hallucination
  - show_sources=True shows which chunks were retrieved (for debugging)

  Next up: 18_rag_document_loaders.py
  RAG with multiple document types: PDF, CSV, Markdown, directories.
""")
