# =============================================================================
# FILE: 16_vector_stores.py
# PART: 4 - Documents & Embeddings  |  LEVEL: Intermediate
# =============================================================================
#
# THE STORY:
#   You have the embeddings. Now you need a warehouse to store them
#   and a search engine to find the most relevant ones.
#
#   FAISS (Facebook AI Similarity Search) is your fast local warehouse.
#   No server. No database. Just a file on your machine.
#   Search a million vectors in milliseconds.
#
#   Chroma is your persistent local database.
#   It stores vectors to disk — they survive restarts.
#   Same speed, but data lives forever until you delete it.
#
#   Both speak the same LangChain language — swap one for the other with
#   a single line change.
#
# WHAT YOU WILL LEARN:
#   1. FAISS.from_documents() — create from scratch
#   2. similarity_search() — find top-k matching documents
#   3. similarity_search_with_score() — search with distance scores
#   4. save_local() / load_local() — persist FAISS to disk
#   5. Chroma — persistent vector store (survives restarts)
#   6. add_documents() — add new documents to existing store
#   7. as_retriever() — convert to the Retriever interface for RAG
#
# HOW THIS CONNECTS:
#   Previous: 15_embeddings.py — creating embedding vectors
#   Next:     17_rag_basic.py — the complete RAG pipeline
# =============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

print("=" * 60)
print("  CHAPTER 16: Vector Stores (FAISS and Chroma)")
print("=" * 60)

# =============================================================================
# SETUP: Embeddings and Sample Documents
# =============================================================================
# The same embeddings config used throughout our project

embeddings = OpenAIEmbeddings(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/text-embedding-3-small",
)

# Sample knowledge base — a mix of topics
sample_docs = [
    Document(page_content="LangChain is a framework for building LLM-powered applications.",
             metadata={"source": "intro", "topic": "langchain"}),
    Document(page_content="LCEL (LangChain Expression Language) uses the pipe operator to chain Runnables.",
             metadata={"source": "lcel", "topic": "langchain"}),
    Document(page_content="RAG connects LLMs to external knowledge bases to reduce hallucinations.",
             metadata={"source": "rag", "topic": "ai"}),
    Document(page_content="FAISS is a library for fast similarity search on dense vectors.",
             metadata={"source": "faiss", "topic": "vector_db"}),
    Document(page_content="Chroma is a persistent vector database that stores embeddings to disk.",
             metadata={"source": "chroma", "topic": "vector_db"}),
    Document(page_content="Agents use LLMs as reasoning engines to decide which tools to call.",
             metadata={"source": "agents", "topic": "ai"}),
    Document(page_content="LangGraph builds stateful, graph-based AI workflows with cycles and branching.",
             metadata={"source": "langgraph", "topic": "ai"}),
    Document(page_content="Embeddings convert text into numeric vectors that capture semantic meaning.",
             metadata={"source": "embeddings", "topic": "ai"}),
]

# =============================================================================
# SECTION 1: FAISS.from_documents() — Create a Vector Store from Scratch
# =============================================================================
# This is the main constructor you'll use.
# It embeds ALL documents and stores them in a FAISS index in memory.
# The first call takes 1-3 seconds (API call to embed all docs).

print("\n--- Section 1: Creating a FAISS Vector Store ---")
print("\nEmbedding all documents and creating FAISS index...")

vectorstore = FAISS.from_documents(sample_docs, embeddings)

print(f"FAISS index created with {vectorstore.index.ntotal} vectors")
print(f"Embedding dimension: {vectorstore.index.d}")

# =============================================================================
# SECTION 2: similarity_search() — Find Top-k Matching Documents
# =============================================================================
# The core search method. Give it a query string.
# It embeds the query, searches for the k nearest vectors, returns Documents.

print("\n--- Section 2: similarity_search() ---")

query = "How does RAG improve AI accuracy?"
results = vectorstore.similarity_search(query, k=3)

print(f"\nQuery: '{query}'")
print(f"\nTop {len(results)} results:")
for i, doc in enumerate(results, 1):
    print(f"\n  Result {i}:")
    print(f"    Content: {doc.page_content}")
    print(f"    Source : {doc.metadata.get('source')}")

# =============================================================================
# SECTION 3: similarity_search_with_score() — Search with Distance Scores
# =============================================================================
# Returns (Document, score) tuples.
# Score meaning depends on the index type:
#   - FAISS default (L2): LOWER score = MORE similar (0 = identical)
#   - Cosine similarity: HIGHER score = MORE similar (1 = identical)

print("\n--- Section 3: similarity_search_with_score() ---")

query2 = "What tools does LangChain use for chaining?"
results_with_scores = vectorstore.similarity_search_with_score(query2, k=4)

print(f"\nQuery: '{query2}'")
print(f"\nResults with L2 distance scores (lower = more similar):")
for doc, score in results_with_scores:
    print(f"  Score: {score:.4f}  |  {doc.page_content[:70]}...")

# =============================================================================
# SECTION 4: save_local() and load_local() — Persist FAISS to Disk
# =============================================================================
# FAISS is in-memory by default — it vanishes when the script ends.
# save_local() writes the index to two files: .faiss and .pkl
# load_local() reads them back and restores the full vector store.

print("\n--- Section 4: Saving and Loading FAISS ---")

index_path = "Langchain/faiss_index"

# Save the current vector store to disk
vectorstore.save_local(index_path)
print(f"\nFAISS index saved to: {index_path}/")
print(f"Files created:")

import os as os_module
if os_module.path.exists(index_path):
    for f in os_module.listdir(index_path):
        size = os_module.path.getsize(f"{index_path}/{f}")
        print(f"  {f} ({size} bytes)")

# Load the saved index
print(f"\nLoading FAISS index from disk...")
loaded_vectorstore = FAISS.load_local(
    index_path,
    embeddings,
    allow_dangerous_deserialization=True  # Required flag — only use with trusted files
)

# Verify it works
test_results = loaded_vectorstore.similarity_search("vector database", k=2)
print(f"Loaded store works: found {len(test_results)} results for 'vector database'")
for doc in test_results:
    print(f"  - {doc.page_content[:60]}...")

# =============================================================================
# SECTION 5: add_documents() — Update an Existing Store
# =============================================================================
# You can add new documents to an existing FAISS store.
# The new documents are embedded and added to the existing index.

print("\n--- Section 5: Adding Documents to Existing Store ---")

new_docs = [
    Document(
        page_content="Production agents need retry logic, fallbacks, and proper logging.",
        metadata={"source": "production", "topic": "engineering"}
    ),
    Document(
        page_content="Async execution with ainvoke() and abatch() dramatically speeds up throughput.",
        metadata={"source": "async", "topic": "performance"}
    ),
]

print(f"\nStore before: {vectorstore.index.ntotal} vectors")
vectorstore.add_documents(new_docs)
print(f"Store after : {vectorstore.index.ntotal} vectors")
print(f"Successfully added {len(new_docs)} new documents!")

# =============================================================================
# SECTION 6: as_retriever() — The RAG Interface
# =============================================================================
# as_retriever() wraps the vector store in a Retriever interface.
# A Retriever has one method: .invoke(query) → List[Document]
# This is what the RAG chains use internally.

print("\n--- Section 6: as_retriever() — Convert to Retriever ---")

# Basic retriever — returns top 4 documents
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Use it
retrieved = retriever.invoke("How do agents work in LangChain?")
print(f"\nRetriever found {len(retrieved)} documents for agent query:")
for doc in retrieved:
    print(f"  [{doc.metadata.get('topic')}] {doc.page_content[:60]}...")

# MMR retriever — diversity-focused retrieval
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
)

print(f"\nMMR (diversity) retriever for same query:")
mmr_results = mmr_retriever.invoke("How do agents work in LangChain?")
for doc in mmr_results:
    print(f"  [{doc.metadata.get('topic')}] {doc.page_content[:60]}...")

# Threshold retriever — only return results above a quality score
threshold_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5, "k": 5}
)

# =============================================================================
# SECTION 7: Chroma — Persistent Vector Store
# =============================================================================
print("\n--- Section 7: Chroma (Persistent Vector Store) ---")
print("""
  Chroma stores vectors to disk — data persists when the script restarts.
  Install: pip install chromadb

  from langchain_community.vectorstores import Chroma

  # Create and persist
  chroma_store = Chroma.from_documents(
      documents=docs,
      embedding=embeddings,
      persist_directory="./chroma_db",   # Where to store data
      collection_name="langchain_docs"   # Name for this collection
  )

  # Load existing store (no need to re-embed!)
  loaded_store = Chroma(
      persist_directory="./chroma_db",
      embedding_function=embeddings,
      collection_name="langchain_docs"
  )

  # All the same methods work: similarity_search, add_documents, as_retriever()

  When to use FAISS vs Chroma:
    FAISS  → Quick experiments, no persistence needed, max speed
    Chroma → Development projects, need persistence, occasional updates
    Pinecone/Weaviate → Production, massive scale, cloud hosting
""")

# =============================================================================
# SECTION 8: FAISS vs CHROMA COMPARISON
# =============================================================================
print("--- Section 8: Choosing the Right Vector Store ---")
print("""
  ┌─────────────────┬──────────────────┬──────────────────┐
  │ Feature         │ FAISS            │ Chroma           │
  ├─────────────────┼──────────────────┼──────────────────┤
  │ Persistence     │ Manual (save_)   │ Automatic        │
  │ Speed           │ Extremely fast   │ Fast             │
  │ Setup           │ pip install      │ pip install      │
  │ Server needed   │ No               │ No               │
  │ Filtering       │ Limited          │ Full metadata    │
  │ Best for        │ Experiments      │ Persistent apps  │
  └─────────────────┴──────────────────┴──────────────────┘
""")

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 16")
print("=" * 60)
print("""
  1. FAISS.from_documents(docs, embeddings) creates a vector store in memory
  2. similarity_search(query, k=N) returns top-N matching Documents
  3. similarity_search_with_score() adds distance scores to results
  4. save_local() / load_local() persists FAISS to disk (.faiss + .pkl files)
  5. add_documents() extends an existing store without rebuilding
  6. as_retriever() wraps the store for use in RAG chains
  7. Chroma is persistent by default — great for dev/production apps

  PART 4 COMPLETE — Documents, splitting, embeddings, and storage mastered!

  Next up: 17_rag_basic.py
  The complete RAG pipeline — the most important LangChain pattern.
""")
