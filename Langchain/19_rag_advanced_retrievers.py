# =============================================================================
# FILE: 19_rag_advanced_retrievers.py
# PART: 5 - RAG  |  LEVEL: Advanced
# =============================================================================
#
# THE STORY:
#   Basic RAG asks: "Which documents are most similar to my query?"
#   But what if your query is phrased badly?
#   What if two documents say the same thing (redundancy)?
#   What if retrieved documents contain only 10% relevant text?
#
#   Advanced retrievers are smarter:
#     MultiQueryRetriever   → Rephrase the query 3 ways, search for all
#     MMR Retriever         → Diversity: avoid redundant results
#     Compression Retriever → Strip irrelevant sentences from retrieved docs
#     Threshold Retriever   → Only return results above a quality score
#
# WHAT YOU WILL LEARN:
#   1. MMR (Maximal Marginal Relevance) — diverse retrieval
#   2. MultiQueryRetriever — auto-generate multiple query phrasings
#   3. ContextualCompressionRetriever — compress retrieved docs to relevant parts
#   4. similarity_score_threshold — quality filter for retrieval
#   5. When to use each strategy and why
#
# HOW THIS CONNECTS:
#   Previous: 18_rag_document_loaders.py — multi-source RAG
#   Next:     20_rag_conversational.py — RAG with conversation history
# =============================================================================

import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Suppress verbose logger output from LangChain internals
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.WARNING)

load_dotenv()

print("=" * 60)
print("  CHAPTER 19: Advanced Retrieval Strategies")
print("=" * 60)

# =============================================================================
# SETUP: Build a knowledge base for demonstration
# =============================================================================
# A larger, more varied knowledge base to show retrieval differences

knowledge_docs = [
    Document(page_content="LangChain is a framework for building LLM applications with tools and memory.", metadata={"topic": "langchain"}),
    Document(page_content="LangChain provides abstractions for prompts, LLMs, chains, and agents.", metadata={"topic": "langchain"}),
    Document(page_content="LCEL (LangChain Expression Language) uses | to compose Runnables.", metadata={"topic": "lcel"}),
    Document(page_content="RAG retrieves relevant context before asking the LLM to answer.", metadata={"topic": "rag"}),
    Document(page_content="FAISS is a fast in-memory vector store for similarity search.", metadata={"topic": "vector_db"}),
    Document(page_content="Chroma is a persistent vector database that stores embeddings to disk.", metadata={"topic": "vector_db"}),
    Document(page_content="Agents use the ReAct pattern: Thought → Action → Observation → Thought.", metadata={"topic": "agents"}),
    Document(page_content="LangGraph supports graph-based workflows with cycles and conditional branching.", metadata={"topic": "langgraph"}),
    Document(page_content="OpenAI embeddings encode semantic meaning into 1536-dimensional vectors.", metadata={"topic": "embeddings"}),
    Document(page_content="InMemoryChatMessageHistory stores conversation history in RAM.", metadata={"topic": "memory"}),
    Document(page_content="FileChatMessageHistory persists conversation history to a JSON file.", metadata={"topic": "memory"}),
    Document(page_content="RunnableWithMessageHistory auto-manages reading and writing conversation history.", metadata={"topic": "memory"}),
    Document(page_content="OutputFixingParser auto-corrects malformed LLM output using another LLM call.", metadata={"topic": "parsers"}),
    Document(page_content="with_structured_output() uses tool calling to enforce JSON schemas natively.", metadata={"topic": "structured_output"}),
    Document(page_content="Production agents need retry logic, fallbacks, rate limiting, and monitoring.", metadata={"topic": "production"}),
]

print(f"\nKnowledge base: {len(knowledge_docs)} documents")

# Create embeddings and vector store
embeddings = OpenAIEmbeddings(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/text-embedding-3-small",
)

vectorstore = FAISS.from_documents(knowledge_docs, embeddings)

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
    temperature=0.0,
)

# =============================================================================
# SECTION 1: BASIC RETRIEVER (BASELINE)
# =============================================================================
# Standard similarity search — our baseline to compare against

print("\n--- Section 1: Basic Retriever (Baseline) ---")

basic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
query = "How does memory work in LangChain?"

basic_results = basic_retriever.invoke(query)
print(f"\nQuery: '{query}'")
print(f"Basic retriever returned {len(basic_results)} docs:")
for i, doc in enumerate(basic_results, 1):
    print(f"  {i}. [{doc.metadata.get('topic')}] {doc.page_content[:70]}...")

# =============================================================================
# SECTION 2: MMR — MAXIMAL MARGINAL RELEVANCE
# =============================================================================
# Problem with basic retrieval: if two docs say similar things,
# both get returned — wasting your context window on redundancy.
#
# MMR balances: relevance to query + diversity from other results.
# lambda_mult: 0 = maximize diversity, 1 = maximize relevance, 0.5 = balanced

print("\n--- Section 2: MMR Retriever (Diversity-Aware) ---")

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,           # Return 3 results
        "fetch_k": 10,    # First fetch 10 candidates from FAISS
        "lambda_mult": 0.5,  # Balance: 0=diverse, 1=relevant, 0.5=balanced
    }
)

mmr_results = mmr_retriever.invoke(query)
print(f"\nMMR retriever (diverse) returned {len(mmr_results)} docs:")
for i, doc in enumerate(mmr_results, 1):
    print(f"  {i}. [{doc.metadata.get('topic')}] {doc.page_content[:70]}...")

print("\nMMR avoids redundant results by penalizing documents similar to already-selected ones.")

# =============================================================================
# SECTION 3: MultiQueryRetriever — Query Rephrasing
# =============================================================================
# Problem: Users phrase queries poorly or ambiguously.
# "How does LangChain remember stuff?" → vague, might miss relevant docs.
#
# MultiQueryRetriever asks the LLM to generate 3 variants of your query,
# runs all 3 retrievals, and deduplicates the results.
# You get more coverage with very little extra cost.

print("\n--- Section 3: MultiQueryRetriever (Automatic Query Expansion) ---")

from langchain.retrievers.multi_query import MultiQueryRetriever

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=llm,
)

vague_query = "How does LangChain remember stuff between conversations?"
print(f"\nOriginal query: '{vague_query}'")
print("\n(MultiQueryRetriever will generate 3 query variants internally)")

multi_results = multi_retriever.invoke(vague_query)
print(f"\nMultiQueryRetriever returned {len(multi_results)} unique docs:")
for i, doc in enumerate(multi_results, 1):
    print(f"  {i}. [{doc.metadata.get('topic')}] {doc.page_content[:70]}...")

print("""
  How it works internally:
  1. LLM generates 3 paraphrases of your query
  2. All 3 paraphrases are sent to the base retriever
  3. Results from all 3 are merged and deduplicated
  4. You get broader coverage than a single-query retrieval
""")

# =============================================================================
# SECTION 4: ContextualCompressionRetriever — Strip Irrelevant Content
# =============================================================================
# Problem: Retrieved chunks often contain irrelevant sentences.
# Example: You ask about LangChain memory, but the retrieved chunk
# mentions memory + vector stores + tools. You only need the memory part.
#
# ContextualCompressionRetriever sends retrieved docs to the LLM and asks it
# to extract ONLY the relevant portions. The result is cleaner, more focused context.

print("--- Section 4: ContextualCompressionRetriever ---")

try:
    from langchain.retrievers.document_compressors import LLMChainExtractor
    from langchain.retrievers import ContextualCompressionRetriever

    # The compressor: asks the LLM to extract only relevant sentences
    compressor = LLMChainExtractor.from_llm(llm)

    # Wrap any base retriever with compression
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    )

    specific_query = "What is LangGraph used for?"
    print(f"\nQuery: '{specific_query}'")

    print("\nBase retriever results (uncompressed):")
    base_docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(specific_query)
    for i, doc in enumerate(base_docs, 1):
        print(f"  {i}. {doc.page_content}")

    print("\nCompression retriever results (only relevant parts):")
    compressed_docs = compression_retriever.invoke(specific_query)
    for i, doc in enumerate(compressed_docs, 1):
        print(f"  {i}. {doc.page_content}")
    print("\n(Compressor removed sentences unrelated to LangGraph!)")

except ImportError as e:
    print(f"\nNote: Contextual compression requires langchain. Error: {e}")

# =============================================================================
# SECTION 5: SIMILARITY SCORE THRESHOLD
# =============================================================================
# Problem: Basic retrieval always returns k results — even if they're irrelevant.
# If no document is relevant, you still get 3 bad ones.
#
# similarity_score_threshold only returns results ABOVE a quality score.
# Bad queries return 0 results (better to say "I don't know" than hallucinate).

print("\n--- Section 5: Similarity Score Threshold ---")

threshold_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.5,  # Only return docs with similarity >= 0.5
        "k": 5,                  # Max results (if all pass the threshold)
    }
)

# Good query — should get results
good_query = "How does the RAG pipeline work?"
good_results = threshold_retriever.invoke(good_query)
print(f"\nGood query '{good_query}':")
print(f"  Results above threshold: {len(good_results)}")
for doc in good_results:
    print(f"    - {doc.page_content[:60]}...")

# Off-topic query — might get 0 results
off_query = "What is the best pizza recipe?"
off_results = threshold_retriever.invoke(off_query)
print(f"\nOff-topic query '{off_query}':")
print(f"  Results above threshold: {len(off_results)}")
if not off_results:
    print("  (0 results — correct! The knowledge base doesn't have pizza info)")

# =============================================================================
# SECTION 6: CHOOSING THE RIGHT RETRIEVER
# =============================================================================
print("\n--- Section 6: When to Use Each Retriever ---")
print("""
  BASIC SIMILARITY SEARCH:
    Use when: Simple use cases, small knowledge bases, speed is critical
    Pros: Fastest, most predictable
    Cons: Can return redundant results

  MMR (Maximal Marginal Relevance):
    Use when: Large knowledge bases with redundant content
    Pros: Diverse results, avoids echo chamber of similar docs
    Cons: Slightly slower than basic

  MultiQueryRetriever:
    Use when: Users phrase queries vaguely or ambiguously
    Pros: Best recall, finds docs that basic search might miss
    Cons: 1 extra LLM call to generate queries (small cost)

  ContextualCompressionRetriever:
    Use when: Chunks are large and only partially relevant
    Pros: Cleaner context = better LLM answers
    Cons: 1 LLM call per retrieved doc (can be slow/expensive)

  similarity_score_threshold:
    Use when: You'd rather say "I don't know" than hallucinate
    Pros: High precision, no bad results
    Cons: Can return 0 results for valid but poorly-phrased queries

  COMBINATION STRATEGY (production):
    multi_retriever → threshold_retriever → send to LLM
    This gives: broad coverage + quality filter + no hallucination
""")

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 19")
print("=" * 60)
print("""
  1. Basic retrieval always returns k results — even irrelevant ones
  2. MMR: diversity=lambda_mult=0.5, fetch more than k, pick diverse subset
  3. MultiQueryRetriever: auto-generates 3 query variants for better coverage
  4. ContextualCompressionRetriever: LLM extracts only relevant parts
  5. score_threshold: quality filter — returns 0 results for bad queries
  6. In production: combine strategies for best results

  Next up: 20_rag_conversational.py
  RAG + Memory: The AI remembers your conversation AND retrieves from docs.
""")
