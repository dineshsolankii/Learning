# =============================================================================
# FILE: 32_caching_and_optimization.py
# PART: 9 - Production  |  LEVEL: Expert
# =============================================================================
#
# THE STORY:
#   Your app asks the same question 50 times a day.
#   "What is machine learning?" costs tokens every single time.
#   That's expensive. And slow. And wasteful.
#
#   Caching stores the answer the first time.
#   Every subsequent call: instant response, zero API cost.
#   For embeddings: creating the same vector twice is pure waste.
#   CacheBackedEmbeddings makes it happen only once.
#
# WHAT YOU WILL LEARN:
#   1. InMemoryCache — instant caching (lives in RAM)
#   2. SQLiteCache — persistent caching (survives restarts)
#   3. How LangChain cache works under the hood
#   4. CacheBackedEmbeddings — never embed the same text twice
#   5. LocalFileStore — file-based embedding cache
#   6. Token optimization strategies
#
# HOW THIS CONNECTS:
#   Previous: 31_async_and_batching.py — concurrent execution
#   Next:     33_production_patterns.py — full production app structure
# =============================================================================

import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain_community.cache import SQLiteCache

load_dotenv()

print("=" * 60)
print("  CHAPTER 32: Caching and Optimization")
print("=" * 60)

# Base LLM and chain for demos
base_llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
    temperature=0.0,  # Temperature 0 is required for caching to work reliably
)

prompt = ChatPromptTemplate.from_template("Define {concept} in one sentence.")
chain = prompt | base_llm | StrOutputParser()

# =============================================================================
# SECTION 1: InMemoryCache — FASTEST, RAM-BASED
# =============================================================================
# LangChain has a global LLM cache.
# set_llm_cache() activates it for ALL LLM calls.
# Cache key = (prompt_text, model_name, temperature)
# Same key = instant cached response, no API call.

print("\n--- Section 1: InMemoryCache ---")
print("""
  How it works:
  First call  → sends API request, stores (prompt, response) in RAM
  Second call → same prompt? Return cached response instantly.
  Process ends → cache is gone (in-memory only)

  REQUIREMENT: temperature=0.0
  Non-zero temperature means different outputs each time,
  so caching wouldn't make sense semantically.
""")

# Activate in-memory cache globally
set_llm_cache(InMemoryCache())

QUESTION = "machine learning"

print(f"Calling LLM twice with: '{QUESTION}'")
print()

# First call — hits the API
t1_start = time.time()
result1 = chain.invoke({"concept": QUESTION})
t1_end = time.time()
first_call_time = t1_end - t1_start
print(f"Call 1 (API call): {first_call_time:.2f}s")
print(f"  Answer: {result1[:80]}...")

# Second call — exact same prompt, returns from cache instantly
t2_start = time.time()
result2 = chain.invoke({"concept": QUESTION})
t2_end = time.time()
second_call_time = t2_end - t2_start
print(f"\nCall 2 (cached): {second_call_time:.3f}s")
print(f"  Answer: {result2[:80]}...")

if first_call_time > 0:
    speedup = first_call_time / second_call_time if second_call_time > 0 else float('inf')
    print(f"\nSpeedup: {speedup:.0f}x faster (cached vs API)")
print(f"Results identical: {result1 == result2}")

# =============================================================================
# SECTION 2: SQLiteCache — PERSISTENT CACHING
# =============================================================================
# SQLiteCache stores responses in a local SQLite database file.
# Survives process restarts — cache persists between runs.
# Perfect for development (don't re-pay for the same prompts).

print("\n--- Section 2: SQLiteCache (Persistent) ---")
print("""
  SQLiteCache stores responses in .langchain.db (SQLite file).
  After the first run, repeated runs are instant even after restart.
  Clears only when you delete the file or call cache.clear().
""")

cache_db_path = "/tmp/langchain_cache_demo.db"
sqlite_cache = SQLiteCache(database_path=cache_db_path)
set_llm_cache(sqlite_cache)

OTHER_QUESTION = "neural networks"

print(f"\nFirst call to SQLite cache: '{OTHER_QUESTION}'")
start = time.time()
result = chain.invoke({"concept": OTHER_QUESTION})
elapsed = time.time() - start
print(f"  Time: {elapsed:.2f}s | Result: {result[:70]}...")

print(f"\nSecond call (cached in SQLite):")
start = time.time()
result2 = chain.invoke({"concept": OTHER_QUESTION})
elapsed = time.time() - start
print(f"  Time: {elapsed:.3f}s | Cached: {result == result2}")
print(f"  Cache file: {cache_db_path}")

# Reset to in-memory for remaining sections
set_llm_cache(InMemoryCache())

# =============================================================================
# SECTION 3: HOW CACHE KEYS WORK
# =============================================================================
print("\n--- Section 3: Cache Key Rules ---")
print("""
  Cache key = hash(prompt_text + model + temperature + other params)

  CACHE HIT (same key):
  - Exact same prompt text
  - Same model name
  - Same temperature

  CACHE MISS (different key — new API call):
  - Different prompt (even one extra space)
  - Different model ("gpt-4o" vs "gpt-4o-mini")
  - Different temperature (0.0 vs 0.1)

  PRACTICAL IMPLICATION:
  If you cache with temperature=0, a call with temperature=0.5
  for the same prompt will MISS the cache (goes to API).
  Keep temperature=0 for cacheable workflows.
""")

# Demonstrate cache miss with different prompt
print("Demonstrating cache miss (different concept):")
start = time.time()
chain.invoke({"concept": "deep learning"})  # Different — cache miss, API call
elapsed = time.time() - start
print(f"  'deep learning': {elapsed:.2f}s (cache miss → API)")

start = time.time()
chain.invoke({"concept": "deep learning"})  # Same — cache hit
elapsed = time.time() - start
print(f"  'deep learning': {elapsed:.3f}s (cache hit → instant)")

# =============================================================================
# SECTION 4: CacheBackedEmbeddings — NEVER EMBED TWICE
# =============================================================================
# Embedding is expensive: each call costs tokens AND time.
# If you embed the same document twice (e.g., after app restart),
# you're wasting money.
#
# CacheBackedEmbeddings wraps your embeddings model:
# - First call: embed + store to cache
# - Subsequent calls: return from cache instantly

print("\n--- Section 4: CacheBackedEmbeddings ---")
print("""
  CacheBackedEmbeddings wraps any embeddings model with a cache store.
  Cache key = hash(text content)
  Same text = same vector, never re-embedded.

  This is CRITICAL for RAG systems where documents are
  embedded once but the app might restart multiple times.
""")

try:
    from langchain.embeddings import CacheBackedEmbeddings
    from langchain.storage import LocalFileStore

    # LocalFileStore: stores embedding vectors as files in a directory
    store = LocalFileStore("/tmp/embedding_cache_demo/")

    base_embeddings = OpenAIEmbeddings(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="openai/text-embedding-3-small",
    )

    # Wrap with cache
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=base_embeddings,
        document_embedding_cache=store,
        namespace=base_embeddings.model,  # Namespace by model name
    )

    TEXTS = [
        "LangChain is a framework for building LLM applications.",
        "FAISS is a library for efficient similarity search.",
        "RAG combines retrieval with LLM generation.",
    ]

    print(f"\nEmbedding {len(TEXTS)} texts — first time (API calls):")
    start = time.time()
    vectors1 = cached_embedder.embed_documents(TEXTS)
    t1 = time.time() - start
    print(f"  First run: {t1:.2f}s ({len(vectors1)} vectors, dim={len(vectors1[0])})")

    print(f"\nEmbedding same {len(TEXTS)} texts — second time (from cache):")
    start = time.time()
    vectors2 = cached_embedder.embed_documents(TEXTS)
    t2 = time.time() - start
    print(f"  Second run: {t2:.3f}s")
    speedup = t1 / t2 if t2 > 0 else float('inf')
    print(f"  Speedup: {speedup:.0f}x | Vectors identical: {vectors1 == vectors2}")
    print(f"  Cache dir: /tmp/embedding_cache_demo/")

except ImportError as e:
    print(f"  Note: CacheBackedEmbeddings requires langchain package.")
    print(f"  from langchain.embeddings import CacheBackedEmbeddings")
    print(f"  from langchain.storage import LocalFileStore")
    print(f"  Error: {e}")

# =============================================================================
# SECTION 5: TOKEN OPTIMIZATION TIPS
# =============================================================================
print("\n--- Section 5: Token Optimization Tips ---")
print("""
  REDUCE TOKENS IN PROMPTS:
  ✓ Remove redundant instructions ("Please kindly..." → "...")
  ✓ Use bullet points instead of paragraphs in system prompts
  ✓ Truncate retrieved docs to max_tokens per chunk
  ✓ trim_messages() to drop old history beyond window size

  REDUCE TOKENS IN RESPONSES:
  ✓ Add "Be concise. Answer in X sentences." to system prompt
  ✓ max_tokens parameter limits response length
  ✓ Use structured output (with_structured_output) — no verbose prose

  EMBEDDING OPTIMIZATION:
  ✓ Use CacheBackedEmbeddings (never embed twice)
  ✓ text-embedding-3-small vs ada-002: cheaper, often as good
  ✓ Batch embed_documents() calls rather than one-at-a-time

  LLM CALL OPTIMIZATION:
  ✓ Cache deterministic queries (temperature=0)
  ✓ Use gpt-4o-mini for classification/extraction tasks
  ✓ Reserve gpt-4o/claude for complex reasoning only
  ✓ Batch with chain.batch() instead of sequential invoke()
""")

# =============================================================================
# SECTION 6: MEASURING COST WITH get_openai_callback
# =============================================================================
print("--- Section 6: Token Counting with get_openai_callback ---")

from langchain_community.callbacks import get_openai_callback

# Disable cache for this demo to actually make API call
set_llm_cache(None)

print("\nMeasuring token usage:")
with get_openai_callback() as cb:
    result = chain.invoke({"concept": "attention mechanism"})
    print(f"  Answer: {result[:80]}...")
    print(f"\n  Token Usage:")
    print(f"    Prompt tokens  : {cb.prompt_tokens}")
    print(f"    Completion tkns: {cb.completion_tokens}")
    print(f"    Total tokens   : {cb.total_tokens}")
    print(f"    Total cost     : ${cb.total_cost:.6f}")
    print(f"    Successful reqs: {cb.successful_requests}")

# Restore cache
set_llm_cache(InMemoryCache())

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 32")
print("=" * 60)
print("""
  1. set_llm_cache(InMemoryCache()) — global cache for all LLM calls
  2. SQLiteCache — persistent cache (survives restarts), great for dev
  3. Cache key = hash(prompt + model + temperature) — must match exactly
  4. temperature=0 is required for caching to be semantically correct
  5. CacheBackedEmbeddings — never re-embed the same text (huge savings)
  6. LocalFileStore — file-based embedding cache (persists to disk)
  7. get_openai_callback() — measure exact token usage and cost
  8. Token optimization: concise prompts, trim_messages, max_tokens

  WHEN TO CACHE:
  → Deterministic QA (temperature=0) → cache everything
  → Document embeddings in RAG → always use CacheBackedEmbeddings
  → Development: SQLiteCache (skip re-paying for the same prompts)
  → Production: Redis or Memcached for distributed cache

  Next up: 33_production_patterns.py
  Full production app structure: config, logging, validation, testing.
""")
