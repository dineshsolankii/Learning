# =============================================================================
# FILE: 15_embeddings.py
# PART: 4 - Documents & Embeddings  |  LEVEL: Intermediate
# =============================================================================
#
# THE STORY:
#   Every piece of text can be represented as a point in a 1536-dimensional space.
#   Two sentences about the same topic land CLOSE together.
#   Two sentences about completely different things land FAR apart.
#
#   "Machine learning is a subset of AI" and "Deep learning uses neural networks"
#   are neighbors in this space.
#   "Pizza is a popular Italian dish" is far away from both.
#
#   This is the magic that makes semantic search possible.
#   You don't match keywords — you match MEANING.
#
# WHAT YOU WILL LEARN:
#   1. What embeddings are (numbers that capture meaning)
#   2. OpenAIEmbeddings via OpenRouter
#   3. embed_query() vs embed_documents()
#   4. Cosine similarity by hand with numpy (proven semantic search)
#   5. Embedding dimension (1536 for text-embedding-3-small)
#   6. The pizza outlier — proving that unrelated text scores low
#
# HOW THIS CONNECTS:
#   Previous: 14_text_splitters.py — splitting text into chunks
#   Next:     16_vector_stores.py — storing and searching embeddings with FAISS
# =============================================================================

import os
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

print("=" * 60)
print("  CHAPTER 15: Embeddings")
print("=" * 60)

# =============================================================================
# SECTION 1: WHAT IS AN EMBEDDING?
# =============================================================================
print("\n--- Section 1: What Is an Embedding? ---")
print("""
  An embedding is a LIST OF NUMBERS (a vector) that represents text.

  "Hello world"  →  [0.023, -0.412, 0.891, ..., 0.034]  (1536 numbers)
  "Hi there"     →  [0.019, -0.398, 0.887, ..., 0.031]  (similar numbers!)
  "Pizza recipe" →  [-0.234, 0.671, -0.123, ..., 0.556] (very different!)

  The key insight: SIMILAR TEXT → SIMILAR NUMBERS → NEARBY IN SPACE

  This lets us search by MEANING instead of exact keyword matching.
  "Show me documents about machine learning"
  matches "AI uses gradient descent" even without shared keywords.
""")

# =============================================================================
# SECTION 2: SETTING UP OpenAIEmbeddings
# =============================================================================
# We use OpenRouter's endpoint with the text-embedding-3-small model.
# This is the same embeddings model used throughout our RAG files.

print("--- Section 2: Setting Up OpenAIEmbeddings ---")

embeddings = OpenAIEmbeddings(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/text-embedding-3-small",
)

print("\nEmbeddings model: openai/text-embedding-3-small")
print("Embedding dimensions: 1536 (each text → 1536 numbers)")

# =============================================================================
# SECTION 3: embed_query() — Embed a Single Query
# =============================================================================
# embed_query() is optimized for short, query-like text.
# Use this when embedding the USER'S QUESTION before searching.

print("\n--- Section 3: embed_query() ---")

query = "What is machine learning?"
query_vector = embeddings.embed_query(query)

print(f"\nQuery : '{query}'")
print(f"Vector length: {len(query_vector)} dimensions")
print(f"First 5 values: {[round(v, 4) for v in query_vector[:5]]}")
print(f"Last  5 values: {[round(v, 4) for v in query_vector[-5:]]}")
print(f"\n(All {len(query_vector)} numbers together encode the MEANING of the query)")

# =============================================================================
# SECTION 4: embed_documents() — Embed Multiple Documents
# =============================================================================
# embed_documents() is optimized for longer, document-like text.
# Use this when embedding your KNOWLEDGE BASE before storing.
# It sends all texts in one batch API call — more efficient.

print("\n--- Section 4: embed_documents() ---")

# A set of sentences — notice the pizza outlier
sentences = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Supervised learning requires labeled training data.",
    "Gradient descent is used to optimize neural network weights.",
    "Pizza is a popular Italian dish with tomato sauce and cheese.",  # THE OUTLIER
]

print(f"\nEmbedding {len(sentences)} sentences...")
doc_vectors = embeddings.embed_documents(sentences)

print(f"Each sentence → vector of {len(doc_vectors[0])} numbers")
print(f"Total embeddings generated: {len(doc_vectors)}")

# =============================================================================
# SECTION 5: COSINE SIMILARITY — PROVING SEMANTIC SEARCH WORKS
# =============================================================================
# Cosine similarity measures the angle between two vectors.
# Range: -1 (opposite) to 1 (identical)
# Higher = more similar in meaning.
#
# We'll show that ML sentences have HIGH similarity to the ML query,
# while the pizza sentence has LOW similarity.

print("\n--- Section 5: Cosine Similarity (Proving Semantic Search) ---")

def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """
    Calculate cosine similarity between two vectors.

    Cosine similarity = (A · B) / (|A| × |B|)

    Result range: -1 (opposite) to 1 (identical)
    - 1.0 = exactly the same direction
    - 0.0 = completely perpendicular (unrelated)
    - -1.0 = exactly opposite directions
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print(f"\nQuery: '{query}'")
print(f"\nCosine similarity to each sentence:")
print("-" * 65)

scores = []
for sentence, vector in zip(sentences, doc_vectors):
    score = cosine_similarity(query_vector, vector)
    scores.append((score, sentence))
    bar = "█" * int(score * 20)  # Visual bar (scaled)
    print(f"{score:.4f}  {bar:<20}  {sentence[:55]}...")

print("-" * 65)

# Sort by similarity
sorted_scores = sorted(scores, reverse=True)
print(f"\nMost similar     : {sorted_scores[0][1][:60]}...")
print(f"Least similar    : {sorted_scores[-1][1][:60]}...")
print(f"\n(Notice the pizza sentence scores much lower — it's semantically distant!)")

# =============================================================================
# SECTION 6: FINDING THE MOST SIMILAR DOCUMENT
# =============================================================================
# This is exactly what a vector store does under the hood:
# 1. Embed the query
# 2. Calculate similarity to all stored vectors
# 3. Return top-k most similar documents

print("\n--- Section 6: Simulating a Vector Search ---")

def find_most_similar(query: str, docs: list, n: int = 3) -> list:
    """Find the n most similar documents to a query using cosine similarity."""
    q_vec = embeddings.embed_query(query)
    d_vecs = embeddings.embed_documents(docs)

    similarities = [
        (cosine_similarity(q_vec, d_vec), doc)
        for doc, d_vec in zip(docs, d_vecs)
    ]
    return sorted(similarities, reverse=True)[:n]

knowledge_base = [
    "Python is a high-level programming language known for its simplicity.",
    "LangChain helps connect LLMs with tools and memory.",
    "FAISS is a library for efficient similarity search on dense vectors.",
    "Neural networks are inspired by biological brain structures.",
    "The Eiffel Tower is located in Paris, France.",
    "Gradient descent minimizes the loss function during training.",
]

new_query = "How do I train a neural network?"
print(f"\nSearching for: '{new_query}'")
print(f"\nTop 3 most relevant documents:")

top_results = find_most_similar(new_query, knowledge_base, n=3)
for i, (score, doc) in enumerate(top_results, 1):
    print(f"  {i}. [{score:.4f}] {doc}")

# =============================================================================
# SECTION 7: EMBEDDING COST AWARENESS
# =============================================================================
print("\n--- Section 7: Cost and Performance Notes ---")
print("""
  text-embedding-3-small pricing (as of 2024):
    $0.020 per 1M tokens
    Average sentence ≈ 15 tokens
    1000 sentences ≈ 15,000 tokens ≈ $0.0003

  Performance:
    embed_query(): 1 API call, 50-200ms
    embed_documents(100 docs): 1 batched API call, 200-500ms

  Caching (see 32_caching_and_optimization.py):
    CacheBackedEmbeddings stores results locally
    Identical text → instant return, no API call
    Essential for large knowledge bases that rarely change

  Dimensions:
    text-embedding-3-small: 1536 dimensions
    text-embedding-3-large: 3072 dimensions (better, 5x more expensive)
    ada-002: 1536 dimensions (older, slightly weaker)
""")

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 15")
print("=" * 60)
print("""
  1. Embeddings convert text → numeric vectors (1536 numbers)
  2. Similar text → similar vectors → low cosine distance
  3. embed_query(): for short queries; embed_documents(): for knowledge base docs
  4. Cosine similarity: range -1 to 1; higher = more similar meaning
  5. This enables semantic search — match by meaning, not keywords
  6. The pizza outlier proves: unrelated text = low similarity score
  7. text-embedding-3-small is cheap ($0.02/1M tokens) and fast

  Next up: 16_vector_stores.py
  Store millions of embeddings and search them instantly with FAISS and Chroma.
""")
