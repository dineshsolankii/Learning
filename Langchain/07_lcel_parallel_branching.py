# =============================================================================
# FILE: 07_lcel_parallel_branching.py
# PART: 2 - LCEL Deep Dive  |  LEVEL: Intermediate
# =============================================================================
#
# THE STORY:
#   What if you want to ask 3 experts the same question simultaneously?
#   RunnableParallel does exactly that — fans out to multiple chains
#   and collects all answers, running them AT THE SAME TIME.
#
#   Then: what if you want to route to DIFFERENT experts based on the question?
#   RunnableBranch picks a chain based on a condition — like a switch statement.
#
# WHAT YOU WILL LEARN:
#   1. RunnableParallel — run multiple chains on the same input simultaneously
#   2. Dict shorthand for RunnableParallel
#   3. Combining parallel outputs in a final step
#   4. RunnableBranch — conditional routing to different chains
#   5. Real use case: multi-aspect document analysis
#
# HOW THIS CONNECTS:
#   Previous: 06_lcel_advanced_chains.py — sequential chains, RunnableLambda
#   Next:     08_lcel_runnables.py — retry, fallback, config, bind
# =============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnablePassthrough

load_dotenv()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
)

parser = StrOutputParser()

print("=" * 60)
print("  CHAPTER 7: Parallel Execution & Conditional Branching")
print("=" * 60)

# =============================================================================
# SECTION 1: RunnableParallel — Fan Out to Multiple Chains
# =============================================================================
# RunnableParallel runs multiple Runnables on the SAME input simultaneously.
# It returns a dict where each key contains that chain's output.
# Under the hood, it uses threading for actual concurrency.

print("\n--- Section 1: RunnableParallel — Fan-Out Analysis ---")

# Scenario: Analyze a product review from 3 different angles at once
review_text = (
    "The phone has a beautiful AMOLED display and blazing-fast processor, "
    "but the battery only lasts 4 hours and the camera app crashes frequently."
)

# Define three specialist chains
sentiment_chain = (
    ChatPromptTemplate.from_template(
        "What is the overall sentiment of this review? (POSITIVE/NEGATIVE/MIXED)\n"
        "Review: {review}\nRespond with one word."
    ) | llm | parser
)

pros_chain = (
    ChatPromptTemplate.from_template(
        "List only the POSITIVE aspects mentioned. Be brief.\nReview: {review}"
    ) | llm | parser
)

cons_chain = (
    ChatPromptTemplate.from_template(
        "List only the NEGATIVE aspects mentioned. Be brief.\nReview: {review}"
    ) | llm | parser
)

# RunnableParallel runs all three SIMULTANEOUSLY
# Dict shorthand: {"key": chain} is equivalent to RunnableParallel(key=chain)
analysis_pipeline = RunnableParallel(
    sentiment=sentiment_chain,
    pros=pros_chain,
    cons=cons_chain,
)

print("\nAnalyzing review with 3 specialist chains in parallel...")
print("(All three LLM calls happen at the same time)")

analysis_result = analysis_pipeline.invoke({"review": review_text})

print(f"\nReview    : {review_text[:80]}...")
print(f"\nSentiment : {analysis_result['sentiment'].strip()}")
print(f"\nPros      : {analysis_result['pros'].strip()}")
print(f"\nCons      : {analysis_result['cons'].strip()}")

# =============================================================================
# SECTION 2: COMBINING PARALLEL OUTPUTS IN A FINAL STEP
# =============================================================================
# After parallel analysis, feed all outputs into a final summarizer chain.
# This is the "gather and synthesize" pattern.

print("\n--- Section 2: Parallel → Final Synthesis ---")

# Final chain takes the dict output from RunnableParallel
synthesis_prompt = ChatPromptTemplate.from_template(
    "Based on this analysis, write a 1-sentence verdict for a buyer:\n"
    "Sentiment: {sentiment}\n"
    "Pros: {pros}\n"
    "Cons: {cons}"
)

# Full pipeline: parallel analysis → synthesis
full_review_pipeline = analysis_pipeline | synthesis_prompt | llm | parser

verdict = full_review_pipeline.invoke({"review": review_text})
print(f"\nBuyer Verdict: {verdict.strip()}")

# =============================================================================
# SECTION 3: RunnableBranch — Conditional Routing
# =============================================================================
# RunnableBranch routes to different chains based on conditions.
# It works like if/elif/else — checks conditions in order, runs the first match.
# Structure: RunnableBranch((condition_fn, chain), ..., default_chain)

print("\n--- Section 3: RunnableBranch — Language Router ---")

# Define specialist chains for different languages
english_chain = (
    ChatPromptTemplate.from_template("You are an English assistant. Answer: {query}")
    | llm | parser
)

spanish_chain = (
    ChatPromptTemplate.from_template("Eres un asistente en español. Responde: {query}")
    | llm | parser
)

french_chain = (
    ChatPromptTemplate.from_template("Tu es un assistant en français. Réponds: {query}")
    | llm | parser
)

# Simple language detector (in production, you'd use an LLM for this)
def is_spanish(x: dict) -> bool:
    keywords = ["hola", "que", "como", "gracias", "por", "favor", "qué", "cómo"]
    return any(word in x.get("query", "").lower() for word in keywords)

def is_french(x: dict) -> bool:
    keywords = ["bonjour", "comment", "merci", "que", "est", "le", "la", "les", "je", "tu"]
    return any(word in x.get("query", "").lower() for word in keywords)

# Build the branch router
language_router = RunnableBranch(
    (is_spanish, spanish_chain),   # If Spanish detected → Spanish chain
    (is_french, french_chain),     # If French detected → French chain
    english_chain,                  # Default: English chain
)

# Test the router
test_queries = [
    {"query": "What is the speed of light?"},
    {"query": "Hola, como estas? Que haces hoy?"},
    {"query": "Bonjour! Comment vous appelez-vous?"},
]

print("\nRouting queries to specialist language chains:")
for test in test_queries:
    result = language_router.invoke(test)
    lang = "Spanish" if is_spanish(test) else ("French" if is_french(test) else "English")
    print(f"\n  [{lang}] Query  : {test['query'][:50]}")
    print(f"  [{lang}] Answer : {result.strip()[:100]}")

# =============================================================================
# SECTION 4: REAL WORLD — INTENT-BASED ROUTING
# =============================================================================
# A more sophisticated router using an LLM to classify the intent,
# then routing to a specialist chain.

print("\n--- Section 4: LLM-Based Intent Router ---")

# Step 1: Classify the intent with a small LLM call
intent_chain = (
    ChatPromptTemplate.from_template(
        "Classify this query into exactly one category: MATH, HISTORY, or SCIENCE.\n"
        "Query: {query}\nRespond with ONLY the category name."
    ) | llm | parser
)

# Step 2: Specialist answer chains
math_chain = (
    ChatPromptTemplate.from_template(
        "You are a math expert. Solve or explain: {query}"
    ) | llm | parser
)

history_chain = (
    ChatPromptTemplate.from_template(
        "You are a history professor. Answer: {query}"
    ) | llm | parser
)

science_chain = (
    ChatPromptTemplate.from_template(
        "You are a scientist. Explain: {query}"
    ) | llm | parser
)

# Step 3: Build the full intent-routed pipeline
def route_by_intent(inputs: dict) -> str:
    """Classify intent and route to the correct chain."""
    intent = intent_chain.invoke({"query": inputs["query"]}).strip().upper()
    print(f"  (Detected intent: {intent})")

    if "MATH" in intent:
        return math_chain.invoke(inputs)
    elif "HISTORY" in intent:
        return history_chain.invoke(inputs)
    else:
        return science_chain.invoke(inputs)

from langchain_core.runnables import RunnableLambda

intent_router_chain = RunnableLambda(route_by_intent)

print("\nTesting intent router:")
for q in [
    {"query": "What is the Pythagorean theorem?"},
    {"query": "When did World War II end?"},
    {"query": "How does photosynthesis work?"},
]:
    print(f"\nQuestion: {q['query']}")
    answer = intent_router_chain.invoke(q)
    print(f"Answer  : {answer[:150].strip()}")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 7")
print("=" * 60)
print("""
  1. RunnableParallel runs multiple chains simultaneously on same input
  2. Dict shorthand: {"a": chain1, "b": chain2} creates a parallel runner
  3. Parallel outputs become a dict → can feed into a synthesis chain
  4. RunnableBranch: (condition_fn, chain)... with a default fallback
  5. Conditions are plain Python functions returning bool
  6. LLM-based routing: classify intent first, then dispatch to specialist

  Next up: 08_lcel_runnables.py
  The full Runnable toolkit: retry, fallback, bind, config, and async.
""")
