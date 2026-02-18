# =============================================================================
# FILE: 06_lcel_advanced_chains.py
# PART: 2 - LCEL Deep Dive  |  LEVEL: Intermediate
# =============================================================================
#
# THE STORY:
#   You learned to build a straight road (prompt | llm | parser).
#   Now you build a city with intersections and merging lanes.
#
#   Real workflows aren't linear. You might need to:
#   - Pass the original input alongside a processed result
#   - Run a Python function in the middle of a chain
#   - Chain two LLM calls where output 1 becomes input 2
#
#   RunnablePassthrough and RunnableLambda are the tools for this.
#
# WHAT YOU WILL LEARN:
#   1. Multi-step sequential chains (chain1 | chain2)
#   2. RunnablePassthrough — pass input through unchanged
#   3. RunnableLambda — wrap any Python function as a Runnable
#   4. Dict-based branching: {"key1": chain1, "key2": chain2}
#   5. .batch() for sending multiple inputs concurrently
#
# HOW THIS CONNECTS:
#   Previous: 05_few_shot_prompting.py — teaching the LLM by example
#   Next:     07_lcel_parallel_branching.py — RunnableParallel and RunnableBranch
# =============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
)

parser = StrOutputParser()

print("=" * 60)
print("  CHAPTER 6: Advanced LCEL Chains")
print("=" * 60)

# =============================================================================
# SECTION 1: TWO-STAGE SEQUENTIAL CHAIN
# =============================================================================
# Chain the output of one LLM call into a second LLM call.
# Stage 1: Generate a topic from a broad category
# Stage 2: Write a joke about that specific topic

print("\n--- Section 1: Two-Stage Sequential Chain ---")

# Stage 1: Category → Topic
stage1_prompt = ChatPromptTemplate.from_template(
    "Name ONE specific topic within the category: {category}. "
    "Just the topic name, nothing else."
)
stage1_chain = stage1_prompt | llm | parser

# Stage 2: Topic → Joke
stage2_prompt = ChatPromptTemplate.from_template(
    "Write a short, funny joke about: {topic}"
)
stage2_chain = stage2_prompt | llm | parser

# How to combine them?
# The output of stage1 is a string, but stage2 needs a dict {"topic": ...}
# We use RunnableLambda to wrap a simple function that does the conversion

def wrap_as_topic(text: str) -> dict:
    """Wrap a plain string as the 'topic' key for stage 2."""
    return {"topic": text}

# Full two-stage pipeline
full_chain = stage1_chain | RunnableLambda(wrap_as_topic) | stage2_chain

print("\nRunning two-stage chain: category → topic → joke")
result = full_chain.invoke({"category": "technology"})
print(f"\nJoke about technology:\n{result}")

# =============================================================================
# SECTION 2: RunnablePassthrough — Keep the Original Input
# =============================================================================
# Problem: After stage1, we lose the original "category" input.
# What if we need BOTH the category AND the generated topic in stage2?
#
# RunnablePassthrough.assign() adds new keys to the existing dict
# while keeping the original keys intact.

print("\n--- Section 2: RunnablePassthrough.assign() ---")

# This chain outputs {"category": "...", "topic": "..."}
enriched_chain = RunnablePassthrough.assign(
    topic=stage1_chain  # Run stage1 and add result as "topic"
)

# Then we can use both in the next stage
full_enriched_prompt = ChatPromptTemplate.from_template(
    "Write a joke about {topic} (which is a subtopic of {category})."
)

full_enriched_chain = enriched_chain | full_enriched_prompt | llm | parser

result2 = full_enriched_chain.invoke({"category": "science"})
print(f"\nJoke (with category context): {result2}")

# =============================================================================
# SECTION 3: RunnableLambda — Python Functions in the Chain
# =============================================================================
# Any Python function can be a Runnable using RunnableLambda.
# This lets you add preprocessing, postprocessing, or logging mid-chain.

print("\n--- Section 3: RunnableLambda — Python Functions in Chains ---")

# Function that adds word count metadata to a result
def add_word_count(text: str) -> dict:
    """Add word count info to the generated text."""
    words = text.split()
    return {
        "text": text,
        "word_count": len(words),
        "is_long": len(words) > 50,
    }

# Function that formats the output for display
def format_output(data: dict) -> str:
    """Format the result dictionary for display."""
    return (
        f"RESPONSE ({data['word_count']} words, "
        f"{'long' if data['is_long'] else 'short'}):\n{data['text']}"
    )

# Build a chain with Python functions in the middle
analysis_chain = (
    ChatPromptTemplate.from_template("Explain {concept} in 2-3 sentences.")
    | llm
    | parser
    | RunnableLambda(add_word_count)
    | RunnableLambda(format_output)
)

result3 = analysis_chain.invoke({"concept": "machine learning"})
print(f"\n{result3}")

# =============================================================================
# SECTION 4: DICT-BASED PARALLEL INPUTS
# =============================================================================
# You can pass a dict of Runnables at any point in the chain.
# All values in the dict run with the same input, producing a dict output.
# This is a powerful way to gather multiple pieces of information at once.

print("\n--- Section 4: Dict-Based Input Routing ---")

# Generate three things about the same concept simultaneously
multi_output_chain = (
    {
        "concept": RunnablePassthrough(),  # Pass the input through unchanged
        "definition": (
            ChatPromptTemplate.from_template("Define {input} in one sentence.")
            | llm | parser
        ),
        "example": (
            ChatPromptTemplate.from_template("Give one real-world example of {input}.")
            | llm | parser
        ),
    }
)

concept_input = {"input": "recursion in programming"}
combined = multi_output_chain.invoke(concept_input)

print(f"\nConcept  : {combined.get('concept', concept_input)}")
print(f"Definition: {combined.get('definition', 'N/A')}")
print(f"Example   : {combined.get('example', 'N/A')}")

# =============================================================================
# SECTION 5: .batch() — CONCURRENT PROCESSING
# =============================================================================
# .batch() sends multiple inputs at the same time.
# Much faster than a Python for loop for many inputs.

print("\n--- Section 5: .batch() for Concurrent Processing ---")

simple_chain = (
    ChatPromptTemplate.from_template("What country is {city} the capital of? One word.")
    | llm | parser
)

cities = [
    {"city": "Tokyo"},
    {"city": "Paris"},
    {"city": "Cairo"},
    {"city": "Ottawa"},
    {"city": "Canberra"},
]

print(f"\nBatching {len(cities)} geography questions at once...")
results = simple_chain.batch(cities)

for city_dict, country in zip(cities, results):
    print(f"  {city_dict['city']:12} → {country.strip()}")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 6")
print("=" * 60)
print("""
  1. Chain two LLM calls: chain1 | lambda_wrapper | chain2
  2. RunnablePassthrough.assign() adds keys without losing existing ones
  3. RunnableLambda() wraps any Python function as a Runnable
  4. Dict at a chain step: runs all values in parallel, merges output
  5. .batch([inputs]) runs all inputs concurrently — much faster than a loop
  6. Use RunnableLambda for logging, transformation, or postprocessing

  Next up: 07_lcel_parallel_branching.py
  Run multiple chains simultaneously and route based on conditions.
""")
