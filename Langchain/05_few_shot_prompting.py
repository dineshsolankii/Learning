# =============================================================================
# FILE: 05_few_shot_prompting.py
# PART: 1 - Foundations  |  LEVEL: Beginner
# =============================================================================
#
# THE STORY:
#   Teaching by example is the oldest method in the world.
#   Instead of writing a rule book ("if the text is positive, say POSITIVE"),
#   you show the model 3 worked examples and it figures out the pattern.
#
#   This is Few-Shot Prompting — and it dramatically improves accuracy
#   for classification, formatting, and style-matching tasks.
#
#   The challenge: picking the RIGHT examples for each query.
#   If you always use the same 3 examples, they might not match the question.
#   SemanticSimilarityExampleSelector picks the MOST RELEVANT examples
#   based on meaning — dynamically, for each new query.
#
# WHAT YOU WILL LEARN:
#   1. What few-shot prompting is and why it works
#   2. FewShotChatMessagePromptTemplate — inject examples into a chat prompt
#   3. Static examples — same examples every time
#   4. SemanticSimilarityExampleSelector — pick best examples per query
#   5. When to use few-shot vs fine-tuning
#
# HOW THIS CONNECTS:
#   Previous: 04_output_parsers.py — getting structured output from LLMs
#   Next:     06_lcel_advanced_chains.py — multi-step chains, RunnablePassthrough
# =============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
    temperature=0.0,  # Deterministic — we want consistent classifications
)

print("=" * 60)
print("  CHAPTER 5: Few-Shot Prompting")
print("=" * 60)

# =============================================================================
# SECTION 1: THE EXAMPLES — THE TEACHER'S WORKED PROBLEMS
# =============================================================================
# Each example is a dict with "input" (what you show the model) and
# "output" (the correct answer you want it to learn from).
# These examples will be injected into the prompt before the real question.

# Scenario: Sentiment classification
sentiment_examples = [
    {"input": "I absolutely love this product! Best purchase ever.",   "output": "POSITIVE"},
    {"input": "This is complete garbage. Waste of money.",              "output": "NEGATIVE"},
    {"input": "It's okay. Does what it says, nothing special.",        "output": "NEUTRAL"},
    {"input": "Amazing quality! Shipped fast, exceeded expectations.", "output": "POSITIVE"},
    {"input": "Broke after 2 days. Very disappointed.",               "output": "NEGATIVE"},
    {"input": "Average product. Neither great nor terrible.",          "output": "NEUTRAL"},
    {"input": "Outstanding customer service! They fixed my issue.",    "output": "POSITIVE"},
    {"input": "It was neither bad nor good. Just mediocre.",           "output": "NEUTRAL"},
]

print(f"\nWe have {len(sentiment_examples)} examples to teach from.")

# =============================================================================
# SECTION 2: FewShotChatMessagePromptTemplate — Static Examples
# =============================================================================
# FewShotChatMessagePromptTemplate wraps each example as a Human/AI exchange.
# When inserted into the final prompt, the model sees:
#   Human: "I love this!"   →   AI: "POSITIVE"
#   Human: "It's terrible." →   AI: "NEGATIVE"
#   Human: <your real question>
# This pattern teaches the model the format through demonstration.

print("\n--- Section 2: Static Few-Shot (same examples every time) ---")

# Define how each example maps to a Human-AI message pair
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

# Create the few-shot template with ALL examples (static — same every time)
few_shot_template = FewShotChatMessagePromptTemplate(
    examples=sentiment_examples,
    example_prompt=example_prompt,
)

# Wrap it in a full chat prompt: System + Examples + Real Question
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sentiment classifier. Classify text as POSITIVE, NEGATIVE, or NEUTRAL. "
               "Respond with ONLY the label, nothing else."),
    few_shot_template,      # <-- All examples injected here
    ("human", "{input}"),   # <-- The real question at the end
])

# Build the chain
chain = final_prompt | llm | StrOutputParser()

# Test it
test_reviews = [
    "The packaging was damaged but the product still worked fine.",
    "I can't believe how good this is! 10/10 would buy again!",
    "Meh. It does the job but I expected better.",
]

print("\nClassifying test reviews:")
for review in test_reviews:
    label = chain.invoke({"input": review})
    print(f"  '{review[:50]}...' → {label}")

# =============================================================================
# SECTION 3: SemanticSimilarityExampleSelector — Smart Example Selection
# =============================================================================
# With 8 examples, static selection is fine.
# But what if you have 500 examples? You can't fit them all in the prompt.
# SemanticSimilarityExampleSelector picks the k most semantically similar
# examples to your current query — dynamically, each time.
#
# It works by:
#   1. Embedding all examples with OpenAI embeddings
#   2. Storing them in a FAISS vector store
#   3. At query time: embed the query, find k nearest examples

print("\n--- Section 3: SemanticSimilarityExampleSelector ---")

try:
    from langchain_core.example_selectors import SemanticSimilarityExampleSelector
    from langchain_community.vectorstores import FAISS

    # Embeddings model (same OpenRouter config)
    embeddings = OpenAIEmbeddings(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="openai/text-embedding-3-small",
    )

    # Build the selector — it embeds all examples and stores in FAISS
    print("\nBuilding semantic example selector (embedding all examples)...")
    selector = SemanticSimilarityExampleSelector.from_examples(
        examples=sentiment_examples,
        embeddings=embeddings,
        vectorstore_cls=FAISS,
        k=2,  # Pick the 2 most similar examples for each query
    )

    # Test: what examples does it pick for a shipping complaint?
    test_query = "The delivery was extremely late and the box was crushed."
    selected = selector.select_examples({"input": test_query})
    print(f"\nFor query: '{test_query}'")
    print(f"Selector chose {len(selected)} most relevant examples:")
    for ex in selected:
        print(f"  Input : {ex['input']}")
        print(f"  Output: {ex['output']}")
        print()

    # Build a dynamic few-shot template using the selector
    dynamic_few_shot = FewShotChatMessagePromptTemplate(
        example_selector=selector,   # <-- Uses selector instead of static examples
        example_prompt=example_prompt,
    )

    dynamic_chain_prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify text as POSITIVE, NEGATIVE, or NEUTRAL. One word only."),
        dynamic_few_shot,
        ("human", "{input}"),
    ])

    dynamic_chain = dynamic_chain_prompt | llm | StrOutputParser()
    dynamic_result = dynamic_chain.invoke({"input": test_query})
    print(f"Dynamic few-shot classification: {dynamic_result}")

except ImportError as e:
    print(f"\nNote: SemanticSimilarityExampleSelector requires faiss-cpu.")
    print(f"Install with: pip install faiss-cpu")
    print(f"Error: {e}")

# =============================================================================
# SECTION 4: WHEN TO USE FEW-SHOT VS FINE-TUNING
# =============================================================================
print("\n--- Section 4: Few-Shot vs Fine-Tuning ---")
print("""
  FEW-SHOT PROMPTING:
    + No training required — change examples instantly
    + Great for classification, formatting, style matching
    + Works immediately
    - Uses tokens for every request (examples add to context length)
    - Limited to ~10-20 examples that fit in the context window

  FINE-TUNING:
    + Train on thousands of examples
    + No tokens used at inference for examples
    - Requires training time and cost
    - Takes days to iterate
    - Use when few-shot accuracy isn't good enough
""")

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 5")
print("=" * 60)
print("""
  1. Few-shot prompting shows the LLM worked examples before the real question
  2. FewShotChatMessagePromptTemplate formats examples as Human/AI pairs
  3. Static selection: same examples every time (simple, good for small sets)
  4. SemanticSimilarityExampleSelector: picks k most relevant examples per query
  5. The selector uses embeddings + FAISS to find semantic similarity
  6. Use few-shot when accuracy needs a boost without model training

  Next up: 06_lcel_advanced_chains.py
  Multi-step chains, RunnablePassthrough, RunnableLambda, and .batch().
""")
