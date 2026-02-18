# =============================================================================
# FILE: 08_lcel_runnables.py
# PART: 2 - LCEL Deep Dive  |  LEVEL: Intermediate
# =============================================================================
#
# THE STORY:
#   Every Runnable has hidden superpowers beyond .invoke().
#   It can retry itself when the API fails.
#   It can fall back to a backup chain if the primary breaks.
#   It can carry metadata and tags for observability.
#   It can pre-bind parameters so you don't repeat yourself.
#
#   These tools transform a fragile script into a resilient production chain.
#
# WHAT YOU WILL LEARN:
#   1. .bind() — pre-attach parameters to a Runnable
#   2. .with_retry() — auto-retry on transient failures
#   3. .with_fallbacks() — graceful degradation to backup chains
#   4. RunnableConfig — pass tags and metadata into chains
#   5. .with_config() — set permanent config on a chain
#   6. The async interface: ainvoke, abatch, astream (preview)
#
# HOW THIS CONNECTS:
#   Previous: 07_lcel_parallel_branching.py — parallel and branching chains
#   Next:     09_output_parsers_advanced.py — XMLOutputParser, OutputFixingParser
# =============================================================================

import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

load_dotenv()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
)

parser = StrOutputParser()
prompt = ChatPromptTemplate.from_template("Answer this question: {question}")
chain = prompt | llm | parser

print("=" * 60)
print("  CHAPTER 8: The Full Runnable Toolkit")
print("=" * 60)

# =============================================================================
# SECTION 1: .bind() — Pre-Attach Parameters
# =============================================================================
# .bind() lets you permanently attach keyword arguments to a Runnable.
# The classic use case: binding stop sequences to an LLM so it stops
# generating after specific tokens — great for structured output.

print("\n--- Section 1: .bind() — Pre-Attached Parameters ---")

# Create an LLM that always stops at "END" — useful for enforcing format
llm_with_stop = llm.bind(stop=["END", "STOP"])

stop_prompt = ChatPromptTemplate.from_template(
    "List 3 fruits. After each one, write END.\n"
    "Format: 1. FruitName END"
)

stop_chain = stop_prompt | llm_with_stop | parser
result = stop_chain.invoke({})
print(f"\nResult with stop=['END']: '{result.strip()}'")
print("(The LLM stopped generating when it hit 'END')")

# Another use: bind temperature per-chain without re-creating the LLM
creative_llm = llm.bind(temperature=1.5)
precise_llm = llm.bind(temperature=0.0)

creative_prompt = ChatPromptTemplate.from_template("Invent a fictional product name for: {category}")

print(f"\nCreative (temp=1.5): {(creative_prompt | creative_llm | parser).invoke({'category': 'kitchen gadgets'}).strip()}")
print(f"Precise  (temp=0.0): {(creative_prompt | precise_llm | parser).invoke({'category': 'kitchen gadgets'}).strip()}")

# =============================================================================
# SECTION 2: .with_retry() — Resilience Against API Failures
# =============================================================================
# LLM APIs can fail: rate limits, timeouts, transient server errors.
# .with_retry() automatically re-attempts the call with exponential backoff.
# You set max attempts and which exceptions trigger a retry.

print("\n--- Section 2: .with_retry() — Auto-Retry on Failure ---")

# Build a resilient chain that retries up to 3 times on any exception
resilient_chain = (
    prompt
    | llm.with_retry(
        stop_after_attempt=3,           # Max 3 attempts total
        wait_exponential_jitter=True,   # Add random jitter to backoff
        retry_if_exception_type=(Exception,),  # Retry on any exception
    )
    | parser
)

# This call will work normally (no failure to trigger retry)
result = resilient_chain.invoke({"question": "What is 2 + 2?"})
print(f"\nResilient chain answer: {result.strip()}")
print("(If the API had failed, it would have retried up to 3 times)")

# =============================================================================
# SECTION 3: .with_fallbacks() — Graceful Degradation
# =============================================================================
# When the primary chain fails (after all retries), .with_fallbacks()
# tries a backup chain instead. This is your safety net.
# Common pattern: expensive model → cheap model → hardcoded response

print("\n--- Section 3: .with_fallbacks() — Backup Chains ---")

# Primary chain: the real answer chain
primary_chain = prompt | llm | parser

# Fallback 1: A simpler, more reliable prompt
fallback_prompt = ChatPromptTemplate.from_template(
    "Please answer briefly: {question}"
)
fallback_chain_1 = fallback_prompt | llm | parser

# Fallback 2: Hardcoded response (absolute last resort)
from langchain_core.runnables import RunnableLambda
fallback_chain_2 = RunnableLambda(
    lambda x: "I'm sorry, I couldn't process your question right now. Please try again."
)

# Build the chain with fallbacks — tries primary, then fallback_1, then fallback_2
robust_chain = primary_chain.with_fallbacks(
    [fallback_chain_1, fallback_chain_2]
)

result = robust_chain.invoke({"question": "What is the speed of light?"})
print(f"\nRobust chain answer: {result.strip()}")
print("(If primary had failed, fallback_chain_1 would have run)")

# =============================================================================
# SECTION 4: RunnableConfig — Tags and Metadata
# =============================================================================
# RunnableConfig lets you attach metadata to any chain execution.
# This is used for observability, logging, and tracing.
# Tags show up in LangSmith traces; metadata is passed to callbacks.

print("\n--- Section 4: RunnableConfig — Tagging Chain Runs ---")

config = RunnableConfig(
    tags=["v2.0", "production", "user-facing"],
    metadata={
        "user_id": "dinesh_123",
        "session_id": "session_abc",
        "feature_flag": "new_answer_format",
    },
    run_name="ProductionAnswerChain",  # Shows up in traces
)

result = chain.invoke(
    {"question": "What is Python programming language?"},
    config=config,
)
print(f"\nTagged chain answer: {result.strip()[:100]}")
print("(Tags and metadata are attached to this run for monitoring)")

# =============================================================================
# SECTION 5: .with_config() — Permanent Config on a Chain
# =============================================================================
# Instead of passing config every time in .invoke(), use .with_config()
# to bake the config into the chain permanently.

print("\n--- Section 5: .with_config() — Baked-In Configuration ---")

# Create a chain that always has these tags applied
tagged_chain = chain.with_config({
    "tags": ["analytics", "chatbot"],
    "run_name": "ChatbotChain",
})

# Now every invoke has these tags — no need to pass config manually
result = tagged_chain.invoke({"question": "What is machine learning?"})
print(f"\nTagged chain (permanent config): {result.strip()[:100]}")

# =============================================================================
# SECTION 6: ASYNC PREVIEW — ainvoke and astream
# =============================================================================
# Every Runnable has async versions: ainvoke, abatch, astream.
# Essential for web applications that need non-blocking LLM calls.
# Full async deep-dive is in 31_async_and_batching.py.

print("\n--- Section 6: Async Preview ---")

async def async_demo():
    print("\nRunning chain asynchronously with ainvoke()...")
    # ainvoke is the async version of invoke
    result = await chain.ainvoke({"question": "Name one planet in our solar system."})
    print(f"Async result: {result.strip()}")

    print("\nStreaming asynchronously with astream()...")
    print("AI: ", end="", flush=True)
    async for chunk in chain.astream({"question": "Write a one-sentence poem about Python."}):
        print(chunk, end="", flush=True)
    print()

asyncio.run(async_demo())

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 8")
print("=" * 60)
print("""
  1. .bind(key=value) pre-attaches parameters (stop tokens, temperature)
  2. .with_retry(stop_after_attempt=N) auto-retries on API failures
  3. .with_fallbacks([backup_chain]) tries backups if primary fails
  4. RunnableConfig(tags=[...], metadata={...}) attaches observability data
  5. .with_config({...}) bakes config permanently into a chain
  6. Async: ainvoke(), abatch(), astream() for non-blocking execution
  7. These tools transform demo code into production-ready chains

  Next up: 09_output_parsers_advanced.py
  OutputFixingParser, streaming JSON, and custom parsers.
""")
