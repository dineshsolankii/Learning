# =============================================================================
# FILE: 00_why_langchain.py
# PART: 1 - Foundations  |  LEVEL: Beginner (Absolute Zero)
# =============================================================================
#
# THE STORY:
#   Welcome, Dinesh. You are about to walk into the LangChain Academy.
#   Day 1. The teacher draws two columns on the whiteboard.
#   Left: "Raw OpenAI API Call" — verbose, brittle, manual.
#   Right: "LangChain Way" — clean, composable, production-ready.
#
#   By the end of this file, you will understand WHY LangChain exists,
#   WHAT it gives you, and HOW to make your very first LLM call.
#
# WHAT YOU WILL LEARN:
#   1. What LangChain is and the problem it solves
#   2. How to configure a ChatOpenAI model (via OpenRouter)
#   3. Your first llm.invoke() call — the "Hello World" of AI
#   4. The AIMessage object and its fields (content, metadata, usage)
#   5. How to switch models by changing ONE string
#   6. Environment variable loading with python-dotenv
#
# HOW THIS CONNECTS:
#   Previous: (This is the beginning!)
#   Next:     01_llm_and_chat_models.py — Messages, streaming, batching
# =============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load API keys from the .env file
load_dotenv()

# =============================================================================
# SECTION 1: WHY LANGCHAIN EXISTS
# =============================================================================
# Before LangChain, calling an AI model meant writing lots of boilerplate:
# - Build the HTTP request manually
# - Handle authentication headers
# - Parse the JSON response
# - Handle rate limits and retries
# - Build prompts as raw strings (error-prone)
#
# LangChain wraps all of that into clean, reusable components.
# The fundamental building block is the "Runnable" — anything with .invoke()
# LLMs, prompts, parsers — they all have the same interface. That's the magic.

print("=" * 60)
print("  CHAPTER 0: Why LangChain?")
print("=" * 60)
print()

# =============================================================================
# SECTION 2: CONFIGURE THE LLM
# =============================================================================
# We use OpenRouter — a gateway that gives us access to GPT-4o-mini, Claude,
# Gemini, and 100+ other models through one API.
#
# To switch models, ONLY change the 'model' string below. Nothing else changes.
# Examples:
#   "openai/gpt-4o-mini"       — Fast and cheap (what we use)
#   "openai/gpt-4o"            — Most capable OpenAI model
#   "anthropic/claude-3-5-sonnet" — Anthropic's best
#   "google/gemini-pro"        — Google's model

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",      # OpenRouter endpoint
    api_key=os.getenv("OPENROUTER_API_KEY"),       # From your .env file
    model="openai/gpt-4o-mini",                   # Model name — change to switch!
    temperature=0.7,                               # 0 = deterministic, 2 = wild
)

print("LLM configured successfully.")
print(f"Model: openai/gpt-4o-mini via OpenRouter")
print()

# =============================================================================
# SECTION 3: YOUR FIRST LLM CALL — llm.invoke()
# =============================================================================
# .invoke() is the core method every LangChain Runnable has.
# You give it input, it gives you output.
#
# Input: a string OR a list of messages
# Output: an AIMessage object (not just a plain string!)

print("-" * 40)
print("Calling the LLM with a simple string...")
print("-" * 40)

response = llm.invoke("In one sentence, what is LangChain?")

# response is an AIMessage object — not a plain string!
print(f"\nRaw response type: {type(response)}")
print(f"\nResponse CONTENT (what you normally want):")
print(f"  {response.content}")

# =============================================================================
# SECTION 4: EXPLORING THE AIMessage OBJECT
# =============================================================================
# Every LLM response is wrapped in an AIMessage. Let's explore its fields.

print()
print("-" * 40)
print("Exploring the AIMessage object...")
print("-" * 40)

print(f"\n1. content         : {response.content[:80]}...")
print(f"2. type            : {response.type}")   # always "ai"

# response_metadata contains token usage, model info, finish reason
if response.response_metadata:
    print(f"3. response_metadata keys: {list(response.response_metadata.keys())}")
    if "token_usage" in response.response_metadata:
        usage = response.response_metadata["token_usage"]
        print(f"   - prompt_tokens    : {usage.get('prompt_tokens', 'N/A')}")
        print(f"   - completion_tokens: {usage.get('completion_tokens', 'N/A')}")
        print(f"   - total_tokens     : {usage.get('total_tokens', 'N/A')}")

# usage_metadata is the standardized LangChain format (available on newer versions)
if hasattr(response, "usage_metadata") and response.usage_metadata:
    print(f"\n4. usage_metadata: {response.usage_metadata}")

print()

# =============================================================================
# SECTION 5: CALLING WITH A LIST OF MESSAGES
# =============================================================================
# Instead of a plain string, you can pass a list of messages.
# This lets you give the LLM context: who is speaking, what role they play.

print("-" * 40)
print("Calling with a HumanMessage object...")
print("-" * 40)

message = HumanMessage(content="Who invented Python (the language)?")
response2 = llm.invoke([message])
print(f"\nAnswer: {response2.content}")

# =============================================================================
# SECTION 6: THE KEY INSIGHT
# =============================================================================
print()
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 0")
print("=" * 60)
print("""
  1. LangChain wraps LLM APIs with a clean, unified interface
  2. llm.invoke() works with strings, messages, or lists of messages
  3. The response is an AIMessage object — not a raw string
  4. .content gives you the text. .response_metadata gives you stats
  5. To switch AI providers: change ONE line (the model string)
  6. The OpenRouter base_url lets us access 100+ models

  Next up: 01_llm_and_chat_models.py
  We'll explore HumanMessage, SystemMessage, streaming, and batching.
""")
