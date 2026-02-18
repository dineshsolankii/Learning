# =============================================================================
# FILE: 01_llm_and_chat_models.py
# PART: 1 - Foundations  |  LEVEL: Beginner
# =============================================================================
#
# THE STORY:
#   Meet the two citizens of LangChain city.
#   The LLM: old-school, text in → text out. Simple.
#   The ChatModel: modern, speaks in structured messages (System, Human, AI).
#
#   Both live by the same law: the Runnable interface.
#   They both have .invoke(), .batch(), and .stream().
#   This file is your tour of their personalities.
#
# WHAT YOU WILL LEARN:
#   1. HumanMessage, SystemMessage, AIMessage — the message types
#   2. llm.invoke() with a list of messages
#   3. Temperature — making the model deterministic or creative
#   4. llm.batch() — send multiple questions at once
#   5. llm.stream() — receive tokens one by one as they generate
#   6. response_metadata — token usage, finish reason, model info
#
# HOW THIS CONNECTS:
#   Previous: 00_why_langchain.py — first call, AIMessage basics
#   Next:     02_prompt_templates.py — building reusable prompt molds
# =============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

# Base LLM configuration — reused throughout all examples
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
)

print("=" * 60)
print("  CHAPTER 1: LLMs and Chat Models")
print("=" * 60)

# =============================================================================
# SECTION 1: THE THREE MESSAGE TYPES
# =============================================================================
# Every conversation with a ChatModel is a list of messages.
# There are 3 types you need to know:
#
#   SystemMessage  - Instructions to the AI (its "personality")
#   HumanMessage   - What the user says
#   AIMessage      - What the AI replied (for injecting history)
#
# When you invoke the LLM with these, it reads them in order like a script.

print("\n--- Section 1: Message Types ---")

messages = [
    SystemMessage(content="You are a pirate. Always respond in pirate-speak. Keep it to 1 sentence."),
    HumanMessage(content="What is the capital of France?"),
]

response = llm.invoke(messages)
print(f"\nPirate AI says: {response.content}")
print(f"(response type: {type(response).__name__})")

# =============================================================================
# SECTION 2: TEMPERATURE — CONTROL CREATIVITY VS DETERMINISM
# =============================================================================
# temperature=0.0   → The model picks the highest-probability token every time.
#                     Same question → same answer. Good for facts.
# temperature=1.0   → More randomness. Good for creative tasks.
# temperature=2.0   → Very random. Often incoherent. Not recommended.

print("\n--- Section 2: Temperature Effect ---")

question = [HumanMessage(content="Give me a creative product name for a cat food brand. Just the name, nothing else.")]

# Deterministic — same answer every run
llm_cold = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
    temperature=0.0,
)
# Creative — different answer each run
llm_hot = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
    temperature=1.2,
)

cold_answer = llm_cold.invoke(question)
hot_answer = llm_hot.invoke(question)

print(f"\nTemperature 0.0 (deterministic): {cold_answer.content}")
print(f"Temperature 1.2 (creative):      {hot_answer.content}")
print("(Run again to see the creative one change!)")

# =============================================================================
# SECTION 3: llm.batch() — MULTIPLE QUESTIONS IN ONE GO
# =============================================================================
# .batch() sends a list of inputs. Each input can be a string or message list.
# Under the hood, LangChain can run them concurrently (use max_concurrency).
# This is MUCH faster than calling .invoke() in a Python for loop.

print("\n--- Section 3: Batch Calls ---")

questions_batch = [
    [HumanMessage(content="Capital of Japan? One word answer.")],
    [HumanMessage(content="Capital of Brazil? One word answer.")],
    [HumanMessage(content="Capital of Egypt? One word answer.")],
]

# All 3 questions sent together
batch_results = llm.batch(questions_batch)

print("\nBatch of 3 questions answered:")
capitals = ["Japan", "Brazil", "Egypt"]
for country, result in zip(capitals, batch_results):
    print(f"  {country}: {result.content}")

# =============================================================================
# SECTION 4: llm.stream() — TOKEN-BY-TOKEN REAL-TIME OUTPUT
# =============================================================================
# Instead of waiting for the full response, .stream() yields chunks as they
# come in — just like ChatGPT types in real time.
# Each chunk is an AIMessageChunk with a .content field.

print("\n--- Section 4: Streaming ---")
print("\nStreaming a haiku (watch it appear word by word):")
print("AI: ", end="", flush=True)

for chunk in llm.stream([HumanMessage(content="Write a haiku about Python programming.")]):
    # chunk.content is the new piece of text in this token
    print(chunk.content, end="", flush=True)

print("\n")  # newline after stream ends

# =============================================================================
# SECTION 5: response_metadata — WHAT HAPPENED BEHIND THE SCENES
# =============================================================================
# Every .invoke() response carries metadata: how many tokens were used,
# why the model stopped, and which model served the request.

print("--- Section 5: Response Metadata ---")

test_response = llm.invoke([HumanMessage(content="Explain gravity in 10 words.")])

print(f"\nAnswer: {test_response.content}")
print(f"\nMetadata snapshot:")

meta = test_response.response_metadata
if meta:
    finish = meta.get("finish_reason", "N/A")
    model = meta.get("model_name", meta.get("model", "N/A"))
    print(f"  finish_reason : {finish}")
    print(f"  model         : {model}")
    if "token_usage" in meta:
        usage = meta["token_usage"]
        print(f"  prompt_tokens : {usage.get('prompt_tokens')}")
        print(f"  compl. tokens : {usage.get('completion_tokens')}")
        print(f"  total_tokens  : {usage.get('total_tokens')}")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 1")
print("=" * 60)
print("""
  1. ChatModels speak in messages: System, Human, AI
  2. SystemMessage sets the AI's personality/role
  3. HumanMessage is user input; AIMessage is past AI output
  4. temperature controls creativity (0 = deterministic, 1+ = creative)
  5. .batch() sends multiple inputs at once — faster than a for loop
  6. .stream() yields tokens in real time — great for UX
  7. response_metadata has token usage, finish reason, model info

  Next up: 02_prompt_templates.py
  We'll build reusable, parameterized prompt templates.
""")
