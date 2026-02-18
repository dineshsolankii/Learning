# =============================================================================
# FILE: 03_lcel_first_chain.py
# PART: 1 - Foundations  |  LEVEL: Beginner
# =============================================================================
#
# THE STORY:
#   The pipe | is the heartbeat of LangChain.
#   Imagine a factory assembly line:
#     Raw material (your query) → Station 1: Prompt → Station 2: LLM → Station 3: Parser
#     Finished product: a clean string answer.
#
#   This is LCEL — LangChain Expression Language.
#   The | operator links any two Runnables. Since prompts, LLMs, and parsers
#   are ALL Runnables (they all have .invoke()), they can be chained.
#
# WHAT YOU WILL LEARN:
#   1. The | pipe operator — chaining Runnables
#   2. The three-stage chain: prompt | llm | output_parser
#   3. StrOutputParser — strips AIMessage wrapper, returns plain string
#   4. get_openai_callback() — track token usage and API cost
#   5. Stepping through the chain manually to understand each stage
#
# HOW THIS CONNECTS:
#   Previous: 02_prompt_templates.py — building prompt templates
#   Next:     04_output_parsers.py — JSON, Pydantic, and advanced parsers
# =============================================================================

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback

load_dotenv()

# =============================================================================
# SECTION 1: THE THREE COMPONENTS OF A CHAIN
# =============================================================================
# Every LCEL chain has at least three components:
#   1. Prompt  — formats the user query into proper messages for the LLM
#   2. LLM     — sends messages to the AI, receives an AIMessage response
#   3. Parser  — transforms AIMessage into a usable Python type (string, dict, etc.)

# The LLM — GPT-4o-mini via OpenRouter
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini"
)

# The Prompt Template — {query} gets filled in at chain.invoke() time
prompt = ChatPromptTemplate.from_template("{query}")

# The Output Parser — extracts .content from AIMessage → returns plain string
# Without this, chain.invoke() returns an AIMessage object, not a string
output_parser = StrOutputParser()

# =============================================================================
# SECTION 2: BUILDING THE CHAIN WITH |
# =============================================================================
# The | operator connects Runnables. Each component's OUTPUT → next INPUT:
#
#   prompt.invoke({"query": "..."}) → ChatPromptValue (formatted messages)
#   llm.invoke(ChatPromptValue)     → AIMessage(content="...")
#   output_parser.invoke(AIMessage) → "..." (plain Python string)
#
# The full chain bundles all three into a single Runnable.
chain = prompt | llm | output_parser

# =============================================================================
# SECTION 3: RUNNING THE CHAIN WITH TOKEN TRACKING
# =============================================================================
# get_openai_callback() is a context manager that counts tokens and cost.
# Any LLM calls inside the `with` block are tracked automatically.

print("=" * 60)
print("  CHAPTER 3: Your First LCEL Chain  (prompt | llm | parser)")
print("=" * 60)

user_prompt = input("\nEnter your query: ")

print(f"\nRunning: prompt | llm | output_parser ...")
print()

with get_openai_callback() as cb:
    # One .invoke() runs prompt, llm, AND parser — in sequence
    result = chain.invoke({"query": user_prompt})

    print("AI Answer:")
    print("-" * 40)
    print(result)  # result is a plain Python string
    print()
    print("--- Token Usage ---")
    print(f"Prompt Tokens     : {cb.prompt_tokens}")
    print(f"Completion Tokens : {cb.completion_tokens}")
    print(f"Total Tokens      : {cb.total_tokens}")
    print(f"Total Cost (USD)  : ${cb.total_cost:.6f}")

# =============================================================================
# SECTION 4: STEPPING THROUGH EACH STAGE MANUALLY
# =============================================================================
# Let's peel back the curtain — manually call each component to see
# what the | operator is passing between stages.

print("\n--- Section 4: Anatomy of the Chain (manual step-through) ---")

sample_input = {"query": "What color is the sky?"}

# Stage 1: Prompt formats the raw input
stage1_output = prompt.invoke(sample_input)
print(f"\nStage 1 - prompt.invoke() type : {type(stage1_output).__name__}")
print(f"  Messages: {[m.content for m in stage1_output.to_messages()]}")

# Stage 2: LLM receives the formatted messages and returns AIMessage
stage2_output = llm.invoke(stage1_output)
print(f"\nStage 2 - llm.invoke() type    : {type(stage2_output).__name__}")
print(f"  Content: {stage2_output.content}")

# Stage 3: Parser strips the AIMessage wrapper → plain string
stage3_output = output_parser.invoke(stage2_output)
print(f"\nStage 3 - parser.invoke() type : {type(stage3_output).__name__}")
print(f"  Result: {stage3_output}")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 3")
print("=" * 60)
print("""
  1. LCEL uses | to chain Runnables: prompt | llm | parser
  2. Each component's output is passed to the next component's input
  3. chain.invoke({"query": "..."}) runs all three steps at once
  4. StrOutputParser converts AIMessage → plain Python string
  5. get_openai_callback() tracks token usage and cost
  6. You can inspect each step manually to understand the data flow

  Next up: 04_output_parsers.py
  Go beyond strings — JSON dicts, Pydantic models, typed Python objects.
""")
