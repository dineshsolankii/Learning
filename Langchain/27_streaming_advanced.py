# =============================================================================
# FILE: 27_streaming_advanced.py
# PART: 7 - Callbacks & Streaming  |  LEVEL: Advanced
# =============================================================================
#
# THE STORY:
#   Nobody wants to stare at a loading spinner for 10 seconds.
#   Streaming sends tokens one by one as the LLM generates them —
#   just like ChatGPT types in real time.
#
#   For web applications, you need ASYNC streaming.
#   astream() is the async version of stream() — non-blocking.
#   astream_events() is the power tool — fine-grained events for every
#   LLM token, tool call, and chain step.
#
# WHAT YOU WILL LEARN:
#   1. Async streaming with astream() — non-blocking token delivery
#   2. astream_events() — fine-grained event streaming (v0.2+)
#   3. Distinguishing event types: on_chat_model_stream, on_tool_start, etc.
#   4. Building a real-time CLI chat interface
#   5. How to integrate streaming into web frameworks (FastAPI pattern)
#
# HOW THIS CONNECTS:
#   Previous: 26_callbacks_and_streaming.py — callbacks and basic streaming
#   Next:     28_langgraph_intro.py — graph-based workflows with LangGraph
# =============================================================================

import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
)

prompt = ChatPromptTemplate.from_template("Answer this question thoughtfully: {question}")
parser = StrOutputParser()
chain = prompt | llm | parser

print("=" * 60)
print("  CHAPTER 27: Advanced Streaming")
print("=" * 60)

# =============================================================================
# SECTION 1: SYNC STREAMING (Quick Recap)
# =============================================================================
# Synchronous streaming — blocks the thread until complete.
# Fine for scripts and CLI tools. Not for web servers.

print("\n--- Section 1: Sync Streaming (Quick Recap) ---")

print("\nSync stream (blocks thread, fine for scripts):")
print("AI: ", end="", flush=True)

for token in chain.stream({"question": "Name 3 programming languages in a comma-separated list."}):
    print(token, end="", flush=True)

print("\n")

# =============================================================================
# SECTION 2: ASYNC STREAMING WITH astream()
# =============================================================================
# astream() is the async version — non-blocking.
# Use this in FastAPI, Starlette, or any async web framework.
# The event loop handles multiple requests concurrently.

print("--- Section 2: Async Streaming with astream() ---")

async def async_stream_demo():
    """Demonstrate async token streaming."""
    print("\nAsync stream (non-blocking):")
    print("AI: ", end="", flush=True)

    # astream() returns an async generator — use `async for`
    async for token in chain.astream({"question": "Explain recursion in programming in 3 sentences."}):
        print(token, end="", flush=True)

    print("\n")

asyncio.run(async_stream_demo())

# =============================================================================
# SECTION 3: astream_events() — FINE-GRAINED EVENT STREAMING
# =============================================================================
# astream_events() is the most powerful streaming API.
# It fires named events for EVERY action in the chain:
#   on_chain_start       → when a chain begins
#   on_chat_model_start  → when LLM starts
#   on_chat_model_stream → each new token from LLM
#   on_chat_model_end    → when LLM finishes
#   on_tool_start        → when a tool is called
#   on_tool_end          → when a tool returns
#   on_chain_end         → when chain completes

print("--- Section 3: astream_events() — Named Event Streaming ---")

async def stream_events_demo():
    """Show each named event as it fires."""
    print("\nStreaming events (showing all event types):")
    print("-" * 50)

    token_buffer = []
    events_seen = []

    async for event in chain.astream_events(
        {"question": "What is Python? Answer in 2 sentences."},
        version="v1"  # Use v1 event format
    ):
        event_type = event.get("event", "unknown")

        if event_type == "on_chain_start":
            chain_name = event.get("name", "chain")
            if chain_name not in events_seen:
                events_seen.append(chain_name)
                print(f"  [CHAIN START] {chain_name}")

        elif event_type == "on_chat_model_start":
            print(f"  [LLM START  ] Model starting...")
            print(f"  AI: ", end="", flush=True)

        elif event_type == "on_chat_model_stream":
            # This fires for every token from the LLM
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                print(chunk.content, end="", flush=True)
                token_buffer.append(chunk.content)

        elif event_type == "on_chat_model_end":
            print(f"\n  [LLM END    ] Finished ({len(token_buffer)} tokens streamed)")

        elif event_type == "on_tool_start":
            tool_name = event.get("name", "tool")
            tool_input = event.get("data", {}).get("input", "")
            print(f"\n  [TOOL START ] {tool_name}({str(tool_input)[:50]})")

        elif event_type == "on_tool_end":
            tool_output = event.get("data", {}).get("output", "")
            print(f"  [TOOL END   ] Result: {str(tool_output)[:80]}")

        elif event_type == "on_chain_end":
            pass  # Skip chain end clutter

    print("-" * 50)

asyncio.run(stream_events_demo())

# =============================================================================
# SECTION 4: CONCURRENT ASYNC STREAMING
# =============================================================================
# The real power of async: stream multiple questions SIMULTANEOUSLY.
# In a web app, this handles multiple users at the same time.

print("\n--- Section 4: Concurrent Async Streaming ---")

async def stream_one_question(question: str, label: str) -> str:
    """Stream one question and return the full answer."""
    result = ""
    async for token in chain.astream({"question": question}):
        result += token
    return f"[{label}]: {result.strip()[:80]}..."

async def concurrent_streaming():
    """Stream 3 questions concurrently — asyncio.gather runs them together."""
    print("\nStreaming 3 questions CONCURRENTLY (asyncio.gather):")

    import time
    start = time.time()

    # All 3 start at the same time!
    results = await asyncio.gather(
        stream_one_question("What is Python?", "Q1"),
        stream_one_question("What is Java?", "Q2"),
        stream_one_question("What is Rust?", "Q3"),
    )

    elapsed = time.time() - start
    print(f"\nAll 3 completed in {elapsed:.2f}s (faster than sequential!)")
    for result in results:
        print(f"  {result}")

asyncio.run(concurrent_streaming())

# =============================================================================
# SECTION 5: STREAMING CLI CHAT INTERFACE
# =============================================================================
# A real-time chat interface using sync streaming.
# Tokens appear as they're generated — smooth UX.

print("\n--- Section 5: Real-Time Streaming CLI Chat ---")

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder

# Chat prompt with history for context
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Be concise."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

chat_chain = chat_prompt | llm | parser

# Add memory
chat_store = {}
def get_history(session_id):
    if session_id not in chat_store:
        chat_store[session_id] = InMemoryChatMessageHistory()
    return chat_store[session_id]

chat_with_memory = RunnableWithMessageHistory(
    chat_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

print("\nStreaming Chat with Memory (type 'exit' to stop)\n")

session_cfg = {"configurable": {"session_id": "streaming_session"}}

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    print("AI: ", end="", flush=True)

    # Stream the response token by token
    for token in chat_with_memory.stream(
        {"input": user_input},
        config=session_cfg
    ):
        print(token, end="", flush=True)

    print("\n")  # New line after streaming complete

# =============================================================================
# SECTION 6: FastAPI STREAMING PATTERN
# =============================================================================
print("\n--- Section 6: FastAPI Streaming Pattern ---")
print("""
  In a FastAPI server, use async generators for streaming:

  from fastapi import FastAPI
  from fastapi.responses import StreamingResponse

  app = FastAPI()

  @app.get("/chat")
  async def chat_stream(question: str):
      async def generate():
          async for token in chain.astream({"question": question}):
              # Send each token as a Server-Sent Event
              yield f"data: {token}\\n\\n"
          yield "data: [DONE]\\n\\n"

      return StreamingResponse(
          generate(),
          media_type="text/event-stream",
          headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
      )

  # Frontend reads the SSE stream and appends tokens in real time
  # This is how ChatGPT's UI works!
""")

# =============================================================================
# SECTION 7: WHEN TO USE WHICH STREAMING METHOD
# =============================================================================
print("--- Section 7: Choosing the Right Streaming Method ---")
print("""
  ┌──────────────────────┬──────────────────────────────────────┐
  │ Method               │ Use Case                             │
  ├──────────────────────┼──────────────────────────────────────┤
  │ llm.stream()         │ Simple token streaming, CLI tools    │
  │ chain.stream()       │ Full chain streaming, CLI apps       │
  │ llm.astream()        │ Single async stream in web handlers  │
  │ chain.astream()      │ Full async chain streaming in web    │
  │ astream_events()     │ Need to distinguish event types      │
  │                      │ (LLM tokens vs tool calls)           │
  └──────────────────────┴──────────────────────────────────────┘

  GOLDEN RULE:
  → CLI / scripts → sync: stream()
  → Web apps / concurrent → async: astream(), astream_events()
  → Need tool call visibility → astream_events()
""")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 27")
print("=" * 60)
print("""
  1. stream() is synchronous — blocks the thread (fine for CLI)
  2. astream() is async — non-blocking (required for web apps)
  3. astream_events() fires named events for every action (LLM token, tool, chain)
  4. asyncio.gather() runs multiple streams concurrently (handles multiple users)
  5. Use `async for event in chain.astream_events(..., version='v1'):`
  6. FastAPI + StreamingResponse + async generator = real-time chat UI

  PART 7 (CALLBACKS & STREAMING) COMPLETE!

  Next up: 28_langgraph_intro.py
  Graph-based workflows that support cycles, branching, and state.
""")
