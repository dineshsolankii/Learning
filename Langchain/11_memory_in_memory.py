# =============================================================================
# FILE: 11_memory_in_memory.py
# PART: 3 - Memory  |  LEVEL: Intermediate
# =============================================================================
#
# THE STORY:
#   Your chatbot has goldfish memory. Every message is a fresh start.
#   Ask it "What did I just say?" and it genuinely doesn't know.
#
#   InMemoryChatMessageHistory is the notepad it keeps between turns.
#   RunnableWithMessageHistory is the wrapper that reads from the notepad,
#   injects the history into the prompt, and writes new messages back to it.
#
#   But what happens when the notepad gets too full?
#   After 50 messages, you're sending thousands of tokens on every call.
#   Today you learn to trim old messages and isolate multiple users.
#
# WHAT YOU WILL LEARN:
#   1. InMemoryChatMessageHistory — store messages in RAM
#   2. RunnableWithMessageHistory — auto-manage history in any chain
#   3. session_id — isolate conversations per user
#   4. Inspecting the store — what's inside the memory
#   5. trim_messages() — prevent context window overflow
#   6. Multi-user session isolation demo
#
# HOW THIS CONNECTS:
#   Previous: 10_structured_output.py — getting typed output from LLMs
#   Next:     12_memory_persistent.py — history that survives process restarts
# =============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages, HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini"
)

print("=" * 60)
print("  CHAPTER 11: In-Memory Conversation History")
print("=" * 60)

# =============================================================================
# SECTION 1: THE CORE SETUP — 3 COMPONENTS
# =============================================================================
# To build a chatbot with memory, you need 3 things:
#
#   1. A prompt with MessagesPlaceholder — slot where history gets injected
#   2. A store dict — maps session_id → InMemoryChatMessageHistory
#   3. RunnableWithMessageHistory — the glue that reads/writes the store

# Component 1: Prompt with history slot
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use past conversation context to answer."),
    MessagesPlaceholder(variable_name="history"),  # ← History injected here
    ("human", "{input}"),                           # ← Latest user message
])

# Component 2: The store — a dict mapping session_id to its history object
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return the history for a given session, creating it if new."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# The chain (without memory)
chain = prompt | llm

# Component 3: Wrap with memory management
chat_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",     # Which key has the user's message
    history_messages_key="history", # Which key in prompt gets the history
)

# =============================================================================
# SECTION 2: RUNNING A MULTI-TURN CONVERSATION
# =============================================================================
# When you invoke with the same session_id, history accumulates.
# The AI remembers everything from earlier in the session.

print("\n--- Section 2: Multi-Turn Conversation (Dinesh's session) ---")

session_config = {"configurable": {"session_id": "dinesh_session"}}

# Turn 1: Introduce yourself
r1 = chat_with_memory.invoke({"input": "Hi! My name is Dinesh and I work in AI automation."}, config=session_config)
print(f"\nYou: Hi! My name is Dinesh and I work in AI automation.")
print(f"AI : {r1.content}")

# Turn 2: Ask about a topic
r2 = chat_with_memory.invoke({"input": "What is LangChain good for?"}, config=session_config)
print(f"\nYou: What is LangChain good for?")
print(f"AI : {r2.content[:200]}...")

# Turn 3: Test memory — does it remember the name?
r3 = chat_with_memory.invoke({"input": "What is my name and what do I do for work?"}, config=session_config)
print(f"\nYou: What is my name and what do I do for work?")
print(f"AI : {r3.content}")

# =============================================================================
# SECTION 3: INSPECTING THE STORE
# =============================================================================
# Let's look inside the store to see what got saved.

print("\n--- Section 3: Inspecting the Store ---")

session_history = store["dinesh_session"]
messages = session_history.messages

print(f"\nTotal messages in store: {len(messages)}")
print("\nMessage history:")
for i, msg in enumerate(messages):
    role = "Human" if isinstance(msg, HumanMessage) else "AI"
    print(f"  [{i+1}] {role:5}: {msg.content[:80]}...")

# =============================================================================
# SECTION 4: MULTI-USER SESSION ISOLATION
# =============================================================================
# Each session_id gets its own isolated history.
# User A and User B can chat simultaneously without seeing each other's messages.

print("\n--- Section 4: Multi-User Session Isolation ---")

# Alice's session
alice_config = {"configurable": {"session_id": "alice"}}
chat_with_memory.invoke({"input": "I am Alice and I love cooking."}, config=alice_config)
chat_with_memory.invoke({"input": "My favorite cuisine is Italian."}, config=alice_config)

# Bob's session
bob_config = {"configurable": {"session_id": "bob"}}
chat_with_memory.invoke({"input": "I am Bob and I love hiking."}, config=bob_config)

# Test: Does Alice's context leak into Bob's session?
alice_r = chat_with_memory.invoke({"input": "What do you know about me?"}, config=alice_config)
bob_r = chat_with_memory.invoke({"input": "What do you know about me?"}, config=bob_config)

print(f"\nAlice's session answer: {alice_r.content[:200]}")
print(f"\nBob's session answer  : {bob_r.content[:200]}")
print("\n(Note: They have completely separate contexts!)")

# Show all sessions in the store
print(f"\nSessions in store: {list(store.keys())}")
for session_id, history in store.items():
    print(f"  {session_id}: {len(history.messages)} messages")

# =============================================================================
# SECTION 5: CLEARING HISTORY
# =============================================================================
# You can clear a session's history at any time.
# Useful for "start new conversation" buttons in a chat app.

print("\n--- Section 5: Clearing History ---")

print(f"\nBefore clear: {len(store.get('alice', InMemoryChatMessageHistory()).messages)} messages in Alice's session")
if "alice" in store:
    store["alice"].clear()
print(f"After clear : {len(store.get('alice', InMemoryChatMessageHistory()).messages)} messages in Alice's session")

# =============================================================================
# SECTION 6: TRIMMING OLD MESSAGES (PREVENT CONTEXT OVERFLOW)
# =============================================================================
# Problem: After 50+ messages, your prompt gets huge and costs a lot.
# Solution: trim_messages() keeps only the last N messages.
# Build a chain that automatically trims before injecting history.

print("\n--- Section 6: Message Trimming ---")

from langchain_core.runnables import RunnableLambda

def trim_history(input_dict: dict) -> dict:
    """Keep only the last 6 messages to prevent context overflow."""
    if "history" in input_dict and input_dict["history"]:
        # Keep last 6 messages (3 turns)
        input_dict["history"] = input_dict["history"][-6:]
    return input_dict

# This trimming function runs BEFORE the prompt sees the history
trimming_chain = RunnableLambda(trim_history) | chain

trim_chat = RunnableWithMessageHistory(
    trimming_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

print("\nTrimming chain ready — keeps only last 6 messages.")
print("(In a real app, this prevents expensive long context windows)")

# =============================================================================
# SECTION 7: INTERACTIVE CHAT LOOP
# =============================================================================
print("\n--- Section 7: Interactive Chat (type 'exit' to stop) ---\n")

session_id = "user_1"

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    response = chat_with_memory.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )

    print("AI:", response.content)
    print()

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 11")
print("=" * 60)
print("""
  1. InMemoryChatMessageHistory stores messages in a Python dict (RAM only)
  2. RunnableWithMessageHistory auto-reads and writes to the history store
  3. session_id isolates each user's conversation — no cross-contamination
  4. store["session_id"].messages shows the full raw message list
  5. store["session_id"].clear() resets a conversation
  6. Trim history to prevent context window overflow in long conversations
  7. This memory VANISHES when the script ends — see file 12 for persistence

  Next up: 12_memory_persistent.py
  History that survives process restarts using files and databases.
""")
