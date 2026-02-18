# =============================================================================
# FILE: 29_langgraph_stateful_agent.py
# PART: 8 - LangGraph  |  LEVEL: Advanced
# =============================================================================
#
# THE STORY:
#   AgentExecutor is a black box. You give it a task, it returns an answer.
#   You can't see the loop. You can't pause it. You can't replay steps.
#
#   LangGraph's agent is a glass box.
#   Every node is visible. Every state transition is inspectable.
#   You can add a human approval step. You can replay from any checkpoint.
#   This is how PRODUCTION agents are built.
#
# WHAT YOU WILL LEARN:
#   1. MessagesState — pre-built state for message-based agents
#   2. ToolNode — pre-built node that executes tool calls
#   3. tools_condition — pre-built edge that routes based on tool calls
#   4. MemorySaver — LangGraph's built-in checkpointer (persistent memory)
#   5. Thread-based session isolation (thread_id)
#   6. Streaming graph events node by node
#
# HOW THIS CONNECTS:
#   Previous: 28_langgraph_intro.py — graph basics with StateGraph
#   Next:     30_langgraph_multi_agent.py — supervisor + specialist pattern
# =============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools.base import ToolException
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
import math
from datetime import datetime

load_dotenv()

print("=" * 60)
print("  CHAPTER 29: Stateful Agent with LangGraph")
print("=" * 60)

# =============================================================================
# SECTION 1: MESSAGESSTATE — PRE-BUILT STATE FOR AGENTS
# =============================================================================
# MessagesState is a pre-built TypedDict that has one field:
#   messages: List[BaseMessage]
# It's the standard state for message-based agents.
# LangGraph automatically APPENDS to the messages list (uses Annotated[list, add_messages])
# rather than replacing it — so history accumulates naturally.

print("\n--- Section 1: MessagesState ---")
print("""
  MessagesState is equivalent to:

  from typing import Annotated
  from langgraph.graph.message import add_messages

  class AgentState(TypedDict):
      messages: Annotated[list, add_messages]
      # add_messages: reducer that APPENDS new messages (doesn't replace)

  When a node returns {"messages": [new_message]},
  LangGraph ADDS it to the list, not replaces.
  This is how conversation history accumulates automatically.
""")

# =============================================================================
# SECTION 2: DEFINE TOOLS
# =============================================================================

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Supports +, -, *, /, **, sqrt, pi."""
    safe_env = {"sqrt": math.sqrt, "abs": abs, "round": round, "pi": math.pi, "e": math.e}
    try:
        import ast
        result = eval(compile(ast.parse(expression, mode='eval'), '<string>', 'eval'),
                      {"__builtins__": {}}, safe_env)
        return f"{expression} = {result}"
    except Exception as e:
        raise ToolException(f"Math error: {e}")

@tool
def get_today() -> str:
    """Get today's date and day of the week."""
    return datetime.now().strftime("%A, %B %d, %Y")

@tool
def capital_lookup(country: str) -> str:
    """Get the capital city of a country."""
    capitals = {
        "france": "Paris", "germany": "Berlin", "japan": "Tokyo",
        "india": "New Delhi", "usa": "Washington D.C.", "australia": "Canberra",
        "canada": "Ottawa", "china": "Beijing", "brazil": "Brasília",
        "uk": "London", "united kingdom": "London",
    }
    capital = capitals.get(country.lower().strip())
    if not capital:
        raise ToolException(f"Capital for '{country}' not found. Try the full country name.")
    return f"The capital of {country.title()} is {capital}."

tools = [calculator, get_today, capital_lookup]

# =============================================================================
# SECTION 3: DEFINE THE AGENT NODE
# =============================================================================
# The agent node calls the LLM with the current message history.
# If the LLM wants to call a tool, it returns an AIMessage with tool_calls.
# If the LLM has the final answer, it returns an AIMessage with content.

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
    temperature=0.0,
)

# Bind tools to the LLM — now the LLM knows what tools are available
llm_with_tools = llm.bind_tools(tools)

def call_model(state: MessagesState) -> dict:
    """
    Agent node: calls the LLM with current message history.
    Returns a new AIMessage that either has:
    - tool_calls: [ToolCall] → means the LLM wants to call a tool
    - content: str          → means the LLM has the final answer
    """
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    # Return as a list — MessagesState will APPEND this to the history
    return {"messages": [response]}

# =============================================================================
# SECTION 4: BUILD THE AGENT GRAPH
# =============================================================================
print("--- Section 4: Building the Agent Graph ---")

# ToolNode is a pre-built node that:
# 1. Reads tool_calls from the last AIMessage in state["messages"]
# 2. Executes the appropriate tool functions
# 3. Returns ToolMessages with the results (appended to messages)
tool_node = ToolNode(tools)

# Build the graph
graph_builder = StateGraph(MessagesState)

# Add nodes
graph_builder.add_node("agent", call_model)  # LLM reasoning
graph_builder.add_node("tools", tool_node)   # Tool execution

# Always start with the agent
graph_builder.add_edge(START, "agent")

# tools_condition is a pre-built routing function:
# - If last message has tool_calls → route to "tools"
# - Otherwise (final answer) → route to END
graph_builder.add_conditional_edges(
    "agent",
    tools_condition,  # Pre-built: checks for tool_calls in last message
)

# After tools execute → always loop back to agent
# This is the ReAct loop! agent → tools → agent → tools → agent → END
graph_builder.add_edge("tools", "agent")

print("  Graph structure:")
print("  START → agent → (has tool calls?) → tools → agent → ...")
print("                    ↓ (final answer)")
print("                   END")

# =============================================================================
# SECTION 5: ADD MEMORY WITH MemorySaver
# =============================================================================
# MemorySaver is LangGraph's built-in checkpointer.
# It saves the full state after each node execution.
# Conversations persist as long as the process runs (in-memory).
#
# For database persistence, use SqliteSaver or PostgresSaver.

memory = MemorySaver()

# Compile the graph with the checkpointer
agent_graph = graph_builder.compile(checkpointer=memory)

print("\n  MemorySaver attached — conversations persist via thread_id")

# =============================================================================
# SECTION 6: VISUALIZE THE GRAPH
# =============================================================================
print("\n--- Section 6: Graph Visualization ---")
try:
    print(agent_graph.get_graph().draw_mermaid())
except Exception:
    print("  Graph: START → agent ⟷ tools (conditional) → END")

# =============================================================================
# SECTION 7: RUNNING WITH THREAD-BASED MEMORY
# =============================================================================
# thread_id isolates conversations — each thread has its own history.
# Same thread_id across calls = same conversation continues.

print("\n--- Section 7: Multi-Turn Conversation with Memory ---")

# Session 1 for Dinesh
dinesh_config = {"configurable": {"thread_id": "dinesh_session_1"}}

print("\n[Dinesh's Conversation]")

# Turn 1
result1 = agent_graph.invoke(
    {"messages": [HumanMessage("What is the capital of Japan?")]},
    config=dinesh_config
)
print(f"\nYou: What is the capital of Japan?")
print(f"AI : {result1['messages'][-1].content}")

# Turn 2 — agent remembers the previous exchange (same thread_id)
result2 = agent_graph.invoke(
    {"messages": [HumanMessage("How many letters does that city name have?")]},
    config=dinesh_config
)
print(f"\nYou: How many letters does that city name have?")
print(f"AI : {result2['messages'][-1].content}")

# Turn 3 — complex multi-tool task
result3 = agent_graph.invoke(
    {"messages": [HumanMessage("What is today's date? Also what is 2025 minus 1991?")]},
    config=dinesh_config
)
print(f"\nYou: What is today's date? Also what is 2025 minus 1991?")
print(f"AI : {result3['messages'][-1].content}")

# Inspect the full message history for this thread
print(f"\n--- Message history for thread 'dinesh_session_1' ---")
state = agent_graph.get_state(dinesh_config)
messages = state.values.get("messages", [])
print(f"Total messages: {len(messages)}")
for i, msg in enumerate(messages):
    role = type(msg).__name__
    content_preview = str(msg.content)[:60] if msg.content else "(tool call)"
    print(f"  [{i+1}] {role:20}: {content_preview}...")

# =============================================================================
# SECTION 8: STREAMING GRAPH EVENTS
# =============================================================================
print("\n--- Section 8: Streaming Graph Events ---")

# A fresh thread for streaming demo
stream_config = {"configurable": {"thread_id": "stream_demo"}}

print("\nStreaming: 'What is sqrt(256) and what day is today?'")
print()

for event in agent_graph.stream(
    {"messages": [HumanMessage("What is sqrt(256) and what day is today?")]},
    config=stream_config,
    stream_mode="updates",   # Only stream state UPDATES (not full state)
):
    for node_name, node_update in event.items():
        print(f"  [{node_name}] output:")
        if "messages" in node_update:
            for msg in node_update["messages"]:
                msg_type = type(msg).__name__
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    print(f"    {msg_type}: calling tools → {[tc['name'] for tc in msg.tool_calls]}")
                elif hasattr(msg, "content") and msg.content:
                    print(f"    {msg_type}: {str(msg.content)[:100]}")
                elif hasattr(msg, "name"):
                    print(f"    ToolMessage({msg.name}): {str(msg.content)[:80]}")

# =============================================================================
# SECTION 9: INTERACTIVE AGENT
# =============================================================================
print()
print("=" * 50)
print("  Interactive LangGraph Agent")
print("  (Memory persists via thread_id)")
print("  Type 'exit' to stop, 'new' to start fresh thread")
print("=" * 50 + "\n")

thread_counter = 100

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    if user_input.lower() == "new":
        thread_counter += 1
        dinesh_config = {"configurable": {"thread_id": f"session_{thread_counter}"}}
        print(f"  (New thread: session_{thread_counter})\n")
        continue

    result = agent_graph.invoke(
        {"messages": [HumanMessage(user_input)]},
        config=dinesh_config
    )
    print(f"AI: {result['messages'][-1].content}\n")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 29")
print("=" * 60)
print("""
  1. MessagesState: pre-built state with messages list (auto-appends)
  2. ToolNode: pre-built node that executes LLM tool calls automatically
  3. tools_condition: pre-built router — tools if tool_calls, END if final
  4. MemorySaver: checkpointer that saves state after every node
  5. thread_id: isolates conversations — same ID = same conversation
  6. The agent loop: agent → tools → agent (cycles until final answer)
  7. graph.get_state(config) shows full conversation history for a thread
  8. stream_mode="updates" shows only state changes (cleaner output)

  Next up: 30_langgraph_multi_agent.py
  Supervisor + specialist agents working together on complex tasks.
""")
