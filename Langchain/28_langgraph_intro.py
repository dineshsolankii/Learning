# =============================================================================
# FILE: 28_langgraph_intro.py
# PART: 8 - LangGraph  |  LEVEL: Advanced
# =============================================================================
#
# THE STORY:
#   A chain is a one-way street. Input → output. No loops. No choices.
#   But real AI workflows loop: the agent tries, checks the result,
#   retries if wrong, takes a different path if conditions change.
#
#   LangGraph builds these loop-capable workflows as explicit GRAPHS.
#   Every possible path is visible. You can add state. You can branch.
#   You can cycle back. This is the architecture of production AI agents.
#
#   CORE CONCEPTS:
#     StateGraph : The graph itself (holds the workflow)
#     State      : A TypedDict that flows through the graph (the "memory")
#     Nodes      : Python functions that transform the state
#     Edges      : Connections between nodes (fixed or conditional)
#     START/END  : Special nodes for entry and exit
#
# WHAT YOU WILL LEARN:
#   1. StateGraph and TypedDict state
#   2. add_node() and add_edge() — building the graph structure
#   3. add_conditional_edges() — routing based on state values
#   4. graph.compile() and graph.invoke()
#   5. graph.get_graph().draw_mermaid() — visualize the graph
#
# HOW THIS CONNECTS:
#   Previous: 27_streaming_advanced.py — streaming and async
#   Next:     29_langgraph_stateful_agent.py — real agent with ToolNode
# =============================================================================

import os
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

print("=" * 60)
print("  CHAPTER 28: LangGraph Introduction")
print("=" * 60)

# =============================================================================
# SECTION 1: WHY GRAPHS INSTEAD OF CHAINS?
# =============================================================================
print("""
  CHAIN (Linear):        prompt → llm → parser → DONE
  GRAPH (Cyclic/Branched):
                         START → detect_language
                                 ↓ (if not English)
                         translate → answer → END
                                 ↓ (if English)
                         answer → END

  Graphs let you:
  - Branch based on content (if/else in the workflow)
  - Loop (retry if the answer is wrong)
  - Have multiple start/end points
  - Maintain STATE across steps
  - Build multi-agent systems (each agent = a node)
""")

# =============================================================================
# SECTION 2: DEFINING THE STATE
# =============================================================================
# The State is the data that flows through the graph.
# Every node receives the FULL state and returns a PARTIAL update.
# LangGraph merges the update into the current state automatically.
#
# Use TypedDict to define what's in the state.

print("--- Section 2: Defining the State ---")

class DocumentState(TypedDict):
    """
    State for our document processing pipeline.
    Each field represents a piece of data that nodes can read or write.
    """
    input_text: str             # The original text to process
    language: str               # Detected language (set by detect_language_node)
    needs_translation: bool     # Whether translation is needed
    translated_text: str        # Translation result (if needed)
    summary: str                # Final summary
    word_count: int             # Word count of the final text

print("\nDocumentState defined:")
print("  - input_text: str")
print("  - language: str")
print("  - needs_translation: bool")
print("  - translated_text: str")
print("  - summary: str")
print("  - word_count: int")

# =============================================================================
# SECTION 3: DEFINING THE NODES
# =============================================================================
# Nodes are plain Python functions.
# Input:  the full current state (a dict)
# Output: a dict with ONLY the keys you want to update (partial update)
# LangGraph merges your returned dict into the state automatically.

print("\n--- Section 3: Defining Nodes ---")

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
    temperature=0.3,
)

def detect_language_node(state: DocumentState) -> dict:
    """
    Node: Detect the language of the input text.
    Updates: language, needs_translation
    """
    text = state["input_text"]

    # Use the LLM to detect language
    response = llm.invoke([
        HumanMessage(content=f"Detect the language of this text. Reply with ONLY the language name (e.g., 'English', 'Spanish', 'French').\n\nText: {text[:200]}")
    ])
    language = response.content.strip()

    print(f"  [detect_language_node] Detected: {language}")

    return {
        "language": language,
        "needs_translation": language.lower() not in ["english"],
    }

def translate_node(state: DocumentState) -> dict:
    """
    Node: Translate non-English text to English.
    Updates: translated_text
    """
    text = state["input_text"]
    language = state["language"]

    print(f"  [translate_node] Translating from {language} to English...")

    response = llm.invoke([
        HumanMessage(content=f"Translate this {language} text to English. Return ONLY the translation, nothing else.\n\nText: {text}")
    ])

    return {"translated_text": response.content.strip()}

def summarize_node(state: DocumentState) -> dict:
    """
    Node: Summarize the text (using translated version if available).
    Updates: summary, word_count
    """
    # Use translated text if we translated, otherwise use original
    text_to_summarize = state.get("translated_text") or state["input_text"]

    print(f"  [summarize_node] Summarizing text ({len(text_to_summarize)} chars)...")

    response = llm.invoke([
        HumanMessage(content=f"Summarize this text in 2 sentences:\n\n{text_to_summarize}")
    ])

    summary = response.content.strip()
    word_count = len(summary.split())

    return {
        "summary": summary,
        "word_count": word_count,
    }

print("  Nodes defined: detect_language_node, translate_node, summarize_node")

# =============================================================================
# SECTION 4: DEFINING THE ROUTING FUNCTION
# =============================================================================
# For conditional edges, you provide a function that receives the state
# and returns a string — the name of the next node to go to.

def route_by_language(state: DocumentState) -> str:
    """
    Routing function: decides which node to go to next based on state.
    Returns the NAME of the next node (must match add_node() names).
    """
    if state["needs_translation"]:
        print(f"  [ROUTER] Non-English detected → going to translate_node")
        return "translate"
    else:
        print(f"  [ROUTER] English detected → going directly to summarize")
        return "summarize"

# =============================================================================
# SECTION 5: BUILDING THE GRAPH
# =============================================================================
print("\n--- Section 5: Building the Graph ---")

# Step 1: Create the StateGraph with our state schema
graph_builder = StateGraph(DocumentState)

# Step 2: Add nodes (name → function)
graph_builder.add_node("detect_language", detect_language_node)
graph_builder.add_node("translate", translate_node)
graph_builder.add_node("summarize", summarize_node)

# Step 3: Add edges (from → to)
# Always start at START → first node
graph_builder.add_edge(START, "detect_language")

# Conditional edge: after detect_language, call route_by_language()
# The result ("translate" or "summarize") determines the next node
graph_builder.add_conditional_edges(
    "detect_language",          # From this node
    route_by_language,          # Call this function to decide next node
    {
        "translate": "translate",   # If function returns "translate" → go here
        "summarize": "summarize",   # If function returns "summarize" → go here
    }
)

# After translate → always go to summarize
graph_builder.add_edge("translate", "summarize")

# After summarize → always go to END
graph_builder.add_edge("summarize", END)

# Step 4: Compile the graph
graph = graph_builder.compile()

print("  Graph compiled successfully!")
print("  Flow: START → detect_language → (translate?) → summarize → END")

# =============================================================================
# SECTION 6: VISUALIZE THE GRAPH
# =============================================================================
print("\n--- Section 6: Graph Visualization ---")
print("\nMermaid diagram of the graph:")
print("(Paste this at https://mermaid.live to see a visual diagram)")
print()
try:
    print(graph.get_graph().draw_mermaid())
except Exception as e:
    print(f"Visualization: {e}")
    print("""
  Equivalent Mermaid diagram:
  graph TD
      __start__ --> detect_language
      detect_language -->|needs translation| translate
      detect_language -->|English| summarize
      translate --> summarize
      summarize --> __end__
  """)

# =============================================================================
# SECTION 7: RUNNING THE GRAPH
# =============================================================================
print("--- Section 7: Running the Graph ---")

# Test 1: English text (should skip translation)
print("\n[Test 1]: English Input (no translation needed)")
english_input = {
    "input_text": "LangChain is a framework for building applications powered by large language models. It provides tools for prompt management, chain composition, and agent construction.",
    "language": "",
    "needs_translation": False,
    "translated_text": "",
    "summary": "",
    "word_count": 0,
}

result1 = graph.invoke(english_input)
print(f"\nSummary     : {result1['summary']}")
print(f"Language    : {result1['language']}")
print(f"Translated  : {'Yes' if result1['translated_text'] else 'No'}")
print(f"Word Count  : {result1['word_count']}")

# Test 2: Non-English text (should trigger translation)
print("\n[Test 2]: Spanish Input (translation needed)")
spanish_input = {
    "input_text": "LangChain es un framework para construir aplicaciones con modelos de lenguaje. Permite crear cadenas de componentes y agentes inteligentes.",
    "language": "",
    "needs_translation": False,
    "translated_text": "",
    "summary": "",
    "word_count": 0,
}

result2 = graph.invoke(spanish_input)
print(f"\nOriginal Language: {result2['language']}")
print(f"Translated text  : {result2['translated_text'][:100]}...")
print(f"Summary          : {result2['summary']}")
print(f"Word Count       : {result2['word_count']}")

# =============================================================================
# SECTION 8: STREAMING GRAPH EXECUTION
# =============================================================================
print("\n--- Section 8: Streaming Graph Execution (Step by Step) ---")

print("\nStreaming graph execution — see each node's output:")

for event in graph.stream(english_input):
    for node_name, node_output in event.items():
        if node_name not in ["__start__", "__end__"]:
            print(f"\n  Node '{node_name}' output:")
            for key, value in node_output.items():
                if value:  # Only show non-empty values
                    print(f"    {key}: {str(value)[:80]}")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 28")
print("=" * 60)
print("""
  1. LangGraph StateGraph = a graph where nodes transform a shared state
  2. State is a TypedDict — defines all data flowing through the graph
  3. Nodes = Python functions that receive full state, return partial update
  4. add_edge(A, B) = always go from A to B
  5. add_conditional_edges(A, fn, {result: node}) = dynamic routing
  6. route_by_language() returns a string = name of next node
  7. graph.stream() lets you see each node's output as it executes
  8. draw_mermaid() generates a diagram — great for documentation

  Next up: 29_langgraph_stateful_agent.py
  Build a proper agent loop with ToolNode, MemorySaver, and streaming.
""")
