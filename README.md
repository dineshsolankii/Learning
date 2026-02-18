# LangChain Mastery — From Zero to Production

A complete, story-driven LangChain curriculum. 34 files. 9 parts. Beginner to Expert.

Each file reads like a lesson — with a narrative, worked code, and clear connections to what came before and what comes next.

---

## What This Is

This is not a collection of copy-paste snippets. It is a **structured learning path** through the entire LangChain ecosystem, built so that every concept builds naturally on the last.

You will go from:

```python
llm.invoke("Hello")  # Chapter 0
```

To building:

```python
# Chapter 30 — Multi-agent research pipeline
multi_agent_graph.invoke({"topic": "Impact of LLMs on Software"})
```

Every concept is explained **inside the code file itself**, with comments written like a teacher talking to you — not a documentation page.

---

## Prerequisites

- Python 3.10+
- An [OpenRouter](https://openrouter.ai) API key (free tier available)
- Basic Python knowledge (functions, classes, loops)

---

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd Langchain

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
echo "OPENROUTER_API_KEY=your_key_here" > .env

# 5. Run your first file
python Langchain/00_why_langchain.py
```

---

## Full Curriculum

### Part 1 — Foundations

> Build the mental model first. Understand what LangChain is for and how its pieces connect.

| File | Topic | Level | Key Concepts |
|------|-------|-------|--------------|
| [00_why_langchain.py](Langchain/00_why_langchain.py) | Why LangChain? | Beginner | `llm.invoke()`, `AIMessage`, model switching, raw API vs LangChain |
| [01_llm_and_chat_models.py](Langchain/01_llm_and_chat_models.py) | LLMs and Chat Models | Beginner | `HumanMessage`, `SystemMessage`, temperature, `batch()`, `stream()` |
| [02_prompt_templates.py](Langchain/02_prompt_templates.py) | Prompt Templates | Beginner | `PromptTemplate`, `ChatPromptTemplate`, `MessagesPlaceholder`, partial templates |
| [03_lcel_first_chain.py](Langchain/03_lcel_first_chain.py) | Your First LCEL Chain | Beginner | `\|` pipe operator, `prompt \| llm \| parser`, `get_openai_callback()` |
| [04_output_parsers.py](Langchain/04_output_parsers.py) | Output Parsers | Beginner | `StrOutputParser`, `JsonOutputParser`, `PydanticOutputParser`, `CommaSeparatedListOutputParser` |
| [05_few_shot_prompting.py](Langchain/05_few_shot_prompting.py) | Few-Shot Prompting | Beginner | `FewShotChatMessagePromptTemplate`, `SemanticSimilarityExampleSelector`, FAISS |

---

### Part 2 — LCEL Deep Dive

> LangChain Expression Language is the backbone of everything. Master it completely.

| File | Topic | Level | Key Concepts |
|------|-------|-------|--------------|
| [06_lcel_advanced_chains.py](Langchain/06_lcel_advanced_chains.py) | Advanced Chains | Intermediate | Sequential chains, `RunnablePassthrough`, `RunnableLambda`, `.batch()` |
| [07_lcel_parallel_branching.py](Langchain/07_lcel_parallel_branching.py) | Parallel & Branching | Intermediate | `RunnableParallel`, `RunnableBranch`, fan-out, conditional routing |
| [08_lcel_runnables.py](Langchain/08_lcel_runnables.py) | Runnables Deep Dive | Intermediate | `.bind()`, `.with_retry()`, `.with_fallbacks()`, `RunnableConfig` |
| [09_output_parsers_advanced.py](Langchain/09_output_parsers_advanced.py) | Advanced Parsers | Intermediate | `OutputFixingParser`, streaming JSON, custom `BaseOutputParser` |
| [10_structured_output.py](Langchain/10_structured_output.py) | Structured Output | Intermediate | `llm.with_structured_output()`, Pydantic, TypedDict, JSON Schema, `include_raw=True` |

---

### Part 3 — Memory

> Let your app remember what was said. Without memory, every conversation starts from zero.

| File | Topic | Level | Key Concepts |
|------|-------|-------|--------------|
| [11_memory_in_memory.py](Langchain/11_memory_in_memory.py) | In-Memory History | Intermediate | `InMemoryChatMessageHistory`, `RunnableWithMessageHistory`, session isolation, `trim_messages()` |
| [12_memory_persistent.py](Langchain/12_memory_persistent.py) | Persistent Memory | Intermediate | `FileChatMessageHistory`, JSON file storage, summarization strategy |

---

### Part 4 — Documents & Embeddings

> Before you can build RAG, you need to understand how documents become vectors.

| File | Topic | Level | Key Concepts |
|------|-------|-------|--------------|
| [13_document_loaders.py](Langchain/13_document_loaders.py) | Document Loaders | Intermediate | `TextLoader`, `CSVLoader`, `JSONLoader`, `WebBaseLoader`, `DirectoryLoader`, `Document` object |
| [14_text_splitters.py](Langchain/14_text_splitters.py) | Text Splitting | Intermediate | `RecursiveCharacterTextSplitter`, chunk overlap, `MarkdownHeaderTextSplitter` |
| [15_embeddings.py](Langchain/15_embeddings.py) | Embeddings | Intermediate | `OpenAIEmbeddings`, `embed_query()`, cosine similarity, semantic distance demo |
| [16_vector_stores.py](Langchain/16_vector_stores.py) | Vector Stores | Intermediate | `FAISS`, `similarity_search()`, `save_local()`, `as_retriever()`, MMR, Chroma |

---

### Part 5 — RAG (Retrieval-Augmented Generation)

> The most impactful pattern in LLM applications. Answers grounded in your own data.

| File | Topic | Level | Key Concepts |
|------|-------|-------|--------------|
| [17_rag_basic.py](Langchain/17_rag_basic.py) | Basic RAG | Intermediate | Full 8-step pipeline, intermediate state printing, "I don't know" fallback |
| [18_rag_document_loaders.py](Langchain/18_rag_document_loaders.py) | Multi-Source RAG | Intermediate | Multiple loaders, source metadata, `load_any_file()` auto-detect |
| [19_rag_advanced_retrievers.py](Langchain/19_rag_advanced_retrievers.py) | Advanced Retrievers | Advanced | MMR retriever, `MultiQueryRetriever`, `ContextualCompressionRetriever`, score threshold |
| [20_rag_conversational.py](Langchain/20_rag_conversational.py) | Conversational RAG | Advanced | `create_history_aware_retriever()`, `create_retrieval_chain()`, follow-up questions |

---

### Part 6 — Tools & Agents

> Give the LLM hands. It can now search the web, run code, query databases.

| File | Topic | Level | Key Concepts |
|------|-------|-------|--------------|
| [21_tools_builtin.py](Langchain/21_tools_builtin.py) | Built-in Tools | Advanced | `@tool` decorator, tool introspection, Pydantic `args_schema`, `DuckDuckGoSearchRun` |
| [22_tools_custom.py](Langchain/22_tools_custom.py) | Custom Tools | Advanced | CRM lookup, safe calculator, date tools, `ToolException`, mock email sender |
| [23_agents_react.py](Langchain/23_agents_react.py) | ReAct Agents | Advanced | `create_react_agent`, `AgentExecutor`, Thought/Action/Observation loop, `verbose=True` |
| [24_agents_openai_functions.py](Langchain/24_agents_openai_functions.py) | Tool-Calling Agents | Advanced | `create_tool_calling_agent`, streaming agent steps, ReAct vs tool-calling comparison |
| [25_agents_multi_tool.py](Langchain/25_agents_multi_tool.py) | Multi-Tool Agent | Advanced | 5-tool customer support agent, `RunnableWithMessageHistory` + `AgentExecutor` |

---

### Part 7 — Callbacks & Streaming

> Control every token as it flows. Build real-time UIs. Monitor everything.

| File | Topic | Level | Key Concepts |
|------|-------|-------|--------------|
| [26_callbacks_and_streaming.py](Langchain/26_callbacks_and_streaming.py) | Callbacks | Advanced | `BaseCallbackHandler`, timing/cost callback, `on_llm_start`, `get_openai_callback()` |
| [27_streaming_advanced.py](Langchain/27_streaming_advanced.py) | Advanced Streaming | Advanced | `astream()`, `astream_events(version="v1")`, `asyncio.gather()`, FastAPI pattern |

---

### Part 8 — LangGraph

> Escape the black box. Build agents as visible, inspectable, resumable graphs.

| File | Topic | Level | Key Concepts |
|------|-------|-------|--------------|
| [28_langgraph_intro.py](Langchain/28_langgraph_intro.py) | Graph Basics | Advanced | `StateGraph`, `TypedDict` state, nodes, edges, `add_conditional_edges()`, Mermaid visualization |
| [29_langgraph_stateful_agent.py](Langchain/29_langgraph_stateful_agent.py) | Stateful Agent | Advanced | `MessagesState`, `ToolNode`, `tools_condition`, `MemorySaver`, `thread_id`, streaming updates |
| [30_langgraph_multi_agent.py](Langchain/30_langgraph_multi_agent.py) | Multi-Agent System | Expert | Supervisor pattern, shared state, Researcher/Writer/Critic agents, revision cycles |

---

### Part 9 — Production

> Ship it. Make it fast, reliable, observable, and testable.

| File | Topic | Level | Key Concepts |
|------|-------|-------|--------------|
| [31_async_and_batching.py](Langchain/31_async_and_batching.py) | Async & Batching | Expert | `chain.batch()`, `asyncio.gather()`, `Semaphore`, sequential vs parallel timing |
| [32_caching_and_optimization.py](Langchain/32_caching_and_optimization.py) | Caching | Expert | `InMemoryCache`, `SQLiteCache`, `CacheBackedEmbeddings`, `LocalFileStore`, token optimization |
| [33_production_patterns.py](Langchain/33_production_patterns.py) | Production Patterns | Expert | `pydantic_settings`, structured logging, input validation, `LangChainApp` class, pytest |

---

## Architecture Diagram

```
                        LANGCHAIN ECOSYSTEM
    ┌───────────────────────────────────────────────────────────┐
    │                                                           │
    │  Input         Prompt Template         LLM               │
    │  ──────  ───►  ───────────────  ───►  ──────────  ───►   │
    │  str/dict      ChatPromptTemplate      ChatOpenAI         │
    │                                                           │
    │                                        │                  │
    │                                        ▼                  │
    │                                  Output Parser            │
    │                                  StrOutputParser          │
    │                                  PydanticOutputParser     │
    │                                                           │
    │  MEMORY: RunnableWithMessageHistory wraps the chain       │
    │  RAG:    Retriever injects context before LLM             │
    │  AGENTS: LLM decides which Tool to call next             │
    │  GRAPH:  StateGraph connects nodes with conditional edges  │
    └───────────────────────────────────────────────────────────┘

    LCEL Chain: prompt | llm | output_parser
    RAG Chain:  retriever | prompt | llm | parser
    Agent Loop: llm → tool_calls? → tools → llm → ... → answer
    LangGraph:  START → node_A → node_B → (condition?) → END
```

---

## How to Run

Each file is standalone. Run any file directly:

```bash
# Start from the beginning
python Langchain/00_why_langchain.py

# Jump to any topic
python Langchain/17_rag_basic.py
python Langchain/29_langgraph_stateful_agent.py
python Langchain/33_production_patterns.py
```

Files 09, 27, 29 have interactive chat loops — type `exit` to stop.

---

## Key Concepts Quick Reference

| Concept | File | One-Line Description |
|---------|------|----------------------|
| LCEL | 03, 06-09 | Chain runnables with `\|` operator |
| RAG | 17-20 | Retrieve relevant docs, then generate with them as context |
| ReAct | 23 | Thought → Action → Observation loop until final answer |
| LangGraph | 28-30 | Define AI workflows as a graph with nodes and edges |
| MessagesState | 29 | Pre-built state that auto-appends messages |
| MemorySaver | 29 | Checkpointer — saves state after every node execution |
| ToolNode | 29 | Pre-built LangGraph node that executes LLM tool calls |
| tools_condition | 29 | Routes to tools if tool_calls present, else END |
| CacheBackedEmbeddings | 32 | Never embed the same text twice |
| with_structured_output | 10 | Schema-enforced JSON output from LLM |

---

## Dependencies

```
langchain              # Core framework
langchain-core         # Base abstractions (Runnables, Messages, etc.)
langchain-openai       # ChatOpenAI, OpenAIEmbeddings
langchain-community    # Community integrations (FAISS, loaders, caches)
openai                 # Underlying OpenAI client
python-dotenv          # Load .env files
faiss-cpu              # Local vector store (no server needed)
langgraph              # Graph-based agent framework
numpy                  # Cosine similarity calculations
pypdf                  # PDF document loading
pydantic-settings      # Config management from env vars
pytest                 # Testing framework
wikipedia              # Wikipedia search tool integration
duckduckgo-search      # Web search tool integration
```

Full list in [requirements.txt](requirements.txt).

---

## Project Structure

```
Langchain/
├── Langchain/
│   ├── data/
│   │   ├── knowledge.txt          # Sample knowledge base for RAG demos
│   │   ├── sample.csv             # Sample CSV for loader demos
│   │   └── sample.json            # Sample JSON for loader demos
│   ├── 00_why_langchain.py        # Part 1 — Foundations
│   ├── 01_llm_and_chat_models.py
│   ├── 02_prompt_templates.py
│   ├── 03_lcel_first_chain.py
│   ├── 04_output_parsers.py
│   ├── 05_few_shot_prompting.py
│   ├── 06_lcel_advanced_chains.py # Part 2 — LCEL Deep Dive
│   ├── 07_lcel_parallel_branching.py
│   ├── 08_lcel_runnables.py
│   ├── 09_output_parsers_advanced.py
│   ├── 10_structured_output.py
│   ├── 11_memory_in_memory.py     # Part 3 — Memory
│   ├── 12_memory_persistent.py
│   ├── 13_document_loaders.py     # Part 4 — Documents & Embeddings
│   ├── 14_text_splitters.py
│   ├── 15_embeddings.py
│   ├── 16_vector_stores.py
│   ├── 17_rag_basic.py            # Part 5 — RAG
│   ├── 18_rag_document_loaders.py
│   ├── 19_rag_advanced_retrievers.py
│   ├── 20_rag_conversational.py
│   ├── 21_tools_builtin.py        # Part 6 — Tools & Agents
│   ├── 22_tools_custom.py
│   ├── 23_agents_react.py
│   ├── 24_agents_openai_functions.py
│   ├── 25_agents_multi_tool.py
│   ├── 26_callbacks_and_streaming.py  # Part 7 — Callbacks & Streaming
│   ├── 27_streaming_advanced.py
│   ├── 28_langgraph_intro.py      # Part 8 — LangGraph
│   ├── 29_langgraph_stateful_agent.py
│   ├── 30_langgraph_multi_agent.py
│   ├── 31_async_and_batching.py   # Part 9 — Production
│   ├── 32_caching_and_optimization.py
│   └── 33_production_patterns.py
├── requirements.txt
└── README.md
```

---

## Author

Built as a comprehensive LangChain learning resource — from first LLM call to production-ready multi-agent systems.
