# ⚡ GraphWorkspace AI: Enterprise Multi-Agent Telemetry Workspace

GraphWorkspace AI is a production-grade, stateful multi-agent orchestrator built natively on top of **LangGraph**, **Anthropic Claude 3.5 Sonnet**, and **Streamlit**. It features real-time, animated graph node execution tracking, asynchronous live tool invocation cards, streaming document citations, and an integrated SQLite durable session engine paired with **LangSmith** cloud telemetry tracing.

The system is specifically engineered to handle complex, non-deterministic agentic loops (such as conditional routing between local RAG vectors and open-web search utilities) while maintaining strict thread safety and delivering an interactive, workspace UI experience.

---

## 🚀 Key Feature Deep-Dive

### 🌀 Animated Node Execution
The engine intercepts the background compilation streaming layers using LangGraph updates (`stream_mode="updates"`). As the engine hops boundaries from the main reasoning node (`chat_node`) into action blocks (`tools`), the UI injects targeted CSS keyframe layers to provide immediate visibility into the underlying state adjustments.

### 🛠️ Active Tool Execution Cards
When an agent determines a tool call is required, the layout creates isolated diagnostic visualization cards. These break down the exact parameters, string keys, and payload objects passed down by the LLM before execution begins, displaying explicit transition states from `Processing` to `Resolved`.

### 📄 Low-Latency Streaming Citations
When querying data structures extracted from your uploaded source documents, the streaming generation token processor isolates bracketed index footprints (e.g., `[Page 4]`). It formats these strings into isolated HTML blocks with clean hovering styles on the fly, avoiding any rendering latency or text jitter.

### 💾 Persistent Thread Checkpointing
Utilizes an absolute SQLite connection loop to checkpoint state snapshots (`SqliteSaver`). If a connection breaks or the application reboots, the entire multi-turn thread context, message array list, and variable maps are restored safely.

### 📊 Transparent LangSmith Integration
Exposes absolute telemetry variables into the base runtime, pushing full nested trace loops, tokens consumed, node routing weights, tool return codes, and latency profiles straight to your monitoring dashboard.

---

## 🛠️ Complete Technical Architecture

### System Execution Graph
The workflow engine operates via a stateful directed acyclic graph (DAG) configuration. The message array passes through conditional gates to determine whether the graph should terminate or execute external functional nodes:

```text
               ┌──────────────────────────────────────┐
               │     Streamlit Front-End Panel        │
               │        (frontend_chatbot.py)         │
               └──────────────────┬───────────────────┘
                                  │
                   (User Prompt / UUID Thread Context)
                                  ▼
               ┌──────────────────────────────────────┐
               │     LangGraph Compiled StateGraph    │
               │        (langgraph_backend.py)        │
               └──────────────────┬───────────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
┌─────────────────┐                               ┌─────────────────┐
│    chat_node    ├───────[Should Continue?]─────►│   tools_node    │
│ (Claude-Sonnet) │                               │ (FAISS / Tavily)│
└────────┬────────┘◄──────[Return Value Array]────┴────────┬────────┘
         │                                                 │
         └────────────────────────┬────────────────────────┘
                                  │
                    (Low-Latency Telemetry Hooks)
                                  ▼
               ┌──────────────────────────────────────┐
               │   LangSmith Cloud Trace Analytics    │
               └──────────────────────────────────────┘
