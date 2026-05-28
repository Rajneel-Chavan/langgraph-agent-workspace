# ⚡ AI Agent Workspace

> Enterprise-grade multi-agent AI workspace powered by **Claude Sonnet + LangGraph + Tavily + FAISS** — with persistent memory, multi-thread management, PDF RAG, real-time web search, and LangSmith observability.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-AI%20Agent%20Workspace-blue?style=for-the-badge)](https://langgraph-agent-workspace-svaappym3a6je9wczbhepmg.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Latest-orange?style=for-the-badge)](https://langchain.com)
[![Claude](https://img.shields.io/badge/Claude%20Sonnet-4.5-purple?style=for-the-badge)](https://anthropic.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red?style=for-the-badge)](https://streamlit.io)

---

## 🎯 Overview

AI Agent Workspace is a production-grade agentic AI system where Claude Sonnet orchestrates 4 real-world tools through a **LangGraph StateGraph** — deciding autonomously which tools to call, when to call them, and how to chain results into coherent responses.

Unlike simple chatbots, this system maintains **persistent conversation memory across sessions** using SQLite checkpointing, manages **isolated thread workspaces** with AI-generated titles, and supports **multi-PDF knowledge bases** per thread using FAISS vector stores.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│               Streamlit Frontend (Split Pane)            │
│   Chat Pane (65%)              Knowledge Base (35%)      │
│   Thread Management            PDF Upload + Indexing     │
│   Suggestion Pills             Tool Status Display       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              LangGraph StateGraph                        │
│                                                          │
│   START → chat_node ──→ tools_condition                  │
│               ↑               │                          │
│               └── tool_node ←─┘                          │
│                                                          │
│   SqliteSaver checkpointer (chatbot.db)                  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    4 Tools                               │
│                                                          │
│  🔍 Tavily Search    📈 Alpha Vantage Stock              │
│  🧮 Calculator       📄 PDF RAG (FAISS per thread)       │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Claude Sonnet 4.5 (Anthropic)               │
│         Temperature=0 for consistent responses           │
│         Tool binding via bind_tools()                    │
└─────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 🤖 Agentic Intelligence
- **LangGraph StateGraph** with conditional edges — Claude decides autonomously when to call tools
- **4 real-world tools**: Tavily web search, Alpha Vantage stock prices, calculator, PDF RAG
- **Live tool status** — shows which tool is running during agent execution
- **Corrupted state recovery** — auto-detects and fixes orphaned tool_use/tool_result pairs

### 🧵 Thread Management
- **Isolated thread workspaces** — each conversation is completely separate
- **AI-generated titles** — Claude summarizes first message into a 3-4 word thread title
- **Persistent SQLite memory** — thread history survives restarts via LangGraph checkpointing
- **Thread deletion** — permanently removes checkpoints and metadata

### 📚 PDF Knowledge Base
- **Multi-PDF stacking** — upload multiple PDFs per thread, all indexed in one FAISS store
- **Per-thread vector stores** — completely isolated between threads
- **Page citations** — RAG responses include [Page X] references
- **Chunk overlap** — 1000 char chunks with 200 char overlap for context continuity

### 🎨 UI/UX
- **Split pane layout** — chat (65%) + knowledge base/tools (35%)
- **GitHub dark theme** — #0d1117 background with #161b22 sidebar
- **Suggestion pills** — 3 follow-up prompts after each response
- **Copy popover** — one-click copy of any assistant response
- **Export chat** — download full conversation as markdown
- **LangSmith observability** — all chains traced automatically

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Anthropic Claude Sonnet 4.5 (temperature=0) |
| **Agent Framework** | LangGraph StateGraph + ToolNode + tools_condition |
| **Memory** | LangGraph SqliteSaver checkpointer (chatbot.db) |
| **Embeddings** | OpenAI text-embedding-3-small |
| **Vector Store** | FAISS (per-thread, multi-PDF stacking) |
| **Web Search** | Tavily Search API (max_results=3) |
| **Stock Data** | Alpha Vantage GLOBAL_QUOTE API |
| **PDF Parsing** | LangChain PyPDFLoader |
| **Text Splitting** | RecursiveCharacterTextSplitter (1000/200) |
| **Frontend** | Streamlit (wide layout, dark theme) |
| **Observability** | LangSmith tracing |
| **Deployment** | Streamlit Cloud |

---

## 📋 Tools Detail

| Tool | Description | API |
|---|---|---|
| `tavily_search` | Real-time web search for current events and facts | Tavily API |
| `get_stock_price` | Fetch live stock quote for any symbol (AAPL, RELIANCE.BSE etc) | Alpha Vantage |
| `calculator` | Arithmetic operations: add, sub, mul, div | Built-in |
| `rag_tool` | Semantic search over uploaded PDFs with page citations | FAISS + OpenAI |

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.10+
Anthropic API Key
OpenAI API Key
Tavily API Key
LangSmith API Key (optional, for observability)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Rajneel-Chavan/langgraph-agent-workspace.git
cd langgraph-agent-workspace

# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create `.env` file:
```env
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=LangGraph-Agent-Workspace
```

### Run

```bash
streamlit run frontend_chatbot.py
```

---

## 🌐 Deployment

**Live App:**
```
https://langgraph-agent-workspace-svaappym3a6je9wczbhepmg.streamlit.app/
```

For Streamlit Cloud deployment, add all API keys under **Settings → Secrets**:
```toml
ANTHROPIC_API_KEY = "your_key"
OPENAI_API_KEY = "your_key"
TAVILY_API_KEY = "your_key"
LANGCHAIN_API_KEY = "your_key"
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_PROJECT = "LangGraph-Agent-Workspace"
```

---

## 💬 Example Interactions

```
User: How much did Vaibhav Suryavanshi score yesterday?
→ Agent uses tavily_search
→ Returns real-time match result with stats

User: What is AAPL stock price today?
→ Agent uses get_stock_price("AAPL")
→ Returns live Alpha Vantage quote

User: What is 15% of 85000?
→ Agent uses calculator(85000, 0.15, "mul")
→ Returns 12750.0

User: [Upload PDF] What does page 34 say about MCP?
→ Agent uses rag_tool with thread_id
→ Returns relevant chunks with [Page 34] citation
```

---

## 📁 Project Structure

```
langgraph-agent-workspace/
├── langgraph_backend.py    # LangGraph graph, tools, utilities
├── frontend_chatbot.py     # Streamlit UI
├── requirements.txt        # Dependencies
├── .env                    # API keys (not in repo)
├── chatbot.db              # SQLite checkpoints (auto-created)
└── README.md
```

---

## 🔑 Key Technical Decisions

**Why LangGraph over simple tool calling?**
LangGraph's StateGraph enables a proper agent loop — the LLM decides whether to call tools, which tools, and loops back until it has a final answer. Simple bind_tools() without a graph doesn't support multi-step tool chaining.

**Why per-thread FAISS stores?**
Each thread represents an isolated workspace. A user uploading a financial report in Thread A shouldn't pollute Thread B's context. Per-thread stores with add_documents() for multi-PDF stacking gives both isolation and flexibility.

**Why SqliteSaver over in-memory checkpointing?**
Production apps need memory that survives restarts. SqliteSaver persists the full message graph state to disk, enabling thread history recovery after server restarts — essential for Streamlit Cloud which spins down on inactivity.

**Why corrupted state detection?**
When Streamlit reruns mid-stream (which happens frequently), LangGraph can save a tool_use message before the tool_result arrives. The next request then fails with Anthropic's 400 error. The chat_node cleans orphaned tool_use messages before invoking the LLM.

---

## 🤝 Connect

**Rajneel Chavan**
- GitHub: [@Rajneel-Chavan](https://github.com/Rajneel-Chavan)
- LinkedIn: [linkedin.com/in/rajneelchavan](https://linkedin.com/in/rajneelchavan)
- Email: rajneelchavan16@gmail.com

---

## 📄 License

MIT License

---

*Built as part of an AI Engineering learning journey — exploring production-grade Agentic AI with LangGraph, Claude, and real-world tool integration.*
