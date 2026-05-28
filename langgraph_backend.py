from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

# ─────────────────────────────────────────────
# 1. ENV INIT — must happen before all imports
# ─────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# Load Streamlit Cloud secrets if available
try:
    import streamlit as st
    for key, val in st.secrets.items():
        os.environ[key] = str(val)
except Exception:
    pass

# Force LangSmith env vars
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv(
    "LANGCHAIN_TRACING_V2", "true"
)
os.environ["LANGCHAIN_API_KEY"] = os.getenv(
    "LANGCHAIN_API_KEY", ""
)
os.environ["LANGCHAIN_PROJECT"] = os.getenv(
    "LANGCHAIN_PROJECT", "LangGraph-Agent-Workspace"
)

# ─────────────────────────────────────────────
# 2. IMPORTS
# ─────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_tavily import TavilySearch
from langchain_community.vectorstores import FAISS
from langchain_core.messages import (
    BaseMessage, SystemMessage, ToolMessage, AIMessage
)
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

# ─────────────────────────────────────────────
# 3. LLM + EMBEDDINGS
# ─────────────────────────────────────────────
llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0,
    api_key=os.getenv("ANTHROPIC_API_KEY")
)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

# ─────────────────────────────────────────────
# 4. PDF VECTOR CACHE (per thread)
# ─────────────────────────────────────────────
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, list[dict]] = {}


def _get_retriever(thread_id: Optional[str]):
    if thread_id and str(thread_id) in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[str(thread_id)]
    return None


def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None
) -> dict:
    """Builds or appends onto existing FAISS index for a thread."""
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf"
    ) as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        key = str(thread_id)
        existing = _get_retriever(key)

        if existing is not None:
            existing.vectorstore.add_documents(chunks)
            vector_store = existing.vectorstore
        else:
            vector_store = FAISS.from_documents(chunks, embeddings)

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        _THREAD_RETRIEVERS[key] = retriever

        meta_entry = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        _THREAD_METADATA.setdefault(key, []).append(meta_entry)
        return meta_entry

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# ─────────────────────────────────────────────
# 5. TOOLS
# ─────────────────────────────────────────────
search_tool = TavilySearch(
    max_results=3,
    api_key=os.getenv("TAVILY_API_KEY")
)


@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str
) -> dict:
    """
    Perform basic arithmetic on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            return {"result": first_num + second_num}
        elif operation == "sub":
            return {"result": first_num - second_num}
        elif operation == "mul":
            return {"result": first_num * second_num}
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero"}
            return {"result": first_num / second_num}
        else:
            return {"error": f"Unsupported operation '{operation}'"}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol.
    Examples: AAPL, TSLA, RELIANCE.BSE
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}"
        "&apikey=C9PE94QUEW9VWGFM"
    )
    try:
        r = requests.get(url, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@tool
def rag_tool(
    query: str,
    thread_id: Optional[str] = None
) -> dict:
    """
    Retrieve relevant chunks from uploaded PDFs
    for this thread. Always pass the thread_id.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No documents indexed yet. "
                     "Ask user to upload a PDF first."
        }
    results = retriever.invoke(query)
    context_data = [
        {
            "content": doc.page_content,
            "page": doc.metadata.get("page", 0) + 1
        }
        for doc in results
    ]
    return {"query": query, "references": context_data}


tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)


# ─────────────────────────────────────────────
# 6. GRAPH STATE
# ─────────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ─────────────────────────────────────────────
# 7. CHAT NODE — with corrupted state cleanup
# ─────────────────────────────────────────────
def chat_node(state: ChatState, config=None):
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are Claude, an elite multi-utility AI assistant.\n"
            f"Active Thread ID: `{thread_id}`\n\n"
            "GUIDELINES:\n"
            "1. For questions about uploaded documents, "
            f"use `rag_tool` with thread_id='{thread_id}'.\n"
            "2. When citing from rag_tool, reference page "
            "numbers as [Page X].\n"
            "3. For calculations use calculator tool.\n"
            "4. For current news or real-time info "
            "use search_tool.\n"
            "5. For stock prices use get_stock_price tool.\n"
            "6. Be concise, accurate, and helpful.\n"
            "7. Always respond with clean plain text."
        )
    )

    # ── Clean corrupted message history ──
    # Remove orphaned tool_use messages that have
    # no corresponding tool_result after them
    raw_messages = list(state["messages"])
    cleaned = []
    i = 0
    while i < len(raw_messages):
        msg = raw_messages[i]

        # Detect if this AIMessage has tool_use blocks
        has_tool_use = False
        if hasattr(msg, "content"):
            content = msg.content
            if isinstance(content, list):
                has_tool_use = any(
                    (isinstance(b, dict)
                     and b.get("type") == "tool_use")
                    or (hasattr(b, "type")
                        and b.type == "tool_use")
                    for b in content
                )

        if has_tool_use:
            # Check if next message is a tool result
            has_result = False
            if i + 1 < len(raw_messages):
                next_msg = raw_messages[i + 1]
                # LangGraph ToolMessage
                if isinstance(next_msg, ToolMessage):
                    has_result = True
                # Content-based tool_result
                elif hasattr(next_msg, "content"):
                    nc = next_msg.content
                    if isinstance(nc, list):
                        has_result = any(
                            (isinstance(b, dict)
                             and b.get("type") == "tool_result")
                            or (hasattr(b, "type")
                                and b.type == "tool_result")
                            for b in nc
                        )

            if has_result:
                cleaned.append(msg)
            else:
                # Skip orphaned tool_use — prevents 400 error
                i += 1
                continue
        else:
            cleaned.append(msg)
        i += 1

    messages_to_send = [system_message] + cleaned
    response = llm_with_tools.invoke(
        messages_to_send, config=config
    )
    return {"messages": [response]}


tool_node = ToolNode(tools)

# ─────────────────────────────────────────────
# 8. CHECKPOINTER + GRAPH
# ─────────────────────────────────────────────
conn = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False
)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
chatbot = graph.compile(checkpointer=checkpointer)

# ─────────────────────────────────────────────
# 9. THREAD UTILITIES
# ─────────────────────────────────────────────
with sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False
) as setup_conn:
    setup_conn.execute("""
        CREATE TABLE IF NOT EXISTS thread_metadata (
            thread_id TEXT PRIMARY KEY,
            title TEXT
        )
    """)


def save_thread_title(thread_id: str, title: str):
    try:
        with sqlite3.connect(
            database="chatbot.db",
            check_same_thread=False
        ) as c:
            c.execute(
                "INSERT OR REPLACE INTO thread_metadata "
                "(thread_id, title) VALUES (?, ?)",
                (str(thread_id), title)
            )
            c.commit()
    except Exception:
        pass


def retrieve_all_thread_titles() -> dict[str, str]:
    titles = {}
    try:
        with sqlite3.connect(
            database="chatbot.db",
            check_same_thread=False
        ) as c:
            cursor = c.cursor()
            cursor.execute(
                "SELECT thread_id, title FROM thread_metadata"
            )
            for row in cursor.fetchall():
                titles[row[0]] = row[1]
    except Exception:
        pass
    return titles


def generate_thread_title(first_message: str) -> str:
    try:
        response = llm.invoke(
            f"Summarize this request into exactly 3 to 4 words. "
            f"No quotes or punctuation:\n{first_message}"
        )
        return response.content.strip()
    except Exception:
        return first_message[:25] + "..."


def retrieve_all_threads() -> list[str]:
    all_threads = set()
    try:
        for checkpoint in checkpointer.list(None):
            t_id = checkpoint.config.get(
                "configurable", {}
            ).get("thread_id")
            if t_id:
                all_threads.add(str(t_id))
    except Exception:
        pass
    return list(all_threads)


def thread_document_metadata(thread_id: str) -> list[dict]:
    return _THREAD_METADATA.get(str(thread_id), [])


def clear_thread_history(thread_id: str):
    try:
        with sqlite3.connect(
            database="chatbot.db",
            check_same_thread=False
        ) as c:
            cursor = c.cursor()
            try:
                cursor.execute(
                    "DELETE FROM checkpoints "
                    "WHERE thread_id = ?",
                    (str(thread_id),)
                )
            except Exception:
                pass
            try:
                cursor.execute(
                    "DELETE FROM checkpoint_writes "
                    "WHERE thread_id = ?",
                    (str(thread_id),)
                )
            except Exception:
                pass
            cursor.execute(
                "DELETE FROM thread_metadata "
                "WHERE thread_id = ?",
                (str(thread_id),)
            )
            c.commit()
    except Exception:
        pass
    _THREAD_RETRIEVERS.pop(str(thread_id), None)
    _THREAD_METADATA.pop(str(thread_id), None)


def fix_corrupted_thread(thread_id: str):
    """
    Fix corrupted thread state by clearing checkpoints.
    Call when tool_use/tool_result mismatch error occurs.
    """
    try:
        with sqlite3.connect(
            database="chatbot.db",
            check_same_thread=False
        ) as c:
            cursor = c.cursor()
            try:
                cursor.execute(
                    "DELETE FROM checkpoints "
                    "WHERE thread_id = ?",
                    (str(thread_id),)
                )
            except Exception:
                pass
            try:
                cursor.execute(
                    "DELETE FROM checkpoint_writes "
                    "WHERE thread_id = ?",
                    (str(thread_id),)
                )
            except Exception:
                pass
            c.commit()
    except Exception:
        pass
    _THREAD_RETRIEVERS.pop(str(thread_id), None)