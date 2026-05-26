from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

# -------------------------------------------------------------
# 1. CRITICAL INITIALIZATION: Force Environment Mapping First!
# -------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

# Explicitly push keys to os.environ to guarantee LangSmith hooks attach natively
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "LangGraph-Agent-Workspace")

# Now import the core framework modules safely
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings  
from langchain_anthropic import ChatAnthropic    
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

# -------------------
# 2. LLM + Embeddings
# -------------------
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# -------------------
# 3. PDF Vector Cache Store
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, list[dict]] = {}


def _get_retriever(thread_id: Optional[str]):
    if thread_id and str(thread_id) in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[str(thread_id)]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """Builds or appends onto an existing FAISS retriever index for a thread."""
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        key = str(thread_id)
        existing_retriever = _get_retriever(key)

        if existing_retriever is not None:
            existing_retriever.vectorstore.add_documents(chunks)
            vector_store = existing_retriever.vectorstore
        else:
            vector_store = FAISS.from_documents(chunks, embeddings)

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
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


# -------------------
# 4. Agent Tools (With Safe Direct API Key Bindings)
# -------------------
search_tool = TavilySearchResults(
    max_results=3,
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform a basic arithmetic operation on two numbers."""
    try:
        if operation == "add": result = first_num + second_num
        elif operation == "sub": result = first_num - second_num
        elif operation == "mul": result = first_num * second_num
        elif operation == "div":
            if second_num == 0: return {"error": "Division by zero"}
            result = first_num / second_num
        else: return {"error": f"Unsupported operation '{operation}'"}
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a symbol using Alpha Vantage."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    try:
        r = requests.get(url, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """Retrieve highly contextual chunks from all uploaded thread PDFs."""
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {"error": "No documents indexed yet. Ask user to upload a PDF."}

    result = retriever.invoke(query)
    context_data = [
        {"content": doc.page_content, "page": doc.metadata.get("page", 0) + 1}
        for doc in result
    ]
    return {"query": query, "references": context_data}


tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 5. Graph State Definition
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 6. Graph Nodes and Workflow compilation
# -------------------
def chat_node(state: ChatState, config=None):
    thread_id = config.get("configurable", {}).get("thread_id") if config else None
    system_message = SystemMessage(
        content=(
            "You are Claude, an elite, helpful multi-utility AI assistant built by Anthropic.\n"
            f"Active Thread Context ID: `{thread_id}`.\n\n"
            "OPERATING GUIDELINES:\n"
            "1. For questions matching uploaded files or context, execute `rag_tool`. "
            f"Always map argument `thread_id` to explicitly be '{thread_id}'.\n"
            "2. When writing responses based on data from `rag_tool`, always reference what page number "
            "the data came from using clean text citations (e.g., '[Page X]').\n"
            "3. If an operation requires calculation, search, or stock parsing, utilize tools automatically."
        )
    )
    messages = [system_message] + state["messages"]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
chatbot = graph.compile(checkpointer=checkpointer)


# -------------------
# 7. Database Workspace Storage Utilities
# -------------------
with sqlite3.connect(database="chatbot.db", check_same_thread=False) as setup_conn:
    setup_conn.execute("""
        CREATE TABLE IF NOT EXISTS thread_metadata (
            thread_id TEXT PRIMARY KEY,
            title TEXT
        )
    """)


def save_thread_title(thread_id: str, title: str):
    """Saves a thread title permanently to the SQLite database."""
    try:
        with sqlite3.connect(database="chatbot.db", check_same_thread=False) as c:
            c.execute(
                "INSERT OR REPLACE INTO thread_metadata (thread_id, title) VALUES (?, ?)",
                (str(thread_id), title)
            )
            c.commit()
    except Exception:
        pass


def retrieve_all_thread_titles() -> dict[str, str]:
    """Retrieves all permanently saved thread titles from the database."""
    titles = {}
    try:
        with sqlite3.connect(database="chatbot.db", check_same_thread=False) as c:
            cursor = c.cursor()
            cursor.execute("SELECT thread_id, title FROM thread_metadata")
            for row in cursor.fetchall():
                titles[row[0]] = row[1]
    except Exception:
        pass
    return titles


def generate_thread_title(first_message: str) -> str:
    """Uses Claude to summarize the initial request into a clear title."""
    try:
        response = llm.invoke(f"Summarize this request into exactly 3 to 4 words. Use no quotes or tracking dots:\n{first_message}")
        return response.content.strip()
    except Exception:
        return first_message[:25] + "..."


def retrieve_all_threads() -> list[str]:
    all_threads = set()
    try:
        for checkpoint in checkpointer.list(None):
            t_id = checkpoint.config.get("configurable", {}).get("thread_id")
            if t_id: all_threads.add(str(t_id))
    except Exception: pass
    return list(all_threads)


def thread_document_metadata(thread_id: str) -> list[dict]:
    return _THREAD_METADATA.get(str(thread_id), [])


def clear_thread_history(thread_id: str):
    """Deletes checkpoints and titles from disk completely."""
    try:
        with sqlite3.connect(database="chatbot.db", check_same_thread=False) as c:
            cursor = c.cursor()
            cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (str(thread_id),))
            cursor.execute("DELETE FROM thread_metadata WHERE thread_id = ?", (str(thread_id),))
            c.commit()
    except Exception:
        pass