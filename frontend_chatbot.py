import uuid
import streamlit as st
from langchain_core.messages import (
    AIMessage, HumanMessage, ToolMessage
)

from langgraph_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
    generate_thread_title,
    clear_thread_history,
    fix_corrupted_thread,
    save_thread_title,
    retrieve_all_thread_titles,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="AI Agent Workspace",
    page_icon="⚡"
)

st.markdown("""
<style>
.stApp { background-color: #0d1117; color: #e6edf3; }
div[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}
.stButton>button {
    border-radius: 6px;
    font-weight: 500;
    transition: all 0.2s ease;
}
.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.sandbox-container {
    border: 1px solid #30363d;
    padding: 20px;
    border-radius: 8px;
    background-color: #161b22;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────
def generate_thread_id_str():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id_str()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    st.session_state["active_suggestions"] = []
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def extract_text_from_content(content) -> str:
    """
    Safely extract plain text from any content format.
    Handles str, list of dicts, list of objects.
    Never returns raw JSON or tool blocks.
    """
    if isinstance(content, str):
        text = content.strip()
        # Skip if it looks like JSON (starts with { or [)
        if text.startswith(('{', '[')):
            return ""
        return text
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type", "")
                # Skip tool_use, tool_result, and other non-text blocks
                if block_type in ("tool_use", "tool_result", "json"):
                    continue
                if (block_type == "text"
                        and block.get("text", "").strip()):
                    parts.append(block["text"].strip())
            elif hasattr(block, "type"):
                block_type = getattr(block, "type", "")
                # Skip tool_use, tool_result, and other non-text blocks
                if block_type in ("tool_use", "tool_result", "json"):
                    continue
                if (block_type == "text"
                        and hasattr(block, "text")
                        and block.text
                        and block.text.strip()):
                    parts.append(block.text.strip())
        result = " ".join(parts).strip()
        # Additional safety: skip if looks like JSON
        if result.startswith(('{', '[')):
            return ""
        return result
    return ""


def yield_text_from_chunk(content):
    """
    Generator — yields only text tokens from
    AIMessage content. Skips tool_use blocks.
    """
    if isinstance(content, str):
        if content.strip():
            yield content
        return
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type in ("tool_use", "tool_result"):
                    continue
                if block_type == "text":
                    text = block.get("text", "")
                    if text and text.strip():
                        yield text
            elif hasattr(block, "type"):
                if block.type in ("tool_use", "tool_result"):
                    continue
                if (block.type == "text"
                        and hasattr(block, "text")
                        and block.text
                        and block.text.strip()):
                    yield block.text


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id_str()
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()
if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}
if "active_suggestions" not in st.session_state:
    st.session_state["active_suggestions"] = []
if "titles" not in st.session_state:
    st.session_state["titles"] = retrieve_all_thread_titles()

thread_key = str(st.session_state["thread_id"])
if thread_key not in st.session_state["chat_threads"]:
    st.session_state["chat_threads"].append(thread_key)

thread_docs = st.session_state["ingested_docs"].setdefault(
    thread_key, []
)
threads = st.session_state["chat_threads"][::-1]


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ Agent Command Center")
    st.markdown(f"**Session:** `{thread_key[:8]}...`")

    if st.button(
        "➕ New Thread",
        use_container_width=True,
        type="primary"
    ):
        reset_chat()
        st.rerun()

    st.divider()
    st.subheader("💬 Past Threads")

    for t_id in threads:
        display_title = st.session_state["titles"].get(
            t_id, f"Thread {t_id[:6]}"
        )
        c1, c2 = st.columns([0.82, 0.18])
        with c1:
            btn_type = "primary" if t_id == thread_key \
                else "secondary"
            if st.button(
                display_title,
                key=f"sel-{t_id}",
                use_container_width=True,
                type=btn_type
            ):
                st.session_state["thread_id"] = t_id
                state = chatbot.get_state(
                    config={"configurable": {"thread_id": t_id}}
                )
                messages = state.values.get("messages", [])
                st.session_state["message_history"] = []
                for m in messages:
                    # Skip ToolMessage objects entirely
                    if isinstance(m, ToolMessage):
                        continue
                    # Only process HumanMessage and AIMessage
                    if m.content:
                        extracted = extract_text_from_content(
                            m.content
                        )
                        if extracted.strip():
                            st.session_state["message_history"].append(
                                {
                                    "role": "user"
                                        if isinstance(m, HumanMessage)
                                        else "assistant",
                                    "content": extracted
                                }
                            )
                st.session_state["active_suggestions"] = []
                st.rerun()
        with c2:
            if st.button(
                "🗑️",
                key=f"del-{t_id}",
                help="Delete this thread"
            ):
                clear_thread_history(t_id)
                if t_id in st.session_state["chat_threads"]:
                    st.session_state["chat_threads"].remove(t_id)
                if t_id in st.session_state["titles"]:
                    del st.session_state["titles"][t_id]
                if t_id == thread_key:
                    reset_chat()
                st.rerun()


# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
chat_pane, telemetry_pane = st.columns(
    [0.65, 0.35], gap="large"
)

# ── RIGHT PANE ──
with telemetry_pane:
    st.markdown(
        "<div class='sandbox-container'>",
        unsafe_allow_html=True
    )
    st.subheader("📊 Knowledge Base")
    st.caption(
        "Upload PDFs to ground the agent in your documents. "
        "Multiple PDFs stack into one vector store per thread."
    )

    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_pdf:
        already_done = any(
            d.get("filename") == uploaded_pdf.name
            for d in thread_docs
        )
        if not already_done:
            with st.status(
                "Indexing PDF into vector store...",
                expanded=True
            ) as status_box:
                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                )
                thread_docs.append(summary)
                status_box.update(
                    label="✅ PDF indexed successfully",
                    state="complete",
                    expanded=False
                )
            st.rerun()
        else:
            st.info(f"`{uploaded_pdf.name}` already indexed.")

    st.divider()
    st.caption("**Indexed Documents**")

    if thread_docs:
        ui_data = [
            {
                "Filename": d.get("filename"),
                "Chunks": d.get("chunks"),
                "Pages": d.get("documents")
            }
            for d in thread_docs
        ]
        st.data_editor(
            ui_data,
            use_container_width=True,
            disabled=True,
            key=f"df-{thread_key}"
        )
    else:
        st.info(
            "No PDFs uploaded yet. "
            "Agent will use web search and tools."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='sandbox-container'>",
        unsafe_allow_html=True
    )
    st.subheader("🛠️ Available Tools")
    st.caption("🔍 **Tavily** — real-time web search")
    st.caption("📈 **Stock Price** — Alpha Vantage API")
    st.caption("🧮 **Calculator** — arithmetic operations")
    st.caption(
        "📄 **PDF RAG** — document Q&A with page citations"
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ── LEFT PANE ──
with chat_pane:
    st.title("⚡ AI Agent Workspace")
    st.caption(
        "Claude Sonnet + LangGraph + Tavily + FAISS · "
        "LangSmith observability enabled"
    )
    st.divider()

    # Render chat history
    for message in st.session_state["message_history"]:
        with st.chat_message(message["role"]):
            content = message.get("content", "")
            if content and content.strip():
                st.markdown(content)
            if message["role"] == "assistant" and content:
                with st.popover("📋 Copy"):
                    st.code(content, language="markdown")

    # Suggestion pills
    if st.session_state["active_suggestions"]:
        st.markdown("💡 **Suggested follow-ups:**")
        cols = st.columns(
            len(st.session_state["active_suggestions"])
        )
        for i, suggestion in enumerate(
            st.session_state["active_suggestions"]
        ):
            if cols[i].button(
                suggestion,
                key=f"pill-{i}-{thread_key}",
                use_container_width=True
            ):
                st.session_state["pending_input"] = suggestion
                st.session_state["active_suggestions"] = []
                st.rerun()

    # Chat input
    user_input = st.chat_input(
        "Ask anything — search web, query PDFs, "
        "get stock prices, calculate..."
    )

    # Handle suggestion pill input
    if "pending_input" in st.session_state:
        user_input = st.session_state.pop("pending_input")

    # ─────────────────────────────────────────
    # PROCESS MESSAGE
    # ─────────────────────────────────────────
    if user_input:
        # Generate thread title on first message
        if thread_key not in st.session_state["titles"]:
            title = generate_thread_title(user_input)
            st.session_state["titles"][thread_key] = title
            save_thread_title(thread_key, title)

        st.session_state["message_history"].append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            status_container = {"widget": None}

            def stream_response():
                for chunk, _ in chatbot.stream(
                    {
                        "messages": [
                            HumanMessage(content=user_input)
                        ]
                    },
                    config={
                        "configurable": {
                            "thread_id": thread_key
                        }
                    },
                    stream_mode="messages",
                ):
                    # ── Tool status widget ──
                    if isinstance(chunk, ToolMessage):
                        tool_name = getattr(
                            chunk, "name", "tool"
                        )
                        if status_container["widget"] is None:
                            status_container["widget"] = (
                                st.status(
                                    f"⚡ Using `{tool_name}`...",
                                    expanded=True
                                )
                            )
                        else:
                            status_container["widget"].update(
                                label=(
                                    f"⚡ Using `{tool_name}`..."
                                ),
                                state="running"
                            )

                    # ── Stream text only ──
                    if isinstance(chunk, AIMessage):
                        yield from yield_text_from_chunk(
                            chunk.content
                        )

            try:
                ai_response = st.write_stream(
                    stream_response()
                )
            except Exception as e:
                error_str = str(e)
                # Handle corrupted thread state
                if (
                    "tool_use" in error_str
                    and "tool_result" in error_str
                ):
                    fix_corrupted_thread(thread_key)
                    st.warning(
                        "⚠️ Thread state was corrupted and "
                        "has been fixed automatically. "
                        "Please send your message again."
                    )
                    ai_response = ""
                else:
                    ai_response = f"Error: {error_str}"
                    st.error(ai_response)

            # Close tool status widget
            if status_container["widget"] is not None:
                status_container["widget"].update(
                    label="✅ Done",
                    state="complete",
                    expanded=False
                )

        # Save clean response to history
        if ai_response:
            final_text = (
                ai_response
                if isinstance(ai_response, str)
                else extract_text_from_content(ai_response)
            )
            if final_text and final_text.strip():
                st.session_state["message_history"].append(
                    {
                        "role": "assistant",
                        "content": final_text
                    }
                )

        st.session_state["active_suggestions"] = [
            "Summarise key points",
            "Explain in simple terms",
            "What are the next steps?"
        ]
        st.rerun()

    # ─────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────
    if st.session_state["message_history"]:
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            chat_md = "\n".join([
                f"### {m['role'].upper()}\n{m['content']}\n"
                for m in st.session_state["message_history"]
                if isinstance(m.get("content"), str)
                and m["content"].strip()
            ])
            st.download_button(
                "📥 Export Chat (.md)",
                data=chat_md,
                file_name=f"chat-{thread_key[:6]}.md",
                mime="text/markdown",
                use_container_width=True
            )
        with c2:
            if st.button(
                "🧼 Clear Chat",
                use_container_width=True
            ):
                st.session_state["message_history"] = []
                st.session_state["active_suggestions"] = []
                st.rerun()