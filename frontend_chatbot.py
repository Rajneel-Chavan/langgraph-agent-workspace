import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
    generate_thread_title,
    clear_thread_history,
    save_thread_title,
    retrieve_all_thread_titles
)

# Workspace Theme Injections & Minimal Clean Styling
st.set_page_config(layout="wide", page_title="AI Agent Intelligence Workspace", page_icon="⚡")
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    div[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    .stButton>button { border-radius: 6px; font-weight: 500; transition: all 0.2s ease; }
    .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
    .sandbox-container { border: 1px solid #30363d; padding: 20px; border-radius: 8px; background-color: #161b22; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

def generate_thread_id_str():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id_str()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)
    st.session_state["active_suggestions"] = []

# ======================= Session Initialization ===================
if "message_history" not in st.session_state: st.session_state["message_history"] = []
if "thread_id" not in st.session_state: st.session_state["thread_id"] = generate_thread_id_str()
if "chat_threads" not in st.session_state: st.session_state["chat_threads"] = retrieve_all_threads()
if "ingested_docs" not in st.session_state: st.session_state["ingested_docs"] = {}
if "active_suggestions" not in st.session_state: st.session_state["active_suggestions"] = []

# Persistent local lookup pull from database values
if "titles" not in st.session_state: 
    st.session_state["titles"] = retrieve_all_thread_titles()

thread_key = st.session_state["thread_id"]
if thread_key not in st.session_state["chat_threads"]: st.session_state["chat_threads"].append(thread_key)

thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, [])
threads = st.session_state["chat_threads"][::-1]

# ============================ Sidebar ============================
st.sidebar.title("⚡ Agent Command Center")
st.sidebar.markdown(f"**Active Session:** `{thread_key[:8]}...`")

if st.sidebar.button("➕ Open New Isolated Thread", use_container_width=True, type="primary"):
    reset_chat()
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("💬 Historic Workspaces")
for t_id in threads:
    display_title = st.session_state["titles"].get(t_id, f"Environment: {t_id[:6]}")
    c1, c2 = st.sidebar.columns([0.82, 0.18])
    with c1:
        btn_type = "primary" if t_id == thread_key else "secondary"
        if st.button(display_title, key=f"sel-{t_id}", use_container_width=True, type=btn_type):
            st.session_state["thread_id"] = t_id
            state = chatbot.get_state(config={"configurable": {"thread_id": t_id}})
            messages = state.values.get("messages", [])
            st.session_state["message_history"] = [{"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content} for m in messages if m.content]
            st.session_state["active_suggestions"] = []
            st.rerun()
    with c2:
        if st.button("🗑️", key=f"del-{t_id}", help="Purge thread from database"):
            clear_thread_history(t_id)
            if t_id in st.session_state["chat_threads"]: st.session_state["chat_threads"].remove(t_id)
            if t_id in st.session_state["titles"]: del st.session_state["titles"][t_id]
            if t_id == st.session_state["thread_id"]: reset_chat()
            st.rerun()

# ============================ Main Split Workspace Layout ========================
chat_pane, telemetry_pane = st.columns([0.65, 0.35], gap="large")

with telemetry_pane:
    st.markdown("<div class='sandbox-container'>", unsafe_allow_html=True)
    st.subheader("📊 Thread Knowledge Base")
    st.markdown("Ground your operational environment by uploading contextual document files.")
    
    uploaded_pdf = st.file_uploader("Drop context metrics documents here", type=["pdf"], label_visibility="collapsed")
    if uploaded_pdf:
        already_processed = any(d.get('filename') == uploaded_pdf.name for d in thread_docs)
        if not already_processed:
            with st.status("Gating context into Vector Stores...", expanded=True) as status_box:
                summary = ingest_pdf(uploaded_pdf.getvalue(), thread_id=thread_key, filename=uploaded_pdf.name)
                thread_docs.append(summary)
                status_box.update(label="✅ Memory Array Synced", state="complete")
            st.rerun()
        else:
            st.sidebar.warning("Document signature matches an active file index.")
            
    st.divider()
    st.caption("Active Embedded Memory References")
    if thread_docs:
        ui_dataframe = [{
            "Document Filename": d.get("filename"),
            "Vector Chunks": d.get("chunks"),
            "Total Pages": d.get("documents")
        } for d in thread_docs]
        st.data_editor(ui_dataframe, use_container_width=True, disabled=True, key=f"df-{thread_key}")
    else:
        st.info("No text matrices linked to this environment yet. System operating via standard web tools.")
    st.markdown("</div>", unsafe_allow_html=True)

with chat_pane:
    st.title("Enterprise Multi-Agent Workspace")
    
    for idx, message in enumerate(st.session_state["message_history"]):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                with st.popover("📋 Capture Snippet"):
                    st.caption("Click the icon in the code box corner to clone clean content.")
                    st.code(message["content"], language="markdown")

    if st.session_state["active_suggestions"]:
        st.markdown("💡 *Suggested Follow-ups:*")
        pill_cols = st.columns(len(st.session_state["active_suggestions"]))
        for i, option in enumerate(st.session_state["active_suggestions"]):
            if pill_cols[i].button(option, key=f"pill-{i}", use_container_width=True):
                st.session_state["active_submission"] = option
                st.rerun()

    input_value = st.chat_input("Prompt agent, interrogate document vectors, calculate equations...")
    if "active_submission" in st.session_state:
        input_value = st.session_state.pop("active_submission")

    if input_value:
        if thread_key not in st.session_state["titles"]:
            generated_title = generate_thread_title(input_value)
            st.session_state["titles"][thread_key] = generated_title
            save_thread_title(thread_key, generated_title)

        st.session_state["message_history"].append({"role": "user", "content": input_value})
        with st.chat_message("user"):
            st.markdown(input_value)

        with st.chat_message("assistant"):
            status_container = {"widget": None}

            def live_agent_stream():
                for chunk, _ in chatbot.stream(
                    {"messages": [HumanMessage(content=input_value)]},
                    config={"configurable": {"thread_id": thread_key}},
                    stream_mode="messages",
                ):
                    if isinstance(chunk, ToolMessage):
                        tool_label = getattr(chunk, "name", "agent_node")
                        if status_container["widget"] is None:
                            status_container["widget"] = st.status(f"⚡ Running Pipeline: `{tool_label}`", expanded=True)
                        else:
                            status_container["widget"].update(label=f"⏳ Processing Node Logic: `{tool_label}`")

                    # Structured object extractor logic optimized explicitly for Anthropic stream packets
                    if isinstance(chunk, AIMessage):
                        if isinstance(chunk.content, str) and chunk.content:
                            yield chunk.content
                        elif isinstance(chunk.content, list):
                            for block in chunk.content:
                                if isinstance(block, dict):
                                    if block.get("type") == "text" and "text" in block:
                                        yield block["text"]
                                elif hasattr(block, "text"):
                                    yield block.text

            ai_response = st.write_stream(live_agent_stream())
            if status_container["widget"] is not None:
                status_container["widget"].update(label="✨ Process Stream Settled", state="complete", expanded=False)

        st.session_state["message_history"].append({"role": "assistant", "content": ai_response})
        st.session_state["active_suggestions"] = ["Break down key metrics", "Draft complete summary", "Audit data sources"]
        st.rerun()

    if st.session_state["message_history"]:
        st.divider()
        footer_c1, footer_c2 = st.columns([0.5, 0.5])
        with footer_c1:
            chat_markdown_output = "\n".join([f"### {m['role'].upper()}\n{m['content']}\n" for m in st.session_state["message_history"]])
            st.download_button("📥 Export Workspace Artifact (.md)", data=chat_markdown_output, file_name=f"agent-export-{thread_key[:6]}.md", mime="text/markdown", use_container_width=True)
        with footer_c2:
            if st.button("🧼 Wipe Timeline Clear", use_container_width=True):
                st.session_state["message_history"] = []
                st.session_state["active_suggestions"] = []
                st.rerun()