#app.py

import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from core.agent import build_graph, chat
from core.rag.rag import ingest, list_indexes
from core.memory.db import (
       init_db,
    create_conversation,
    get_conversations,
    load_messages,
    delete_conversation,
    search_conversations,
)

init_db()

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="FinCopilot",
    page_icon="💼",
    layout="wide"
)
st.markdown(
    """
    <style>
    /* Hide ONLY the dropdown chevron arrow in popover */
    button[aria-expanded="false"] > div:last-child,
    button[aria-expanded="true"] > div:last-child {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────
# Session Initialization
# ─────────────────────────────────────────────

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "chat_history" not in st.session_state:
    if "session_id" in st.session_state:
        stored = load_messages(st.session_state.session_id)
        st.session_state.chat_history = stored
    else:
        st.session_state.chat_history = []

if "session_id" not in st.session_state:
    new_session = create_conversation()
    st.session_state.session_id = new_session

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("FinCopilot")

    # 🔍 Search Chats
    search_query = st.text_input("Search chats", placeholder="Search...")

    # ➕ New Chat
    if st.button("➕ New Chat", use_container_width=True):
        new_session = create_conversation()
        st.session_state.session_id = new_session
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    # Load Conversations
    if search_query:
        conversations = search_conversations(search_query)
    else:
        conversations = get_conversations()

    for conv_id, title, _ in conversations:
        display_title = title if title else "Untitled Chat"

        is_active = conv_id == st.session_state.session_id

        container = st.container()

        with container:
            col1, col2 = st.columns([0.85, 0.15])

            # Chat Title Button
            with col1:
                if st.button(
                    display_title,
                    key=f"open_{conv_id}",
                    use_container_width=True,
                ):
                    st.session_state.session_id = conv_id
                    stored = load_messages(conv_id)
                    st.session_state.chat_history = stored
                    st.rerun()

            # 3-dot style menu
            with col2:
                with st.popover("⋮"):
                    if st.button("Delete Chat", key=f"delete_{conv_id}"):
                        delete_conversation(conv_id)

                        if st.session_state.session_id == conv_id:
                            new_session = create_conversation()
                            st.session_state.session_id = new_session
                            st.session_state.chat_history = []

                        st.rerun()

    st.divider()

    indexes = list_indexes()
    if indexes:
        st.subheader("Indexes")
        for idx in indexes:
            st.write(idx)


    st.divider()
    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "Drag & Drop or Browse Files",
        type=["pdf", "txt", "docx", "csv", "html"],
        accept_multiple_files=True
    )

    company = st.text_input("Company Name")
    doc_type = st.text_input("Document Type (e.g., 10-K)")
    year = st.text_input("Year")

    if st.button("Ingest Documents"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        elif not company or not doc_type or not year:
            st.warning("Please provide Company, Document Type, and Year.")
        else:
            file_paths = []
            os.makedirs("data/uploads", exist_ok=True)

            for uploaded_file in uploaded_files:
                file_path = os.path.join("data/uploads", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)

            metadata = {
                "company": company,
                "doc_type": doc_type,
                "year": year
            }

            result = ingest(file_paths, metadata)

            if result["status"] == "success":
                st.success(f"Index '{result['index_name']}' {result['action']}")
            else:
                st.error(f"Ingestion failed: {result['reason']}")

# ─────────────────────────────────────────────
# Main Chat Area
# ─────────────────────────────────────────────

st.title("💬 Chat")

# Display chat history
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.text(message)   # IMPORTANT: use text, not markdown

# Chat input
user_input = st.chat_input("Ask about your documents...")

if user_input:

    # Save user message
    st.session_state.chat_history.append(("user", user_input))

    with st.chat_message("user"):
        st.text(user_input)

    # Get response (no streaming)
    response = chat(
        user_input,
        st.session_state.session_id,
        st.session_state.graph
    )

    # Save assistant message
    st.session_state.chat_history.append(("assistant", response))

    with st.chat_message("assistant"):
        st.text(response)