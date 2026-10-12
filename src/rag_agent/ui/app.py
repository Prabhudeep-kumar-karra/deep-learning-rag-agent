"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
- Left sidebar: Document ingestion and corpus browser
- Centre: Document viewer
- Right: Chat interface

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------

@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """Return the singleton VectorStoreManager."""
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_graph():
    """Return the compiled LangGraph agent."""
    return get_compiled_graph()


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------

def initialise_session_state() -> None:
    """Initialise all st.session_state keys on first run."""
    defaults = {
        "chat_history": [],
        "ingested_documents": [],
        "selected_document": None,
        "last_ingestion_result": None,
        "thread_id": "default-session",
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel (Sidebar)
# ---------------------------------------------------------------------------

def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """Render the document ingestion panel in the sidebar."""
    st.sidebar.header("📂 Corpus Ingestion")

    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["pdf", "md"],
        accept_multiple_files=True,
    )

    if st.sidebar.button("Ingest Documents", disabled=not uploaded_files):
        if uploaded_files:
            saved_paths = []
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    temp_path = Path(temp_dir) / uploaded_file.name
                    temp_path.write_bytes(uploaded_file.getvalue())
                    saved_paths.append(temp_path)

                try:
                    chunks = chunker.chunk_files(saved_paths)
                    result = store.ingest(chunks)
                    st.session_state["last_ingestion_result"] = result
                    st.sidebar.success(
                        f"Added {result.ingested} chunks, skipped {result.skipped} duplicates."
                    )
                except Exception as e:
                    st.sidebar.error(f"Ingestion failed: {e}")

    # Render ingested documents list below uploader
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Corpus Documents")
    docs = store.list_documents()
    st.session_state["ingested_documents"] = docs

    if not docs:
        st.sidebar.info("No documents ingested yet.")
    else:
        for doc in docs:
            col_info, col_del = st.sidebar.columns([4, 1])
            with col_info:
                st.text(f"{doc['source']} ({doc['chunk_count']} chunks)")
            with col_del:
                if st.button("🗑", key=f"del_{doc['source']}"):
                    store.delete_document(doc["source"])
                    st.rerun()


def render_corpus_stats(store: VectorStoreManager) -> None:
    """Render a compact corpus health summary in the sidebar."""
    stats = store.get_collection_stats()
    st.sidebar.markdown("---")
    st.sidebar.metric("Total Chunks", stats["total_chunks"])
    if stats["topics"]:
        st.sidebar.write("Topics:", ", ".join(stats["topics"]))
    if stats["bonus_topics_present"]:
        st.sidebar.success("✅ Bonus topics present")
    else:
        st.sidebar.warning("⚠️ No bonus topics yet")


# ---------------------------------------------------------------------------
# Document Viewer Panel (Centre)
# ---------------------------------------------------------------------------

def render_document_viewer(store: VectorStoreManager) -> None:
    """Render the document viewer in the main centre column."""
    st.subheader("📄 Document Viewer")

    docs = st.session_state.get("ingested_documents", [])
    if not docs:
        st.info("Ingest documents using the sidebar to view content here.")
        return

    doc_sources = [doc["source"] for doc in docs]
    selected = st.selectbox(
        "Select document to view",
        options=doc_sources,
        key="selected_document_selectbox",
    )

    if selected:
        st.session_state["selected_document"] = selected
        chunks = store.get_document_chunks(selected)

        st.markdown(f"**Showing {len(chunks)} chunks for `{selected}`**")
        with st.container(height=450):
            for idx, chunk in enumerate(chunks):
                meta = chunk.metadata
                st.markdown(
                    f"**Chunk {idx + 1}** | Topic: `{meta.topic}` | Difficulty: `{meta.difficulty}`"
                )
                st.info(chunk.chunk_text)
                st.markdown("---")


# ---------------------------------------------------------------------------
# Chat Interface Panel (Right)
# ---------------------------------------------------------------------------

def render_chat_interface(graph) -> None:
    """Render the chat interface in the right column."""
    st.subheader("💬 Interview Prep Chat")

    # Filters
    docs = st.session_state.get("ingested_documents", [])
    available_topics = list(set(d["topic"] for d in docs))

    col_topic, col_diff = st.columns(2)
    with col_topic:
        topic_filter = st.selectbox(
            "Filter Topic",
            options=["All"] + available_topics,
            key="chat_topic_filter",
        )
        st.session_state["topic_filter"] = None if topic_filter == "All" else topic_filter

    with col_diff:
        diff_filter = st.selectbox(
            "Filter Difficulty",
            options=["All", "beginner", "intermediate", "advanced"],
            key="chat_diff_filter",
        )
        st.session_state["difficulty_filter"] = None if diff_filter == "All" else diff_filter

    # Chat history display container
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("📎 Sources"):
                        for source in message["sources"]:
                            st.caption(source)
                if message.get("no_context_found"):
                    st.warning("⚠️ No relevant content found in corpus.")

    # Chat input
    if query := st.chat_input("Ask about a deep learning topic..."):
        # 1. Append user message
        st.session_state.chat_history.append({"role": "user", "content": query})

        # 2. Invoke LangGraph agent
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        inputs = {"messages": [HumanMessage(content=query)]}

        try:
            result = graph.invoke(inputs, config=config)

            if isinstance(result, dict):
                final_response = result.get("final_response")
                if final_response is not None and hasattr(final_response, "answer"):
                    answer = final_response.answer
                    sources = getattr(final_response, "sources", [])
                    no_context = getattr(final_response, "no_context_found", False)
                else:
                    messages = result.get("messages", [])
                    last_message = messages[-1] if messages else None
                    answer = getattr(last_message, "content", "No response generated.") if last_message else "No response generated."
                    sources = []
                    no_context = False
            else:
                answer = str(result)
                sources = []
                no_context = False

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "no_context_found": no_context,
                }
            )
        except Exception as e:
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": f"Error communicating with agent graph: {e}",
                    "sources": [],
                    "no_context_found": True,
                }
            )

        st.rerun()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

def main() -> None:
    """Application entry point."""
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption(
        "RAG-powered interview preparation — built with LangChain, LangGraph, and ChromaDB"
    )

    initialise_session_state()

    # Instantiate shared backend resources
    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    # Sidebar
    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    # Main content
    left_col, right_col = st.columns([1.3, 1.0])
    with left_col:
        render_document_viewer(store)
    with right_col:
        render_chat_interface(graph)


if __name__ == "__main__":
    main()