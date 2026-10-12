"""
nodes.py
========
LangGraph node functions for the RAG interview preparation agent.

Each function in this module is a node in the agent state graph.
Nodes receive the current AgentState, perform their operation,
and return a dict of state fields to update.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages

from rag_agent.agent.prompts import (
    QUERY_REWRITE_PROMPT,
    QUESTION_GENERATION_PROMPT,
    SYSTEM_PROMPT,
)
from rag_agent.agent.state import AgentResponse, AgentState, RetrievedChunk
from rag_agent.config import LLMFactory, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


def _state_get(state, key, default=None):
    """Support both dict-backed LangGraph state and typed AgentState objects."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


# ---------------------------------------------------------------------------
# Node: Query Rewriter
# ---------------------------------------------------------------------------


def query_rewrite_node(state: AgentState) -> dict:
    """Rewrite the user's query to maximise retrieval effectiveness."""
    messages = _state_get(state, "messages", [])
    original_query = ""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            original_query = str(message.content)
            break

    if not original_query:
        original_query = _state_get(state, "original_query", "") or ""

    if not original_query:
        return {"original_query": "", "rewritten_query": ""}

    try:
        settings = get_settings()
        llm = LLMFactory(settings).create()
        rewrite_prompt = QUERY_REWRITE_PROMPT.format(original_query=original_query)
        result = llm.invoke([HumanMessage(content=rewrite_prompt)])
        rewritten = str(result.content).strip()
    except Exception:
        rewritten = original_query

    return {
        "original_query": original_query,
        "rewritten_query": rewritten or original_query,
    }


# ---------------------------------------------------------------------------
# Node: Retriever
# ---------------------------------------------------------------------------


def retrieval_node(state: AgentState) -> dict:
    """Retrieve relevant chunks from ChromaDB based on the rewritten query."""
    query_text = _state_get(state, "rewritten_query", "") or _state_get(state, "original_query", "") or ""
    if not query_text:
        return {"retrieved_chunks": [], "no_context_found": True}

    try:
        manager = VectorStoreManager()
        chunks = manager.query(
            query_text=query_text,
            topic_filter=_state_get(state, "topic_filter", None),
            difficulty_filter=_state_get(state, "difficulty_filter", None),
        )
    except Exception:
        chunks = []

    return {
        "retrieved_chunks": chunks,
        "no_context_found": not bool(chunks),
    }


# ---------------------------------------------------------------------------
# Node: Generator
# ---------------------------------------------------------------------------


def generation_node(state: AgentState) -> dict:
    """Generate the final response using retrieved chunks as context."""
    settings = get_settings()
    no_context_found = _state_get(state, "no_context_found", False)
    retrieved_chunks = _state_get(state, "retrieved_chunks", [])
    rewritten_query = _state_get(state, "rewritten_query", "")
    messages = _state_get(state, "messages", [])

    if no_context_found:
        no_context_message = (
            "I was unable to find relevant information in the corpus for your query. "
            "This may mean the topic is not yet covered in the study material, or "
            "your query may need to be rephrased. Please try a more specific "
            "deep learning topic such as 'LSTM forget gate' or 'CNN pooling layers'."
        )
        response = AgentResponse(
            answer=no_context_message,
            sources=[],
            confidence=0.0,
            no_context_found=True,
            rewritten_query=rewritten_query,
        )
        return {
            "final_response": response,
            "messages": [AIMessage(content=no_context_message)],
        }

    context_lines = []
    sources = []
    scores = []
    for chunk in retrieved_chunks:
        citation = f"[SOURCE: {chunk.metadata.topic} | {chunk.metadata.source}]"
        sources.append(citation)
        scores.append(chunk.score)
        context_lines.append(f"{citation}\n{chunk.chunk_text}\n")

    context_text = "\n".join(context_lines)
    avg_confidence = sum(scores) / len(scores) if scores else 0.0
    user_question = _state_get(state, "original_query", "") or rewritten_query

    try:
        llm = LLMFactory(settings).create()
        system_message = SystemMessage(content=SYSTEM_PROMPT)
        context_message = HumanMessage(
            content=(
                "Use the following context to answer the question. "
                "Cite every factual statement using the source format.\n\n"
                f"{context_text}\n\nQuestion: {user_question}"
            )
        )

        conversation = list(messages)
        if len(conversation) > 0:
            conversation = trim_messages(
                conversation,
                max_tokens=settings.max_context_tokens,
                strategy="last",
                token_counter=lambda msgs: sum(len(str(m.content).split()) for m in msgs),
            )

        payload = [system_message, context_message, *conversation, HumanMessage(content=user_question)]
        result = llm.invoke(payload)
        answer = str(result.content).strip()
    except Exception:
        answer = (
            "I found relevant corpus material for this topic. "
            "The model backend is not configured for live generation in this environment, "
            "so I can only return the retrieved source context and citations."
        )
        answer += "\n\n" + context_text

    response = AgentResponse(
        answer=answer,
        sources=sources,
        confidence=avg_confidence,
        no_context_found=False,
        rewritten_query=rewritten_query,
    )
    return {"final_response": response, "messages": [AIMessage(content=answer)]}


# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------


def should_retry_retrieval(state: AgentState) -> str:
    """Decide whether to continue to generation or stop after failed retrieval."""
    if _state_get(state, "no_context_found", False):
        return "end"
    return "generate"
