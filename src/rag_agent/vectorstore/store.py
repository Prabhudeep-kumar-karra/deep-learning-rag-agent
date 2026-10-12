"""
store.py
========
ChromaDB vector store management.

Handles all interactions with the persistent ChromaDB collection:
initialisation, ingestion, duplicate detection, and retrieval.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """
    Manages the ChromaDB persistent vector store for the corpus.

    All corpus ingestion and retrieval operations pass through this class.
    It is the single point of contact between the application and ChromaDB.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        """
        Create or connect to the persistent ChromaDB client and collection.

        Creates the chroma_db_path directory if it does not exist.
        Uses PersistentClient so data survives between application restarts.
        """
        try:
            Path(self._settings.chroma_db_path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._settings.chroma_db_path)
            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"}  # cosine similarity
            )
            logger.info(
                f"Successfully initialized ChromaDB collection '{self._settings.chroma_collection_name}' "
                f"with {self._collection.count()} items at '{self._settings.chroma_db_path}'."
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise RuntimeError(f"ChromaDB initialization failed: {e}") from e

    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """
        Generate a deterministic chunk ID from source filename and content.
        """
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        """
        Check whether a chunk with this ID already exists in the collection.
        """
        result = self._collection.get(ids=[chunk_id])
        return len(result["ids"]) > 0

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """
        Embed and store a list of DocumentChunks in ChromaDB.
        """
        result = IngestionResult()
        for chunk in chunks:
            if self.check_duplicate(chunk.chunk_id):
                result.skipped += 1
                continue

            try:
                embedding = self._embeddings.embed_documents([chunk.chunk_text])[0]
                self._collection.upsert(
                    ids=[chunk.chunk_id],
                    embeddings=[embedding],
                    documents=[chunk.chunk_text],
                    metadatas=[chunk.metadata.to_dict()],
                )
                result.ingested += 1
            except Exception as e:
                logger.error(f"Error ingesting chunk {chunk.chunk_id}: {e}")
                result.errors.append(f"{chunk.chunk_id}: {e}")

        logger.info(
            f"Ingestion summary: {result.ingested} ingested, "
            f"{result.skipped} skipped, {len(result.errors)} errored."
        )
        return result

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.
        """
        k = k or self._settings.retrieval_k
        
        filters = []
        if topic_filter:
            filters.append({"topic": topic_filter})
        if difficulty_filter:
            filters.append({"difficulty": difficulty_filter})

        where_clause = None
        if len(filters) == 1:
            where_clause = filters[0]
        elif len(filters) > 1:
            where_clause = {"$and": filters}

        query_embedding = self._embeddings.embed_query(query_text)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        if results and results["ids"] and results["ids"][0]:
            for cid, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                score = 1 - dist  # Cosine distance to similarity score
                if score >= self._settings.similarity_threshold:
                    chunk_meta = ChunkMetadata.from_dict(meta)
                    retrieved.append(
                        RetrievedChunk(
                            chunk_id=cid,
                            chunk_text=doc,
                            metadata=chunk_meta,
                            score=score,
                        )
                    )

        retrieved.sort(key=lambda x: x.score, reverse=True)
        return retrieved

    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """
        Return a list of all unique source documents in the collection.
        """
        data = self._collection.get(include=["metadatas"])
        docs_map = {}
        if data and data["metadatas"]:
            for meta in data["metadatas"]:
                source = meta.get("source", "unknown")
                topic = meta.get("topic", "General")
                if source not in docs_map:
                    docs_map[source] = {"source": source, "topic": topic, "chunk_count": 0}
                docs_map[source]["chunk_count"] += 1
        return list(docs_map.values())

    def get_document_chunks(self, source: str) -> list[DocumentChunk]:
        """
        Retrieve all chunks belonging to a specific source document.
        """
        data = self._collection.get(where={"source": source}, include=["documents", "metadatas"])
        chunks = []
        if data and data["ids"]:
            for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
                chunk_meta = ChunkMetadata.from_dict(meta)
                chunks.append(
                    DocumentChunk(
                        chunk_id=cid,
                        chunk_text=doc,
                        metadata=chunk_meta,
                        source=source,
                    )
                )
        return chunks

    def get_collection_stats(self) -> dict:
        """
        Return summary statistics about the current collection.
        """
        docs = self.list_documents()
        total_chunks = self._collection.count()
        topics = list(set(d["topic"] for d in docs))
        sources = [d["source"] for d in docs]
        return {
            "total_chunks": total_chunks,
            "topics": topics,
            "sources": sources,
            "bonus_topics_present": False,
        }

    def delete_document(self, source: str) -> int:
        """
        Remove all chunks from a specific source document.
        """
        existing = self.get_document_chunks(source)
        count = len(existing)
        if count > 0:
            self._collection.delete(where={"source": source})
        return count