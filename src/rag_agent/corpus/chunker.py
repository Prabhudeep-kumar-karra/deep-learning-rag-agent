"""
chunker.py
==========
Document loading and chunking pipeline.

Handles ingestion of raw files (PDF and Markdown) into structured
DocumentChunk objects ready for embedding and vector store storage.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


class DocumentChunker:
    """
    Loads raw documents and splits them into DocumentChunk objects.

    Supports PDF and Markdown file formats. Chunking strategy uses
    recursive character splitting with configurable chunk size and
    overlap — both are interview-defensible parameters.
    """

    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -----------------------------------------------------------------------
    # Public Interface
    # -----------------------------------------------------------------------

    def chunk_file(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[DocumentChunk]:
        """
        Load a file and split it into DocumentChunks.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".md":
            raw_chunks = self._chunk_markdown(path, chunk_size, chunk_overlap)
        elif suffix == ".pdf":
            raw_chunks = self._chunk_pdf(path, chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        metadata = self._infer_metadata(path, metadata_overrides)

        chunks = []
        for raw in raw_chunks:
            text = raw["text"]
            chunk_id = VectorStoreManager.generate_chunk_id(path.name, text)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    metadata=metadata,
                )
            )

        logger.info(f"Chunked '{path.name}' into {len(chunks)} chunks.")
        return chunks

    def chunk_files(
        self,
        file_paths: list[Path],
        metadata_overrides: dict | None = None,
    ) -> list[DocumentChunk]:
        """
        Chunk multiple files in a single call.
        """
        all_chunks = []
        for fp in file_paths:
            try:
                chunks = self.chunk_file(fp, metadata_overrides=metadata_overrides)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to chunk file {fp}: {e}")
        return all_chunks

    # -----------------------------------------------------------------------
    # Format-Specific Loaders
    # -----------------------------------------------------------------------

    def _chunk_pdf(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a PDF file.
        """
        # PDF support is optional for this workshop demo
        raise NotImplementedError("PDF ingestion is not enabled for this workshop configuration.")

    def _chunk_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a Markdown file using header-aware splitting.
        """
        text = file_path.read_text(encoding="utf-8")

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        header_splits = markdown_splitter.split_text(text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        splits = text_splitter.split_documents(header_splits)

        raw_chunks = []
        for split in splits:
            raw_chunks.append({"text": split.page_content})
        return raw_chunks

    # -----------------------------------------------------------------------
    # Metadata Inference
    # -----------------------------------------------------------------------

    def _infer_metadata(
        self,
        file_path: Path,
        overrides: dict | None = None,
    ) -> ChunkMetadata:
        """
        Infer chunk metadata from filename conventions and apply overrides.
        Filename convention: <topic>_<difficulty>.md
        """
        stem = file_path.stem  # e.g., 'lstm_intermediate'
        parts = stem.split("_")

        topic = parts[0].capitalize() if len(parts) > 0 else "General"
        difficulty = parts[1].lower() if len(parts) > 1 else "beginner"

        bonus_topics = {"som", "boltzmannmachine", "gan"}
        is_bonus = topic.lower() in bonus_topics or stem.lower() in bonus_topics

        metadata = ChunkMetadata(
            source=file_path.name,
            topic=topic,
            difficulty=difficulty,
            is_bonus=is_bonus,
            type="note",
        )

        if overrides:
            for key, value in overrides.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)

        return metadata