"""Armazenamento vetorial compatível com pgvector (memória por padrão)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .embeddings import EmbeddingProvider, similaridade_coseno
from .models import Chunk


class VectorStore(Protocol):
    """Contrato de busca e armazenamento de chunks vetorizados."""

    def adicionar(self, chunks: list[Chunk]) -> None:
        """Armazena chunks com seus embeddings."""

    def buscar(self, consulta: str, provedor: EmbeddingProvider, top_k: int = 5) -> list[Chunk]:
        """Retorna os top_k chunks mais similares à consulta."""


@dataclass
class InMemoryVectorStore:
    """Implementação em memória do VectorStore (testes e execução offline)."""

    chunks: dict[str, Chunk] = field(default_factory=dict)

    def adicionar(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self.chunks[chunk.id] = chunk

    def buscar(self, consulta: str, provedor: EmbeddingProvider, top_k: int = 5) -> list[Chunk]:
        if not self.chunks:
            return []
        vetor_consulta = provedor.embed(consulta)
        pontuados = [
            (similaridade_coseno(vetor_consulta, chunk.embedding), chunk)
            for chunk in self.chunks.values()
            if chunk.embedding
        ]
        pontuados.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in pontuados[:top_k]]


class PgVectorStore:
    """Adaptador PostgreSQL/pgvector (opcional).

    Requer AGENTE_PG_DSN e o driver psycopg instalado. Não é usado pelos
    testes, que rodam sobre InMemoryVectorStore.
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def conectar(self):
        try:
            import psycopg  # type: ignore
        except ImportError as exc:
            raise RuntimeError("psycopg não instalado; não é possível usar PgVectorStore") from exc
        return psycopg.connect(self.dsn)
