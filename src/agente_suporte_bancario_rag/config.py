"""Configuração do agente de suporte bancário RAG."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Parâmetros de ambiente com valores default determinísticos."""

    app_name: str = "agente-suporte-bancario-rag"
    app_version: str = "0.1.0"
    chunk_size: int = 200
    chunk_overlap: int = 40
    embedding_dimension: int = 384
    top_k: int = 5
    similarity_threshold: float = 0.30
    pg_dsn: str | None = None

    @classmethod
    def from_env(cls) -> Settings:
        """Lê variáveis de ambiente, com fallbacks determinísticos."""

        return cls(
            chunk_size=int(os.getenv("AGENTE_CHUNK_SIZE", "200")),
            chunk_overlap=int(os.getenv("AGENTE_CHUNK_OVERLAP", "40")),
            embedding_dimension=int(os.getenv("AGENTE_EMBEDDING_DIM", "384")),
            top_k=int(os.getenv("AGENTE_TOP_K", "5")),
            similarity_threshold=float(os.getenv("AGENTE_SIM_THRESHOLD", "0.30")),
            pg_dsn=os.getenv("AGENTE_PG_DSN"),
        )
