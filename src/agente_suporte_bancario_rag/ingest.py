"""Ingestão de documentos bancários: leitura, chunking e indexação."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .embeddings import EmbeddingProvider
from .models import Chunk
from .vector_store import VectorStore


@dataclass(frozen=True)
class Documento:
    """Documento bruto a ser ingerido."""

    id: str
    texto: str
    fonte: str


def particionar(texto: str, tamanho: int = 200, sobreposicao: int = 40) -> list[str]:
    """Particiona texto em chunks de número fixo de palavras.

    A partição é determinística: mesma entrada produz sempre os mesmos chunks.
    """

    if tamanho <= 0:
        raise ValueError("tamanho do chunk deve ser positivo")
    if sobreposicao < 0 or sobreposicao >= tamanho:
        raise ValueError("sobreposição deve ser >= 0 e < tamanho")

    palavras = texto.split()
    if not palavras:
        return []

    partes: list[str] = []
    inicio = 0
    while inicio < len(palavras):
        fim = min(inicio + tamanho, len(palavras))
        partes.append(" ".join(palavras[inicio:fim]))
        if fim >= len(palavras):
            break
        inicio = fim - sobreposicao
    return partes


def ingestar(
    documentos: Iterable[Documento],
    loja: VectorStore,
    provedor: EmbeddingProvider,
    tamanho: int = 200,
    sobreposicao: int = 40,
) -> list[Chunk]:
    """Ingere documentos: particiona, embed e armazena na loja vetorial."""

    chunks: list[Chunk] = []
    contador = 0
    for documento in documentos:
        for indice, trecho in enumerate(particionar(documento.texto, tamanho, sobreposicao)):
            contador += 1
            chunk_id = f"{documento.id}::{indice:04d}"
            embedding = provedor.embed(trecho)
            chunks.append(Chunk.criar(id=chunk_id, texto=trecho, fonte=documento.fonte, embedding=embedding))
    loja.adicionar(chunks)
    return chunks
