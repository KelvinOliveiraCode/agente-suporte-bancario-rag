"""Extração e formatação de citações nas respostas do agente."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .models import Chunk, Citation

CITACAO_RE = re.compile(r"\[fonte:\s*([^\]]+)\]")


def extrair_citacoes(texto: str) -> list[str]:
    """Extrai identificadores de fonte marcados como [fonte: id] no texto."""

    return [match.group(1).strip() for match in CITACAO_RE.finditer(texto)]


def montar_citacoes(ids_fonte: Iterable[str], chunks: dict[str, Chunk]) -> list[Citation]:
    """Constrói objetos Citation a partir de IDs e chunks disponíveis.

    Ignora IDs desconhecidos e remove duplicados, preservando a ordem.
    """

    citacoes: list[Citation] = []
    vistos: set[str] = set()
    for chunk_id in ids_fonte:
        if chunk_id in vistos or chunk_id not in chunks:
            continue
        vistos.add(chunk_id)
        chunk = chunks[chunk_id]
        citacoes.append(Citation(chunk_id=chunk_id, fonte=chunk.fonte, trecho=chunk.texto[:200]))
    return citacoes
