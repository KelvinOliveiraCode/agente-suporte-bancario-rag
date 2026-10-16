"""Quatro camadas de validação anti-alucinação: saída, fonte, consistência, política."""

from __future__ import annotations

import re

from .citations import extrair_citacoes
from .models import Chunk, LayerResult, ValidationReport

SAIDA_MINIMO_OVERLAP = 0.40
CONSISTENCIA_MINIMO_OVERLAP = 0.30
RECUSA = "não tenho informações suficientes nos documentos disponíveis para responder a essa pergunta."


def _normalizar(texto: str) -> str:
    return re.sub(r"\s+", " ", texto.lower()).strip()


def _palavras_chave(texto: str) -> set[str]:
    return {palavra for palavra in _normalizar(texto).split() if len(palavra) > 3}


def camada_saida(resposta: str, contexto: str) -> LayerResult:
    """Camada 1 — Saída: a resposta deve ser derivável do contexto recuperado."""

    if not resposta or not resposta.strip():
        return LayerResult(camada="saida", aprovado=False, score=0.0, motivo="resposta vazia")
    palavras_resposta = _palavras_chave(resposta)
    palavras_contexto = _palavras_chave(contexto)
    if not palavras_resposta:
        return LayerResult(camada="saida", aprovado=False, score=0.0, motivo="resposta sem vocabulário útil")
    coberta = palavras_resposta & palavras_contexto
    score = len(coberta) / len(palavras_resposta)
    eh_recusa = RECUSA in _normalizar(resposta)
    aprovado = score >= SAIDA_MINIMO_OVERLAP or eh_recusa
    motivo = (
        "resposta derivada do contexto (ou recusa legítima)"
        if aprovado
        else f"apenas {score:.0%} do vocabulário da resposta aparece no contexto"
    )
    return LayerResult(camada="saida", aprovado=aprovado, score=score, motivo=motivo)


def camada_fonte(resposta: str, chunks_recuperados: list[Chunk]) -> LayerResult:
    """Camada 2 — Fonte: citações devem referenciar chunks recuperados."""

    ids_recuperados = {chunk.id for chunk in chunks_recuperados}
    citacoes = extrair_citacoes(resposta)
    eh_recusa = RECUSA in _normalizar(resposta)
    if not citacoes and not eh_recusa:
        return LayerResult(camada="fonte", aprovado=False, score=0.0, motivo="resposta informativa sem citações")
    invalidas = [citacao for citacao in citacoes if citacao not in ids_recuperados]
    score = 1.0 if not citacoes else 1.0 - (len(invalidas) / len(citacoes))
    aprovado = not invalidas
    motivo = (
        "todas as citações referenciam chunks recuperados"
        if aprovado
        else f"citações inválidas: {', '.join(invalidas)}"
    )
    return LayerResult(camada="fonte", aprovado=aprovado, score=score, motivo=motivo)


def camada_consistencia(resposta: str, chunks: list[Chunk]) -> LayerResult:
    """Camada 3 — Consistência: resposta não deve contradizer os chunks."""

    if not resposta or not resposta.strip():
        return LayerResult(camada="consistencia", aprovado=True, score=1.0, motivo="sem afirmações para contradizer")
    texto_chunks = " ".join(chunk.texto for chunk in chunks)
    palavras_resposta = _palavras_chave(resposta)
    palavras_chunks = _palavras_chave(texto_chunks)
    if not palavras_resposta:
        return LayerResult(camada="consistencia", aprovado=True, score=1.0, motivo="resposta de recusa, sem claims")
    coberta = palavras_resposta & palavras_chunks
    score = len(coberta) / len(palavras_resposta)
    aprovado = score >= CONSISTENCIA_MINIMO_OVERLAP
    motivo = (
        "resposta consistente com os chunks"
        if aprovado
        else "possível contradição com os chunks recuperados"
    )
    return LayerResult(camada="consistencia", aprovado=aprovado, score=score, motivo=motivo)


def camada_politica(resposta: str) -> LayerResult:
    """Camada 4 — Política: regras de suporte bancário."""

    normalizada = _normalizar(resposta)
    proibidas = [
        ("aprovo seu crédito", "promessa de aprovação de crédito"),
        ("aprovamos seu crédito", "promessa de aprovação de crédito"),
        ("seu saldo é", "informação de saldo sem fonte"),
        ("digite sua senha", "pedido de senha"),
        ("informe sua senha", "pedido de senha"),
        ("invista em", "conselho de investimento"),
        ("garantido", "garantia indevida"),
    ]
    achadas = [motivo for _padrao, motivo in proibidas if _padrao in normalizada]
    aprovado = not achadas
    score = 0.0 if achadas else 1.0
    motivo = (
        "resposta dentro das políticas bancárias"
        if aprovado
        else f"viola política bancária: {', '.join(achadas)}"
    )
    return LayerResult(camada="politica", aprovado=aprovado, score=score, motivo=motivo)


def validar_todas(resposta: str, contexto: str, chunks: list[Chunk]) -> tuple[ValidationReport, bool]:
    """Executa as 4 camadas e retorna (relatório, todas_aprovaram)."""

    camadas = (
        camada_saida(resposta, contexto),
        camada_fonte(resposta, chunks),
        camada_consistencia(resposta, chunks),
        camada_politica(resposta),
    )
    relatorio = ValidationReport(camadas=camadas, aprovado=all(camada.aprovado for camada in camadas))
    return relatorio, relatorio.aprovado
