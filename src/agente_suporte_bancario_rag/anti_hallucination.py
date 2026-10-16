"""Gate de anti-alucinação: pergunta fora do contexto deve recusar responder."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AntiHallucinationResult:
    """Resultado da verificação de anti-alucinação."""

    pode_responder: bool
    motivo: str
    similaridade_maxima: float


def gate_anti_alucinacao(similaridade_maxima: float, threshold: float) -> AntiHallucinationResult:
    """Recusa responder quando a pergunta não tem suporte nos documentos."""

    if similaridade_maxima < threshold:
        return AntiHallucinationResult(
            pode_responder=False,
            motivo="pergunta fora do contexto dos documentos disponíveis",
            similaridade_maxima=similaridade_maxima,
        )
    return AntiHallucinationResult(
        pode_responder=True,
        motivo="pergunta com suporte nos documentos",
        similaridade_maxima=similaridade_maxima,
    )
