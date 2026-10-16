"""Provedores de embedding determinísticos (sem modelos externos nem rede)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol


class EmbeddingProvider(Protocol):
    """Contrato de geração de vetores de embedding."""

    def embed(self, texto: str) -> tuple[float, ...]:
        """Retorna o vetor de embedding do texto."""


@dataclass(frozen=True)
class HashEmbedding:
    """Embedding determinístico baseado em hash de tokens.

    O mesmo texto produz sempre o mesmo vetor, sem aleatoriedade, sem
    pesos de modelo e sem chamadas de rede. A similaridade de coseno entre
    dois textos cresce quando eles compartilham tokens, o que torna a
    recuperação por similaridade funcional offline e reproduzível em testes.
    """

    dimensao: int = 384

    def _vetor_token(self, token: str) -> tuple[float, ...]:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return tuple(
            (digest[i % len(digest)] - 128) / 128.0 for i in range(self.dimensao)
        )

    def embed(self, texto: str) -> tuple[float, ...]:
        tokens = texto.split()
        if not tokens:
            return tuple(0.0 for _ in range(self.dimensao))
        soma = [0.0] * self.dimensao
        for token in tokens:
            vetor = self._vetor_token(token)
            for i in range(self.dimensao):
                soma[i] += vetor[i]
        n = len(tokens)
        media = [valor / n for valor in soma]
        norma = sum(valor * valor for valor in media) ** 0.5
        if norma == 0.0:
            return tuple(media)
        return tuple(valor / norma for valor in media)


def similaridade_coseno(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Similaridade de coseno entre dois vetores (determinística)."""

    if len(a) != len(b):
        raise ValueError("vetores com dimensões diferentes")
    if not a:
        return 0.0
    produto = sum(x * y for x, y in zip(a, b))
    norma_a = sum(x * x for x in a) ** 0.5
    norma_b = sum(y * y for y in b) ** 0.5
    if norma_a == 0.0 or norma_b == 0.0:
        return 0.0
    return produto / (norma_a * norma_b)
