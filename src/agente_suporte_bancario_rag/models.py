"""Modelos de dados do agente (dataclasses imutáveis)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    """Trecho de documento indexado com seu vetor de embedding."""

    id: str
    texto: str
    fonte: str
    embedding: tuple[float, ...] = field(default_factory=tuple)
    metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def criar(cls, id: str, texto: str, fonte: str, embedding: tuple[float, ...]) -> Chunk:
        return cls(id=id, texto=texto, fonte=fonte, embedding=embedding)


@dataclass(frozen=True)
class Citation:
    """Citação de um chunk recuperado usado na resposta."""

    chunk_id: str
    fonte: str
    trecho: str


@dataclass(frozen=True)
class LayerResult:
    """Resultado de uma camada de validação."""

    camada: str  # saida | fonte | consistencia | politica
    aprovado: bool
    score: float
    motivo: str


@dataclass(frozen=True)
class ValidationReport:
    """Relatório consolidado das 4 camadas de validação."""

    camadas: tuple[LayerResult, ...]
    aprovado: bool

    @property
    def motivos_reprovacao(self) -> tuple[str, ...]:
        return tuple(
            f"{resultado.camada}: {resultado.motivo}"
            for resultado in self.camadas
            if not resultado.aprovado
        )


@dataclass(frozen=True)
class RAGResponse:
    """Resposta final do agente ao cliente."""

    pergunta: str
    resposta: str
    citacoes: tuple[Citation, ...]
    aprovado: bool
    relatorio_validacao: ValidationReport
    recusado: bool
    motivo_recusa: str = ""
