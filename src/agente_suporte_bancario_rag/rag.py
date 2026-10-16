"""Pipeline RAG: recuperação, geração e validação em 4 camadas."""

from __future__ import annotations

from .anti_hallucination import gate_anti_alucinacao
from .citations import montar_citacoes
from .config import Settings
from .embeddings import EmbeddingProvider, similaridade_coseno
from .models import Chunk, RAGResponse, ValidationReport
from .validation import validar_todas
from .vector_store import VectorStore

RESPOSTA_RECUSA = "Não tenho informações suficientes nos documentos disponíveis para responder a essa pergunta."
TAMANHO_RESPOSTA = 600


class RAGAgent:
    """Agente de RAG com anti-alucinação e 4 camadas de validação."""

    def __init__(self, loja: VectorStore, provedor: EmbeddingProvider, settings: Settings) -> None:
        self.loja = loja
        self.provedor = provedor
        self.settings = settings

    def responder(self, pergunta: str) -> RAGResponse:
        if not pergunta or not pergunta.strip():
            return self._recusada(pergunta, "pergunta vazia")

        chunks = self.loja.buscar(pergunta, self.provedor, top_k=self.settings.top_k)
        vetor_pergunta = self.provedor.embed(pergunta)
        similaridade_maxima = max(
            (
                similaridade_coseno(vetor_pergunta, chunk.embedding)
                for chunk in chunks
                if chunk.embedding
            ),
            default=0.0,
        )

        gate = gate_anti_alucinacao(similaridade_maxima, self.settings.similarity_threshold)
        if not gate.pode_responder:
            return self._recusada(pergunta, gate.motivo)

        contexto = "\n\n".join(f"[{chunk.id}] {chunk.texto}" for chunk in chunks)
        resposta_texto, ids_citacao = self._gerar(chunks)

        citacoes = montar_citacoes(ids_citacao, {chunk.id: chunk for chunk in chunks})
        relatorio, aprovado = validar_todas(resposta_texto, contexto, chunks)

        if not aprovado:
            return self._recusada(pergunta, "validação reprovada: " + "; ".join(relatorio.motivos_reprovacao))

        return RAGResponse(
            pergunta=pergunta,
            resposta=resposta_texto,
            citacoes=tuple(citacoes),
            aprovado=True,
            relatorio_validacao=relatorio,
            recusado=False,
            motivo_recusa="",
        )

    def _gerar(self, chunks: list[Chunk]) -> tuple[str, list[str]]:
        """Geração determinística e extrativa com citações (sem LLM externo)."""

        partes = [chunk.texto for chunk in chunks]
        corpo = " ".join(partes)
        if len(corpo) > TAMANHO_RESPOSTA:
            corpo = corpo[:TAMANHO_RESPOSTA].rsplit(" ", 1)[0] + "..."
        ids = [chunk.id for chunk in chunks]
        citacoes_marcadas = "".join(f" [fonte: {chunk_id}]" for chunk_id in ids)
        return f"Com base nos documentos disponíveis: {corpo}{citacoes_marcadas}", ids

    def _recusada(self, pergunta: str, motivo: str) -> RAGResponse:
        return RAGResponse(
            pergunta=pergunta,
            resposta=RESPOSTA_RECUSA,
            citacoes=(),
            aprovado=False,
            relatorio_validacao=ValidationReport(camadas=(), aprovado=False),
            recusado=True,
            motivo_recusa=motivo,
        )
