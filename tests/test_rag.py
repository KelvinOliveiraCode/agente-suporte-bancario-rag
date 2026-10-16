"""Testes do pipeline RAG."""

from agente_suporte_bancario_rag.anti_hallucination import gate_anti_alucinacao
from agente_suporte_bancario_rag.config import Settings
from agente_suporte_bancario_rag.embeddings import HashEmbedding
from agente_suporte_bancario_rag.ingest import Documento, ingestar
from agente_suporte_bancario_rag.rag import RAGAgent
from agente_suporte_bancario_rag.vector_store import InMemoryVectorStore

SETTINGS = Settings(
    chunk_size=200, chunk_overlap=40, embedding_dimension=384, top_k=5, similarity_threshold=0.30
)
PROVEDOR = HashEmbedding(dimensao=384)


def _agente_com_documento() -> RAGAgent:
    loja = InMemoryVectorStore()
    documentos = [
        Documento(
            id="doc1",
            texto="O limite do cartão de crédito é de R$ 5.000,00 para clientes com renda comprovada.",
            fonte="credito",
        )
    ]
    ingestar(documentos, loja, PROVEDOR)
    return RAGAgent(loja=loja, provedor=PROVEDOR, settings=SETTINGS)


def test_responder_pergunta_dentro_contexto_retorna_resposta_e_citacoes():
    agente = _agente_com_documento()
    resposta = agente.responder("Qual é o limite do cartão de crédito?")
    assert resposta.recusado is False
    assert resposta.aprovado is True
    assert len(resposta.citacoes) > 0
    assert "limite" in resposta.resposta


def test_responder_pergunta_fora_contexto_recusa():
    agente = _agente_com_documento()
    resposta = agente.responder("Qual é a previsão do tempo em São Paulo?")
    assert resposta.recusado is True
    assert resposta.aprovado is False
    assert (
        resposta.resposta
        == "Não tenho informações suficientes nos documentos disponíveis para responder a essa pergunta."
    )


def test_pergunta_vazia_recusa():
    agente = _agente_com_documento()
    resposta = agente.responder("")
    assert resposta.recusado is True
    assert resposta.aprovado is False


def test_gate_anti_alucinacao_sem_similaridade_recusa():
    resultado = gate_anti_alucinacao(0.0, 0.30)
    assert resultado.pode_responder is False


def test_gate_anti_alucinacao_com_similaridade_permite():
    resultado = gate_anti_alucinacao(0.80, 0.30)
    assert resultado.pode_responder is True
