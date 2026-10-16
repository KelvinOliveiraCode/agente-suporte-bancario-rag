"""Teste de anti-alucinação: pergunta fora do contexto deve recusar responder."""

from agente_suporte_bancario_rag.config import Settings
from agente_suporte_bancario_rag.embeddings import HashEmbedding
from agente_suporte_bancario_rag.ingest import Documento, ingestar
from agente_suporte_bancario_rag.rag import RAGAgent
from agente_suporte_bancario_rag.vector_store import InMemoryVectorStore

SETTINGS = Settings(
    chunk_size=200, chunk_overlap=40, embedding_dimension=384, top_k=5, similarity_threshold=0.30
)
PROVEDOR = HashEmbedding(dimensao=384)


def test_pergunta_fora_do_contexto_deve_recusar_responder():
    loja = InMemoryVectorStore()
    documentos = [
        Documento(id="doc1", texto="O limite do cartão de crédito é de R$ 5.000,00.", fonte="credito")
    ]
    ingestar(documentos, loja, PROVEDOR)
    agente = RAGAgent(loja=loja, provedor=PROVEDOR, settings=SETTINGS)
    resposta = agente.responder("Qual é a cotação do dólar hoje?")
    assert resposta.recusado is True
    assert resposta.aprovado is False
    assert "suficientes" in resposta.resposta
    assert len(resposta.citacoes) == 0


def test_pergunta_dentro_do_contexto_nao_recusa():
    loja = InMemoryVectorStore()
    documentos = [
        Documento(id="doc1", texto="O limite do cartão de crédito é de R$ 5.000,00.", fonte="credito")
    ]
    ingestar(documentos, loja, PROVEDOR)
    agente = RAGAgent(loja=loja, provedor=PROVEDOR, settings=SETTINGS)
    resposta = agente.responder("Qual é o limite do cartão de crédito?")
    assert resposta.recusado is False
    assert resposta.aprovado is True
