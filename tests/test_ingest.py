"""Testes de ingestão e particionamento de documentos."""

from agente_suporte_bancario_rag.embeddings import HashEmbedding
from agente_suporte_bancario_rag.ingest import Documento, ingestar, particionar
from agente_suporte_bancario_rag.vector_store import InMemoryVectorStore

PROVEDOR = HashEmbedding(dimensao=384)


def test_particionar_tamanho_fixo():
    texto = " ".join(f"palavra{i}" for i in range(10))
    assert particionar(texto, tamanho=4, sobreposicao=1) == [
        "palavra0 palavra1 palavra2 palavra3",
        "palavra3 palavra4 palavra5 palavra6",
        "palavra6 palavra7 palavra8 palavra9",
    ]


def test_particionar_deterministico():
    texto = "Um texto qualquer para testar determinismo do particionamento."
    assert particionar(texto) == particionar(texto)


def test_particionar_texto_vazio():
    assert particionar("") == []


def test_particionar_tamanho_invalido_erro():
    try:
        particionar("texto", tamanho=0)
    except ValueError:
        return
    raise AssertionError("deveria levantar ValueError para tamanho <= 0")


def test_ingestar_cria_chunks_com_ids_previsiveis():
    documentos = [
        Documento(id="doc1", texto="Um pouco de texto sobre cartão de crédito com limite.", fonte="credito")
    ]
    loja = InMemoryVectorStore()
    chunks = ingestar(documentos, loja, PROVEDOR)
    assert len(chunks) >= 1
    assert chunks[0].id == "doc1::0000"
    assert chunks[0].fonte == "credito"
    assert len(chunks[0].embedding) == 384
    assert loja.buscar("cartão crédito", PROVEDOR, top_k=1)[0].id == "doc1::0000"
