"""Testes de extração e montagem de citações."""

from agente_suporte_bancario_rag.citations import extrair_citacoes, montar_citacoes
from agente_suporte_bancario_rag.models import Chunk


def test_extrair_citacoes():
    texto = "Resposta com [fonte: doc1::0000] e [fonte: doc2::0003]."
    assert extrair_citacoes(texto) == ["doc1::0000", "doc2::0003"]


def test_extrair_citacoes_sem_citacoes():
    assert extrair_citacoes("sem citações aqui") == []


def test_montar_citacoes_ignora_desconhecidos_e_duplicados():
    chunks = {
        "doc1::0000": Chunk(id="doc1::0000", texto="texto do chunk um", fonte="fonte A"),
        "doc1::0001": Chunk(id="doc1::0001", texto="texto do chunk dois", fonte="fonte B"),
    }
    citacoes = montar_citacoes(["doc1::0000", "doc1::0000", "doc9::9999"], chunks)
    assert [c.chunk_id for c in citacoes] == ["doc1::0000"]
    assert citacoes[0].fonte == "fonte A"
    assert citacoes[0].trecho == "texto do chunk um"
