"""Testes da loja vetorial em memória."""

from agente_suporte_bancario_rag.embeddings import HashEmbedding, similaridade_coseno
from agente_suporte_bancario_rag.models import Chunk
from agente_suporte_bancario_rag.vector_store import InMemoryVectorStore

PROVEDOR = HashEmbedding(dimensao=384)
EIXO = (1.0,) + (0.0,) * 383


def _chunk(id_: str, texto: str, embedding: tuple[float, ...], fonte: str = "fonte") -> Chunk:
    return Chunk(id=id_, texto=texto, fonte=fonte, embedding=embedding)


def test_busca_loja_vazia_retorna_vazio():
    loja = InMemoryVectorStore()
    assert loja.buscar("qualquer coisa", PROVEDOR) == []


def test_busca_retorna_chunks_ordenados_por_similaridade():
    loja = InMemoryVectorStore()
    chunks = {
        "a": _chunk("a", "texto a", EIXO),
        "b": _chunk("b", "texto b", (0.5,) + (0.0,) * 383),
        "c": _chunk("c", "texto c", (0.1,) + (0.0,) * 383),
    }
    loja.adicionar(list(chunks.values()))
    vetor_consulta = PROVEDOR.embed("consulta")
    esperado = sorted(
        chunks,
        key=lambda id_: similaridade_coseno(vetor_consulta, chunks[id_].embedding),
        reverse=True,
    )
    ids = [chunk.id for chunk in loja.buscar("consulta", PROVEDOR, top_k=2)]
    assert len(ids) == 2
    assert ids == esperado[:2]


def test_busca_respeita_top_k():
    loja = InMemoryVectorStore()
    for i in range(5):
        loja.adicionar([_chunk(f"c{i}", f"texto {i}", EIXO)])
    assert len(loja.buscar("consulta", PROVEDOR, top_k=3)) == 3
