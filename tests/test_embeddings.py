"""Testes do provedor de embedding determinístico."""

from agente_suporte_bancario_rag.embeddings import HashEmbedding, similaridade_coseno


def test_embedding_deterministico():
    provedor = HashEmbedding(dimensao=384)
    texto = "O limite do cartão de crédito é de R$ 5.000,00."
    assert provedor.embed(texto) == provedor.embed(texto)


def test_embedding_dimensao_fixa():
    provedor = HashEmbedding(dimensao=384)
    assert len(provedor.embed("qualquer texto de teste")) == 384


def test_embedding_texto_vazio_retorna_zeros():
    provedor = HashEmbedding(dimensao=32)
    assert provedor.embed("") == tuple(0.0 for _ in range(32))


def test_similaridade_coseno_vetor_igual_a_si_mesmo_um():
    vetor = (0.5, -0.25, 1.0, 0.0)
    assert similaridade_coseno(vetor, vetor) == 1.0


def test_similaridade_coseno_vetores_vazios():
    assert similaridade_coseno((), ()) == 0.0


def test_similaridade_coseno_dimensoes_diferentes_erro():
    try:
        similaridade_coseno((1.0, 2.0), (1.0,))
    except ValueError:
        return
    raise AssertionError("deveria levantar ValueError para dimensões diferentes")


def test_textos_com_tokens_comuns_tem_maior_similaridade():
    provedor = HashEmbedding(dimensao=384)
    base = provedor.embed("limite cartão crédito")
    relacionada = provedor.embed("O limite do cartão de crédito é alto.")
    nao_relacionada = provedor.embed("previsão do tempo em São Paulo")
    assert similaridade_coseno(base, relacionada) > similaridade_coseno(base, nao_relacionada)
