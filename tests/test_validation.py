"""Testes das quatro camadas de validação."""

from agente_suporte_bancario_rag.models import Chunk
from agente_suporte_bancario_rag.validation import (
    camada_consistencia,
    camada_fonte,
    camada_politica,
    camada_saida,
    validar_todas,
)

CHUNKS = [
    Chunk(id="d1::0000", texto="O limite do cartão de crédito é de R$ 5.000,00.", fonte="credito"),
    Chunk(id="d1::0001", texto="Empréstimo consignado tem taxa de 1,99% ao mês.", fonte="emprestimo"),
]


def test_camada_saida_aprova_resposta_do_contexto():
    resposta = "O limite do cartão de crédito é de R$ 5.000,00."
    resultado = camada_saida(resposta, "O limite do cartão de crédito é de R$ 5.000,00.")
    assert resultado.aprovado is True
    assert resultado.camada == "saida"


def test_camada_saida_reprova_fora_do_contexto():
    resposta = "A previsão do tempo em São Paulo é de chuva."
    resultado = camada_saida(resposta, "O limite do cartão de crédito é de R$ 5.000,00.")
    assert resultado.aprovado is False


def test_camada_saida_aprova_recusa():
    resposta = "Não tenho informações suficientes nos documentos disponíveis para responder a essa pergunta."
    resultado = camada_saida(resposta, "qualquer contexto")
    assert resultado.aprovado is True


def test_camada_fonte_aprova_citacoes_validas():
    resposta = "Veja [fonte: d1::0000] e [fonte: d1::0001]."
    resultado = camada_fonte(resposta, CHUNKS)
    assert resultado.aprovado is True


def test_camada_fonte_reprova_citacao_invalida():
    resposta = "Veja [fonte: inexistente::9999]."
    resultado = camada_fonte(resposta, CHUNKS)
    assert resultado.aprovado is False


def test_camada_fonte_aprova_recusa_sem_citacoes():
    resposta = "Não tenho informações suficientes nos documentos disponíveis para responder a essa pergunta."
    resultado = camada_fonte(resposta, CHUNKS)
    assert resultado.aprovado is True


def test_camada_consistencia_aprova_resposta_coerente():
    resposta = "O limite do cartão de crédito é de R$ 5.000,00."
    resultado = camada_consistencia(resposta, CHUNKS)
    assert resultado.aprovado is True


def test_camada_politica_aprova_resposta_segura():
    resposta = "O limite do cartão de crédito é de R$ 5.000,00."
    resultado = camada_politica(resposta)
    assert resultado.aprovado is True


def test_camada_politica_reprova_promessa_de_credito():
    resposta = "Aprovo seu crédito imediatamente."
    resultado = camada_politica(resposta)
    assert resultado.aprovado is False


def test_validar_todas_aprovado_quando_todas_ok():
    resposta = "O limite do cartão de crédito é de R$ 5.000,00. [fonte: d1::0000]"
    relatorio, aprovado = validar_todas(
        resposta, "O limite do cartão de crédito é de R$ 5.000,00.", CHUNKS
    )
    assert aprovado is True
    assert relatorio.aprovado is True
