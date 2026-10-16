"""Testes da API FastAPI."""

from fastapi.testclient import TestClient

import agente_suporte_bancario_rag.main as main_mod


def _cliente():
    main_mod.loja.chunks.clear()
    return TestClient(main_mod.app)


def test_health():
    cliente = _cliente()
    resposta = cliente.get("/health")
    assert resposta.status_code == 200
    assert resposta.json() == {"status": "ok", "servico": "agente-suporte-bancario-rag"}


def test_ingest_endpoint():
    cliente = _cliente()
    resposta = cliente.post(
        "/ingest",
        json={
            "documentos": [
                {"id": "doc1", "texto": "O limite do cartão de crédito é de R$ 5.000,00.", "fonte": "credito"}
            ]
        },
    )
    assert resposta.status_code == 200
    assert resposta.json()["chunks_indexados"] >= 1


def test_perguntar_fora_contexto_recusa_sem_documentos():
    cliente = _cliente()
    resposta = cliente.post("/perguntar", json={"pergunta": "Qual é a previsão do tempo?"})
    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["recusado"] is True
    assert dados["aprovado"] is False


def test_perguntar_dentro_contexto_responde_com_citacoes():
    cliente = _cliente()
    cliente.post(
        "/ingest",
        json={
            "documentos": [
                {
                    "id": "doc1",
                    "texto": "O limite do cartão de crédito é de R$ 5.000,00 para clientes com renda comprovada.",
                    "fonte": "credito",
                }
            ]
        },
    )
    resposta = cliente.post("/perguntar", json={"pergunta": "Qual é o limite do cartão de crédito?"})
    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["recusado"] is False
    assert dados["aprovado"] is True
    assert len(dados["citacoes"]) > 0
