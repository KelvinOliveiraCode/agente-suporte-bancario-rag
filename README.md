# Agente de Suporte Bancário RAG

Agente de suporte bancário com RAG (retrieval-augmented generation) e anti-alucinação. Ingestão de documentos bancários, geração de resposta com citações e quatro camadas de validação antes de responder: saída, fonte, consistência e política. Perguntas fora do contexto dos documentos recebem uma recusa direta — o agente não inventa resposta.

Versão 0.1.0. Licença MIT. Autor: Kelvin Oliveira.

## O que o projeto faz

- Indexação de documentos bancários com particionamento determinístico em chunks, embedding por hash (sem modelos externos nem rede) e armazenamento em loja vetorial.
- Respostas a perguntas do cliente usando apenas o conteúdo dos documentos indexados.
- Quatro camadas de validação em cada resposta:
  1. saída — a resposta é derivada do contexto recuperado?
  2. fonte — as citações apontam para chunks efetivamente recuperados?
  3. consistência — a resposta não contradiz os chunks?
  4. política — sem promessa de crédito, sem pedido de senha, sem conselho de investimento, sem garantia indevida?
- Se alguma camada reprovar, ou a pergunta estiver fora do contexto (similaridade máxima abaixo do limite), o agente recusa com a mensagem: "Não tenho informações suficientes nos documentos disponíveis para responder a essa pergunta."

## Stack

- Python 3.10 ou superior
- FastAPI, Pydantic, Uvicorn (API)
- pgvector / psycopg (opcional, PostgreSQL)
- langchain-core, sentence-transformers (opcional)
- Loja vetorial em memória por padrão, adequada a testes offline
- Gerenciamento de dependências com uv; build com hatchling

## Requisitos

- Python 3.10 ou superior
- uv

## Instalação

```
cd agente-suporte-bancario-rag
uv sync --extra dev
```

Para instalar apenas as dependências de runtime:

```
uv sync
```

## Como usar

Inicie a API:

```
uv run agente-suporte-bancario
```

O serviço fica disponível em http://localhost:8000.

### Endpoints

**GET /health**

Checagem do serviço. Retorna `{"status": "ok", "servico": "agente-suporte-bancario-rag"}`.

**POST /ingest**

Indexa documentos. Exemplo de corpo:

```json
{
  "documentos": [
    {"id": "doc1", "texto": "O limite do cartão de crédito é de R$ 5.000,00.", "fonte": "credito"}
  ]
}
```

Retorna `{"chunks_indexados": 1}`.

**POST /perguntar**

Faz uma pergunta. Exemplo de corpo:

```json
{"pergunta": "Qual é o limite do cartão de crédito?"}
```

Exemplo de resposta dentro do contexto:

```json
{
  "pergunta": "Qual é o limite do cartão de crédito?",
  "resposta": "Com base nos documentos disponíveis: ... [fonte: doc1::0000]",
  "citacoes": [{"chunk_id": "doc1::0000", "fonte": "credito", "trecho": "..."}],
  "aprovado": true,
  "recusado": false,
  "motivo_recusa": ""
}
```

Exemplo de pergunta fora do contexto:

```json
{"pergunta": "Qual é a previsão do tempo em São Paulo?"}
```

Responde com recusa (`recusado: true`, `aprovado: false`) e a mensagem padrão de recusa, sem inventar resposta.

### Variáveis de ambiente

| Variável | Padrão | Descrição |
| --- | --- | --- |
| `AGENTE_CHUNK_SIZE` | 200 | Tamanho do chunk em palavras |
| `AGENTE_CHUNK_OVERLAP` | 40 | Sobreposição entre chunks |
| `AGENTE_EMBEDDING_DIM` | 384 | Dimensão do vetor de embedding |
| `AGENTE_TOP_K` | 5 | Quantidade de chunks recuperados por pergunta |
| `AGENTE_SIM_THRESHOLD` | 0.30 | Limite de similaridade para o gate de anti-alucinação |
| `AGENTE_PG_DSN` | — | String de conexão PostgreSQL/pgvector (opcional) |

## Testes

```
uv run --extra dev pytest -q
```

São 39 testes: embeddings determinísticos, loja vetorial, ingestão, citações, quatro camadas de validação, pipeline RAG, anti-alucinação e endpoints da API. O teste `tests/test_anti_hallucination.py` confirma que uma pergunta sobre assunto fora dos documentos gera recusa e não inventa resposta.

Sem `uv`, com o Python do ambiente configurado:

```
python -m pytest -q
```

## Verificação de código

```
uv run --extra dev ruff check .
```

## Estrutura do repositório

```
src/agente_suporte_bancario_rag/
  config.py             Configuração por variável de ambiente
  models.py             Modelos de dados (Chunk, Citation, LayerResult, ValidationReport, RAGResponse)
  embeddings.py         HashEmbedding determinístico e similaridade de coseno
  vector_store.py       InMemoryVectorStore e PgVectorStore (opcional)
  ingest.py             Documento, particionamento e ingestão
  prompts.py            Templates de RAG e das quatro camadas
  citations.py          Extração e montagem de citações
  validation.py         Quatro camadas: saída, fonte, consistência, política
  anti_hallucination.py Gate de recusa fora de contexto
  rag.py                Pipeline RAG: recuperar, gerar, validar, responder ou recusar
  main.py               API FastAPI: /health, /ingest, /perguntar
tests/
  test_embeddings.py
  test_vector_store.py
  test_ingest.py
  test_citations.py
  test_validation.py
  test_rag.py
  test_anti_hallucination.py
  test_main.py
```

## Licença

MIT

## Autor

Kelvin Oliveira
