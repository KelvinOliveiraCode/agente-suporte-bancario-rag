# Banking Support RAG Agent

Banking support agent with RAG (retrieval-augmented generation) and anti-hallucination. Ingests banking documents, generates answers with citations, and runs four validation layers before replying: output, source, consistency, and policy. Questions outside the document context get a direct refusal — the agent does not invent an answer.

Version 0.1.0. MIT license. Author: Kelvin Oliveira.

## What the project does

- Indexing of banking documents with deterministic word-based chunking, hash-based embedding (no external models or network), and vector store storage.
- Answers to customer questions using only the content of indexed documents.
- Four validation layers on every answer:
  1. output — is the answer derived from the retrieved context?
  2. source — do citations point to actually retrieved chunks?
  3. consistency — does the answer avoid contradicting the chunks?
  4. policy — no credit promises, no password requests, no investment advice, no undue guarantees?
- If any layer fails, or the question is out of context (maximum similarity below the threshold), the agent refuses with the message: "I do not have enough information in the available documents to answer this question."

## Stack

- Python 3.10 or later
- FastAPI, Pydantic, Uvicorn (API)
- pgvector / psycopg (optional, PostgreSQL)
- langchain-core, sentence-transformers (optional)
- In-memory vector store by default, suitable for offline tests
- Dependency management with uv; build with hatchling

## Requirements

- Python 3.10 or later
- uv

## Installation

```
cd agente-suporte-bancario-rag
uv sync --extra dev
```

To install runtime dependencies only:

```
uv sync
```

## Usage

Start the API:

```
uv run agente-suporte-bancario
```

The service is available at http://localhost:8000.

### Endpoints

**GET /health**

Service check. Returns `{"status": "ok", "servico": "agente-suporte-bancario-rag"}`.

**POST /ingest**

Indexes documents. Example body:

```json
{
  "documentos": [
    {"id": "doc1", "texto": "O limite do cartão de crédito é de R$ 5.000,00.", "fonte": "credito"}
  ]
}
```

Returns `{"chunks_indexados": 1}`.

**POST /perguntar**

Asks a question. Example body:

```json
{"pergunta": "Qual é o limite do cartão de crédito?"}
```

Example of in-context answer:

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

Example of out-of-context question:

```json
{"pergunta": "Qual é a previsão do tempo em São Paulo?"}
```

Responds with refusal (`recusado: true`, `aprovado: false`) and the standard refusal message, without inventing an answer.

### Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `AGENTE_CHUNK_SIZE` | 200 | Chunk size in words |
| `AGENTE_CHUNK_OVERLAP` | 40 | Overlap between chunks |
| `AGENTE_EMBEDDING_DIM` | 384 | Embedding vector dimension |
| `AGENTE_TOP_K` | 5 | Number of chunks retrieved per question |
| `AGENTE_SIM_THRESHOLD` | 0.30 | Similarity limit for the anti-hallucination gate |
| `AGENTE_PG_DSN` | — | PostgreSQL/pgvector connection string (optional) |

## Tests

```
uv run --extra dev pytest -q
```

39 tests: deterministic embeddings, vector store, ingestion, citations, four validation layers, RAG pipeline, anti-hallucination, and API endpoints. The test `tests/test_anti_hallucination.py` confirms that a question about a subject outside the documents triggers a refusal and no invented answer.

Without uv, with the environment Python configured:

```
python -m pytest -q
```

## Code check

```
uv run --extra dev ruff check .
```

## Repository structure

```
src/agente_suporte_bancario_rag/
  config.py             Configuration from environment variables
  models.py             Data models (Chunk, Citation, LayerResult, ValidationReport, RAGResponse)
  embeddings.py         Deterministic HashEmbedding and cosine similarity
  vector_store.py       InMemoryVectorStore and PgVectorStore (optional)
  ingest.py             Document, partitioning, ingestion
  prompts.py            RAG and four-layer prompt templates
  citations.py          Citation extraction and assembly
  validation.py         Four layers: output, source, consistency, policy
  anti_hallucination.py Out-of-context refusal gate
  rag.py                RAG pipeline: retrieve, generate, validate, answer or refuse
  main.py               FastAPI API: /health, /ingest, /perguntar
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

## License

MIT

## Author

Kelvin Oliveira
