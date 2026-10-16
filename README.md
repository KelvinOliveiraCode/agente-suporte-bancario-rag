# Agente de Suporte Bancário RAG / Banking Support RAG Agent

## PT-BR

### Sobre o projeto

Agente de suporte bancário com RAG (retrieval-augmented generation) e anti-alucinação. Ingestão de documentos bancários, geração de resposta com citações e quatro camadas de validação antes de responder: saída, fonte, consistência e política. Perguntas fora do contexto dos documentos recebem uma recusa direta — o agente não inventa resposta.

Stack: Python, FastAPI, pgvector, LangChain-lite (opcional), prompts.

### O que ele faz

- Indexa documentos bancários: texto em chunks determinísticos, embedding, loja vetorial.
- Responde perguntas do cliente só com o que está nos documentos indexados.
- Executa 4 camadas em cada resposta:
  1. Saída — a resposta vem do contexto recuperado?
  2. Fonte — as citações apontam para chunks efetivamente recuperados?
  3. Consistência — a resposta não contradiz os chunks?
  4. Política — sem promessa de crédito, sem pedido de senha, sem conselho de investimento?
- Se alguma camada reprovar, ou a pergunta estiver fora do contexto, o agente recusa: "Não tenho informações suficientes nos documentos disponíveis para responder a essa pergunta."

### Como usar

Requisitos: Python 3.10+ e uv.

```
cd agente-suporte-bancario-rag
uv sync --extra dev
uv run agente-suporte-bancario   # inicia a API em http://localhost:8000
```

Para instalar só as dependências de runtime:

```
uv sync
```

#### Endpoints

**POST /ingest** — indexa documentos:

```json
{
  "documentos": [
    {"id": "doc1", "texto": "O limite do cartão de crédito é de R$ 5.000,00.", "fonte": "credito"}
  ]
}
```

**POST /perguntar** — faz uma pergunta:

```json
{"pergunta": "Qual é o limite do cartão de crédito?"}
```

Resposta dentro do contexto:

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

Pergunta fora do contexto:

```json
{"pergunta": "Qual é a previsão do tempo em São Paulo?"}
```

Responde com recusa (`recusado: true`, `aprovado: false`).

**GET /health** — checagem do serviço.

### Testes

```
uv run --extra dev pytest -q
```

ou, com o Python do ambiente já configurado:

```
python -m pytest -q
```

São 39 testes: embeddings determinísticos, loja vetorial, ingestão, citações, 4 camadas de validação, pipeline RAG, anti-alucinação e endpoints da API. O teste de anti-alucinação (`tests/test_anti_hallucination.py`) confirma que uma pergunta sobre assunto fora dos documentos gera recusa e não inventa resposta.

### Verificação de código

```
uv run --extra dev ruff check .
```

### Licença

MIT

### Autor

Kelvin Oliveira

---

## EN

### About the project

Banking support agent with RAG (retrieval-augmented generation) and anti-hallucination. Ingests banking documents, generates answers with citations, and runs four validation layers before replying: output, source, consistency, and policy. Questions outside the document context get a direct refusal — the agent does not invent an answer.

Stack: Python, FastAPI, pgvector, LangChain-lite (optional), prompts.

### What it does

- Indexes banking documents: text in deterministic chunks, embedding, vector store.
- Answers customer questions using only indexed documents.
- Runs 4 layers on every answer:
  1. Output — does the answer come from the retrieved context?
  2. Source — do citations point to actually retrieved chunks?
  3. Consistency — does the answer avoid contradicting the chunks?
  4. Policy — no credit promises, no password requests, no investment advice?
- If any layer fails, or the question is out of context, the agent refuses: "I do not have enough information in the available documents to answer this question."

### How to use

Requirements: Python 3.10+ and uv.

```
cd agente-suporte-bancario-rag
uv sync --extra dev
uv run agente-suporte-bancario   # starts the API at http://localhost:8000
```

To install runtime dependencies only:

```
uv sync
```

#### Endpoints

**POST /ingest** — indexes documents:

```json
{
  "documentos": [
    {"id": "doc1", "texto": "O limite do cartão de crédito é de R$ 5.000,00.", "fonte": "credito"}
  ]
}
```

**POST /perguntar** — asks a question:

```json
{"pergunta": "Qual é o limite do cartão de crédito?"}
```

In-context answer:

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

Out-of-context question:

```json
{"pergunta": "Qual é a previsão do tempo em São Paulo?"}
```

Responds with refusal (`recusado: true`, `aprovado: false`).

**GET /health** — service check.

### Tests

```
uv run --extra dev pytest -q
```

or, with the environment Python already set up:

```
python -m pytest -q
```

39 tests: deterministic embeddings, vector store, ingestion, citations, 4 validation layers, RAG pipeline, anti-hallucination, and API endpoints. The anti-hallucination test (`tests/test_anti_hallucination.py`) confirms that a question about a subject outside the documents triggers a refusal and no invented answer.

### Code check

```
uv run --extra dev ruff check .
```

### License

MIT

### Author

Kelvin Oliveira
