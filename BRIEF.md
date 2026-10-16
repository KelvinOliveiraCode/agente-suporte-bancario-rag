# BRIEF — agente-suporte-bancario-rag

## Entregas

Projeto "agente-suporte-bancario-rag" (área IA/ML): agente de suporte bancário com RAG e anti-alucinação em 4 camadas de validação.

Stack: Python, FastAPI, pgvector (com loja em memória compatível para testes offline), LangChain-lite como dependência opcional, prompts.

## Arquivos criados

- `pyproject.toml` — dependências, entry point, config pytest/ruff
- `src/agente_suporte_bancario_rag/__init__.py`
- `src/agente_suporte_bancario_rag/config.py` — settings por variável de ambiente
- `src/agente_suporte_bancario_rag/models.py` — Chunk, Citation, LayerResult, ValidationReport, RAGResponse
- `src/agente_suporte_bancario_rag/embeddings.py` — HashEmbedding determinístico + similaridade de coseno
- `src/agente_suporte_bancario_rag/vector_store.py` — InMemoryVectorStore + PgVectorStore (opcional)
- `src/agente_suporte_bancario_rag/ingest.py` — Documento, particionamento, ingestão
- `src/agente_suporte_bancario_rag/prompts.py` — templates de RAG e das 4 camadas
- `src/agente_suporte_bancario_rag/citations.py` — extração e montagem de citações
- `src/agente_suporte_bancario_rag/validation.py` — 4 camadas: saída, fonte, consistência, política
- `src/agente_suporte_bancario_rag/anti_hallucination.py` — gate de anti-alucinação
- `src/agente_suporte_bancario_rag/rag.py` — pipeline RAG (recuperar, gerar, validar, responder ou recusar)
- `src/agente_suporte_bancario_rag/main.py` — API FastAPI: /health, /ingest, /perguntar
- `tests/__init__.py`
- `tests/test_embeddings.py`
- `tests/test_vector_store.py`
- `tests/test_ingest.py`
- `tests/test_citations.py`
- `tests/test_validation.py`
- `tests/test_rag.py`
- `tests/test_anti_hallucination.py`
- `tests/test_main.py`
- `README.md` — bilíngue PT-BR + EN (sobre, como usar, testes, licença MIT, autor)
- `BRIEF.md` — este arquivo

## Caminho do projeto

`C:\Users\Kelvin\AppData\Local\Temp\opencode\github-staging\projects\agente-suporte-bancario-rag\`

## Comando de teste usado

Gate completo (skill `python`):

```
uv run --extra dev ruff check .
uv run --extra dev pytest -q
```

Equivalente sem `uv`:

```
python -m pytest -q
```

## Resultado dos testes

- `ruff check`: All checks passed (0 erros).
- `pytest`: 39 passed, 0 failed, em ~1s.

Inclui o teste de anti-alucinação exigido: pergunta fora do contexto (`Qual é a cotação do dólar hoje?` / previsão do tempo) gera recusa (`recusado: true`, `aprovado: false`) e a mensagem padrão de recusa, sem inventar resposta.

## Caminhos dos documentos de entrega

- README: `C:\Users\Kelvin\AppData\Local\Temp\opencode\github-staging\projects\agente-suporte-bancario-rag\README.md`
- BRIEF: `C:\Users\Kelvin\AppData\Local\Temp\opencode\github-staging\projects\agente-suporte-bancario-rag\BRIEF.md`

## Critérios de aceitação (PLANO_50_PROJETOS.md)

- [x] Testes de unidade
- [x] Teste de anti-alucinação (pergunta fora do contexto recusa responder)
- [x] README completa PT-BR + EN (título, descrição, como usar, testes, licença MIT, autor Kelvin Oliveira)
- [x] 4 camadas de validação: saída, fonte, consistência, política
- [x] Citações nas respostas
- [x] Código limpo, tipado, com docstrings, sem aleatoriedade (determinístico)

## Autor

Kelvin Oliveira
