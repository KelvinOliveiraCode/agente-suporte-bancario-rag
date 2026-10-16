"""API FastAPI do agente de suporte bancário RAG."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .config import Settings
from .embeddings import HashEmbedding
from .ingest import Documento, ingestar
from .rag import RAGAgent
from .vector_store import InMemoryVectorStore

settings = Settings.from_env()
provedor = HashEmbedding(dimensao=settings.embedding_dimension)
loja = InMemoryVectorStore()
agente = RAGAgent(loja=loja, provedor=provedor, settings=settings)

app = FastAPI(title=settings.app_name, version=settings.app_version)


class IngestRequest(BaseModel):
    documentos: list[dict[str, str]]


class PerguntaRequest(BaseModel):
    pergunta: str


class PerguntaResponse(BaseModel):
    pergunta: str
    resposta: str
    citacoes: list[dict[str, str]]
    aprovado: bool
    recusado: bool
    motivo_recusa: str = ""


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "servico": settings.app_name}


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict[str, int]:
    documentos = [
        Documento(id=item["id"], texto=item["texto"], fonte=item["fonte"])
        for item in req.documentos
    ]
    chunks = ingestar(documentos, loja, provedor, settings.chunk_size, settings.chunk_overlap)
    return {"chunks_indexados": len(chunks)}


@app.post("/perguntar", response_model=PerguntaResponse)
def perguntar(req: PerguntaRequest) -> PerguntaResponse:
    resposta = agente.responder(req.pergunta)
    return PerguntaResponse(
        pergunta=resposta.pergunta,
        resposta=resposta.resposta,
        citacoes=[
            {"chunk_id": c.chunk_id, "fonte": c.fonte, "trecho": c.trecho}
            for c in resposta.citacoes
        ],
        aprovado=resposta.aprovado,
        recusado=resposta.recusado,
        motivo_recusa=resposta.motivo_recusa,
    )


def cli() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
