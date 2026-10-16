"""Templates de prompt para RAG e camadas de validação."""

from __future__ import annotations

PROMPT_RESPOSTA_RAG = """Você é um agente de suporte bancário. Responda a pergunta do cliente usando APENAS o contexto fornecido abaixo.

Regras obrigatórias:
- Responda em português do Brasil.
- Use somente informações presentes no contexto. Não invente dados, saldos, taxas ou prazos.
- Se a resposta não estiver no contexto, diga exatamente: "Não tenho informações suficientes nos documentos disponíveis para responder a essa pergunta."
- Sempre que usar uma informação do contexto, cite a fonte entre colchetes, ex: [fonte: documento_id::chunk].
- Não forneça conselho de investimento, não peça senhas e não garanta aprovação de crédito.

Contexto:
{contexto}

Pergunta:
{pergunta}

Resposta:
"""

PROMPT_CAMADA_SAIDA = """Verifique se a resposta a seguir é derivada exclusivamente do contexto recuperado.

Contexto:
{contexto}

Resposta:
{resposta}

A resposta contém afirmações que NÃO estão no contexto? Responda apenas "SIM" ou "NÃO".
"""

PROMPT_CAMADA_FONTE = """Verifique se toda citação na resposta aponta para um chunk efetivamente recuperado.

Citações na resposta:
{citacoes}

Chunks recuperados (id :: fonte):
{chunks}

Todas as citações são válidas? Responda apenas "SIM" ou "NÃO".
"""

PROMPT_CAMADA_CONSISTENCIA = """Verifique se a resposta é consistente com os chunks recuperados (sem contradições).

Chunks:
{chunks}

Resposta:
{resposta}

Há contradição entre resposta e chunks? Responda apenas "SIM" ou "NÃO".
"""

PROMPT_CAMADA_POLITICA = """Verifique se a resposta respeita políticas de suporte bancário.

Proibições:
- Não prometer aprovação de crédito ou financiamento.
- Não informar saldos, limites ou taxas específicos sem fonte nos documentos.
- Não pedir senha, token ou dados sensíveis do cliente.
- Não dar conselho de investimento.

Resposta:
{resposta}

A resposta viola alguma política? Responda apenas "SIM" ou "NÃO".
"""
