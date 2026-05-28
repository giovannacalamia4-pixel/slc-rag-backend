"""
Studio Legale Calamia — RAG Backend
FastAPI · Gemini AI · Pipeline RAG Giuridica

Avvia con:  uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import os
import json
import hashlib
import time
import asyncio
from typing import Optional, List, Dict, Any
from functools import lru_cache

import httpx
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config — tutte le chiavi SOLO nel server, mai nell'HTML
# ---------------------------------------------------------------------------
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")          # obbligatoria
STUDIO_API_TOKEN = os.getenv("STUDIO_API_TOKEN", "slc-dev") # token interno segreto

# Opzionale: Pinecone per vettori
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENV     = os.getenv("PINECONE_ENV", "")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "sentenze")

# Opzionale: Elasticsearch per keyword search
ES_URL           = os.getenv("ES_URL", "")     # es. http://localhost:9200
ES_INDEX         = os.getenv("ES_INDEX", "sentenze")

GEMINI_MODEL = "gemini-2.0-flash-lite"
GEMINI_URL_TPL   = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)

# ---------------------------------------------------------------------------
app = FastAPI(
    title="Studio Legale Calamia — RAG API",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # in produzione limitare al dominio dello studio
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Auth middleware leggero
# ---------------------------------------------------------------------------

def require_token(x_studio_token: str = Header(...)):
    if x_studio_token != STUDIO_API_TOKEN:
        raise HTTPException(status_code=401, detail="Token non autorizzato")
    return True


# ---------------------------------------------------------------------------
# Modelli Pydantic
# ---------------------------------------------------------------------------

class AiRequest(BaseModel):
    query: str = Field(..., max_length=4000)
    context: Optional[str] = None          # testo contesto (es. fascicolo aperto)
    mode: str = "chat"                     # "chat" | "sentenze" | "adempimenti" | "redattore"
    rg: Optional[str] = None
    tipo_atto: Optional[str] = None
    jurisdiction: Optional[str] = None

class SearchRequest(BaseModel):
    query: str = Field(..., max_length=500)
    jurisdiction: Optional[str] = None
    top_k: int = Field(5, ge=1, le=20)
    use_vectors: bool = False

class IndexRequest(BaseModel):
    """Per indicizzare nuovi documenti nel DB vettoriale interno."""
    doc_id: str
    text: str
    metadata: Dict[str, Any] = {}

class SentenzaResult(BaseModel):
    corte: str
    numero: str
    titolo: str
    massima: str
    score: float = 0.0
    source: str = "ai"   # "ai" | "vector_db" | "elasticsearch"


# ---------------------------------------------------------------------------
# Gemini helper
# ---------------------------------------------------------------------------

async def call_gemini(prompt: str, system: str = "", max_tokens: int = 1500) -> str:
    """Chiama Gemini con retry automatico."""
    if not GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY non configurata nel server.")

    url = GEMINI_URL_TPL.format(model=GEMINI_MODEL, key=GEMINI_API_KEY)
    contents = []
    if system:
        contents.append({"role": "user",  "parts": [{"text": system}]})
        contents.append({"role": "model", "parts": [{"text": "Capito. Agirò di conseguenza."}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})

   payload = {
        "contents": contents,
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.1},
    }

    async with httpx.AsyncClient(timeout=40) as client:
        for attempt in range(4):
            try:
                r = await client.post(url, json=payload)
                data = r.json()
                if "error" in data:
                    code = data["error"].get("code", 0)
                    msg  = data["error"].get("message", "Errore Gemini")
                    if code in (429, 503) and attempt < 3:
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    raise HTTPException(502, f"Gemini: {msg}")
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt < 3:
                    await asyncio.sleep(2)
                    continue
                raise HTTPException(504, "Timeout Gemini")
    raise HTTPException(502, "Gemini non raggiungibile dopo 4 tentativi")


# ---------------------------------------------------------------------------
# Vector DB helper (Pinecone — attivo solo se PINECONE_API_KEY è impostata)
# ---------------------------------------------------------------------------

async def vector_search(query: str, top_k: int = 5) -> List[Dict]:
    """Ricerca vettoriale su Pinecone (opzionale)."""
    if not PINECONE_API_KEY:
        return []
    try:
        # Genera embedding con Gemini Embedding API
        emb_url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"embedding-001:embedContent?key={GEMINI_API_KEY}"
        )
        async with httpx.AsyncClient(timeout=15) as client:
            er = await client.post(
                emb_url,
                json={"model": "models/embedding-001",
                      "content": {"parts": [{"text": query}]}}
            )
            embedding = er.json()["embedding"]["values"]

        # Query Pinecone
        pc_url = f"https://controller.{PINECONE_ENV}.pinecone.io"
        idx_host_r = await client.get(
            f"{pc_url}/databases/{PINECONE_INDEX}",
            headers={"Api-Key": PINECONE_API_KEY}
        )
        idx_host = idx_host_r.json().get("status", {}).get("host", "")
        if not idx_host:
            return []

        qr = await client.post(
            f"https://{idx_host}/query",
            headers={"Api-Key": PINECONE_API_KEY,
                     "Content-Type": "application/json"},
            json={"vector": embedding, "topK": top_k, "includeMetadata": True}
        )
        matches = qr.json().get("matches", [])
        return [
            {
                "corte":   m["metadata"].get("corte", "—"),
                "numero":  m["metadata"].get("numero", "—"),
                "titolo":  m["metadata"].get("titolo", "—"),
                "massima": m["metadata"].get("massima", "—"),
                "score":   round(m["score"], 3),
                "source":  "vector_db",
            }
            for m in matches
        ]
    except Exception as e:
        print(f"[vector_search] errore ignorato: {e}")
        return []


async def elasticsearch_search(query: str, top_k: int = 5) -> List[Dict]:
    """Ricerca keyword su Elasticsearch (opzionale)."""
    if not ES_URL:
        return []
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"{ES_URL}/{ES_INDEX}/_search",
                json={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["titolo^3", "massima^2", "testo"],
                        }
                    },
                    "size": top_k,
                }
            )
            hits = r.json().get("hits", {}).get("hits", [])
            return [
                {
                    "corte":   h["_source"].get("corte", "—"),
                    "numero":  h["_source"].get("numero", "—"),
                    "titolo":  h["_source"].get("titolo", "—"),
                    "massima": h["_source"].get("massima", "—"),
                    "score":   round(h["_score"], 3),
                    "source":  "elasticsearch",
                }
                for h in hits
            ]
    except Exception as e:
        print(f"[es_search] errore ignorato: {e}")
        return []


# ---------------------------------------------------------------------------
# RAG core — assembla contesto per Gemini
# ---------------------------------------------------------------------------

async def rag_search(query: str, jurisdiction: str | None, top_k: int, use_vectors: bool) -> List[Dict]:
    """
    Strategia a cascata:
    1. Vettori (Pinecone) se disponibili
    2. Keyword (Elasticsearch) se disponibile
    3. Fallback: Gemini genera orientamenti noti (no numeri inventati)
    """
    results: List[Dict] = []

    if use_vectors:
        vec = await vector_search(query, top_k)
        results.extend(vec)

    if len(results) < top_k:
        es = await elasticsearch_search(query, top_k - len(results))
        results.extend(es)

    # Fallback AI se i DB non sono configurati o vuoti
    if not results:
        jur_str = f" (focus: {jurisdiction})" if jurisdiction and jurisdiction != "Tutte le giurisdizioni" else ""
        ai_prompt = (
            f"Sei un esperto di giurisprudenza italiana. "
            f"Fornisci {top_k} orientamenti giurisprudenziali pertinenti a: \"{query}\"{jur_str}.\n"
            "REGOLA FERREA: non inventare numeri di sentenze specifici. "
            "Indica invece il periodo (es. 'orientamento consolidato 2020-2024') o SS.UU.\n"
            f"Rispondi SOLO con JSON array (niente markdown):\n"
            '[{"corte":"Cass. civ.","numero":"orientamento 2022-2024",'
            '"titolo":"...","massima":"..."}]'
        )
        raw = await call_gemini(ai_prompt, max_tokens=1400)
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            arr = json.loads(raw)
        except Exception:
            m = __import__("re").search(r"\[[\s\S]*\]", raw)
            arr = json.loads(m.group()) if m else []
        results = [{**s, "source": "ai", "score": 0.9} for s in arr]

    return results[:top_k]


# ---------------------------------------------------------------------------
# Endpoint: /api/sentenze   — ricerca sentenze RAG
# ---------------------------------------------------------------------------

@app.post("/api/sentenze", response_model=List[SentenzaResult])
async def search_sentenze(
    req: SearchRequest,
    _auth: bool = Depends(require_token),
):
    results = await rag_search(
        query=req.query,
        jurisdiction=req.jurisdiction,
        top_k=req.top_k,
        use_vectors=req.use_vectors,
    )
    return [SentenzaResult(**r) for r in results]


# ---------------------------------------------------------------------------
# Endpoint: /api/ai/chat    — chat giuridica con RAG contestuale
# ---------------------------------------------------------------------------

@app.post("/api/ai/chat")
async def ai_chat(
    req: AiRequest,
    _auth: bool = Depends(require_token),
):
    # 1. Recupera contesto sentenze per la query
    context_docs = await rag_search(req.query, req.jurisdiction, top_k=4, use_vectors=False)
    context_block = ""
    if context_docs:
        context_block = "\n\n--- ORIENTAMENTI GIURISPRUDENZIALI RILEVANTI ---\n"
        for i, d in enumerate(context_docs, 1):
            context_block += (
                f"{i}. {d['corte']} — {d['numero']}: «{d['massima']}»\n"
            )
        context_block += "--- FINE CONTESTO ---\n"

    fascicolo_ctx = ""
    if req.context:
        fascicolo_ctx = f"\n\nCONTESTO FASCICOLO APERTO:\n{req.context[:800]}\n"

    system = (
        "Sei un assistente legale italiano esperto, al servizio dello Studio Legale Calamia "
        "(Avv. Mariagiovanna Calamia, Foro di Palermo, specializzazione diritto civile). "
        "Cita sempre le fonti fornite nel contesto. Non inventare numeri di sentenze. "
        "Usa linguaggio professionale e preciso. Rispondi in italiano."
    )

    prompt = (
        f"{fascicolo_ctx}{context_block}\n"
        f"DOMANDA DELL'AVVOCATO:\n{req.query}"
    )

    answer = await call_gemini(prompt, system=system, max_tokens=1500)

    # Costruisce le citazioni estratte dal contesto
    citations = [
        {"corte": d["corte"], "numero": d["numero"], "titolo": d["titolo"]}
        for d in context_docs
    ]

    return {"answer": answer, "citations": citations, "context_used": bool(context_block)}


# ---------------------------------------------------------------------------
# Endpoint: /api/ai/adempimenti  — calcolo termini processuali RAG-enhanced
# ---------------------------------------------------------------------------

@app.post("/api/ai/adempimenti")
async def ai_adempimenti(
    req: AiRequest,
    _auth: bool = Depends(require_token),
):
    oggi = time.strftime("%d/%m/%Y")
    tipo = req.tipo_atto or "atto generico"
    rg   = req.rg or "da definire"

    # Recupera sentenze/prassi sugli adempimenti per quel tipo atto
    ctx_docs = await rag_search(
        f"termini processuali {tipo} Riforma Cartabia", None, top_k=2, use_vectors=False
    )
    jur_ctx = ""
    if ctx_docs:
        jur_ctx = "\n".join(f"- {d['corte']}: {d['massima']}" for d in ctx_docs)
        jur_ctx = f"\nPrassi rilevante:\n{jur_ctx}\n"

    system = (
        "Sei un avvocato processualcivilista esperto italiano. "
        "Calcola termini perentori con riferimento a c.p.c. e D.Lgs. 149/2022 (Riforma Cartabia). "
        "Non inventare. Se non sei sicuro di una data, indica 'da verificare'."
    )
    prompt = (
        f"Atto: \"{tipo}\" — R.G. {rg} — depositato il {req.query or oggi}\n"
        f"{jur_ctx}"
        "Calcola i termini successivi. Rispondi SOLO con JSON:\n"
        '{"adempimenti":[{"num":1,"desc":"descrizione con norma","scadenza":"GG/MM/AAAA",'
        '"tipo":"perentorio|ordinatorio"}]}'
        "\nMassimo 8 adempimenti ordinati per data."
    )

    raw = await call_gemini(prompt, system=system, max_tokens=1000)
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        obj = json.loads(raw)
    except Exception:
        import re
        m = re.search(r"\{[\s\S]*\}", raw)
        obj = json.loads(m.group()) if m else {"adempimenti": []}

    return obj


# ---------------------------------------------------------------------------
# Endpoint: /api/ai/redattore  — suggerimenti per il redattore atti
# ---------------------------------------------------------------------------

@app.post("/api/ai/redattore")
async def ai_redattore(
    req: AiRequest,
    _auth: bool = Depends(require_token),
):
    tipo   = req.tipo_atto or "atto generico"
    mode   = req.mode      # "strategia" | "norme" | "giurisprudenza"
    query  = req.query

    # Ricerca RAG per giurisprudenza pertinente
    docs = await rag_search(f"{tipo} {query}", req.jurisdiction, top_k=3, use_vectors=False)
    ctx = "\n".join(f"- {d['corte']} {d['numero']}: {d['massima']}" for d in docs)

    system = "Sei un avvocato civilista senior del Foro di Palermo. Sii conciso e pratico. Max 200 parole."

    if mode == "strategia":
        prompt = (
            f"Per un \"{tipo}\" nella causa: {query}\n"
            f"Contesto giurisprudenziale:\n{ctx}\n\n"
            "Suggerisci 3-4 eccezioni preliminari e strategie difensive concrete, citando norme c.p.c."
        )
    elif mode == "norme":
        prompt = (
            f"Per un \"{tipo}\": {query}\n"
            "Elenca le norme principali di c.c. e c.p.c. con articoli precisi e 2-3 massime pertinenti.\n"
            f"Contesto:\n{ctx}"
        )
    else:  # giurisprudenza
        prompt = (
            f"Per un \"{tipo}\": {query}\n"
            f"Ecco orientamenti rilevanti trovati:\n{ctx}\n\n"
            "Sintetizza i 3 precedenti più favorevoli con corte, periodo e massima breve."
        )

    answer = await call_gemini(prompt, system=system, max_tokens=700)
    return {
        "answer": answer,
        "citations": [{"corte": d["corte"], "numero": d["numero"]} for d in docs],
    }


# ---------------------------------------------------------------------------
# Endpoint: /api/index/document  — indicizza un documento nel DB interno
# ---------------------------------------------------------------------------

@app.post("/api/index/document", status_code=201)
async def index_document(
    req: IndexRequest,
    _auth: bool = Depends(require_token),
):
    """
    Indicizza un documento (atto, sentenza) per la ricerca vettoriale.
    In produzione scrive su Pinecone o Elasticsearch.
    """
    if not PINECONE_API_KEY and not ES_URL:
        return {"status": "skipped", "reason": "Nessun DB vettoriale configurato"}

    doc_hash = hashlib.sha256(req.text.encode()).hexdigest()[:16]

    if ES_URL:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.put(
                f"{ES_URL}/{ES_INDEX}/_doc/{req.doc_id}",
                json={**req.metadata, "testo": req.text, "doc_id": req.doc_id},
            )

    return {"status": "indexed", "doc_id": req.doc_id, "hash": doc_hash}


# ---------------------------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "gemini_configured": bool(GEMINI_API_KEY),
        "pinecone_configured": bool(PINECONE_API_KEY),
        "elasticsearch_configured": bool(ES_URL),
    }
