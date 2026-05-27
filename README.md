# Studio Legale Calamia — RAG Backend
## Guida all'Installazione e Configurazione

---

## Architettura

```
Browser (HTML) ─── rag-bridge.js ──► FastAPI Backend ──► Gemini API
                                          │                 (chiave sicura)
                                          ├──► Pinecone (vettori) [opz.]
                                          └──► Elasticsearch [opz.]
```

La chiave Gemini è **solo sul server**. Il browser non la vede mai.

---

## Avvio Rapido (sviluppo locale)

### Prerequisiti
- Python 3.11+ oppure Docker
- Una chiave Gemini da [Google AI Studio](https://aistudio.google.com/app/apikey)

### 1. Configura l'ambiente

```bash
cd rag-backend
cp .env.example .env
# Apri .env e inserisci la tua GEMINI_API_KEY
nano .env
```

### 2. Installa e avvia

**Con Python diretto:**
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Con Docker:**
```bash
docker compose up --build
```

Il backend sarà disponibile su `http://localhost:8000`
Documentazione API interattiva: `http://localhost:8000/docs`

### 3. Collega il frontend

Apri `StudioLegaleCalamia_v9_fixed.html` e aggiungi **prima di `</body>`**:

```html
<script src="rag-bridge.js"></script>
```

Se i due file sono nella stessa cartella, funziona subito.

---

## Endpoint API

| Metodo | Path | Descrizione |
|--------|------|-------------|
| `POST` | `/api/sentenze` | Ricerca sentenze con RAG |
| `POST` | `/api/ai/chat` | Chat AI con contesto giuridico |
| `POST` | `/api/ai/adempimenti` | Calcolo termini processuali |
| `POST` | `/api/ai/redattore` | Suggerimenti per redazione atti |
| `POST` | `/api/index/document` | Indicizza documento nel DB |
| `GET`  | `/api/health` | Stato del backend |

Tutti gli endpoint (tranne health) richiedono l'header:
```
x-studio-token: <STUDIO_API_TOKEN>
```

---

## Pipeline RAG — Come Funziona

```
Query utente
     │
     ▼
[rag_search()]
     │
     ├── 1. Pinecone (vettori) — se PINECONE_API_KEY configurata
     │        └─ Embedding Gemini → query vettoriale → top-k chunks
     │
     ├── 2. Elasticsearch (keyword) — se ES_URL configurata
     │        └─ multi_match su titolo/massima/testo
     │
     └── 3. Fallback AI (sempre disponibile)
              └─ Gemini genera orientamenti noti (no numeri inventati)
                   │
                   ▼
             Contesto assemblato
                   │
                   ▼
          [call_gemini(prompt + contesto)]
                   │
                   ▼
             Risposta con citazioni fonti
```

---

## Configurazione DB Vettoriale (Pinecone) — Opzionale

Per indicizzare e cercare sentenze reali nei tuoi documenti:

1. Crea account su [pinecone.io](https://www.pinecone.io)
2. Crea un indice con dimensione **768** (embedding Gemini)
3. Aggiungi al `.env`:
   ```
   PINECONE_API_KEY=pc-...
   PINECONE_ENV=us-east-1-aws
   PINECONE_INDEX=sentenze
   ```
4. Per indicizzare una sentenza, chiama:
   ```bash
   curl -X POST http://localhost:8000/api/index/document \
     -H "x-studio-token: slc-dev" \
     -H "Content-Type: application/json" \
     -d '{
       "doc_id": "cass-12345-2024",
       "text": "Testo completo della sentenza...",
       "metadata": {
         "corte": "Cass. civ.",
         "numero": "12345/2024",
         "titolo": "Responsabilità contrattuale",
         "massima": "..."
       }
     }'
   ```

---

## Configurazione Elasticsearch — Opzionale

Per grandi volumi (>1000 sentenze):

```bash
# Con Docker:
docker run -d --name es \
  -e discovery.type=single-node \
  -e xpack.security.enabled=false \
  -p 9200:9200 \
  docker.elastic.co/elasticsearch/elasticsearch:8.13.0
```

Aggiungi al `.env`:
```
ES_URL=http://localhost:9200
ES_INDEX=sentenze
```

---

## Deploy in Produzione

### Opzione A — VPS (Hetzner, DigitalOcean, Contabo)

```bash
# Sul server
git clone ... && cd rag-backend
cp .env.example .env  # compila con dati reali
docker compose up -d

# Nginx come reverse proxy
server {
    listen 443 ssl;
    server_name api.studiolegalecalamia.it;
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
    }
}
```

### Opzione B — Railway.app (più semplice, gratuito fino a 5$/mese)

1. Vai su [railway.app](https://railway.app)
2. "New Project" → "Deploy from GitHub"
3. Aggiungi le variabili d'ambiente dalla UI
4. Railway assegna automaticamente un URL `https://...railway.app`

### Dopo il deploy: aggiorna `rag-bridge.js`

```javascript
var RAG_CONFIG = {
  baseUrl: "https://api.studiolegalecalamia.it",  // ← URL reale
  token: "TOKEN_SEGRETO_32_CARATTERI",              // ← token sicuro
  allowDirectFallback: false,                        // ← disabilita in prod
};
```

---

## Sicurezza

- ✅ Chiave Gemini **mai** nel browser
- ✅ Token di autenticazione su ogni richiesta
- ✅ CORS configurabile per dominio specifico
- ✅ Rate limiting Gemini gestito lato server con retry
- ⚠️ In produzione: cambia `STUDIO_API_TOKEN` con `openssl rand -hex 32`
- ⚠️ In produzione: limita CORS al solo dominio dello studio

---

## Troubleshooting

**"Backend non raggiungibile"**
- Controlla che `uvicorn` sia in esecuzione (`lsof -i :8000`)
- Verifica CORS se usi un dominio diverso da localhost

**"GEMINI_API_KEY non configurata"**
- Assicurati che il file `.env` sia nella stessa cartella di `main.py`
- Usa `python-dotenv`: aggiungi `from dotenv import load_dotenv; load_dotenv()` all'inizio di `main.py`

**Risposte lente**
- Gemini Flash è ottimizzato per velocità; se usi Pro aspettati 3-5s
- Pinecone aggiunge ~200ms ma migliora la qualità del contesto

---

*Studio Legale Calamia — Via Vittorio Veneto 268, Alcamo (TP)*
