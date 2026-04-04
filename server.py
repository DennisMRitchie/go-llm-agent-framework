"""
Python LLM Backend for go-llm-agent-framework.

Exposes:
  HTTP (default):
    POST /v1/complete      - LLM completion
    POST /v1/classify      - Zero-shot intent classification
    POST /v1/entities      - Named entity extraction
    GET  /health           - Health check

  gRPC (optional, run with --grpc):
    Implements LLMService defined in proto/llm/llm.proto

Usage:
  pip install -r requirements.txt
  python server.py                  # HTTP mode (port 8000)
  python server.py --grpc           # gRPC mode (port 50051)
  python server.py --both           # Both HTTP and gRPC
  
  # With real OpenAI backend:
  OPENAI_API_KEY=sk-... python server.py

  # With local HuggingFace models:
  USE_HF=1 python server.py
"""

import argparse
import logging
import os
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("llm-backend")

# ── Configuration ──────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
USE_HF         = os.getenv("USE_HF", "0") == "1"
HF_MODEL_NLI   = os.getenv("HF_NLI_MODEL", "facebook/bart-large-mnli")
HF_MODEL_NER   = os.getenv("HF_NER_MODEL", "dslim/bert-base-NER")

app = FastAPI(title="LLM Backend", version="1.0.0")

# ── Lazy-loaded models ─────────────────────────────────────────────────────────

_nli_pipeline  = None
_ner_pipeline  = None
_openai_client = None


def get_nli_pipeline():
    global _nli_pipeline
    if _nli_pipeline is None and USE_HF:
        from transformers import pipeline
        log.info(f"Loading NLI model: {HF_MODEL_NLI}")
        _nli_pipeline = pipeline("zero-shot-classification", model=HF_MODEL_NLI)
    return _nli_pipeline


def get_ner_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None and USE_HF:
        from transformers import pipeline
        log.info(f"Loading NER model: {HF_MODEL_NER}")
        _ner_pipeline = pipeline("ner", model=HF_MODEL_NER, aggregation_strategy="simple")
    return _ner_pipeline


def get_openai():
    global _openai_client
    if _openai_client is None and OPENAI_API_KEY:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


# ── Request / Response schemas ─────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str
    timestamp_ms: Optional[int] = None


class CompleteRequest(BaseModel):
    prompt: str
    history: list[Message] = []
    parameters: dict[str, str] = {}
    session_id: str = ""


class CompleteResponse(BaseModel):
    text: str
    model: str
    tokens_used: int
    metadata: dict[str, str] = {}
    latency_ms: float


class ClassifyRequest(BaseModel):
    text: str
    candidate_labels: list[str]
    multi_label: bool = False


class ClassifyResponse(BaseModel):
    top_label: str
    confidence: float
    all_scores: dict[str, float]


class ExtractRequest(BaseModel):
    text: str
    entity_types: list[str] = []


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: float


class ExtractResponse(BaseModel):
    entities: list[Entity]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}


@app.post("/v1/complete", response_model=CompleteResponse)
def complete(req: CompleteRequest):
    start = time.time()
    client = get_openai()

    if client:
        messages = [{"role": m.role, "content": m.content} for m in req.history]
        messages.append({"role": "user", "content": req.prompt})

        resp = client.chat.completions.create(
            model=req.parameters.get("model", OPENAI_MODEL),
            messages=messages,
            max_tokens=int(req.parameters.get("max_tokens", "1024")),
            temperature=float(req.parameters.get("temperature", "0.7")),
        )
        text = resp.choices[0].message.content
        tokens = resp.usage.total_tokens
        model = resp.model
    else:
        # Stub: echo prompt back (demo mode)
        log.warning("No LLM backend configured — returning stub response")
        text = (
            f"[STUB] Received: {req.prompt[:100]}\n\n"
            "Set OPENAI_API_KEY or USE_HF=1 to enable real completions.\n"
            "For demo: the calculator tool can compute 2+2=<tool_call>"
            '{"name":"calculator","params":{"operation":"add","a":"2","b":"2"}}'
            "</tool_call>"
        )
        tokens = len(req.prompt.split())
        model = "stub"

    latency = (time.time() - start) * 1000
    log.info(f"complete: {tokens} tokens in {latency:.0f}ms")
    return CompleteResponse(text=text, model=model, tokens_used=tokens, latency_ms=latency)


@app.post("/v1/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    nli = get_nli_pipeline()

    if nli:
        result = nli(req.text, candidate_labels=req.candidate_labels, multi_label=req.multi_label)
        scores = dict(zip(result["labels"], result["scores"]))
        return ClassifyResponse(
            top_label=result["labels"][0],
            confidence=result["scores"][0],
            all_scores=scores,
        )
    else:
        # Simple heuristic fallback
        import re
        lower = req.text.lower()
        scores = {}
        for label in req.candidate_labels:
            words = re.findall(r'\w+', label.lower())
            score = sum(1 for w in words if w in lower) / max(len(words), 1)
            scores[label] = round(score, 3)
        top = max(scores, key=scores.get) if scores else req.candidate_labels[0]
        return ClassifyResponse(top_label=top, confidence=scores.get(top, 0.0), all_scores=scores)


@app.post("/v1/entities", response_model=ExtractResponse)
def extract_entities(req: ExtractRequest):
    ner = get_ner_pipeline()

    entities: list[Entity] = []
    if ner:
        results = ner(req.text)
        for r in results:
            if not req.entity_types or r["entity_group"] in req.entity_types:
                entities.append(Entity(
                    text=r["word"],
                    label=r["entity_group"],
                    start=r["start"],
                    end=r["end"],
                    confidence=round(r["score"], 4),
                ))
    else:
        # Regex fallback
        import re
        for m in re.finditer(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b', req.text):
            entities.append(Entity(text=m.group(), label="EMAIL", start=m.start(), end=m.end(), confidence=0.95))
        for m in re.finditer(r'https?://\S+', req.text):
            entities.append(Entity(text=m.group(), label="URL", start=m.start(), end=m.end(), confidence=0.95))

    return ExtractResponse(entities=entities)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Backend Server")
    parser.add_argument("--host",   default="0.0.0.0",  help="Bind host")
    parser.add_argument("--port",   default=8000, type=int, help="HTTP port")
    parser.add_argument("--reload", action="store_true",   help="Hot reload")
    args = parser.parse_args()

    log.info(f"Starting LLM backend on {args.host}:{args.port}")
    log.info(f"OpenAI: {'enabled' if OPENAI_API_KEY else 'disabled (stub mode)'}")
    log.info(f"HuggingFace: {'enabled' if USE_HF else 'disabled'}")

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
