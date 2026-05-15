"""
main.py – FastAPI service for the SHL Assessment Recommender.

Endpoints:
  GET  /health  →  {"status": "ok"}
  POST /chat    →  {reply, recommendations, end_of_conversation}

Run locally:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

import sys
print("Python version:", sys.version, flush=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)
log = logging.getLogger(__name__)

# ── Lazy globals (populated at startup) ──────────────────────────────────────
_catalog: list[dict] = []
_retriever = None   # CatalogRetriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build catalog and retrieval index at startup."""
    global _catalog, _retriever
    from scraper import build_catalog
    from retriever import CatalogRetriever

    log.info("Building catalog …")
    _catalog = build_catalog()
    log.info("Catalog loaded: %d items", len(_catalog))

    log.info("Building retriever …")
    _retriever = CatalogRetriever(_catalog)
    log.info("Retriever ready")

    yield
    # teardown (nothing needed)


app = FastAPI(
    title="SHL Assessment Recommender",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1, max_length=16)

    @field_validator("messages")
    @classmethod
    def must_end_with_user(cls, msgs: list[Message]) -> list[Message]:
        if msgs[-1].role != "user":
            raise ValueError("Last message must have role='user'")
        return msgs

    @field_validator("messages")
    @classmethod
    def check_turn_cap(cls, msgs: list[Message]) -> list[Message]:
        total_turns = len(msgs)
        if total_turns > 8:
            raise ValueError("Conversation exceeds 8-turn limit")
        return msgs


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Readiness check. Returns 200 once catalog and index are loaded."""
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Service not ready yet")
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Stateless chat endpoint.
    Receives full conversation history; returns agent reply + optional shortlist.
    """
    if _retriever is None or not _catalog:
        raise HTTPException(status_code=503, detail="Service not ready yet")

    from agent import run_agent

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        reply, raw_recs, eoc = run_agent(messages, _retriever, _catalog)
    except Exception as exc:
        log.exception("Agent error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal agent error") from exc

    # Normalise recommendations to schema
    recommendations: list[Recommendation] = []
    for rec in raw_recs:
        test_type = rec.get("test_type", "")
        if isinstance(test_type, list):
            test_type = ", ".join(test_type)
        recommendations.append(
            Recommendation(
                name=rec.get("name", ""),
                url=rec.get("url", ""),
                test_type=test_type,
            )
        )

    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=eoc,
    )
