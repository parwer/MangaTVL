from __future__ import annotations

import datetime as dt
import os
import secrets
import sqlite3
import time
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

UPSTREAM = os.getenv("RELAY_UPSTREAM", "https://openrouter.ai/api/v1")
OWNER_KEY = os.getenv("OWNER_OPENROUTER_KEY", "")
DAILY_PAGES = int(os.getenv("RELAY_DAILY_PAGES", "60"))
ALLOWED_MODELS = set(os.getenv("RELAY_MODELS", "google/gemini-2.5-flash-lite,google/gemini-3-flash-preview").split(","))
DEFAULT_MODEL = os.getenv("RELAY_DEFAULT_MODEL", "google/gemini-2.5-flash-lite")
MAX_TOKENS = 2500
DB = Path(os.getenv("RELAY_DB", Path(__file__).with_name("relay.sqlite3")))

app = FastAPI(title="MangaTVL relay")
_ip_hits: dict[str, list[float]] = {}


def db():
    con = sqlite3.connect(DB)
    con.execute("create table if not exists tokens(token text primary key, created real)")
    con.execute("create table if not exists usage(token text, day text, n int, primary key(token, day))")
    return con


def today() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d")


@app.post("/v1/register")
def register(request: Request):
    ip = request.client.host if request.client else "?"
    hits = [t for t in _ip_hits.get(ip, []) if time.time() - t < 3600]
    if len(hits) >= 5:
        raise HTTPException(429, "too many registrations from this address")
    _ip_hits[ip] = hits + [time.time()]
    tok = "mtvl_" + secrets.token_urlsafe(24)
    with db() as con:
        con.execute("insert into tokens values(?, ?)", (tok, time.time()))
    return {"token": tok, "daily_pages": DAILY_PAGES}


def _check(token: str | None) -> str:
    if not token or not token.startswith("Bearer "):
        raise HTTPException(401, "missing device token")
    tok = token[7:]
    with db() as con:
        if not con.execute("select 1 from tokens where token=?", (tok,)).fetchone():
            raise HTTPException(401, "unknown device token")
    return tok


@app.get("/v1/quota")
def quota(authorization: str | None = Header(None)):
    tok = _check(authorization)
    with db() as con:
        row = con.execute("select n from usage where token=? and day=?", (tok, today())).fetchone()
    return {"used": row[0] if row else 0, "limit": DAILY_PAGES}


@app.post("/v1/chat/completions")
async def chat(request: Request, authorization: str | None = Header(None)):
    tok = _check(authorization)
    body = await request.json()
    is_page = any(isinstance(m.get("content"), list) for m in body.get("messages", []))  # page = has an image
    with db() as con:
        row = con.execute("select n from usage where token=? and day=?", (tok, today())).fetchone()
        used = row[0] if row else 0
        if is_page and used >= DAILY_PAGES:
            raise HTTPException(429, f"daily free quota reached ({DAILY_PAGES} pages) — sign in with OpenRouter for unlimited")
    if body.get("model") not in ALLOWED_MODELS:
        body["model"] = DEFAULT_MODEL  # the owner decides what the free tier costs
    body["max_tokens"] = min(int(body.get("max_tokens") or MAX_TOKENS), MAX_TOKENS)
    if not OWNER_KEY:
        raise HTTPException(503, "relay not configured")
    async with httpx.AsyncClient(timeout=120) as cli:
        r = await cli.post(f"{UPSTREAM}/chat/completions", json=body,
                           headers={"Authorization": f"Bearer {OWNER_KEY}", "X-Title": "MangaTVL relay"})
    if r.status_code == 200 and is_page:
        with db() as con:
            con.execute("insert into usage values(?, ?, 1) on conflict(token, day) do update set n=n+1", (tok, today()))
    return JSONResponse(r.json(), status_code=r.status_code)


if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("RELAY_HOST", "127.0.0.1"), port=int(os.getenv("RELAY_PORT", "8787")))
