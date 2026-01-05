from __future__ import annotations

import re
import time
import logging
from typing import Any, Dict, Tuple, Optional

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WIB_TZ = ZoneInfo("Asia/Jakarta")

AS_OF_RE = re.compile(
    r"\b(Today|Yesterday)\s+\w+\s+(\d{2}):(\d{2})\s+WIB\b",
    re.IGNORECASE,
)

STOCKBIT_SYMBOL_URL = "https://stockbit.com/symbol/{symbol}"
CACHE_TTL_SECONDS = 5

# Very small in-memory cache: {symbol: (expires_at_epoch, payload)}
_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

APP_STARTED_AT = datetime.now(WIB_TZ)

def _now_wib() -> datetime:
    return datetime.now(WIB_TZ)

def parse_stockbit_as_of(text: str, now: Optional[datetime] = None) -> Optional[datetime]:
    """
    Parses strings like:
      "Today Mon 09:15 WIB"
      "Yesterday Fri 15:30 WIB"
    Returns a WIB datetime with inferred date.
    """
    if not text:
        return None

    now = now or _now_wib()
    m = AS_OF_RE.search(" ".join(text.split()))
    if not m:
        return None

    rel = m.group(1).lower()
    hh = int(m.group(2))
    mm = int(m.group(3))

    base_date = now.date()
    if rel == "yesterday":
        base_date = (now - timedelta(days=1)).date()

    return datetime(base_date.year, base_date.month, base_date.day, hh, mm, tzinfo=WIB_TZ)

def _bs(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except ImportError:
        return BeautifulSoup(html, "html.parser")

def _extract_as_of_text_from_html(html: str) -> Optional[str]:
    """
    Extract the visible "Today Mon 09:15 WIB" (or Yesterday ...) line.
    Works with markup similar to:
      <span>Today <time>Mon 09:15 WIB</time></span>
    """
    soup = _bs(html)

    # Find a tag whose text starts with Today/Yesterday and contains a <time>
    def looks_like_asof(tag) -> bool:
        if not hasattr(tag, "get_text"):
            return False
        text = tag.get_text(" ", strip=True)
        if not text:
            return False
        lt = text.lower()
        if not (lt.startswith("today ") or lt.startswith("yesterday ")):
            return False
        return tag.find("time") is not None

    container = soup.find(looks_like_asof)
    if container is None:
        return None

    text = container.get_text(" ", strip=True)
    return " ".join(text.split())

def _parse_price_from_html(html: str) -> int:
    """
    Extracts the current price from the Stockbit symbol page HTML.

    Strategy:
    - Find the as-of container (Today/Yesterday + <time>)
    - Walk up ancestor nodes to find the nearest <h3> containing only digits (price)
    """
    soup = _bs(html)

    def looks_like_asof(tag) -> bool:
        if not hasattr(tag, "get_text"):
            return False
        text = tag.get_text(" ", strip=True)
        if not text:
            return False
        lt = text.lower()
        if not (lt.startswith("today ") or lt.startswith("yesterday ")):
            return False
        return tag.find("time") is not None

    marker = soup.find(looks_like_asof)
    if marker is None:
        # Fallback: any <time> tag
        marker = soup.find("time")

    if marker is None:
        raise ValueError("Could not locate a 'Today/Yesterday' or <time> marker in HTML (layout changed).")

    node = marker
    for _ in range(20):
        if node is None:
            break

        h3 = node.find("h3", string=re.compile(r"^\s*\d{1,9}\s*$"))
        if h3 is not None:
            return int(h3.get_text(strip=True))

        node = node.parent

    raise ValueError("Could not locate numeric price <h3> near marker (layout changed).")

async def _fetch_symbol_html(symbol: str) -> str:
    url = f"https://stockbit.com/symbol/{symbol.upper()}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
        "Connection": "keep-alive",
    }

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
    except httpx.TimeoutException as e:
        raise HTTPException(status_code=502, detail=f"Upstream timeout: {e!s}")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream network error: {e!s}")

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Upstream error: HTTP {resp.status_code}")

    return resp.text

def _ok(data: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": True, "data": data}

def _err(
    *,
    error_type: str,
    message: str,
    status_code: int,
    request: Optional[Request] = None,
    details: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    payload = {
        "ok": False,
        "error": {
            "type": error_type,
            "message": message,
            "details": details or {},
        },
        "meta": {
            "now_unix": int(time.time()),
            "path": str(request.url.path) if request is not None else None,
        },
    }
    return JSONResponse(status_code=status_code, content=payload)

app = FastAPI(title="Price API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    msg = detail if isinstance(detail, str) else "Request failed."
    return _err(
        error_type="HTTP_EXCEPTION",
        message=msg,
        status_code=exc.status_code,
        request=request,
        details={"detail": detail} if not isinstance(detail, str) else {},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return _err(
        error_type="VALIDATION_ERROR",
        message="Invalid request parameters.",
        status_code=422,
        request=request,
        details={"errors": exc.errors()},
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return _err(
        error_type="INTERNAL_ERROR",
        message="Unexpected server error.",
        status_code=500,
        request=request,
        details={"exception": exc.__class__.__name__},
    )

@app.get("/health")
async def health() -> Dict[str, Any]:
    now = _now_wib()
    return _ok(
        {
            "status": "ok",
            "service": "price-api",
            "now_wib": now.isoformat(),
            "started_at_wib": APP_STARTED_AT.isoformat(),
        }
    )

@app.get("/price")
async def get_price(symbol: str = Query(..., min_length=2, max_length=12)) -> Dict[str, Any]:
    sym = symbol.upper().strip()

    # cache
    now_epoch = time.time()
    cached = _cache.get(sym)
    if cached and cached[0] + CACHE_TTL_SECONDS > now_epoch:
        return cached[1]

    fetched_at = _now_wib()

    html = await _fetch_symbol_html(sym)

    # Parse as-of line (Today Mon 09:15 WIB)
    as_of_text = _extract_as_of_text_from_html(html)
    as_of_dt = parse_stockbit_as_of(as_of_text or "", now=fetched_at)

    try:
        price = _parse_price_from_html(html)
    except ValueError as e:
        # Layout changed or parsing failed: treat as upstream parsing issue
        raise HTTPException(status_code=502, detail=str(e))

    payload = _ok(
        {
            "symbol": sym,
            "price": price,
            "currency": "IDR",
            "as_of_text": as_of_text,  # e.g. "Today Mon 09:15 WIB"
            "as_of_unix": int(as_of_dt.timestamp()) if as_of_dt else None,
            "as_of_iso": as_of_dt.isoformat() if as_of_dt else None,
            "fetched_at_unix": int(fetched_at.timestamp()),
            "fetched_at_iso": fetched_at.isoformat(),
            "source": "stockbit.com",
        }
    )

    _cache[sym] = (now_epoch, payload)
    return payload


