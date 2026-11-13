"""Small OpenAlex client: API key, throttling, retries and an on-disk cache."""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from urllib.parse import urlencode

import requests

from .config import CACHE, OPENALEX_API_KEY

BASE = "https://api.openalex.org"


class BudgetExhausted(RuntimeError):
    pass


class OpenAlex:
    def __init__(self, api_key: str | None = OPENALEX_API_KEY, cache_dir: Path = CACHE,
                 max_rps: float = 8.0):
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.min_interval = 1.0 / max_rps
        self._lock = threading.Lock()
        self._last = 0.0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "a-graph/0.1 (researcher knowledge graph)"
        self.requests_made = 0
        self.cache_hits = 0
        self.cost_usd = 0.0
        self.remaining_usd: str | None = None
        self.newest = 0.0  # time of the newest data used (cache file mtime or fetch time)

    # -- helpers ---------------------------------------------------------------
    def _cache_path(self, path: str, params: dict) -> Path:
        key = path + "?" + urlencode(sorted(params.items()))
        h = hashlib.sha1(key.encode()).hexdigest()
        return self.cache_dir / h[:2] / f"{h}.json.gz"

    def _throttle(self):
        with self._lock:
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    # -- public ----------------------------------------------------------------
    def get(self, path: str, params: dict) -> dict:
        cp = self._cache_path(path, params)
        if cp.exists():
            with gzip.open(cp, "rt") as f:
                self.cache_hits += 1
                self.newest = max(self.newest, cp.stat().st_mtime)
                return json.load(f)

        q = dict(params)
        if self.api_key:
            q["api_key"] = self.api_key
        delay = 2.0
        for attempt in range(7):
            self._throttle()
            try:
                r = self.session.get(BASE + path, params=q, timeout=90)
            except requests.RequestException:
                time.sleep(delay)
                delay *= 2
                continue
            with self._lock:
                self.requests_made += 1
                self.cost_usd += float(r.headers.get("x-ratelimit-cost-usd", 0) or 0)
                self.remaining_usd = r.headers.get("x-ratelimit-remaining-usd", self.remaining_usd)
            if r.status_code == 200:
                data = r.json()
                cp.parent.mkdir(parents=True, exist_ok=True)
                tmp = cp.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")  # unique per writer
                self.newest = time.time()
                with gzip.open(tmp, "wt") as f:
                    json.dump(data, f)
                tmp.replace(cp)
                return data
            if r.status_code == 429:
                rem = r.headers.get("x-ratelimit-remaining-usd") or r.headers.get("x-ratelimit-remaining")
                if rem is not None and float(rem) <= 0:
                    raise BudgetExhausted(
                        "OpenAlex daily budget used up. Add OPENALEX_API_KEY to .env "
                        "(free key: https://openalex.org/settings/api) or retry after the reset; "
                        "cached responses are kept, so the run resumes where it stopped.")
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise RuntimeError(f"OpenAlex {r.status_code} for {path} {params}: {r.text[:300]}")
        raise RuntimeError(f"OpenAlex request kept failing: {path} {params}")

    def data_date(self) -> str:
        """Date of the newest OpenAlex data this client returned (ISO), today if none."""
        return dt.date.fromtimestamp(self.newest or time.time()).isoformat()

    def stats(self) -> str:
        s = f"{self.requests_made} API requests, {self.cache_hits} cache hits, ${self.cost_usd:.4f} spent"
        if self.remaining_usd is not None:
            s += f", ${float(self.remaining_usd):.4f} left today"
        return s


def reconstruct_abstract(inv: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the text."""
    if not inv:
        return ""
    pos = [(p, w) for w, ps in inv.items() for p in ps]
    pos.sort()
    return " ".join(w for _, w in pos)
