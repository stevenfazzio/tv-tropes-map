"""Qwen3-Embedding-8B through OpenRouter, pinned to one provider.

Implements Toponymy's TextEmbedderProtocol (`encode(texts, show_progress_bar)`), so
the same object embeds the trope descriptions and, later, Toponymy's keyphrases
and topic names. Every vector is cached on disk, keyed by model, provider, and the
exact text sent (instruction included): re-running any stage never pays for a
string twice.

Requests are pinned to one provider with fallbacks off. The same text embedded by
two of OpenRouter's providers came back at cosine 0.996 rather than 1.0, so a map
built from a mix of providers would carry that noise, and Toponymy's later
embeddings have to come from the provider that embedded the documents.
"""

import hashlib
import os
import random
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from tqdm import tqdm

from common import DATA

API_URL = "https://openrouter.ai/api/v1/embeddings"
MODEL = "qwen/qwen3-embedding-8b"
PROVIDER = "DeepInfra"
CACHE_PATH = DATA / "embedding_cache.sqlite"
# A batch closes at whichever limit it reaches first.
BATCH_ITEMS = 32
BATCH_CHARS = 120_000
MAX_WORKERS = 4
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


def with_instruction(instruction: str | None, text: str) -> str:
    # The format from the Qwen3-Embedding model card.
    return f"Instruct: {instruction}\nQuery:{text}" if instruction else text


class OpenRouterEmbedder:
    def __init__(
        self,
        instruction: str | None = None,
        provider: str = PROVIDER,
        cache_path=CACHE_PATH,
        max_workers: int = MAX_WORKERS,
    ):
        self.instruction = instruction
        self.provider = provider
        self.max_workers = max_workers
        self.tokens = 0
        self.cost_usd = 0.0
        self.session = requests.Session()
        self.session.headers["Authorization"] = (
            f"Bearer {os.environ['OPENROUTER_API_KEY']}"
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)"
        )

    def _key(self, sent: str) -> str:
        return hashlib.sha256(
            f"{MODEL}\x00{self.provider}\x00{sent}".encode()
        ).hexdigest()

    def _batches(self, texts: list[str]):
        batch, chars = [], 0
        for t in texts:
            if batch and (len(batch) == BATCH_ITEMS or chars + len(t) > BATCH_CHARS):
                yield batch
                batch, chars = [], 0
            batch.append(t)
            chars += len(t)
        if batch:
            yield batch

    def _embed_remote(
        self, texts: list[str], max_retries: int = 6
    ) -> tuple[np.ndarray, dict]:
        payload = {
            "model": MODEL,
            "input": texts,
            "encoding_format": "float",
            "provider": {"order": [self.provider], "allow_fallbacks": False},
        }
        for attempt in range(max_retries):
            wait = min(2**attempt * 5, 120) + random.uniform(0, 3)
            try:
                r = self.session.post(API_URL, json=payload, timeout=300)
                if r.status_code in RETRY_STATUS and attempt < max_retries - 1:
                    print(
                        f"  {r.status_code}: {r.text[:200]!r}; retrying in {wait:.0f}s"
                    )
                    time.sleep(wait)
                    continue
                if not r.ok:  # raise_for_status() would drop OpenRouter's message
                    raise requests.HTTPError(
                        f"{r.status_code} from OpenRouter: {r.text[:500]}", response=r
                    )
                body = r.json()
                if "data" not in body:  # an error reported inside a 200
                    if attempt < max_retries - 1:
                        print(
                            f"  no data in response: {str(body)[:200]!r}; retrying in {wait:.0f}s"
                        )
                        time.sleep(wait)
                        continue
                    raise RuntimeError(f"no data in response: {str(body)[:500]}")
                assert body.get("provider") == self.provider, (
                    f"served by {body.get('provider')}, not {self.provider}"
                )
                data = sorted(body["data"], key=lambda d: d["index"])
                assert len(data) == len(texts), (
                    f"{len(data)} vectors for {len(texts)} texts"
                )
                return np.asarray(
                    [d["embedding"] for d in data], dtype=np.float32
                ), body.get("usage", {})
            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt == max_retries - 1:
                    raise
                print(f"  attempt {attempt + 1}: {e!r}; retrying in {wait:.0f}s")
                time.sleep(wait)
        raise RuntimeError("exhausted retries")

    def encode(
        self, texts, show_progress_bar=None, verbose=None, **kwargs
    ) -> np.ndarray:
        # The API rejects an empty string, and Toponymy embeds one whenever naming
        # a region fails, so an empty text is sent as a single space.
        sent = [with_instruction(self.instruction, t) or " " for t in texts]
        keys = [self._key(s) for s in sent]
        cached = {}
        for i in range(0, len(keys), 900):  # stay under SQLite's variable limit
            chunk = keys[i : i + 900]
            rows = self.db.execute(
                f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(chunk))})",
                chunk,
            )
            cached.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})

        # Dedupe so repeated strings (common among keyphrases) are embedded once.
        todo = list(dict.fromkeys(s for s, k in zip(sent, keys) if k not in cached))
        if todo:
            pool = ThreadPoolExecutor(self.max_workers)
            futures = {
                pool.submit(self._embed_remote, b): b for b in self._batches(todo)
            }
            bar = tqdm(
                total=len(todo),
                desc="embedding",
                disable=not (show_progress_bar or verbose),
            )
            try:
                # Network calls run in worker threads; the SQLite connection stays
                # on this thread, committed per batch, so a crash loses at most the
                # batches still in flight.
                for future in as_completed(futures):
                    batch = futures[future]
                    vecs, usage = future.result()
                    rows = [(self._key(s), v.tobytes()) for s, v in zip(batch, vecs)]
                    with self.db:
                        self.db.executemany(
                            "INSERT OR REPLACE INTO emb VALUES (?, ?)", rows
                        )
                    cached.update({k: v for (k, _), v in zip(rows, vecs)})
                    self.tokens += usage.get("prompt_tokens", 0)
                    self.cost_usd += usage.get("cost", 0.0)
                    bar.update(len(batch))
            finally:
                bar.close()
                pool.shutdown(wait=True, cancel_futures=True)

        return np.vstack([cached[k] for k in keys])
