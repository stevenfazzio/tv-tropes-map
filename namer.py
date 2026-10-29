"""Toponymy LLM wrappers: Claude through the Anthropic SDK, and a dry run for costing.

Toponymy's own AnthropicNamer goes through LiteLLM and always sends `temperature`,
which the Messages API rejects on every model as of API version 2026-10-21, with a
small `max_tokens` (128 for a topic name) that leaves no room for adaptive
thinking. This wrapper sends neither and gives each call room to think.

`llm_specific_instructions` is appended to every user prompt, topic naming and
disambiguation alike, as Toponymy's own wrappers do with it.

Toponymy drives async wrappers with a fresh `asyncio.run()` per stage, so the
client and semaphore are rebuilt whenever the running event loop changes; both
bind to the loop they were first used on. Successful responses are cached on disk
by prompt, so re-running a fit never pays twice for the same prompt.
"""

import asyncio
import hashlib
import json
import re
import sqlite3

import anthropic
from toponymy.llm_wrappers import AsyncLLMWrapper

from common import DATA

CACHE_PATH = DATA / "llm_cache.sqlite"
MODEL = "claude-opus-5-5"
# Dollars per million tokens (input, output), for the cost line printed after a fit.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}
# Models that accept the server-side refusal fallback; Haiku 5.5 has none.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}


class AsyncClaudeNamer(AsyncLLMWrapper):
    # Errors an identical retry can't fix; Toponymy's base class fails fast on these.
    FAIL_FAST_EXCEPTIONS = (
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.BadRequestError,
        anthropic.NotFoundError,
    )

    def __init__(
        self,
        model: str = MODEL,
        effort: str = "medium",
        max_concurrent_requests: int = 8,
        max_tokens: int = 16000,
        llm_specific_instructions: str | None = None,
        cache_path=CACHE_PATH,
    ):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_concurrent_requests = max_concurrent_requests
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )
        self.callback = None
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "cached": 0}
        self._loop = None
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS llm (key TEXT PRIMARY KEY, text TEXT)"
        )

    def _resources(self):
        loop = asyncio.get_running_loop()
        if loop is not self._loop:
            self._loop = loop
            self._client = anthropic.AsyncAnthropic(max_retries=4)
            self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        return self._client, self._semaphore

    def _key(self, system: str | None, user: str) -> str:
        raw = "\x00".join([self.model, self.effort, system or "", user])
        return hashlib.sha256(raw.encode()).hexdigest()

    async def _create(self, system: str | None, user: str) -> str:
        key = self._key(system, user)
        row = self.db.execute("SELECT text FROM llm WHERE key = ?", (key,)).fetchone()
        if row is not None:
            self.usage["cached"] += 1
            return row[0]

        client, semaphore = self._resources()
        kwargs = {"system": system} if system else {}
        if self.model in FALLBACK_MODELS:
            kwargs |= {
                "betas": ["server-side-fallback-2026-07-01"],
                "fallbacks": "default",
            }
        async with semaphore:
            msg = await client.beta.messages.create(
                model=self.model,
                # Toponymy's per-call max_tokens sizes only the visible answer; it
                # would truncate thinking, so the wrapper sets its own ceiling.
                max_tokens=self.max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": user}],
                **kwargs,
            )
        self.usage["calls"] += 1
        self.usage["input_tokens"] += msg.usage.input_tokens
        self.usage["output_tokens"] += msg.usage.output_tokens
        if msg.stop_reason == "refusal":
            raise RuntimeError(f"refusal: {msg.stop_details}")
        text = next((b.text for b in msg.content if b.type == "text"), None)
        if text is None:
            raise RuntimeError(f"no text block (stop_reason={msg.stop_reason})")
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO llm VALUES (?, ?)", (key, text))
        return text

    def cost_usd(self) -> float:
        price_in, price_out = PRICES[self.model]
        return (
            self.usage["input_tokens"] * price_in
            + self.usage["output_tokens"] * price_out
        ) / 1e6

    async def _call_single_llm(self, prompt, temperature, max_tokens) -> str:
        return await self._create(None, prompt["combined"] + self.extra_prompting)

    async def _call_single_llm_with_system(
        self, prompt, temperature, max_tokens
    ) -> str:
        return await self._create(
            prompt["system"], prompt["user"] + self.extra_prompting
        )


class DryRunNamer(AsyncLLMWrapper):
    """Records every prompt a fit would send and answers without calling a model.

    A topic-name prompt gets a placeholder name built from the region's own
    keywords, so placeholders differ from one another about as much as real names
    would and the disambiguation pass groups them about as often. A cluster-renaming
    (disambiguation) prompt gets an empty answer, which Toponymy treats as "keep
    the old names".
    """

    def __init__(self):
        self.extra_prompting = ""
        self.callback = None
        self.prompts = []  # (kind, system, user)

    async def _respond(self, system: str | None, user: str, max_tokens: int) -> str:
        # Toponymy sizes a single topic name at 128 tokens and a cluster renaming at 1024.
        if max_tokens < 1024:
            self.prompts.append(("topic", system, user))
            keywords = re.search(r"Keywords for this group include: ([^\n]+)", user)
            name = (
                ", ".join(keywords.group(1).split(", ")[:3])
                if keywords
                else hashlib.sha1(user.encode()).hexdigest()[:8]
            )
            # Toponymy's parser requires both fields.
            return json.dumps({"topic_name": name, "topic_specificity": 0.5})
        self.prompts.append(("disambiguation", system, user))
        return ""

    async def _call_single_llm(self, prompt, temperature, max_tokens) -> str:
        return await self._respond(None, prompt["combined"], max_tokens)

    async def _call_single_llm_with_system(
        self, prompt, temperature, max_tokens
    ) -> str:
        return await self._respond(prompt["system"], prompt["user"], max_tokens)
