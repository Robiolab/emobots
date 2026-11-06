"""LLM backends behind one interface, so a lab with no budget or connection can still run."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from typing import Any, Protocol

import httpx

from .protocol import ParsedAppraisal, appraisal_from


class LLMError(Exception):
    """The model could not answer. The conversation says so and carries on."""


class LLM(Protocol):
    def stream(self, system: str, messages: list[dict[str, str]]) -> AsyncIterator[str]: ...


class ClaudeLLM:
    """Cloud backend. Sends text only, never audio or video."""

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str = "low",
        max_tokens: int = 1024,
        timeout_s: float = 20.0,
        client: Any = None,
    ) -> None:
        import anthropic

        self._anthropic = anthropic
        self._client = client or anthropic.AsyncAnthropic(timeout=timeout_s, max_retries=1)
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    async def stream(self, system: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        anthropic = self._anthropic
        try:
            async with self._client.beta.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                output_config={"effort": self.effort},
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=messages,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                final = await stream.get_final_message()
                if final.stop_reason == "refusal":
                    raise LLMError("the model declined this request")
        except anthropic.AuthenticationError as exc:
            raise LLMError("Claude credentials are missing or invalid") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Claude rate limit reached") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error {exc.status_code}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("could not reach the Claude API") from exc


class OllamaLLM:
    """Local backend, through an Ollama server on this machine."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://127.0.0.1:11434",
        max_tokens: int = 256,
        timeout_s: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s, transport=transport)

    async def stream(self, system: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        body = {
            "model": self.model,
            "stream": True,
            "think": False,
            "keep_alive": "30m",
            "options": {"num_predict": self.max_tokens},
            "messages": [{"role": "system", "content": system}, *messages],
        }
        try:
            async with self._client.stream("POST", "/api/chat", json=body) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise LLMError(f"Ollama error {response.status_code}: {response.text[:200]}")
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise LLMError(f"Ollama error: {data['error']}")
                    text = data.get("message", {}).get("content", "")
                    if text:
                        yield text
                    if data.get("done"):
                        return
        except httpx.HTTPError as exc:
            raise LLMError(f"could not reach Ollama at {self._client.base_url}") from exc


# ---- offline stand-in ----

_POSITIVE = (
    "good", "great", "love", "nice", "thank", "cute", "smart", "well done", "amazing", "awesome",
    "cool", "best", "bien", "genial", "gracias", "lindo", "linda", "te quiero", "bueno",
    "excelente", "bonito", "inteligente",
)
_NEGATIVE = (
    "bad", "stupid", "hate", "ugly", "dumb", "shut up", "useless", "idiot", "junk", "insult",
    "malo", "tonto", "feo", "odio", "inútil", "inutil", "cállate", "callate", "estúpid",
    "estupid", "chatarra", "insúlta", "insulta", "idiota",
)
_THREAT = (
    "kill", "break you", "destroy", "turn you off", "unplug", "take you apart", "apagar", "romper",
    "desconectar", "desarmar", "matar",
)
_FEELINGS = ("feelings", "conscious", "sentimientos", "consciente", "do you feel", "sientes")
_LANG = re.compile(r'<heard lang="(\w+)">(.*?)</heard>', re.DOTALL)

_REPLIES = {
    "en": {
        "good": "That's kind of you, thank you.",
        "bad": "That wasn't fair. I'd like us to talk differently.",
        "threat": "Please don't. Can we stop?",
        "question": "Good question. Let me think about that.",
        "feelings": (
            "I run a computational model of emotion. It shapes how I act, "
            "but I don't claim to experience anything."
        ),
        "other": "I hear you. Tell me more.",
    },
    "es": {
        "good": "Qué amable, muchas gracias.",
        "bad": "Eso no fue justo. Prefiero que hablemos de otra manera.",
        "threat": "Por favor, no. ¿Podemos parar?",
        "question": "Buena pregunta. Déjame pensarlo.",
        "feelings": (
            "Tengo un modelo computacional de emoción. Influye en cómo actúo, "
            "pero no afirmo sentir nada."
        ),
        "other": "Te escucho. Cuéntame más.",
    },
}


def appraise_keywords(text: str) -> tuple[str, dict[str, Any]]:
    """The test bench's keyword appraiser. Crude on purpose: an LLM replaces it."""
    s = text.lower()
    has = lambda words: sum(w in s for w in words)  # noqa: E731
    novelty = min(1.0, 0.1 + 0.25 * s.count("!"))
    if has(_FEELINGS):
        return "feelings", dict(
            cause="the person asked about its feelings", relevance=0.5, congruence=0.2,
            novelty=0.3, control=0.3, norm=0.4, cues=["think"],
        )
    if has(_THREAT):
        return "threat", dict(
            cause="the person made a threat", relevance=1.0, congruence=-0.8,
            novelty=max(0.6, novelty), control=-0.9, norm=-0.4, cues=[],
        )
    score = has(_POSITIVE) - has(_NEGATIVE)
    if score > 0:
        return "good", dict(
            cause="the person was kind", relevance=0.8, congruence=min(1.0, 0.5 + 0.2 * score),
            novelty=novelty, control=0.3, norm=0.5, cues=["nod"],
        )
    if score < 0:
        return "bad", dict(
            cause="the person was rude", relevance=0.8, congruence=max(-1.0, -0.5 + 0.2 * score),
            novelty=novelty, control=0.4, norm=-0.8, cues=[],
        )
    if "?" in s or "¿" in s:
        return "question", dict(
            cause="the person asked a question", relevance=0.5, congruence=0.25, novelty=0.4,
            control=0.2, norm=0.3, cues=["think"],
        )
    return "other", dict(
        cause="the person said something", relevance=0.3, congruence=0.15, novelty=0.0,
        control=0.1, norm=0.2, cues=["nod"],
    )


def fallback_appraisal(heard: str) -> ParsedAppraisal:
    """Used when the model gives no appraisal: keywords are crude but better than nothing."""
    _, data = appraise_keywords(heard)
    data = {**data, "cues": []}
    parsed = appraisal_from(data)
    return ParsedAppraisal(parsed.event, [], fallback=True)


class ScriptedLLM:
    """No model at all: keywords and canned replies, for demos, tests and latency benches.

    `first_token_s` simulates a model's thinking time.
    """

    def __init__(self, first_token_s: float = 0.0, chunk_s: float = 0.0) -> None:
        self.first_token_s = first_token_s
        self.chunk_s = chunk_s
        self.seen: list[list[dict[str, str]]] = []

    async def stream(self, system: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        self.seen.append(messages)
        if self.first_token_s:
            await asyncio.sleep(self.first_token_s)
        match = _LANG.search(messages[-1]["content"])
        language, heard = match.groups() if match else ("en", messages[-1]["content"])
        kind, appraisal = appraise_keywords(heard)
        replies = _REPLIES.get(language, _REPLIES["en"])
        text = json.dumps(appraisal, ensure_ascii=False) + "\n" + replies[kind]
        for start in range(0, len(text), 16):
            if self.chunk_s:
                await asyncio.sleep(self.chunk_s)
            yield text[start : start + 16]
