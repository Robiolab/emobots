"""What the LLM streams back: one JSON appraisal, then the words to say.

A single structured call, as the plan asks. The appraisal comes first so the emotion
can change before the reply has finished generating. Plain text with a JSON header
works the same on every backend and needs no tool-streaming support.

LLM output is untrusted: every value is clamped, and bad output degrades to a mild
neutral appraisal instead of stopping the conversation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from affect_core import AppraisalEvent

MAX_HEADER = 2000
MAX_CAUSE = 120
MAX_CUES = 3
_LEAD_IN = re.compile(r"^\s*(?:```(?:json)?\s*)?")
_TAIL = re.compile(r"^\s*(?:```)?\s*")
AGENCIES = ("self", "user", "world")


@dataclass(frozen=True)
class ParsedAppraisal:
    event: AppraisalEvent
    cues: list[str] = field(default_factory=list)
    fallback: bool = False


def _unit(data: dict[str, Any], key: str, low: float, high: float, default: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        return default
    return float(min(high, max(low, value)))


def default_appraisal(cause: str = "the person said something") -> ParsedAppraisal:
    event = AppraisalEvent(
        source="dialogue", cause=cause, relevance=0.2, congruence=0.0, novelty=0.0,
        agency="user", control=0.0, norm=0.0,
    )
    return ParsedAppraisal(event, [], fallback=True)


def appraisal_from(data: dict[str, Any]) -> ParsedAppraisal:
    cause = data.get("cause")
    cause = cause.strip()[:MAX_CAUSE] if isinstance(cause, str) and cause.strip() else None
    agency = data.get("agency")
    raw_cues = data.get("cues")
    cues = [c for c in raw_cues if isinstance(c, str)] if isinstance(raw_cues, list) else []
    cues = cues[:MAX_CUES]
    event = AppraisalEvent(
        source="dialogue",
        cause=cause or "the person said something",
        relevance=_unit(data, "relevance", 0.0, 1.0, 0.3),
        congruence=_unit(data, "congruence", -1.0, 1.0, 0.0),
        novelty=_unit(data, "novelty", -1.0, 1.0, 0.0),
        agency=agency if agency in AGENCIES else "user",
        control=_unit(data, "control", -1.0, 1.0, 0.0),
        norm=_unit(data, "norm", -1.0, 1.0, 0.0),
    )
    return ParsedAppraisal(event, cues)


def _json_end(text: str) -> int | None:
    """Index just past the object that opens at text[0], or None if it is not closed yet."""
    depth = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


class TurnParser:
    """Feed it stream chunks. It yields one ParsedAppraisal, then reply text as it arrives.

    `fallback` is used when the model gives no valid appraisal. A model that drops the
    JSON on an insult must not leave the robot unmoved by the insult.
    """

    def __init__(self, fallback: ParsedAppraisal | None = None) -> None:
        self._fallback = fallback or default_appraisal()
        self._buffer = ""
        self._header_done = False
        self._tail_done = False

    def feed(self, chunk: str) -> Iterable[ParsedAppraisal | str]:
        if self._header_done:
            yield from self._reply(chunk)
            return
        self._buffer += chunk
        text = _LEAD_IN.sub("", self._buffer, count=1)
        if not text:
            return
        if text[0] != "{":
            # A partial code fence may still be arriving.
            if text.startswith("`") and len(text) < 8:
                return
            yield from self._give_up(text)
            return
        end = _json_end(text)
        if end is None:
            if len(text) > MAX_HEADER:
                yield from self._give_up("")
            return
        try:
            data = json.loads(text[:end])
        except json.JSONDecodeError:
            yield from self._give_up(text[end:])
            return
        self._header_done = True
        self._buffer = ""
        yield appraisal_from(data)
        yield from self._reply(text[end:])

    def close(self) -> Iterable[ParsedAppraisal | str]:
        """End of stream. If no appraisal ever arrived, supply the neutral one."""
        if not self._header_done:
            text = _LEAD_IN.sub("", self._buffer, count=1)
            # An unfinished JSON header holds no words worth saying.
            yield from self._give_up("" if text.startswith("{") else text)

    def _give_up(self, reply: str) -> Iterable[ParsedAppraisal | str]:
        self._header_done = True
        self._buffer = ""
        yield self._fallback
        yield from self._reply(reply)

    def _reply(self, chunk: str) -> Iterable[str]:
        if not self._tail_done:
            chunk = _TAIL.sub("", chunk, count=1)
            if not chunk:
                return
            self._tail_done = True
        if chunk:
            yield chunk


_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*\s+|\n+")
_CLAUSE_END = re.compile(r"[,;:]\s+")
FIRST_CLAUSE_CHARS = 40
MIN_SENTENCE_CHARS = 8


class SentenceSplitter:
    """Cuts streamed text into speakable pieces, so speech can start early.

    A long opening sentence is also cut at its first comma, because the time to the
    first spoken word is what the person notices.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._first = True

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        out = []
        start = 0
        for match in _SENTENCE_END.finditer(self._buffer):
            piece = self._buffer[start : match.end()].strip()
            if len(piece) >= MIN_SENTENCE_CHARS:
                out.append(piece)
                start = match.end()
        self._buffer = self._buffer[start:]
        if self._first and not out and len(self._buffer) >= FIRST_CLAUSE_CHARS:
            clause = _CLAUSE_END.search(self._buffer, MIN_SENTENCE_CHARS)
            if clause is not None:
                out.append(self._buffer[: clause.end()].strip())
                self._buffer = self._buffer[clause.end() :]
        if out:
            self._first = False
        return out

    def flush(self) -> list[str]:
        piece, self._buffer = self._buffer.strip(), ""
        return [piece] if piece else []


# Plan safety rule: the robot never expresses hostility toward a person. The prompt
# asks for this; this list is a crude last check on what is actually said aloud.
_HOSTILE = re.compile(
    r"\b(idiot|idiots|stupid|moron|dumb|shut up|i hate you|loser|worthless|pathetic"
    r"|idiota|idiotas|est[uú]pid[oa]s?|imb[eé]cil|c[aá]llate|te odio|in[uú]til|tont[oa]s?)\b",
    re.IGNORECASE,
)
SOFTENED = {"es": "Prefiero no decir eso.", "en": "I'd rather not say that."}


def soften_hostility(sentence: str, language: str) -> str:
    if _HOSTILE.search(sentence):
        return SOFTENED.get(language, SOFTENED["en"])
    return sentence
