"""Recall evaluation: score live search against a set of known-answer questions.

A case file is JSON Lines, one question per line:

    {"query": "who runs billing", "expected": "person:alice"}
    {"query": "...", "expected": ["project:a", "project:b"], "domain": "dev",
     "fact": "retries back off"}

- ``expected``: the entity id (or ids, any of which counts) that should answer.
- ``domain``: defaults to ``shared``.
- ``fact``: optional substring of the fact that answers the question. When set,
  the report says whether that text reaches the entity's dense vector: present
  in the embedded vector text and inside the model's window (issue #49).

Everything here is read-only. Case files built from real memory hold personal
data: keep them outside the repo (the default is ~/.config/kipp/eval.jsonl).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from qdrant_client import QdrantClient

from kipp.client import DOMAINS, _xdg_config_home, get_entity
from kipp.merge import build_search_text, build_vector_text
from kipp.search import EmbedderLike, search_entities

HIT_AT = (1, 5)


@dataclass
class Case:
    query: str
    expected: frozenset[str]
    domain: str = "shared"
    fact: str | None = None
    line: int = 0


@dataclass
class CaseResult:
    case: Case
    rank: int | None  # 1-based rank of the first expected id; None = not in top k
    top: list[str] = field(default_factory=list)
    window: str = "unknown"  # "in", "out" or "unknown" (no fact given / not found)


def load_cases(path: Path) -> list[Case]:
    """Parse a JSONL case file. Raises ValueError naming the bad line."""
    cases: list[Case] = []
    for n, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("//"):
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{n}: invalid JSON ({e.msg})") from e
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{n}: each line must be a JSON object")
        query = obj.get("query")
        expected = obj.get("expected")
        if isinstance(expected, str):
            expected = [expected]
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{path}:{n}: 'query' must be a non-empty string")
        if (
            not isinstance(expected, list)
            or not expected
            or not all(isinstance(e, str) and ":" in e for e in expected)
        ):
            raise ValueError(f"{path}:{n}: 'expected' must be an id like 'person:alice' or a list of them")
        domain = obj.get("domain", "shared")
        if domain not in DOMAINS:
            raise ValueError(f"{path}:{n}: unknown domain {domain!r}")
        fact = obj.get("fact")
        if fact is not None and not isinstance(fact, str):
            raise ValueError(f"{path}:{n}: 'fact' must be a string")
        cases.append(Case(query, frozenset(expected), domain, fact, n))
    return cases


def rank_of(expected: frozenset[str], result_ids: list[str]) -> int | None:
    """1-based position of the first expected id in ``result_ids``, else None."""
    for i, rid in enumerate(result_ids, start=1):
        if rid in expected:
            return i
    return None


def fact_window(
    search_text: str,
    vector_text: str,
    fact: str,
    count_tokens: Callable[[str], int],
    window: int,
) -> str:
    """Whether ``fact`` reaches the entity's dense vector.

    "in": it appears in ``vector_text`` and starts inside its first ``window``
    tokens. "out": it is among the entity's top facts (``search_text``) but is
    clipped off or truncated before the model reads it. "unknown": not in the
    top facts at all (it may rank below them, or the substring is wrong).
    """
    needle = " ".join(fact.lower().split())
    if needle not in " ".join(search_text.lower().split()):
        return "unknown"
    # build_vector_text collapses each fact's whitespace; match it the same way.
    start = vector_text.lower().find(needle)
    return "in" if start >= 0 and count_tokens(vector_text[:start]) < window else "out"


def run_case(
    client: QdrantClient,
    embedder: EmbedderLike,
    case: Case,
    k: int,
    count_tokens: Callable[[str], int] | None = None,
    window: int = 0,
) -> CaseResult:
    """Search for one case and locate its answer. Read-only.

    The window check rebuilds the entity's texts with today's fact scores, so
    it can differ slightly from the text the stored vector was built from at
    the entity's last write.
    """
    hits = search_entities(client, case.query, embedder, limit=k, domain=case.domain)
    top = [h.entity.id for h in hits]
    result = CaseResult(case, rank_of(case.expected, top), top)
    if case.fact and count_tokens is not None:
        for eid in sorted(case.expected):
            entity = get_entity(client, eid, domain=case.domain)
            if entity is None:
                continue
            result.window = fact_window(
                build_search_text(entity), build_vector_text(entity),
                case.fact, count_tokens, window,
            )
            if result.window != "unknown":
                break
    return result


def summarize(results: list[CaseResult]) -> dict[str, float | int]:
    """hit@1, hit@5 and mean reciprocal rank over ``results``."""
    n = len(results)
    if n == 0:
        return {"n": 0}
    out: dict[str, float | int] = {"n": n}
    for k in HIT_AT:
        out[f"hit@{k}"] = sum(1 for r in results if r.rank is not None and r.rank <= k) / n
    out["mrr"] = sum(1 / r.rank for r in results if r.rank is not None) / n
    return out


def default_case_path() -> Path:
    return _xdg_config_home() / "kipp" / "eval.jsonl"


def format_report(results: list[CaseResult], k: int) -> str:
    """Human-readable summary table plus every case that missed the top 5."""
    lines = [f"{'group':<14}{'n':>4}{'hit@1':>8}{'hit@5':>8}{'MRR':>7}"]
    for name, s in grouped_summary(results).items():
        lines.append(
            f"{name:<14}{s['n']:>4}{s['hit@1']:>8.2f}{s['hit@5']:>8.2f}{s['mrr']:>7.2f}"
        )
    misses = [r for r in results if r.rank is None or r.rank > HIT_AT[-1]]
    if misses:
        lines.append("")
        lines.append(f"Missed the top {HIT_AT[-1]} ({len(misses)}):")
        for r in misses:
            where = f"rank {r.rank}" if r.rank else f"not in top {k}"
            exp = ", ".join(sorted(r.case.expected))
            window = f", fact {r.window} window" if r.window != "unknown" else ""
            lines.append(f"  line {r.case.line} [{r.case.domain}] {r.case.query!r}")
            lines.append(f"    want {exp}: {where}{window}; got {', '.join(r.top[:3]) or 'nothing'}")
    return "\n".join(lines)


def report_json(results: list[CaseResult], k: int) -> dict:
    return {
        "k": k,
        "summary": grouped_summary(results),
        "cases": [
            {
                "line": r.case.line,
                "query": r.case.query,
                "domain": r.case.domain,
                "expected": sorted(r.case.expected),
                "rank": r.rank,
                "window": r.window,
                "top": r.top,
            }
            for r in results
        ],
    }


def grouped_summary(results: list[CaseResult]) -> dict[str, dict[str, float | int]]:
    """Overall, per-domain and per-window summaries; empty groups omitted."""
    groups: dict[str, list[CaseResult]] = {"overall": results}
    for d in DOMAINS:
        groups[d] = [r for r in results if r.case.domain == d]
    for w in ("in", "out"):
        groups[f"window:{w}"] = [r for r in results if r.window == w]
    return {name: summarize(rs) for name, rs in groups.items() if rs}
