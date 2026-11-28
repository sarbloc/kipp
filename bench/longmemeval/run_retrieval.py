"""LongMemEval retrieval benchmark, Track A (issue #81): no LLM, recall only.

For each of the 500 LongMemEval_S questions, ranks that question's haystack
(about 50 sessions, about 500 messages) against the question and checks
whether the evidence is near the top. It measures the layer KIPP's recall
rests on, exact cosine over a local embedding model, not answer accuracy.

Index units are rounds: a user message plus the assistant reply that
follows it. Two texts per round:

- ``round``: ``"User: ...\\n\\nAssistant: ..."``, the text ``kipp-capture``
  writes for a turn (``capture.Turn.text``).
- ``user``: the user message alone.

A session's score is its best round's score. Metrics, over the 470
non-abstention questions:

- ``session_any@k``: an answer session is among the top k sessions.
- ``session_all@k``: every answer session is among the top k.
- ``round_any@k``: a round holding a ``has_answer`` message is among the top
  k rounds (questions with no labelled message are left out).

Search is brute-force cosine in numpy, the same exact search an in-memory
or small Qdrant collection runs, so no Qdrant is needed. Embeddings are
cached by text hash, so a killed run resumes where it stopped.

The dataset is not in the repo. Download ``longmemeval_s_cleaned.json`` from
https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned (MIT) into
``~/.cache/kipp-bench/``. Only aggregate scores are written to the repo.

Run: python bench/longmemeval/run_retrieval.py --model minilm --unit round
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("longmemeval")

CACHE = Path.home() / ".cache" / "kipp-bench"
DATASET = CACHE / "longmemeval_s_cleaned.json"
RESULTS = Path(__file__).parent / "results"
KS = (1, 5, 10)
UNITS = ("round", "user")
SAVE_EVERY = 8192

# Prefixes are each model's documented query instruction. No model here
# needs trust_remote_code.
MODELS: dict[str, dict] = {
    "minilm": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "doc_prefix": "",
    },
    "bge-small": {
        "name": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": "",
    },
    "bge-base": {
        "name": "BAAI/bge-base-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": "",
    },
}


@dataclass
class Round:
    session_id: str
    user: str
    assistant: str
    has_answer: bool

    def text(self, unit: str) -> str:
        if unit == "user":
            return self.user
        parts = []
        if self.user:
            parts.append(f"User: {self.user}")
        if self.assistant:
            parts.append(f"Assistant: {self.assistant}")
        return "\n\n".join(parts)


@dataclass
class Question:
    qid: str
    qtype: str
    question: str
    answer_sessions: set[str]
    rounds: list[Round] = field(default_factory=list)

    @property
    def abstention(self) -> bool:
        return self.qid.endswith("_abs")


def build_rounds(session_id: str, messages: list[dict]) -> list[Round]:
    """Pair each user message with the assistant messages that follow it.

    Assistant messages before any user message form a round of their own,
    as capture does with an assistant reply it can't anchor.
    """
    rounds: list[Round] = []
    current: Round | None = None
    for m in messages:
        content = m.get("content") or ""
        labelled = bool(m.get("has_answer"))
        if m.get("role") == "user":
            current = Round(session_id, content, "", labelled)
            rounds.append(current)
        else:
            if current is None:
                current = Round(session_id, "", "", False)
                rounds.append(current)
            current.assistant = f"{current.assistant}\n\n{content}" if current.assistant else content
            current.has_answer = current.has_answer or labelled
    return rounds


def load_questions(path: Path) -> list[Question]:
    data = json.loads(path.read_text())
    questions = []
    for item in data:
        q = Question(
            qid=item["question_id"],
            qtype=item["question_type"],
            question=item["question"],
            answer_sessions=set(item["answer_session_ids"]),
        )
        for sid, messages in zip(item["haystack_session_ids"], item["haystack_sessions"], strict=True):
            q.rounds.extend(build_rounds(sid, messages))
        questions.append(q)
    return questions


def _key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class EmbeddingCache:
    """Vectors by text hash, saved in parts so a killed run loses one part."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.vectors: dict[str, np.ndarray] = {}
        for part in sorted(directory.glob("part-*.npz")):
            with np.load(part) as z:
                self.vectors.update(zip(z["keys"].tolist(), z["vectors"], strict=True))
        self._parts = len(list(directory.glob("part-*.npz")))

    def missing(self, texts: list[str]) -> list[str]:
        seen: set[str] = set()
        out = []
        for t in texts:
            k = _key(t)
            if k not in self.vectors and k not in seen:
                seen.add(k)
                out.append(t)
        return out

    def add(self, texts: list[str], vectors: np.ndarray) -> None:
        keys = [_key(t) for t in texts]
        np.savez(self.directory / f"part-{self._parts:05d}.npz", keys=np.array(keys), vectors=vectors)
        self._parts += 1
        self.vectors.update(zip(keys, vectors, strict=True))

    def matrix(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.vectors[_key(t)] for t in texts])


def load_model(spec: dict):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(spec["name"])


def encode(model, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
    vectors = model.encode(
        [prefix + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_all(model, cache: EmbeddingCache, texts: list[str], prefix: str, batch_size: int) -> None:
    # Longest first, so batches pad little; the cache doesn't care about order.
    todo = sorted(cache.missing(texts), key=len, reverse=True)
    log.info("%d unique texts, %d to embed", len(cache.vectors) + len(todo), len(todo))
    start = time.monotonic()
    for i in range(0, len(todo), SAVE_EVERY):
        chunk = todo[i : i + SAVE_EVERY]
        cache.add(chunk, encode(model, chunk, prefix, batch_size))
        done = i + len(chunk)
        rate = done / (time.monotonic() - start)
        log.info("embedded %d/%d (%.0f/s)", done, len(todo), rate)


def score_question(q: Question, rounds_m: np.ndarray, query_v: np.ndarray) -> dict[str, float | None]:
    """Recall metrics for one question, given its rounds' vectors in order."""
    scores = rounds_m @ query_v
    order = np.argsort(-scores, kind="stable")

    best: dict[str, float] = {}
    for r, s in zip(q.rounds, scores.tolist(), strict=True):
        if s > best.get(r.session_id, -np.inf):
            best[r.session_id] = s
    sessions = sorted(best, key=lambda sid: -best[sid])

    labelled = any(r.has_answer for r in q.rounds)
    out: dict[str, float | None] = {}
    for k in KS:
        top = set(sessions[:k])
        out[f"session_any@{k}"] = float(bool(q.answer_sessions & top))
        out[f"session_all@{k}"] = float(q.answer_sessions <= top)
        out[f"round_any@{k}"] = (
            float(any(q.rounds[i].has_answer for i in order[:k])) if labelled else None
        )
    return out


def aggregate(per_question: list[tuple[str, dict[str, float | None]]]) -> dict[str, dict[str, float]]:
    """Mean of each metric per question type and overall, ignoring None."""
    groups: dict[str, list[dict[str, float | None]]] = defaultdict(list)
    for qtype, metrics in per_question:
        groups[qtype].append(metrics)
        groups["all"].append(metrics)
    out: dict[str, dict[str, float]] = {}
    for name, rows in sorted(groups.items()):
        summary: dict[str, float] = {"n": float(len(rows))}
        for metric in rows[0]:
            values = [r[metric] for r in rows if r[metric] is not None]
            if values:
                summary[metric] = sum(values) / len(values)
        out[name] = summary
    return out


def truncated_share(model, texts: list[str], prefix: str) -> float:
    """Share of texts longer than the model's window (sampled, tokenizer only)."""
    sample = texts[:: max(1, len(texts) // 5000)]
    window = model.max_seq_length - 2
    over = sum(1 for t in sample if len(model.tokenizer.tokenize(prefix + t)) > window)
    return over / len(sample)


def run(args: argparse.Namespace) -> dict:
    spec = MODELS[args.model]
    questions = load_questions(args.dataset)
    if args.limit:
        questions = questions[: args.limit]
    scored = [q for q in questions if not q.abstention]
    log.info("%d questions, %d scored (abstention left out)", len(questions), len(scored))

    texts = [r.text(args.unit) for q in scored for r in q.rounds]
    model = load_model(spec)
    cache = EmbeddingCache(CACHE / "emb" / f"{args.model}-{args.unit}")
    embed_all(model, cache, texts, spec["doc_prefix"], args.batch_size)
    queries = encode(model, [q.question for q in scored], spec["query_prefix"], args.batch_size)

    per_question = []
    for q, qv in zip(scored, queries, strict=True):
        m = cache.matrix([r.text(args.unit) for r in q.rounds])
        per_question.append((q.qtype, score_question(q, m, qv)))

    unique = sorted(set(texts))
    return {
        "model": spec["name"],
        "unit": args.unit,
        "max_seq_length": model.max_seq_length,
        "questions": len(scored),
        "rounds": len(texts),
        "unique_texts": len(unique),
        "truncated_share": round(truncated_share(model, unique, spec["doc_prefix"]), 4),
        "metrics": aggregate(per_question),
    }


def print_table(result: dict) -> None:
    print(f"{result['model']} / {result['unit']}: {result['questions']} questions, "
          f"{result['truncated_share']:.0%} of texts truncated at {result['max_seq_length']} tokens")
    cols = ["session_any@5", "session_any@10", "session_all@10", "round_any@10"]
    print(f"{'type':28}{'n':>5}" + "".join(f"{c:>16}" for c in cols))
    for name, m in result["metrics"].items():
        cells = "".join(f"{m[c]:>16.3f}" if c in m else f"{'-':>16}" for c in cols)
        print(f"{name:28}{int(m['n']):>5}{cells}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--unit", choices=UNITS, default="round")
    p.add_argument("--dataset", type=Path, default=DATASET)
    p.add_argument("--limit", type=int, default=0, help="first N questions only (smoke test)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--no-save", action="store_true", help="print only; don't write results/")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr)

    if not args.dataset.exists():
        log.error("dataset not found at %s; see the module docstring", args.dataset)
        return 1
    result = run(args)
    print_table(result)
    if not args.no_save and not args.limit:
        RESULTS.mkdir(exist_ok=True)
        out = RESULTS / f"{args.model}-{args.unit}.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
