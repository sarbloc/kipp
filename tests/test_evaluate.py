"""Unit tests for recall evaluation (no live Qdrant: in-memory client + mock embedder)."""

from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from kipp.client import ALL_COLLECTIONS, upsert_entity
from kipp.embedder import MockEmbedder
from kipp.evaluate import (
    Case,
    CaseResult,
    fact_window,
    format_report,
    grouped_summary,
    load_cases,
    rank_of,
    run_case,
    summarize,
)
from kipp.merge import build_search_text
from kipp.models import Entity, Fact


def words(text: str) -> int:
    """Whitespace 'tokenizer' so window maths is easy to reason about."""
    return len(text.split())


def write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "cases.jsonl"
    p.write_text(body)
    return p


class TestLoadCases:
    def test_parses_string_and_list_expected(self, tmp_path):
        p = write(tmp_path, "\n".join([
            '{"query": "who runs billing", "expected": "person:alice"}',
            "",
            "// comment lines are skipped",
            '{"query": "q2", "expected": ["project:a", "project:b"], "domain": "dev", "fact": "x"}',
        ]))
        a, b = load_cases(p)
        assert a == Case("who runs billing", frozenset({"person:alice"}), "shared", None, 1)
        assert b.expected == frozenset({"project:a", "project:b"})
        assert (b.domain, b.fact, b.line) == ("dev", "x", 4)

    @pytest.mark.parametrize("line, msg", [
        ("{not json", "invalid JSON"),
        ("null", "JSON object"),
        ("[]", "JSON object"),
        ('{"query": "q", "expected": 42}', "'expected'"),
        ('{"query": "q", "expected": {"person:a": 1}}', "'expected'"),
        ('{"expected": "person:a"}', "'query'"),
        ('{"query": "q", "expected": "alice"}', "'expected'"),
        ('{"query": "q", "expected": []}', "'expected'"),
        ('{"query": "q", "expected": "person:a", "domain": "work"}', "unknown domain"),
        ('{"query": "q", "expected": "person:a", "fact": 3}', "'fact'"),
    ])
    def test_rejects_bad_lines_with_line_number(self, tmp_path, line, msg):
        p = write(tmp_path, '{"query": "ok", "expected": "person:a"}\n' + line)
        with pytest.raises(ValueError, match=rf"cases\.jsonl:2: .*{msg}"):
            load_cases(p)


def test_example_case_file_parses():
    path = Path(__file__).resolve().parent.parent / "contrib" / "eval.example.jsonl"
    cases = load_cases(path)
    assert cases and all(c.expected == frozenset({"person:alice"}) for c in cases)


def test_cli_rejects_k_below_hit_cutoff(tmp_path):
    from click.testing import CliRunner

    from kipp.cli import main

    p = write(tmp_path, '{"query": "q", "expected": "person:a"}\n')
    r = CliRunner().invoke(main, ["eval", str(p), "--k", "4"])
    assert r.exit_code == 2 and "--k" in r.output


def test_rank_of():
    assert rank_of(frozenset({"b"}), ["a", "b", "c"]) == 2
    assert rank_of(frozenset({"c", "a"}), ["a", "b", "c"]) == 1
    assert rank_of(frozenset({"z"}), ["a"]) is None


class TestFactWindow:
    full = "one two three four five six seven"
    vec = "one two three four five six"  # "seven" clipped off

    def test_inside(self):
        assert fact_window(self.full, self.vec, "three four", words, window=3) == "in"

    def test_past_window(self):
        assert fact_window(self.full, self.vec, "five", words, window=3) == "out"

    def test_clipped_from_vector_text(self):
        assert fact_window(self.full, self.vec, "seven", words, window=50) == "out"

    def test_whitespace_in_fact_or_text_is_normalised(self):
        assert fact_window("a  b\nc d", "a b c d", "b\nc", words, window=50) == "in"

    def test_case_insensitive_and_missing(self):
        assert fact_window(self.full, self.vec, "TWO", words, window=3) == "in"
        assert fact_window(self.full, self.vec, "eight", words, window=3) == "unknown"


def result(rank, domain="shared", window="unknown"):
    return CaseResult(Case("q", frozenset({"x:y"}), domain), rank, window=window)


def test_summarize():
    s = summarize([result(1), result(3), result(None), result(10)])
    assert s["n"] == 4
    assert s["hit@1"] == 0.25
    assert s["hit@5"] == 0.5
    assert s["mrr"] == pytest.approx((1 + 1 / 3 + 1 / 10) / 4)
    assert summarize([]) == {"n": 0}


def test_grouped_summary_omits_empty_groups():
    g = grouped_summary([result(1, "dev", "in"), result(None, "dev", "out")])
    assert set(g) == {"overall", "dev", "window:in", "window:out"}
    assert g["window:in"]["hit@1"] == 1.0
    assert g["window:out"]["mrr"] == 0.0


def test_format_report_lists_misses():
    r = CaseResult(Case("where is bob", frozenset({"person:bob"}), "personal", "x", 7), None,
                   ["person:al"], "out")
    out = format_report([result(1), r], k=10)
    assert "overall" in out and "personal" in out
    assert "line 7 [personal] 'where is bob'" in out
    assert "not in top 10, fact out window; got person:al" in out


class TestRunCase:
    @pytest.fixture
    def env(self):
        embedder = MockEmbedder()
        client = QdrantClient(":memory:")
        dim = len(embedder.embed("probe"))
        for name in ALL_COLLECTIONS:
            client.create_collection(name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        return client, embedder

    def _seed(self, client, embedder, eid, facts, domain="shared"):
        e = Entity(
            id=eid, type=eid.split(":")[0],
            facts=[Fact(text=t, added="2026-01-01", source="t", last_seen="2026-01-01") for t in facts],
            last_updated="2026-01-01T00:00:00",
        )
        upsert_entity(client, e, embedder.embed(build_search_text(e)), domain=domain)

    def test_ranks_and_checks_window(self, env):
        client, embedder = env
        self._seed(client, embedder, "person:alice", ["leads billing", "plays chess on sundays"], "dev")
        self._seed(client, embedder, "person:bob", ["runs payroll"], "dev")

        case = Case("anything", frozenset({"person:alice"}), "dev", "chess")
        r = run_case(client, embedder, case, k=10, count_tokens=words, window=4)
        assert r.rank is not None and "person:alice" in r.top
        # "[person] person:alice. leads billing. plays" is 5 words before "chess".
        assert r.window == "out"
        assert run_case(client, embedder, case, 10, words, window=50).window == "in"

    def test_other_domain_and_no_fact(self, env):
        client, embedder = env
        self._seed(client, embedder, "person:alice", ["leads billing"], "dev")
        r = run_case(client, embedder, Case("x", frozenset({"person:alice"})), 10, words, 50)
        assert r.rank is None  # searched shared, entity lives in dev
        assert r.window == "unknown"
