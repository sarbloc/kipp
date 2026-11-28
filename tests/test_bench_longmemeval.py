"""Scoring logic of the LongMemEval retrieval harness (issue #81), on fixtures."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parent.parent / "bench" / "longmemeval" / "run_retrieval.py"
_spec = importlib.util.spec_from_file_location("run_retrieval", SCRIPT)
rr = importlib.util.module_from_spec(_spec)
# dataclasses look their module up in sys.modules.
sys.modules["run_retrieval"] = rr
_spec.loader.exec_module(rr)


def msg(role, content, has_answer=False):
    return {"role": role, "content": content, "has_answer": has_answer}


def test_rounds_pair_user_with_following_replies():
    rounds = rr.build_rounds("s1", [
        msg("assistant", "orphan"),
        msg("user", "u1"),
        msg("assistant", "a1"),
        msg("assistant", "a1b", has_answer=True),
        msg("user", "u2", has_answer=True),
    ])
    assert [(r.user, r.assistant, r.has_answer) for r in rounds] == [
        ("", "orphan", False),
        ("u1", "a1\n\na1b", True),
        ("u2", "", True),
    ]
    assert all(r.session_id == "s1" for r in rounds)


def test_round_text_matches_capture_and_user_unit():
    r = rr.Round("s", "hi", "hello", False)
    assert r.text("round") == "User: hi\n\nAssistant: hello"
    assert r.text("user") == "hi"
    assert rr.Round("s", "", "only", False).text("round") == "Assistant: only"


def _question(rounds, answers):
    q = rr.Question("q1", "multi-session", "?", set(answers))
    q.rounds = rounds
    return q


def test_session_score_is_best_round_and_recall_counts():
    rounds = [
        rr.Round("a", "", "", False),
        rr.Round("a", "", "", False),
        rr.Round("b", "", "", True),
        rr.Round("c", "", "", False),
    ]
    q = _question(rounds, ["b", "c"])
    query = np.array([1.0, 0.0])
    # Session a holds the two best rounds; b is next, c last.
    m = np.array([[1.0, 0.0], [0.9, 0.1], [0.5, 0.5], [0.0, 1.0]])
    out = rr.score_question(q, m, query)
    assert out["session_any@1"] == 0.0
    assert out["session_any@5"] == 1.0
    # Top 1 session is a; b and c need k >= 3.
    assert out["session_all@1"] == 0.0
    assert out["session_all@5"] == 1.0
    # Round ranks: 0, 1, 2(has_answer), 3.
    assert out["round_any@1"] == 0.0
    assert out["round_any@5"] == 1.0


def test_unlabelled_question_has_no_round_metric():
    q = _question([rr.Round("a", "", "", False)], ["a"])
    out = rr.score_question(q, np.array([[1.0]]), np.array([1.0]))
    assert out["round_any@10"] is None
    assert out["session_any@1"] == 1.0


def test_aggregate_means_by_type_and_skips_none():
    rows = [
        ("x", {"m": 1.0, "r": None}),
        ("x", {"m": 0.0, "r": 1.0}),
        ("y", {"m": 1.0, "r": None}),
    ]
    out = rr.aggregate(rows)
    assert out["x"] == {"n": 2.0, "m": 0.5, "r": 1.0}
    assert out["y"] == {"n": 1.0, "m": 1.0}
    assert out["all"]["m"] == pytest.approx(2 / 3)
    assert out["all"]["r"] == 1.0


def test_embedding_cache_dedupes_and_survives_reload(tmp_path):
    cache = rr.EmbeddingCache(tmp_path)
    assert cache.missing(["a", "b", "a"]) == ["a", "b"]
    cache.add(["a", "b"], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    assert cache.missing(["a", "c"]) == ["c"]
    cache.add(["c"], np.array([[0.5, 0.5]], dtype=np.float32))

    reloaded = rr.EmbeddingCache(tmp_path)
    assert reloaded.missing(["a", "b", "c"]) == []
    np.testing.assert_array_equal(reloaded.matrix(["c", "a"]), [[0.5, 0.5], [1.0, 0.0]])
    # A new part doesn't overwrite an old one after reload.
    reloaded.add(["d"], np.array([[0.0, 0.0]], dtype=np.float32))
    assert len(list(tmp_path.glob("part-*.npz"))) == 3


def test_interrupted_save_leaves_no_part(tmp_path, monkeypatch):
    cache = rr.EmbeddingCache(tmp_path)
    cache.add(["a"], np.array([[1.0]], dtype=np.float32))

    def killed(f, **_):
        f.write(b"half a zip")
        raise KeyboardInterrupt

    monkeypatch.setattr(rr.np, "savez", killed)
    with pytest.raises(KeyboardInterrupt):
        cache.add(["b"], np.array([[0.0]], dtype=np.float32))
    monkeypatch.undo()

    reloaded = rr.EmbeddingCache(tmp_path)
    assert reloaded.missing(["a", "b"]) == ["b"]
    reloaded.add(["b"], np.array([[0.0]], dtype=np.float32))
    assert sorted(p.name for p in tmp_path.glob("part-*.npz")) == ["part-00000.npz", "part-00001.npz"]


def test_load_questions_flattens_sessions(tmp_path):
    import json

    data = [{
        "question_id": "q_abs", "question_type": "temporal-reasoning", "question": "when?",
        "answer_session_ids": ["s2"],
        "haystack_session_ids": ["s1", "s2"],
        "haystack_sessions": [[msg("user", "u")], [msg("user", "v", True), msg("assistant", "w")]],
    }]
    path = tmp_path / "d.json"
    path.write_text(json.dumps(data))
    [q] = rr.load_questions(path)
    assert q.abstention
    assert [(r.session_id, r.text("round")) for r in q.rounds] == [
        ("s1", "User: u"), ("s2", "User: v\n\nAssistant: w"),
    ]
    assert q.answer_sessions == {"s2"}
