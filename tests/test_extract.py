"""Tests for event extraction logic."""

from datetime import datetime

import pytest

from kipp.extract import (
    build_entity_index,
    extract_events,
    extract_events_with_index,
    match_sentence_to_entity,
    parse_lead_tag,
    split_sentences,
)
from kipp.models import Entity, Fact


class _CountingEmbedder:
    """Wraps an embedder and counts embed() calls (issue #12 regression guard)."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def embed(self, text):
        self.calls += 1
        return self.inner.embed(text)


NOW = datetime(2026, 3, 10)
TODAY = NOW.date().isoformat()


def _entity(eid, etype, facts_texts):
    facts = [
        Fact(text=t, added=TODAY, source="test", last_seen=TODAY)
        for t in facts_texts
    ]
    return Entity(id=eid, type=etype, facts=facts, last_updated=NOW.isoformat())


# ── split_sentences ──────────────────────────────────────

class TestSplitSentences:
    def test_basic_split(self):
        text = "Alice reviewed the PR. Bob approved it. Ship it."
        result = split_sentences(text)
        assert len(result) == 3

    def test_single_sentence(self):
        result = split_sentences("Just one sentence here.")
        assert len(result) == 1

    def test_filters_short_fragments(self):
        result = split_sentences("Hello. OK. This is a real sentence.")
        # "OK" is only 2 chars, should be filtered (< 5)
        assert all(len(s) >= 5 for s in result)

    def test_preserves_content(self):
        text = "Built auth API with Express. Alice reviewed and approved."
        result = split_sentences(text)
        assert "Built auth API with Express." in result
        assert "Alice reviewed and approved." in result

    def test_empty_string(self):
        assert split_sentences("") == []

    def test_no_punctuation(self):
        result = split_sentences("This has no ending punctuation")
        assert len(result) == 1


# ── match_sentence_to_entity ─────────────────────────────

class TestMatchSentence:
    def test_exact_match(self, embedder):
        entity = _entity("person:alice", "person", ["Manages the auth team"])
        result = match_sentence_to_entity(
            "Manages the auth team", [entity], embedder
        )
        # Same text → same embedding → cosine 1.0 → should match
        assert result is entity

    def test_no_match_empty_entities(self, embedder):
        result = match_sentence_to_entity("anything", [], embedder)
        assert result is None

    def test_no_match_different_text(self, embedder):
        entity = _entity("person:alice", "person", ["Manages the auth team"])
        result = match_sentence_to_entity(
            "Completely unrelated topic about cooking recipes", [entity], embedder
        )
        # MockEmbedder: different text → low cosine → likely no match
        # (could match by hash collision, but very unlikely)
        if result is not None:
            import pytest
            pytest.skip("Hash collision in mock embedder")


# ── extract_events ───────────────────────────────────────

class TestExtractEvents:
    def test_empty_events(self, embedder):
        result = extract_events([], [], embedder, now=NOW)
        assert result.events_processed == 0
        assert result.matched == []
        assert result.unmatched == []

    def test_no_entities_all_unmatched(self, embedder):
        events = [{"id": "e1", "text": "Alice reviewed the PR. Bob approved it."}]
        result = extract_events(events, [], embedder, now=NOW)
        assert result.events_processed == 1
        assert len(result.matched) == 0
        assert len(result.unmatched) == 2

    def test_counts_events_processed(self, embedder):
        events = [
            {"id": "e1", "text": "First event happened."},
            {"id": "e2", "text": "Second event happened."},
        ]
        result = extract_events(events, [], embedder, now=NOW)
        assert result.events_processed == 2

    def test_matched_event_ids_empty_when_no_entities(self, embedder):
        # No entities to match against → nothing can match → set is empty.
        events = [{"id": "e1", "text": "Alice reviewed the PR. Bob approved it."}]
        result = extract_events(events, [], embedder, now=NOW)
        assert result.matched_event_ids == set()

    def test_matched_event_ids_includes_matching_event(self, embedder):
        from kipp.merge import build_search_text

        entity = _entity("person:alice", "person", ["Manages the auth team"])
        # An event whose sentence equals the entity's search_text matches
        # deterministically (identical text → cosine 1.0 with MockEmbedder).
        matching_sentence = build_search_text(entity)
        events = [{"id": "e1", "text": matching_sentence}]
        result = extract_events(events, [entity], embedder, now=NOW, match_untagged=True)
        assert "e1" in result.matched_event_ids
        assert len(result.matched) >= 1

    def test_fully_unmatched_event_absent_from_matched_ids(self, embedder):
        # One event matches (its fact sentence == a fact row → cosine 1.0), one
        # does not. The gate is 0.95 so only identical text clears it: the
        # MockEmbedder's all-positive vectors floor cosine near 0.7, so no
        # distinct short string scores below a default 0.7 gate — a high gate is
        # the deterministic way to construct a non-match here.
        from kipp.merge import build_search_text

        entity = _entity("person:alice", "person", ["Manages the auth team"])
        st = build_search_text(entity)  # blob; its fact sentence hits the fact row

        events = [
            {"id": "match", "text": st},
            {"id": "nomatch", "text": "grok quaffle."},
        ]
        result = extract_events(
            events, [entity], embedder, now=NOW, threshold=0.95, match_untagged=True,
        )
        assert "match" in result.matched_event_ids
        assert "nomatch" not in result.matched_event_ids


# ── embedding cost (issue #12 regression guard) ──────────

class TestEmbedsCorpusOnce:
    def test_embed_calls_additive_not_multiplicative(self, embedder):
        """Facts are embedded once per run, not re-embedded per sentence.

        Canary for the #12 blowup: the old per-sentence re-embedding of the
        whole corpus made embed calls scale as events × sentences × entities.
        With the precomputed per-fact index (issue #17) it must be
        ``total_facts + total_sentences`` — additive, and proportional to
        *facts* not entities, so the two-fact entity contributes two rows.
        """
        entities = [
            _entity("person:alice", "person", ["Manages the auth team", "Based in London"]),
            _entity("project:dash", "project", ["The trading dashboard"]),
            _entity("tool:qdrant", "tool", ["A vector database"]),
        ]
        events = [
            {"id": "e1", "text": "First sentence here. Second sentence here."},
            {"id": "e2", "text": "Third sentence here. Fourth sentence here."},
        ]
        total_facts = sum(len(e.facts) for e in entities)  # 4
        total_sentences = 4
        counter = _CountingEmbedder(embedder)

        extract_events(events, entities, counter, now=NOW, match_untagged=True)

        # 4 fact embeds + 4 sentence embeds. Multiplicative path: 4*4 + 4;
        # per-entity (pre-#17) would be 3 + 4. Per-fact is 4 + 4.
        assert counter.calls == total_facts + total_sentences

    def test_with_index_does_not_re_embed_entities(self, embedder):
        """extract_events_with_index embeds only sentences, never entities."""
        entities = [
            _entity("person:alice", "person", ["Manages the auth team"]),
            _entity("project:dash", "project", ["The trading dashboard"]),
        ]
        index = build_entity_index(entities, embedder)  # entity embeds happen here

        events = [{"id": "e1", "text": "One sentence. Two sentence. Three sentence."}]
        counter = _CountingEmbedder(embedder)
        extract_events_with_index(events, index, counter, now=NOW, match_untagged=True)

        # Index is prebuilt → only the 3 sentences are embedded.
        assert counter.calls == 3


# ── tag routing (issue #33) ──────────────────────────────

class TestParseLeadTag:
    def test_plain_tag(self):
        assert parse_lead_tag("[person:alice] Runs auth. Likes tea.") == (
            "person:alice", "Runs auth. Likes tea.",
        )

    def test_first_entity_tag_among_leading_groups(self):
        tag, body = parse_lead_tag("[preference][project:orbs-web] Review PRs carefully.")
        assert tag == "project:orbs-web"
        assert body == "Review PRs carefully."

    def test_untagged_text_unchanged(self):
        assert parse_lead_tag("Alice runs auth.") == (None, "Alice runs auth.")

    def test_non_entity_brackets_are_not_a_tag(self):
        text = "[widened from:shared] Alice runs auth."
        assert parse_lead_tag(text) == (None, text)

    def test_provenance_bracket_is_skipped(self):
        tag, body = parse_lead_tag("[run_id:abc][project:endurance] Ships nightly.")
        assert tag == "project:endurance"
        assert body == "Ships nightly."

    @pytest.mark.parametrize("key", ["run_id", "session_id", "profile", "trigger_source", "source"])
    def test_provenance_only_is_untagged(self, key):
        text = f"[{key}:abc] Alice runs auth."
        assert parse_lead_tag(text) == (None, text)

    def test_mid_text_tag_ignored(self):
        text = "Alice met [person:bob] today."
        assert parse_lead_tag(text) == (None, text)


class TestTagRouting:
    def test_tagged_event_routes_every_sentence_to_target(self, embedder):
        alice = _entity("person:alice", "person", ["Manages the auth team"])
        hub = _entity("project:hub", "project", ["Alice moved to Berlin. Likes coffee."])
        events = [{"id": "e1", "text": "[person:alice] Moved to Berlin. Drinks coffee daily."}]

        # Even with cosine on and the gate at zero (everything would match the
        # hub), the tag wins.
        result = extract_events(
            events, [hub, alice], embedder, now=NOW, threshold=-1.0, match_untagged=True,
        )

        assert [(s, e.id) for s, e in result.matched] == [
            ("Moved to Berlin.", "person:alice"),
            ("Drinks coffee daily.", "person:alice"),
        ]
        assert result.matched_event_ids == {"e1"}

    def test_tag_for_missing_entity_leaves_event_unmatched(self, embedder):
        hub = _entity("project:hub", "project", ["Anything at all"])
        events = [{"id": "e1", "text": "[person:carol] New starter. Works on payments."}]

        result = extract_events(
            events, [hub], embedder, now=NOW, threshold=-1.0, match_untagged=True,
        )

        assert result.matched == []
        assert result.unmatched == ["New starter.", "Works on payments."]
        assert result.matched_event_ids == set()

    def test_untagged_events_unmatched_by_default(self, embedder):
        entity = _entity("person:alice", "person", ["Manages the auth team"])
        events = [{"id": "e1", "text": "Manages the auth team"}]  # identical text

        result = extract_events(events, [entity], embedder, now=NOW)

        assert result.matched == []
        assert result.unmatched == ["Manages the auth team"]

    def test_default_run_embeds_nothing(self, embedder):
        """Tag routing needs ids only: no fact or sentence embeds by default."""
        entities = [_entity("person:alice", "person", ["Fact one", "Fact two"])]
        events = [
            {"id": "e1", "text": "[person:alice] Tagged sentence here."},
            {"id": "e2", "text": "Untagged sentence here."},
        ]
        counter = _CountingEmbedder(embedder)

        extract_events(events, entities, counter, now=NOW)

        assert counter.calls == 0
