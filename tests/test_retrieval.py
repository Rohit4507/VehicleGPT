"""Retrieval tests: FTS query safety, ranking, and rank fusion.

**No recall@8 number is asserted here, and none is claimed in the docs.** With
three cards on the shipped store, recall@8 is 1.0 for any query that matches
anything, which measures the corpus, not the retriever. What is tested is that
the mechanism ranks correctly on a synthetic corpus and that free text cannot
become FTS5 syntax.
"""

from __future__ import annotations

import pytest

from vehiclegpt.analytics import retrieval
from vehiclegpt.store import db

CARDS = [
    ("incident", "2026-03-11", "critical",
     "Overheating incident", "The coolant pump stopped and pack temperature climbed."),
    ("incident", "2026-03-15", "info",
     "ADAS disengagement sequence", "Adaptive cruise disengaged three times in seven minutes."),
    ("trend", "2026-03-12", "warning",
     "Consumption rose", "Energy per kilometre rose after the cold front arrived."),
    ("incident", "2026-03-09", "warning",
     "Tyre pressure warning", "Rear left tyre pressure fell below the threshold."),
    ("incident", "2026-03-04", "info",
     "Charging session interrupted", "The charge port latch released early."),
]


@pytest.fixture
def store(tmp_path):
    conn = db.connect(tmp_path / "r.db")
    conn.executemany(
        "INSERT INTO card (kind, start_ns, end_ns, day, severity, title, body, signals, "
        "evidence) VALUES (?,?,?,?,?,?,?,?,?)",
        [(kind, 1, 2, day, sev, title, body, "", "[]")
         for kind, day, sev, title, body in CARDS])
    conn.execute("INSERT INTO card_fts(card_fts) VALUES('rebuild')")
    conn.commit()
    yield conn
    conn.close()


class TestFtsQuery:
    def test_tokens_are_quoted(self):
        assert retrieval.fts_query("pump stopped") == '"pump" OR "stopped"'

    def test_stopwords_are_dropped(self):
        assert retrieval.fts_query("when did the pump stop") == '"pump" OR "stop"'

    def test_a_query_of_only_stopwords_expands_to_nothing(self):
        assert retrieval.fts_query("what was it that they did") == ""

    @pytest.mark.parametrize("hostile", [
        'NEAR(pump temp, 5)',
        'pump*',
        '"unbalanced quote',
        'a OR b AND c',
        'title:secret',
        'pump NOT engine',
        '^start',
        '{}[]()',
    ])
    def test_fts_operators_cannot_survive_as_operators(self, hostile):
        """Free text must not be able to express FTS5 syntax."""
        expanded = retrieval.fts_query(hostile)
        if not expanded:
            return
        # Every token is inside double quotes, so none can act as an operator.
        stripped = expanded.replace(" OR ", "")
        assert stripped.startswith('"') and stripped.endswith('"')
        assert stripped.count('"') % 2 == 0

    def test_a_quote_in_the_input_never_reaches_the_expression(self):
        """The tokeniser cannot emit a quote, so quoting cannot be broken out of."""
        for hostile in ['pu"mp', 'pu""mp', '"', 'a"b"c']:
            expanded = retrieval.fts_query(hostile)
            if not expanded:
                continue
            for piece in expanded.split(" OR "):
                assert piece.startswith('"') and piece.endswith('"')
                assert piece.count('"') == 2, f"unbalanced quoting in {piece!r}"

    def test_an_apostrophe_inside_a_word_is_preserved_safely(self):
        expanded = retrieval.fts_query("pump's stall")
        assert expanded == '"pump\'s" OR "stall"'

    def test_empty_and_punctuation_only(self):
        assert retrieval.fts_query("") == ""
        assert retrieval.fts_query("?!,.") == ""

    def test_a_hostile_query_is_executable_rather_than_fatal(self, store):
        for hostile in ('NEAR(pump temp, 5)', 'pump*', '"unbalanced', 'a OR b AND c'):
            retrieval.search(store, hostile)      # must not raise


class TestSearch:
    def test_the_right_card_ranks_first(self, store):
        hits = retrieval.search(store, "coolant pump temperature")
        assert hits and hits[0]["title"] == "Overheating incident"

    def test_paraphrase_matches_through_stemming(self, store):
        """'disengaged' should find 'disengagement' via the porter stemmer."""
        hits = retrieval.search(store, "why did it disengage")
        assert hits and "ADAS" in hits[0]["title"]

    def test_no_match_returns_empty_rather_than_a_near_miss(self, store):
        assert retrieval.search(store, "windscreen washer fluid") == []

    def test_kind_filter(self, store):
        hits = retrieval.search(store, "consumption", kind="trend")
        assert [h["kind"] for h in hits] == ["trend"]

    def test_limit_is_respected(self, store):
        assert len(retrieval.search(store, "pressure temperature charge", limit=2)) <= 2

    def test_evidence_is_decoded_for_the_caller(self, store):
        hits = retrieval.search(store, "coolant")
        assert isinstance(hits[0]["evidence"], list)

    def test_rank_is_returned_negative_better(self, store):
        hits = retrieval.search(store, "coolant pump")
        assert hits[0]["rank"] <= hits[-1]["rank"]


class TestRankFusion:
    def test_combines_two_rankings(self):
        fused = retrieval.reciprocal_rank_fusion([1, 2, 3], [3, 1, 9])
        assert set(fused) == {1, 2, 3, 9}

    def test_a_document_high_in_both_wins(self):
        fused = retrieval.reciprocal_rank_fusion([1, 2, 3], [1, 9, 8])
        assert fused[0] == 1

    def test_duplicate_within_one_ranking_is_counted_once(self):
        fused = retrieval.reciprocal_rank_fusion([1, 1, 1], [])
        assert fused == [1]

    def test_limit_applies(self):
        assert len(retrieval.reciprocal_rank_fusion(list(range(50)), list(range(50)), limit=3)) == 3

    def test_empty_input(self):
        assert retrieval.reciprocal_rank_fusion() == []

    def test_k_damps_the_head(self):
        """A smaller k makes the top rank dominate more."""
        small = retrieval.reciprocal_rank_fusion([1, 2, 3], [3, 2, 1], k=1)
        assert small[0] == 1


class TestBuildCards:
    """build_cards distils from `event`; it does not preserve hand-inserted rows."""

    @pytest.fixture
    def events(self, tmp_path):
        conn = db.connect(tmp_path / "e.db")
        conn.executemany(
            "INSERT INTO event (ts_ns, kind, severity, message) VALUES (?,?,?,?)",
            [(1_772_400_000 * 10**9, "adas.disengagement", "info", "ADAS disengaged"),
             (1_772_400_400 * 10**9, "adas.disengagement", "info", "ADAS disengaged"),
             (1_772_500_000 * 10**9, "fault.adas_disengagement", "info", "injected")])
        conn.commit()
        yield conn
        conn.close()

    def test_rebuild_is_deterministic_and_does_not_duplicate(self, events):
        first = retrieval.build_cards(events)
        snapshot = [dict(r) for r in events.execute("SELECT id, title, body FROM card ORDER BY id")]
        second = retrieval.build_cards(events)
        assert first == second
        assert snapshot == [dict(r) for r in
                            events.execute("SELECT id, title, body FROM card ORDER BY id")]

    def test_harness_faults_never_become_cards(self, events):
        retrieval.build_cards(events)
        bodies = [r["body"] for r in events.execute("SELECT body FROM card")]
        assert bodies, "the real events should produce a card"
        assert all("injected" not in b for b in bodies)

    def test_cards_are_searchable_after_a_rebuild(self, events):
        retrieval.build_cards(events)
        assert any("adas" in h["title"].lower()
                   for h in retrieval.search(events, "disengagement"))
