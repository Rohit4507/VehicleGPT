"""Card storage and lexical retrieval over it.

Retrieval searches episode *cards*, not samples or events. That is the design
decision the whole layer exists for: on a device with 2 GB of memory you cannot
embed a million samples, but a few hundred cards fit in an index and in cache.

**What is implemented here is BM25 only.** The roadmap pairs FTS5 with a dense
vector index and fuses them with reciprocal rank fusion; the dense half needs
`bge-small-en-v1.5`, and HuggingFace is unreachable from the build sandbox, so
it is not implemented rather than faked. :func:`reciprocal_rank_fusion` is
written and tested against synthetic rankings so that adding the dense side later
is a plug-in rather than a redesign - but no number in this repository is
measured on a dense index that does not exist.

**The query is data, not syntax.** FTS5 has its own query language: `NEAR`,
`AND`, `*`, `:`, and quoted phrases. Passing user text straight into `MATCH`
lets that language through, so a question containing `NEAR(` becomes a syntax
error at best and a different query than the user asked at worst. Every token is
therefore quoted, which makes it a literal - the same principle as the DSL's
"the SQL is a function of the kind alone".
"""

from __future__ import annotations

import json
import re
import sqlite3

from vehiclegpt.analytics.episodes import Episode, distill

#: Runs of word characters and apostrophes; everything else separates tokens.
_TOKEN = re.compile(r"[\w']+", re.UNICODE)

#: Function words that carry no retrieval signal.
#:
#: This list is not decoration. With three cards, BM25's inverse document
#: frequency has almost nothing to work with, so a query like "when did the
#: coolant pump fail" matched a consumption card purely on "the" and "when" -
#: a confident-looking result with nothing behind it. A stopword list is a blunt
#: instrument, but it is the right one at this scale; IDF does the same job once
#: the corpus is large, and the two compose.
STOPWORDS = frozenset((
    "a", "an", "and", "are", "as", "at",
    "be", "been", "before", "but", "by", "did",
    "do", "does", "for", "from", "had", "has",
    "have", "how", "i", "if", "in", "into",
    "is", "it", "its", "me", "my", "of",
    "on", "or", "our", "over", "so", "than",
    "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "to", "was", "we",
    "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "you", "your",
))


def fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    Each token becomes a quoted literal joined with OR, so a deployment can
    retrieve anything containing *any* of the words rather than requiring all of
    them. Quoting is what stops operator syntax in the input from being
    interpreted.
    """
    tokens = [t for t in _TOKEN.findall(text)
              if (len(t) > 1 or t.isdigit()) and t.lower() not in STOPWORDS]
    if not tokens:
        return ""
    quoted = []
    for token in tokens:
        escaped = token.replace('"', '""')
        quoted.append(f'"{escaped}"')
    return " OR ".join(quoted)


def build_cards(conn: sqlite3.Connection) -> int:
    """Distil episodes into the `card` table and rebuild the FTS index.

    Cards are derived data, so this is idempotent: rebuilding produces the same
    rows, and the FTS index is rebuilt from the content table rather than
    patched, so it cannot drift out of sync with the cards it indexes.
    """
    episodes: list[Episode] = distill(conn)
    conn.execute("DELETE FROM card")
    conn.executemany(
        "INSERT INTO card (kind, start_ns, end_ns, day, severity, title, body, "
        "signals, evidence) VALUES (?,?,?,?,?,?,?,?,?)",
        [e.to_row() for e in episodes])
    conn.execute("INSERT INTO card_fts(card_fts) VALUES('rebuild')")
    conn.commit()
    return len(episodes)


def search(conn: sqlite3.Connection, text: str, limit: int = 8,
           kind: str | None = None) -> list[dict]:
    """BM25 search over cards. Empty result for empty or unmatched text."""
    match = fts_query(text)
    if not match:
        return []
    sql = ("SELECT c.id, c.kind, c.day, c.severity, c.title, c.body, c.signals, "
           "       c.evidence, bm25(card_fts) AS rank "
           "FROM card_fts JOIN card c ON c.id = card_fts.rowid "
           "WHERE card_fts MATCH ?")
    params: list = [match]
    if kind:
        sql += " AND c.kind = ?"
        params.append(kind)
    sql += " ORDER BY rank LIMIT ?"       # bm25() is negative-better; ascending is best first
    params.append(limit)
    out = []
    for row in conn.execute(sql, params):
        record = dict(row)
        record["evidence"] = json.loads(record["evidence"])
        out.append(record)
    return out


def reciprocal_rank_fusion(*rankings: list, k: int = 60, limit: int = 8) -> list:
    """Fuse ranked id lists: ``score(d) = sum(1 / (k + rank))``.

    Implemented and tested now so the dense side can be added without changing
    anything downstream. ``k=60`` is the value from the original Cormack et al.
    paper; it damps the influence of very high ranks, which is what makes RRF
    work when the two retrievers disagree.
    """
    scores: dict = {}
    for ranking in rankings:
        seen: set = set()
        for position, item in enumerate(ranking):
            key = item if isinstance(item, (int, str)) else item[0]
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + position + 1)
    return sorted(scores, key=lambda key: (-scores[key], str(key)))[:limit]
