-- VehicleGPT local store.
--
-- Retention follows docs/06 §7: L0 (raw, in RAM) is not persisted; L1 hot is
-- decimated to 1 Hz here; rollups and segments are derived and rebuilt rather
-- than incrementally mutated, so the whole database is reproducible from the
-- source data.

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

-- Signal dimension. Canonical paths from vehiclegpt/schema/signals.yaml.
CREATE TABLE IF NOT EXISTS signal (
    id      INTEGER PRIMARY KEY,
    path    TEXT NOT NULL UNIQUE,
    unit    TEXT NOT NULL,
    tier    TEXT NOT NULL
);

-- L1 hot samples, decimated to 1 Hz by vehiclegpt/store/decimate.py.
--
-- Storage choices, each measured rather than assumed (docs/06 §7):
--   * WITHOUT ROWID with PRIMARY KEY (signal_id, ts_s) makes the table its own
--     index, removing a rowid pointer and a duplicate B-tree. 65.2 -> 24.2 B/row.
--   * ts_s, not ts_ns: a nanosecond epoch needs a 9-byte varint, a seconds epoch
--     needs 5. 24.2 -> 19.3 B/row.
--   * The old standalone index on (ts_ns) was 30% of the database and was never
--     used: every query filters by signal first. Dropped.
-- Together: 65.2 -> 19.3 bytes per sample, a 3.4x reduction with no loss.
CREATE TABLE IF NOT EXISTS sample (
    ts_s      INTEGER NOT NULL,            -- seconds since the Unix epoch
    signal_id INTEGER NOT NULL REFERENCES signal(id),
    value     REAL    NOT NULL,
    quality   INTEGER NOT NULL DEFAULT 0,  -- 0 = ok, 1 = out of bounds
    PRIMARY KEY (signal_id, ts_s)
) WITHOUT ROWID;

-- Discrete events. These are what "show me all X from yesterday" queries hit.
CREATE TABLE IF NOT EXISTS event (
    id       INTEGER PRIMARY KEY,
    ts_ns    INTEGER NOT NULL,
    kind     TEXT    NOT NULL,
    severity TEXT    NOT NULL,             -- info | warning | critical
    message  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_event_kind_ts ON event(kind, ts_ns);
CREATE INDEX IF NOT EXISTS ix_event_ts      ON event(ts_ns);

-- A drive is one contiguous powered period. Sessions are derived, not supplied.
CREATE TABLE IF NOT EXISTS drive (
    id         INTEGER PRIMARY KEY,
    start_ns   INTEGER NOT NULL,
    end_ns     INTEGER NOT NULL,
    distance_km REAL   NOT NULL,
    energy_wh  REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_drive_start ON drive(start_ns);

-- A segment is a stretch of a drive inside one speed regime. This is what makes
-- "city vs highway" answerable from data rather than from the simulator's
-- knowledge of which mode it was in.
CREATE TABLE IF NOT EXISTS segment (
    id          INTEGER PRIMARY KEY,
    drive_id    INTEGER NOT NULL REFERENCES drive(id),
    start_ns    INTEGER NOT NULL,
    end_ns      INTEGER NOT NULL,
    regime      TEXT    NOT NULL,          -- idle | city | highway
    distance_km REAL    NOT NULL,
    energy_wh   REAL    NOT NULL,
    mean_speed_kph REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_segment_regime ON segment(regime);

-- Daily rollup, rebuilt from `sample` and `segment`.
CREATE TABLE IF NOT EXISTS daily (
    day        TEXT PRIMARY KEY,           -- YYYY-MM-DD, UTC
    distance_km REAL,
    energy_wh   REAL,
    wh_per_km   REAL,
    ambient_c   REAL,
    hvac_w      REAL,
    soc_start   REAL,
    soc_end     REAL
);

-- Bookkeeping so a rebuild cannot silently use stale derived rows.
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Episode cards: the searchable unit. Derived from `event` and `daily`, rebuilt
-- by the same pass that builds segments and rollups. See docs/11 §11.8.
CREATE TABLE IF NOT EXISTS card (
    id        INTEGER PRIMARY KEY,
    kind      TEXT    NOT NULL,          -- incident | trend
    start_ns  INTEGER NOT NULL,
    end_ns    INTEGER NOT NULL,
    day       TEXT    NOT NULL,
    severity  TEXT    NOT NULL,
    title     TEXT    NOT NULL,
    body      TEXT    NOT NULL,
    signals   TEXT    NOT NULL,          -- comma-separated paths and event kinds
    evidence  TEXT    NOT NULL           -- JSON provenance pointers
);

CREATE INDEX IF NOT EXISTS ix_card_start ON card(start_ns);
CREATE INDEX IF NOT EXISTS ix_card_kind  ON card(kind);

-- External-content FTS5: the index stores no copy of the text, so it costs only
-- the inverted index. `porter` stemming lets "disengage" match "disengaged".
CREATE VIRTUAL TABLE IF NOT EXISTS card_fts USING fts5(
    title,
    body,
    content='card',
    content_rowid='id',
    tokenize='porter unicode61'
);
