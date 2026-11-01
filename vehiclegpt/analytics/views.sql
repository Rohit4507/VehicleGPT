-- Analytics views: the deterministic layer.
--
-- These answer questions with SQL, not with a language model. That is a
-- deliberate architectural choice, not a shortcut: on a constrained tier the
-- model cannot meet the latency target (docs/10), and a filter or an aggregate
-- does not need one. The numbers here are exact, reproducible and auditable,
-- and the model's job becomes explaining them rather than computing them.
--
-- Every view is derived. Nothing is stored twice; rebuild by dropping the views.

DROP VIEW IF EXISTS v_daily_rolling;
DROP VIEW IF EXISTS v_regime_energy;
DROP VIEW IF EXISTS v_adas_disengagements;
DROP VIEW IF EXISTS v_event_sequence;
DROP VIEW IF EXISTS v_warning_episodes;

-- Distance-weighted consumption over a trailing 7-day window, and over the
-- 7 days before that. Weighting by distance rather than averaging daily Wh/km
-- stops a 7.9 km weekend errand from counting as much as a 40 km commute.
--
-- Days with almost no driving (< 2 km) are excluded outright: a cold-start
-- crawl can read 900 Wh/km and would otherwise dominate the average.
CREATE VIEW v_daily_rolling AS
SELECT
    day,
    distance_km,
    energy_wh,
    wh_per_km,
    ambient_c,
    hvac_w,
    SUM(energy_wh)   OVER w7      / NULLIF(SUM(distance_km) OVER w7, 0)      AS ma7_wh_per_km,
    SUM(energy_wh)   OVER prior7  / NULLIF(SUM(distance_km) OVER prior7, 0)  AS prior7_wh_per_km,
    AVG(ambient_c)   OVER w7                                                  AS ma7_ambient_c,
    AVG(hvac_w)      OVER w7                                                  AS ma7_hvac_w,
    COUNT(*)         OVER prior7                                              AS prior7_days
FROM daily
WHERE distance_km > 2
WINDOW
    w7     AS (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),
    prior7 AS (ORDER BY day ROWS BETWEEN 13 PRECEDING AND 7 PRECEDING);

-- Energy per km by speed regime, over an arbitrary window the caller filters.
-- Consumption is SUM(Wh)/SUM(km), never AVG(Wh/km): averaging a ratio weights a
-- 30-second stint the same as a 15-minute one.
CREATE VIEW v_regime_energy AS
SELECT
    regime,
    COUNT(*)                        AS segments,
    SUM(distance_km)                AS distance_km,
    SUM(energy_wh)                  AS energy_wh,
    SUM(energy_wh) / NULLIF(SUM(distance_km), 0) AS wh_per_km,
    SUM(distance_km) / NULLIF(SUM(energy_wh), 0) * 1000 AS km_per_kwh,
    SUM(distance_km * mean_speed_kph) / NULLIF(SUM(distance_km), 0) AS mean_speed_kph,
    MIN(start_ns)                   AS first_ns,
    MAX(end_ns)                     AS last_ns
FROM segment
WHERE regime <> 'idle'
GROUP BY regime;

-- ADAS disengagement events. Injected faults are excluded: the question asks
-- what the vehicle did, and a row saying "we made this happen" is test harness
-- leakage, not evidence.
CREATE VIEW v_adas_disengagements AS
SELECT
    e.id,
    e.ts_ns,
    date(e.ts_ns / 1000000000, 'unixepoch') AS day,
    time(e.ts_ns / 1000000000, 'unixepoch') AS time_utc,
    e.severity,
    e.message
FROM event e
WHERE e.kind = 'adas.disengagement'
ORDER BY e.ts_ns;

-- Every warning/critical event paired with what came before it.
-- `lead_s` is how far back the predecessor event sits, so the ordering is
-- explicit rather than implied by row order.
CREATE VIEW v_event_sequence AS
SELECT
    w.id           AS warning_id,
    w.ts_ns        AS warning_ns,
    w.kind         AS warning_kind,
    w.severity     AS warning_severity,
    p.id           AS precursor_id,
    p.ts_ns        AS precursor_ns,
    p.kind         AS precursor_kind,
    p.severity     AS precursor_severity,
    p.message      AS precursor_message,
    (w.ts_ns - p.ts_ns) / 1000000000.0 AS lead_s
FROM event w
JOIN event p
  ON p.ts_ns < w.ts_ns
 AND p.ts_ns >= w.ts_ns - 3600 * 1000000000   -- one hour of lookback
WHERE w.severity IN ('warning', 'critical')
ORDER BY w.ts_ns, p.ts_ns;
