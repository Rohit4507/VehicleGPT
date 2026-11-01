#!/usr/bin/env python3
"""Route a question, answer it, and check the answer's numbers.

    python3 tools/phase0/nlu.py --db /tmp/vehiclegpt.db --ask "why is my range down"
    python3 tools/phase0/nlu.py --db /tmp/vehiclegpt.db --eval holdout_v2
    python3 tools/phase0/nlu.py --db /tmp/vehiclegpt.db --grounding
    python3 tools/phase0/nlu.py --db /tmp/vehiclegpt.db --eval design --json

``--ask`` runs the whole path: sanitise, extract a window, route to a handler,
then re-read the finished sentence and verify every number in it against what
the handler computed. A question that routes nowhere abstains rather than
picking the closest handler, and the reason is printed.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from vehiclegpt.analytics import queries  # noqa: E402
from vehiclegpt.nlu import evaluate as ev  # noqa: E402
from vehiclegpt.nlu.guard import verify_numbers  # noqa: E402
from vehiclegpt.nlu.intent import parse_question  # noqa: E402

HANDLERS = {
    "range_reduction": lambda conn, intent: queries.range_reduction(conn),
    "efficiency_drop": lambda conn, intent: queries.efficiency_largest_drop(
        conn, min_km=intent.slots.get("min_km", 10.0)),
    "adas_events": lambda conn, intent: queries.adas_events(conn),
    "before_warning": lambda conn, intent: queries.before_warning(
        conn, lookback_min=intent.slots.get("lookback_min", 60.0)),
    "city_vs_highway": lambda conn, intent: queries.city_vs_highway(conn),
}


def open_store(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise SystemExit(f"no store at {path}; run tools/phase0/build_store.py")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def ask(conn: sqlite3.Connection, question: str, as_json: bool) -> int:
    now = ev.store_now(conn)
    intent = parse_question(question, now)
    if as_json:
        payload: dict = {"question": question, "routing": intent.to_dict()}
    else:
        where = intent.window.label if intent.window else "no window in the question"
        print(f"question  {question}")
        print(f"intent    {intent.name}  (score {intent.score:.1f}, {where})")
        for reason in intent.reasons[:4]:
            print(f"          {reason}")
        for finding in intent.findings:
            print(f"  ! {finding.severity}: {finding.rule} - {finding.excerpt!r}")

    if intent.blocked:
        if as_json:
            payload["refused"] = "input blocked by sanitise"
            print(json.dumps(payload, indent=2))
        else:
            print("\nrefused: the question contains an instruction pattern this "
                  "system will not follow. Nothing was executed.")
        return 2
    if intent.abstained:
        if as_json:
            payload["answer"] = None
            payload["reason"] = "no deterministic handler; would escalate to the model"
            print(json.dumps(payload, indent=2))
        else:
            print("\nno handler: this would escalate to the language model, which "
                  "costs seconds on constrained hardware. Refusing rather than "
                  "guessing.")
        return 1

    answer = HANDLERS[intent.name](conn, intent)
    pack = ev.pack_from_answer(answer)
    verification = verify_numbers(answer.text, pack)

    if as_json:
        payload["answer"] = answer.to_dict()
        payload["grounding"] = verification.to_dict()
        print(json.dumps(payload, indent=2))
    else:
        print(f"\n{answer.text}\n")
        for caveat in answer.caveats:
            print(f"  caveat: {caveat}")
        status = "ok" if verification.ok else "FAILED"
        print(f"\ngrounding {status}: {verification.checked} numbers, "
              f"{len(verification.ungrounded)} untraceable, "
              f"{verification.allowed} allowed forms")
        for miss in verification.ungrounded:
            print(f"  ! {miss['number']}  ...{miss['context']}...")
    return 0 if verification.ok else 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=Path("/tmp/vehiclegpt.db"))
    ap.add_argument("--ask", metavar="QUESTION")
    ap.add_argument("--eval", metavar="SET", nargs="?", const="holdout_v2",
                    choices=sorted(ev.SETS), help="score routing on a corpus")
    ap.add_argument("--grounding", action="store_true",
                    help="measure numeric grounding on the five brief questions")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    conn = open_store(args.db)
    if args.ask:
        return ask(conn, args.ask, args.json)

    if args.eval:
        report = ev.evaluate_routing(ev.store_now(conn), set_name=args.eval)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"{args.eval}: {report.correct}/{report.total} correct "
                  f"= {report.accuracy:.0%}")
            print(f"negatives wrongly claimed: "
                  f"{report.negatives['wrongly_claimed']}/{report.negatives['total']}")
            for name, stats in sorted(report.per_intent.items()):
                print(f"  {name:16s} {stats['correct']}/{stats['total']} "
                      f"recall {stats['recall']:.0%}")
            for miss in report.misses:
                print(f"  miss {miss['question']!r} -> {miss['got']} "
                      f"(wanted {miss['want']})")
        return 0

    if args.grounding:
        report = ev.evaluate_grounding(conn)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"grounding {report.checked - report.ungrounded}/{report.checked} "
                  f"= {report.rate:.1%} across {len(report.rows)} answers")
            for row in report.rows:
                flag = "ok  " if not row["ungrounded"] else "FAIL"
                print(f"  {flag} {row['intent']:24s} {row['checked']:3d} numbers")
                for example in row["examples"]:
                    print(f"        {example['number']}  ...{example['context']}...")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
