"""Línea de comandos.

    python -m trading_ai.cli import mis_trades.csv
    python -m trading_ai.cli stats
    python -m trading_ai.cli diagnose
    python -m trading_ai.cli bias [--ai]
"""
from __future__ import annotations

import argparse
import json

from . import ai_coach, analytics, bias, diagnosis, fundamentals, technical
from .journal import Journal


def cmd_stats(j: Journal, _args) -> None:
    df = j.load()
    print(json.dumps(analytics.summary(df), indent=2, ensure_ascii=False, default=str))
    for col in ("pair", "session", "setup"):
        print(f"\n--- Por {col} ---")
        print(analytics.breakdown(df, col).to_string())


def cmd_diagnose(j: Journal, args) -> None:
    df = j.load()
    rep = diagnosis.Diagnoser(df).pattern_report()
    print("CONCLUSIONES")
    for c in rep["conclusiones"]:
        print(" -", c)
    print("\nCONFLUENCIAS\n", analytics.confluence_lift(df).to_string())
    if args.ai:
        bd = {k: analytics.breakdown(df, k) for k in ("session", "setup", "pair", "emotion")}
        print("\n" + ai_coach.coach_journal(df, analytics.summary(df), rep, analytics.confluence_lift(df), bd))


def cmd_bias(j: Journal, args) -> None:
    tech = {}
    for pair in ("EURUSD", "DXY"):
        try:
            tech[pair] = technical.technical_bias(technical.fetch_ohlc(pair), pair)
        except Exception as e:  # noqa: BLE001
            print(f"[aviso] sin datos técnicos de {pair}: {e}")
            tech[pair] = None
    try:
        impacts = [fundamentals.analyze_event(e) for e in fundamentals.relevant_events(fundamentals.fetch_calendar())]
    except Exception as e:  # noqa: BLE001
        print(f"[aviso] sin calendario: {e}")
        impacts = []
    for i in impacts:
        print(f"* {i.event.date:%a %d %H:%M} {i.event.currency} {i.event.title}: {i.explanation}")
    heads = fundamentals.fetch_headlines()
    res = bias.combine(tech, fundamentals.fundamental_score(impacts), fundamentals.headline_sentiment(heads))
    ai_probs = None
    if args.ai:
        brief = ai_coach.market_brief(
            {p: {"score": t.score, "levels": t.levels} for p, t in tech.items() if t},
            [{"titulo": i.event.title, "divisa": i.event.currency, "real": i.event.actual, "prevision": i.event.forecast}
             for i in impacts],
            [h["title"] for h in heads], {p: b.prob_up for p, b in res.items()})
        print(brief["markdown"])
        ai_probs = brief["prob_up"] or None
        res = bias.combine(tech, fundamentals.fundamental_score(impacts), fundamentals.headline_sentiment(heads), ai_probs)
    for b in res.values():
        print(b.as_text())
        for r in b.reasons:
            print("   -", r)


def main() -> None:
    p = argparse.ArgumentParser(description="IA de trading EURUSD/DXY")
    p.add_argument("--db", help="ruta de la base de datos")
    sub = p.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import")
    imp.add_argument("csv")
    sub.add_parser("stats")
    d = sub.add_parser("diagnose")
    d.add_argument("--ai", action="store_true")
    b = sub.add_parser("bias")
    b.add_argument("--ai", action="store_true")
    args = p.parse_args()
    j = Journal(args.db)
    if args.cmd == "import":
        print(f"{j.import_csv(args.csv)} operaciones importadas")
    else:
        {"stats": cmd_stats, "diagnose": cmd_diagnose, "bias": cmd_bias}[args.cmd](j, args)


if __name__ == "__main__":
    main()
