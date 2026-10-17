"""Línea de comandos.

  python -m futbol_ai.cli finde [--ligas SP1,E0] [--desde AAAA-MM-DD] [--hasta ...] [--html informe.html]
  python -m futbol_ai.cli finde --liga SP1 --partidos "Real Madrid - Barcelona" "Betis - Sevilla"
  python -m futbol_ai.cli partido "Real Madrid" "Barcelona" --liga SP1
  python -m futbol_ai.cli backtest --liga SP1 [--desde 2025-08-01]
  python -m futbol_ai.cli actualizar --ligas SP1,E0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import backtest, config, data, report
from .predictor import LeaguePredictor, build_combo, predict_fixtures, select_picks


def _leagues(s: str | None) -> list[str]:
    return [x.strip().upper() for x in s.split(",")] if s else ["SP1", "E0", "I1", "D1", "F1"]


def _apply_thresholds(preds, args):
    for p in preds:
        p.picks = select_picks(p.rows, args.prob_min, args.cuota_min, args.cuota_max)


def cmd_finde(args):
    d0, d1 = data.weekend_range()
    d0 = pd.Timestamp(args.desde).date() if args.desde else d0
    d1 = pd.Timestamp(args.hasta).date() if args.hasta else d1
    if args.partidos:
        lg = (args.liga or "SP1").upper()
        fixtures = data.parse_fixtures_text("\n".join(args.partidos), lg)
        leagues = [lg]
    else:
        leagues = _leagues(args.ligas)
        print(f"Buscando partidos del {d0} al {d1} en {', '.join(leagues)}...")
        fixtures = data.load_fixtures(leagues, d0, d1)
        if fixtures.empty:
            sys.exit("No se encontraron partidos (sin conexión a football-data.co.uk o no hay jornada). "
                     "Usa --partidos \"Local - Visitante\" --liga SP1.")
    print("Descargando histórico (últimos 5 años)...")
    hist = data.load_leagues(sorted(set(fixtures["Div"]) | set(leagues)))
    preds = predict_fixtures(hist, fixtures)
    _apply_thresholds(preds, args)
    for p in preds:
        print(report.match_text(p))
    combo = build_combo(preds)
    print(report.combo_text(combo))
    if args.html:
        Path(args.html).write_text(report.to_html(preds, combo), encoding="utf-8")
        print(f"\nInforme HTML guardado en {args.html}")


def cmd_partido(args):
    lg = args.liga.upper()
    pred = LeaguePredictor(data.load_league(lg), lg).predict(args.local, args.visitante)
    pred.picks = select_picks(pred.rows, args.prob_min, args.cuota_min, args.cuota_max)
    print(report.match_text(pred))
    if args.todo:
        cols = ["Mercado", "p_model", "p_hist", "p_final", "cuota_justa", "local_casa", "visit_fuera", "h2h"]
        print(pred.rows.sort_values("p_final", ascending=False)[cols].to_string(index=False, float_format="%.2f"))


def cmd_backtest(args):
    lg = args.liga.upper()
    df = data.load_league(lg)
    print(f"Backtest walk-forward de {lg} ({len(df)} partidos de histórico)...")
    res = backtest.run(df, lg, test_from=args.desde)
    print("\nCalibración (probabilidad dicha vs lo que pasó):")
    print(backtest.calibration(res).to_string(index=False, float_format="%.3f"))
    print("\nPicks que habría dado la IA:")
    print(backtest.picks_summary(res).to_string(float_format="%.3f"))
    v = backtest.value_summary(res)
    if not v.empty:
        print("\nApostando solo con valor frente a la casa:")
        print(v.to_string(index=False, float_format="%.3f"))
    print("\nMezcla modelo / histórico (menor log-loss = mejor):")
    print(backtest.tune(res).head(8).to_string(index=False, float_format="%.4f"))


def cmd_actualizar(args):
    for lg in _leagues(args.ligas):
        for s in data.seasons_to_load():
            df = data.download_season(lg, s, force=True)
            print(f"{lg} {s}: {len(df)} partidos")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="futbol_ai", description="Pronósticos de fútbol basados en estadística")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def thresholds(p):
        p.add_argument("--prob-min", type=float, default=config.MIN_PROB)
        p.add_argument("--cuota-min", type=float, default=config.MIN_ODDS)
        p.add_argument("--cuota-max", type=float, default=config.MAX_ODDS)

    f = sub.add_parser("finde", help="Pronósticos de los partidos del fin de semana")
    f.add_argument("--ligas"), f.add_argument("--liga"), f.add_argument("--desde"), f.add_argument("--hasta")
    f.add_argument("--partidos", nargs="*"), f.add_argument("--html")
    thresholds(f)
    f.set_defaults(func=cmd_finde)

    p = sub.add_parser("partido", help="Un partido concreto")
    p.add_argument("local"), p.add_argument("visitante"), p.add_argument("--liga", default="SP1")
    p.add_argument("--todo", action="store_true", help="Muestra todos los mercados")
    thresholds(p)
    p.set_defaults(func=cmd_partido)

    b = sub.add_parser("backtest", help="Comprueba la fiabilidad con temporadas pasadas")
    b.add_argument("--liga", default="SP1"), b.add_argument("--desde")
    b.set_defaults(func=cmd_backtest)

    u = sub.add_parser("actualizar", help="Fuerza la descarga de datos")
    u.add_argument("--ligas")
    u.set_defaults(func=cmd_actualizar)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
