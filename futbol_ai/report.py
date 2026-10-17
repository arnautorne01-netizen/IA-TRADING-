"""Informe del fin de semana en texto (terminal) y en HTML autocontenido."""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd

from .config import LEAGUES
from .markets import SCOPE_NAMES
from .predictor import MatchPrediction


DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]


def _fecha(d) -> str:
    return f"{DIAS[d.weekday()]} {d:%d/%m}" if d is not None else ""


def _pct(p) -> str:
    return "—" if p is None or pd.isna(p) else f"{100 * p:.0f}%"


def _odd(o) -> str:
    return "—" if o is None or pd.isna(o) else f"{o:.2f}"


def most_frequent(pred: MatchPrediction, n: int = 6) -> pd.DataFrame:
    """Lo que más se ha repetido en el histórico de estos equipos (con muestra suficiente),
    sin mercados triviales (cuota justa < 1.15)."""
    r = pred.rows[(pred.rows["n_hist"] >= 8) & (pred.rows["cuota_justa"] >= 1.15)]
    r = r.sort_values("p_hist", ascending=False)
    return r.groupby("family", as_index=False).head(1).head(n)


def match_text(p: MatchPrediction) -> str:
    lg = LEAGUES.get(p.league, (p.league,))[0]
    d = _fecha(p.date)
    lines = [f"\n{'=' * 78}", f"{lg} · {d}   {p.home}  vs  {p.away}", "=" * 78]
    for scope, (h, a) in p.expected.items():
        lines.append(f"  {SCOPE_NAMES[scope]:<9} esperados: {h:.2f} - {a:.2f}  (total {h + a:.2f})")
    lines.append("  Marcadores más probables: " + ", ".join(f"{s} ({_pct(q)})" for s, q in p.top_scores))
    for w in p.warnings:
        lines.append(f"  ⚠ {w}")
    lines.append("\n  Lo que más se ha repetido (local en casa · visitante fuera · H2H):")
    for _, r in most_frequent(p).iterrows():
        lines.append(f"    - {r['Mercado']:<38} {_pct(r['p_hist']):>4}  ({r['local_casa']} · {r['visit_fuera']} · {r['h2h']})")
    if p.picks.empty:
        lines.append("\n  Sin picks: ningún mercado supera el umbral con cuota razonable.")
    else:
        lines.append(f"\n  {'PICK':<40}{'Prob':>6}{'Justa':>7}{'Mín.':>7}{'Casa':>7}  Confianza")
        for _, r in p.picks.iterrows():
            lines.append(f"  {r['Mercado']:<40}{_pct(r['p_final']):>6}{_odd(r['cuota_justa']):>7}"
                         f"{_odd(r['cuota_minima']):>7}{_odd(r['cuota_casa']):>7}  {r['confianza']}")
    return "\n".join(lines)


def combo_text(combo: dict | None) -> str:
    if not combo:
        return ""
    legs = "\n".join(f"    - {l['Partido']}: {l['Mercado']} ({_pct(l['p'])})" for l in combo["legs"])
    return (f"\nCOMBINADA SUGERIDA · probabilidad {_pct(combo['prob'])} · cuota justa {combo['cuota_justa']:.2f}\n{legs}")


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--mute:#667085;--line:#e4e7ec;--acc:#0b7a53;--warn:#b54708;--chip:#ecfdf3}
@media (prefers-color-scheme:dark){:root{--bg:#0f1218;--card:#171b23;--ink:#e6e8ec;--mute:#98a2b3;--line:#2a303b;--acc:#3ccb8f;--warn:#f79009;--chip:#12301f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px}h1{font-size:24px;margin:0 0 4px}.mute{color:var(--mute)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:16px 0}
.card h2{font-size:18px;margin:0}.meta{font-size:13px;color:var(--mute);margin-bottom:8px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--mute);font-weight:600;font-size:12px;text-transform:uppercase}td.n{text-align:right;font-variant-numeric:tabular-nums}
.chip{display:inline-block;padding:1px 8px;border-radius:99px;background:var(--chip);color:var(--acc);font-size:12px;font-weight:600}
.warn{color:var(--warn);font-size:13px}.wrap{overflow-x:auto}details{margin-top:10px}summary{cursor:pointer;color:var(--mute)}
.grid{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:14px;margin:6px 0 10px}
"""


def _table(df: pd.DataFrame, cols: list[tuple[str, str, callable]]) -> str:
    head = "".join(f"<th>{html.escape(t)}</th>" for t, _, _ in cols)
    body = ""
    for _, r in df.iterrows():
        cells = []
        for _, c, f in cols:
            v = f(r[c]) if f else r[c]
            cls = ' class="n"' if f else ""
            cells.append(f"<td{cls}>{v if str(v).startswith('<span') else html.escape(str(v))}</td>")
        body += "<tr>" + "".join(cells) + "</tr>"
    return f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


PICK_COLS = [("Mercado", "Mercado", None), ("Prob.", "p_final", _pct), ("Cuota justa", "cuota_justa", _odd),
             ("Cuota mínima", "cuota_minima", _odd), ("Casa", "cuota_casa", _odd),
             ("Local en casa", "local_casa", None), ("Visit. fuera", "visit_fuera", None), ("H2H", "h2h", None),
             ("Confianza", "confianza", lambda v: f'<span class="chip">{v}</span>')]
ALL_COLS = [("Mercado", "Mercado", None), ("Modelo", "p_model", _pct), ("Histórico", "p_hist", _pct),
            ("Final", "p_final", _pct), ("Cuota justa", "cuota_justa", _odd),
            ("Local en casa", "local_casa", None), ("Visit. fuera", "visit_fuera", None), ("H2H", "h2h", None)]
FREQ_COLS = [("Lo que más se repite", "Mercado", None), ("Frecuencia", "p_hist", _pct),
             ("Local en casa", "local_casa", None), ("Visit. fuera", "visit_fuera", None), ("H2H", "h2h", None)]


def match_html(p: MatchPrediction) -> str:
    lg = LEAGUES.get(p.league, (p.league,))[0]
    d = _fecha(p.date)
    exp = "".join(f"<span><b>{SCOPE_NAMES[s]}</b> {h:.2f} – {a:.2f}</span>" for s, (h, a) in p.expected.items())
    scores = ", ".join(f"{s} ({_pct(q)})" for s, q in p.top_scores)
    warns = "".join(f'<div class="warn">⚠ {html.escape(w)}</div>' for w in p.warnings)
    picks = _table(p.picks, PICK_COLS) if not p.picks.empty else '<p class="mute">Sin picks: ningún mercado supera el umbral con cuota razonable.</p>'
    allm = p.rows.sort_values(["scope", "p_final"], ascending=[True, False])
    return f"""<section class="card"><div class="meta">{html.escape(lg)} · {d}</div>
<h2>{html.escape(p.home)} vs {html.escape(p.away)}</h2>
<div class="grid">{exp}</div><div class="meta">Marcadores más probables: {scores}</div>{warns}
<h3>Picks</h3>{picks}
<details><summary>Lo que más ha ocurrido en el histórico</summary>{_table(most_frequent(p, 8), FREQ_COLS)}</details>
<details><summary>Todos los mercados ({len(allm)})</summary>{_table(allm, ALL_COLS)}</details></section>"""


def to_html(preds: list[MatchPrediction], combo: dict | None = None, title: str = "Pronósticos del fin de semana") -> str:
    combo_html = ""
    if combo:
        rows = "".join(f"<tr><td>{html.escape(l['Partido'])}</td><td>{html.escape(l['Mercado'])}</td><td class='n'>{_pct(l['p'])}</td></tr>"
                       for l in combo["legs"])
        combo_html = (f'<section class="card"><h2>Combinada sugerida</h2><div class="meta">Probabilidad {_pct(combo["prob"])} · '
                      f'cuota justa {combo["cuota_justa"]:.2f}</div><table><tbody>{rows}</tbody></table></section>')
    body = "".join(match_html(p) for p in preds) or '<p class="mute">No hay partidos.</p>'
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>
<h1>{html.escape(title)}</h1><div class="mute">Generado el {datetime.now():%d/%m/%Y %H:%M} · Modelo Dixon-Coles con 5 temporadas + frecuencias históricas.
<b>Cuota justa</b> = 1/probabilidad. Apuesta solo si la casa paga al menos la <b>cuota mínima</b> (justa + 3%).</div>
{combo_html}{body}
<p class="mute">Probabilidades estadísticas, no certezas. Ninguna apuesta es segura; juega con responsabilidad.</p>
</main></body></html>"""
