"""Catálogo de mercados. Cada mercado es una condición sobre (local, visitante) de un "ámbito":
goles a final (ft), goles al descanso (ht), córners o tarjetas. La misma condición sirve para
sumar la matriz de probabilidades del modelo y para contar cuántas veces ocurrió en el histórico."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Market:
    id: str
    label: str
    family: str   # mercados de la misma familia son líneas alternativas (se elige una)
    scope: str    # ft | ht | corners | cards
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray]


def _m(id, label, family, scope, fn):
    return Market(id, label, family, scope, fn)


MARKETS: list[Market] = [
    _m("1", "Gana el local", "1x2", "ft", lambda h, a: h > a),
    _m("X", "Empate", "1x2", "ft", lambda h, a: h == a),
    _m("2", "Gana el visitante", "1x2", "ft", lambda h, a: h < a),
    _m("1X", "Doble oportunidad 1X", "doble", "ft", lambda h, a: h >= a),
    _m("X2", "Doble oportunidad X2", "doble", "ft", lambda h, a: h <= a),
    _m("12", "Doble oportunidad 12 (no hay empate)", "doble", "ft", lambda h, a: h != a),
    _m("BTTS_SI", "Ambos marcan: Sí", "btts", "ft", lambda h, a: (h > 0) & (a > 0)),
    _m("BTTS_NO", "Ambos marcan: No", "btts", "ft", lambda h, a: (h == 0) | (a == 0)),
    _m("L_MARCA", "El local marca", "local_marca", "ft", lambda h, a: h > 0),
    _m("L_O1.5", "El local marca 2 o más", "local_marca", "ft", lambda h, a: h >= 2),
    _m("V_MARCA", "El visitante marca", "visit_marca", "ft", lambda h, a: a > 0),
    _m("V_O1.5", "El visitante marca 2 o más", "visit_marca", "ft", lambda h, a: a >= 2),
    _m("L_U1.5", "El local marca menos de 2", "local_menos", "ft", lambda h, a: h <= 1),
    _m("V_U1.5", "El visitante marca menos de 2", "visit_menos", "ft", lambda h, a: a <= 1),
    _m("1X_O1.5", "1X y más de 1.5 goles", "combo_1x", "ft", lambda h, a: (h >= a) & (h + a >= 2)),
    _m("1X_U3.5", "1X y menos de 3.5 goles", "combo_1x", "ft", lambda h, a: (h >= a) & (h + a <= 3)),
    _m("X2_O1.5", "X2 y más de 1.5 goles", "combo_x2", "ft", lambda h, a: (h <= a) & (h + a >= 2)),
    _m("X2_U3.5", "X2 y menos de 3.5 goles", "combo_x2", "ft", lambda h, a: (h <= a) & (h + a <= 3)),
    _m("1_O1.5", "Gana el local y más de 1.5 goles", "combo_1", "ft", lambda h, a: (h > a) & (h + a >= 2)),
    _m("2_O1.5", "Gana el visitante y más de 1.5 goles", "combo_2", "ft", lambda h, a: (h < a) & (h + a >= 2)),
    _m("HT_O0.5", "Gol en la 1ª parte (más de 0.5)", "ht_over", "ht", lambda h, a: h + a >= 1),
    _m("HT_O1.5", "Más de 1.5 goles en la 1ª parte", "ht_over", "ht", lambda h, a: h + a >= 2),
    _m("HT_U1.5", "Menos de 1.5 goles en la 1ª parte", "ht_under", "ht", lambda h, a: h + a <= 1),
    _m("HT_1X", "Descanso: 1X", "ht_doble", "ht", lambda h, a: h >= a),
    _m("HT_X2", "Descanso: X2", "ht_doble", "ht", lambda h, a: h <= a),
    _m("HT_X", "Descanso: empate", "ht_1x2", "ht", lambda h, a: h == a),
    _m("C_L_MAS", "Más córners el local", "c_1x2", "corners", lambda h, a: h > a),
    _m("C_V_MAS", "Más córners el visitante", "c_1x2", "corners", lambda h, a: h < a),
    _m("CARDS_L_MAS", "Más tarjetas el local", "k_1x2", "cards", lambda h, a: h > a),
    _m("CARDS_V_MAS", "Más tarjetas el visitante", "k_1x2", "cards", lambda h, a: h < a),
]

for _x in (0.5, 1.5, 2.5, 3.5, 4.5):
    MARKETS.append(_m(f"O{_x}", f"Más de {_x} goles", "goles_over", "ft", lambda h, a, x=_x: h + a > x))
for _x in (1.5, 2.5, 3.5, 4.5):
    MARKETS.append(_m(f"U{_x}", f"Menos de {_x} goles", "goles_under", "ft", lambda h, a, x=_x: h + a < x))
for _x in (6.5, 7.5, 8.5, 9.5, 10.5, 11.5):
    MARKETS.append(_m(f"C_O{_x}", f"Más de {_x} córners", "c_over", "corners", lambda h, a, x=_x: h + a > x))
for _x in (8.5, 9.5, 10.5, 11.5, 12.5):
    MARKETS.append(_m(f"C_U{_x}", f"Menos de {_x} córners", "c_under", "corners", lambda h, a, x=_x: h + a < x))
for _x in (2.5, 3.5, 4.5, 5.5):
    MARKETS.append(_m(f"C_L_O{_x}", f"Local más de {_x} córners", "c_local", "corners", lambda h, a, x=_x: h > x))
    MARKETS.append(_m(f"C_V_O{_x}", f"Visitante más de {_x} córners", "c_visit", "corners", lambda h, a, x=_x: a > x))
for _x in (1.5, 2.5, 3.5, 4.5, 5.5):
    MARKETS.append(_m(f"K_O{_x}", f"Más de {_x} tarjetas", "k_over", "cards", lambda h, a, x=_x: h + a > x))
for _x in (3.5, 4.5, 5.5, 6.5):
    MARKETS.append(_m(f"K_U{_x}", f"Menos de {_x} tarjetas", "k_under", "cards", lambda h, a, x=_x: h + a < x))
for _x in (0.5, 1.5, 2.5):
    MARKETS.append(_m(f"K_L_O{_x}", f"Local más de {_x} tarjetas", "k_local", "cards", lambda h, a, x=_x: h > x))
    MARKETS.append(_m(f"K_V_O{_x}", f"Visitante más de {_x} tarjetas", "k_visit", "cards", lambda h, a, x=_x: a > x))

MARKETS_BY_ID = {m.id: m for m in MARKETS}

# Columnas del histórico que alimentan cada ámbito (local, visitante)
SCOPE_COLS = {"ft": ("FTHG", "FTAG"), "ht": ("HTHG", "HTAG"), "corners": ("HC", "AC"), "cards": ("HK", "AK")}
SCOPE_NAMES = {"ft": "Goles", "ht": "1ª parte", "corners": "Córners", "cards": "Tarjetas"}
SCOPE_MAX = {"ft": 10, "ht": 7, "corners": 25, "cards": 14}

# Cuotas de casa disponibles en football-data.co.uk para comparar
BOOK_ODDS = {"1": ["B365H", "AvgH", "PSH"], "X": ["B365D", "AvgD", "PSD"], "2": ["B365A", "AvgA", "PSA"],
             "O2.5": ["B365>2.5", "Avg>2.5"], "U2.5": ["B365<2.5", "Avg<2.5"]}


def add_cards(df):
    """Tarjetas por equipo = amarillas + rojas (la convención más habitual en las casas)."""
    if {"HY", "AY"}.issubset(df.columns):
        df = df.copy()
        df["HK"] = df["HY"] + (df["HR"].fillna(0) if "HR" in df.columns else 0)
        df["AK"] = df["AY"] + (df["AR"].fillna(0) if "AR" in df.columns else 0)
    return df


def market_probs(matrix: np.ndarray, scope: str) -> dict[str, float]:
    g = np.arange(matrix.shape[0])
    h, a = np.meshgrid(g, np.arange(matrix.shape[1]), indexing="ij")
    return {m.id: float(matrix[m.fn(h, a)].sum()) for m in MARKETS if m.scope == scope}


def market_outcomes(hvals, avals, scope: str) -> dict[str, np.ndarray]:
    h = np.asarray(hvals, dtype=float)
    a = np.asarray(avals, dtype=float)
    return {m.id: m.fn(h, a) for m in MARKETS if m.scope == scope}
