"""Combina análisis técnico + fundamental en un sesgo diario con probabilidad
aproximada de subida/bajada para EURUSD y DXY.

IMPORTANTE: las probabilidades son una estimación heurística, no una garantía.
Se limitan al rango 20%-80% a propósito: en FX nada es seguro. Con el registro
diario (bias_log) puedes medir qué tan acertado es el modelo con el tiempo.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .technical import TechnicalBias, score_label

TECH_WEIGHT = 0.30
FUND_WEIGHT = 0.60
NEWS_WEIGHT = 0.10
# Peso de la confirmación inversa entre EURUSD y DXY
CROSS_WEIGHT = 0.30
STEEPNESS = 2.2
PROB_FLOOR, PROB_CEIL = 0.20, 0.80


def to_probability(score: float) -> float:
    p = 1 / (1 + np.exp(-STEEPNESS * score))
    return float(np.clip(p, PROB_FLOOR, PROB_CEIL))


@dataclass
class DailyBias:
    pair: str
    technical: float
    fundamental: float
    news: float
    combined: float
    prob_up: float
    reasons: list[str] = field(default_factory=list)

    @property
    def prob_down(self) -> float:
        return 1 - self.prob_up

    @property
    def label(self) -> str:
        return score_label(self.combined)

    def as_text(self) -> str:
        return (f"{self.pair}: sesgo {self.label.upper()} — probabilidad aprox. de subida {self.prob_up:.0%} / "
                f"bajada {self.prob_down:.0%} (técnico {self.technical:+.2f}, fundamental {self.fundamental:+.2f}, "
                f"titulares {self.news:+.2f})")


def combine(tech: dict[str, TechnicalBias | None], fund: dict[str, float], news: dict[str, float],
            ai_prob: dict[str, float] | None = None, ai_weight: float = 0.35) -> dict[str, DailyBias]:
    """tech: {'EURUSD': TechnicalBias, 'DXY': TechnicalBias}
    fund/news: {'EURUSD': score, 'DXY': score} en -1..1
    ai_prob: probabilidades de subida opcionales devueltas por Claude (0..1)."""
    t_eur = tech.get("EURUSD").score if tech.get("EURUSD") else None
    t_dxy = tech.get("DXY").score if tech.get("DXY") else None
    # EURUSD y DXY están muy correlacionados de forma inversa: cada uno confirma al otro
    if t_eur is not None and t_dxy is not None:
        t_eur_adj = (1 - CROSS_WEIGHT) * t_eur + CROSS_WEIGHT * (-t_dxy)
        t_dxy_adj = (1 - CROSS_WEIGHT) * t_dxy + CROSS_WEIGHT * (-t_eur)
    else:
        t_eur_adj = t_eur if t_eur is not None else (-t_dxy if t_dxy is not None else 0.0)
        t_dxy_adj = t_dxy if t_dxy is not None else (-t_eur if t_eur is not None else 0.0)

    out = {}
    for pair, t in (("EURUSD", t_eur_adj), ("DXY", t_dxy_adj)):
        f, n = fund.get(pair, 0.0), news.get(pair, 0.0)
        combined = TECH_WEIGHT * t + FUND_WEIGHT * f + NEWS_WEIGHT * n
        p = to_probability(combined)
        reasons = []
        tb = tech.get(pair)
        if tb:
            strongest = sorted(tb.signals, key=lambda s: -abs(s["valor"] * s["peso"]))[:3]
            reasons += [s["lectura"] for s in strongest]
        if t_eur is not None and t_dxy is not None:
            agree = np.sign(t_eur) == -np.sign(t_dxy)
            reasons.append("EURUSD y DXY se confirman mutuamente (correlación inversa)." if agree
                           else "EURUSD y DXY NO se confirman: señal de menor calidad, reduce tamaño o espera.")
        if abs(f) > 0.05:
            reasons.append(f"Las noticias publicadas favorecen {'subidas' if f > 0 else 'bajadas'} en {pair}.")
        if ai_prob and pair in ai_prob and ai_prob[pair] is not None:
            p = float(np.clip((1 - ai_weight) * p + ai_weight * ai_prob[pair], PROB_FLOOR, PROB_CEIL))
            reasons.append(f"Ajustado con el análisis de contexto mundial de la IA ({ai_prob[pair]:.0%} subida).")
        out[pair] = DailyBias(pair=pair, technical=round(t, 3), fundamental=round(f, 3), news=round(n, 3),
                              combined=round(combined, 3), prob_up=round(p, 3), reasons=reasons)
    return out


def evaluate_bias_log(log: pd.DataFrame) -> dict:
    """Mide el acierto del sesgo diario registrado frente al movimiento real."""
    done = log.dropna(subset=["actual_change"])
    done = done[(done["prob_up"] - 0.5).abs() > 0.03]
    if done.empty:
        return {"evaluados": 0, "acierto": None, "brier": None}
    hit = np.sign(done["prob_up"] - 0.5) == np.sign(done["actual_change"])
    outcome = (done["actual_change"] > 0).astype(float)
    brier = float(((done["prob_up"] - outcome) ** 2).mean())
    return {"evaluados": int(len(done)), "acierto": round(100 * hit.mean(), 1), "brier": round(brier, 3)}
