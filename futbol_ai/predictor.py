"""Pronóstico de un partido: modelo + frecuencias históricas → probabilidad final, cuota justa y picks."""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .markets import (BOOK_ODDS, MARKETS, SCOPE_COLS, SCOPE_MAX, add_cards, market_outcomes,
                      market_probs)
from .model import fit_from_df


@dataclass
class MatchPrediction:
    league: str
    home: str
    away: str
    date: pd.Timestamp | None
    expected: dict[str, tuple[float, float]]      # ámbito -> (local, visitante) esperados
    top_scores: list[tuple[str, float]]
    rows: pd.DataFrame                              # todos los mercados
    picks: pd.DataFrame                             # los seleccionados
    warnings: list[str] = field(default_factory=list)
    history: dict[str, int] = field(default_factory=dict)  # tamaños de muestra


class LeaguePredictor:
    """Ajusta los modelos de una liga con los partidos anteriores a `ref_date`."""

    def __init__(self, df: pd.DataFrame, league: str = "", ref_date=None,
                 blend_m: float = config.BLEND_M, n_recent: int = config.N_RECENT,
                 h2h_weight: float = config.H2H_WEIGHT):
        ref = pd.Timestamp(ref_date) if ref_date is not None else df["Date"].max() + pd.Timedelta(days=1)
        self.df = add_cards(df[df["Date"] < ref]).reset_index(drop=True)
        self.league = league
        self.ref = ref
        self.blend_m = blend_m
        self.n_recent = n_recent
        self.h2h_weight = h2h_weight
        self.models = {}
        for scope, (hc, ac) in SCOPE_COLS.items():
            m = fit_from_df(self.df, hc, ac, ref, dixon_coles=scope in ("ft", "ht"))
            if m is not None:
                self.models[scope] = m
        self.teams = sorted(set(self.df["HomeTeam"]) | set(self.df["AwayTeam"]))
        recent = self.df[self.df["Date"] >= ref - pd.Timedelta(days=300)]
        self.current_teams = sorted(set(recent["HomeTeam"]) | set(recent["AwayTeam"]))

    # --- nombres -----------------------------------------------------------------------
    def resolve_team(self, name: str) -> str:
        if name in self.teams:
            return name
        low = {t.lower(): t for t in self.teams}
        if name.lower() in low:
            return low[name.lower()]
        for t in self.teams:
            if name.lower() in t.lower() or t.lower() in name.lower():
                return t
        match = difflib.get_close_matches(name, self.teams, n=1, cutoff=0.6)
        return match[0] if match else name

    # --- evidencia histórica -----------------------------------------------------------
    def _samples(self, home: str, away: str) -> dict[str, pd.DataFrame]:
        d = self.df
        h2h = d[((d["HomeTeam"] == home) & (d["AwayTeam"] == away)) | ((d["HomeTeam"] == away) & (d["AwayTeam"] == home))]
        return {"home": d[d["HomeTeam"] == home], "away": d[d["AwayTeam"] == away], "h2h": h2h}

    def _evidence(self, samples: dict[str, pd.DataFrame], home: str, scope: str) -> dict[str, tuple[np.ndarray, int]]:
        hc, ac = SCOPE_COLS[scope]
        out = {}
        for src, s in samples.items():
            if hc not in s.columns:
                out[src] = ({}, 0)
                continue
            s = s.dropna(subset=[hc, ac])
            if src != "h2h":
                s = s.tail(self.n_recent)
            flip = (s["HomeTeam"] != home).to_numpy()
            hv = np.where(flip, s[ac], s[hc])
            av = np.where(flip, s[hc], s[ac])
            out[src] = (market_outcomes(hv, av, scope), len(s))
        return out

    # --- pronóstico --------------------------------------------------------------------
    def predict(self, home: str, away: str, odds: dict | None = None, date=None) -> MatchPrediction:
        home, away = self.resolve_team(home), self.resolve_team(away)
        warnings = []
        for t in (home, away):
            if t not in self.teams:
                warnings.append(f"{t}: sin histórico en esta liga (se trata como recién ascendido).")
            elif "ft" in self.models and self.models["ft"].n_matches.get(t, 0) < 10:
                warnings.append(f"{t}: pocos partidos recientes, la estimación es menos fiable.")

        samples = self._samples(home, away)
        expected, model_p, evidence = {}, {}, {}
        top_scores = []
        for scope, model in self.models.items():
            lam, mu = model.rates(home, away)
            expected[scope] = (lam, mu)
            mat = model.matrix(lam, mu, SCOPE_MAX[scope])
            model_p.update(market_probs(mat, scope))
            evidence[scope] = self._evidence(samples, home, scope)
            if scope == "ft":
                flat = [(f"{i}-{j}", mat[i, j]) for i in range(5) for j in range(5)]
                top_scores = sorted(flat, key=lambda x: -x[1])[:5]

        rows = []
        for m in MARKETS:
            if m.id not in model_p:
                continue
            ev = evidence[m.scope]
            k = {src: int(np.sum(ev[src][0][m.id])) if ev[src][1] else 0 for src in ev}
            n = {src: ev[src][1] for src in ev}
            K = k["home"] + k["away"] + self.h2h_weight * k["h2h"]
            N = n["home"] + n["away"] + self.h2h_weight * n["h2h"]
            p_model = model_p[m.id]
            p_hist = K / N if N else np.nan
            p_final = (K + self.blend_m * p_model) / (N + self.blend_m)
            book = _book_odds(odds, m.id)
            rows.append({
                "id": m.id, "Mercado": m.label, "family": m.family, "scope": m.scope,
                "p_model": p_model, "p_hist": p_hist, "p_final": p_final,
                "local_casa": f"{k['home']}/{n['home']}", "visit_fuera": f"{k['away']}/{n['away']}",
                "h2h": f"{k['h2h']}/{n['h2h']}", "n_hist": N,
                "k_home": k["home"], "n_home": n["home"], "k_away": k["away"], "n_away": n["away"],
                "k_h2h": k["h2h"], "n_h2h": n["h2h"],
                "cuota_justa": 1 / p_final if p_final > 0 else np.inf,
                "cuota_minima": (1 + config.VALUE_EDGE) / p_final if p_final > 0 else np.inf,
                "cuota_casa": book,
                "valor": p_final * book - 1 if book else np.nan,
            })
        rows = pd.DataFrame(rows)
        hist = {"local_en_casa": len(samples["home"].tail(self.n_recent)),
                "visitante_fuera": len(samples["away"].tail(self.n_recent)), "h2h": len(samples["h2h"])}
        pred = MatchPrediction(self.league, home, away, pd.Timestamp(date) if date is not None else None,
                               expected, top_scores, rows, pd.DataFrame(), warnings, hist)
        pred.picks = select_picks(rows)
        return pred


def _book_odds(odds: dict | None, market_id: str) -> float | None:
    if not odds:
        return None
    for col in BOOK_ODDS.get(market_id, []):
        v = odds.get(col)
        if v is not None and not pd.isna(v) and float(v) > 1:
            return float(v)
    return None


def select_picks(rows: pd.DataFrame, min_prob: float = config.MIN_PROB, min_odds: float = config.MIN_ODDS,
                 max_odds: float = config.MAX_ODDS, max_picks: int = 5) -> pd.DataFrame:
    """Mercados con probabilidad alta, cuota justa razonable y respaldo del histórico.
    En cada familia (p. ej. líneas de goles) se queda la línea más exigente que siga
    superando el umbral, que es la que mejor cuota paga."""
    if rows.empty:
        return rows
    c = rows[(rows["p_final"] >= min_prob) & (rows["cuota_justa"] >= min_odds) & (rows["cuota_justa"] <= max_odds)]
    # El histórico tiene que estar de acuerdo (si hay muestra suficiente)
    c = c[(c["n_hist"] < 6) | (c["p_hist"] >= min_prob - 0.07)]
    # Si la casa paga menos que la cuota justa, no es buena apuesta
    c = c[c["valor"].isna() | (c["valor"] > -0.02)]
    if c.empty:
        return c
    c = c.sort_values("p_final").groupby("family", as_index=False).head(1)
    c = c.assign(confianza=np.where((c["p_final"] >= 0.72) & (c["p_hist"].fillna(0) >= 0.70), "Alta", "Media"))
    return c.sort_values("p_final", ascending=False).head(max_picks).reset_index(drop=True)


def build_combo(preds: list[MatchPrediction], min_odds: float = 1.8, max_odds: float = 3.2,
                max_legs: int = 4) -> dict | None:
    """Combinada con el pick más probable de varios partidos, hasta llegar a la cuota objetivo."""
    legs = []
    for p in preds:
        if p.picks.empty:
            continue
        best = p.picks.iloc[0]
        legs.append({"Partido": f"{p.home} - {p.away}", "Mercado": best["Mercado"], "p": best["p_final"]})
    legs.sort(key=lambda x: -x["p"])
    chosen, prob = [], 1.0
    for leg in legs:
        if len(chosen) >= max_legs:
            break
        if 1 / (prob * leg["p"]) > max_odds:
            continue
        chosen.append(leg)
        prob *= leg["p"]
        if 1 / prob >= min_odds:
            break
    if len(chosen) < 2:
        return None
    return {"legs": chosen, "prob": prob, "cuota_justa": 1 / prob}


def predict_fixtures(data: dict[str, pd.DataFrame], fixtures: pd.DataFrame, **kw) -> list[MatchPrediction]:
    preds, cache = [], {}
    for _, fx in fixtures.iterrows():
        lg = fx["Div"]
        if lg not in data or data[lg].empty:
            continue
        if lg not in cache:
            cache[lg] = LeaguePredictor(data[lg], lg, ref_date=fixtures["Date"].min(), **kw)
        odds = {c: fx[c] for c in fx.index if c not in ("Date", "Div", "HomeTeam", "AwayTeam")}
        preds.append(cache[lg].predict(fx["HomeTeam"], fx["AwayTeam"], odds, fx["Date"]))
    return preds
