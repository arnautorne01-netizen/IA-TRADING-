"""Backtest walk-forward: cada semana se reajusta con los partidos anteriores y se pronostica la siguiente.
Sirve para comprobar que las probabilidades están bien calibradas (si decimos 70%, que pase ~70%)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .markets import MARKETS_BY_ID, SCOPE_COLS, add_cards, market_outcomes
from .predictor import LeaguePredictor, select_picks


def run(df: pd.DataFrame, league: str = "", test_from=None, window_days: int = 5 * 365 + 60,
        **kw) -> pd.DataFrame:
    """Devuelve una fila por (partido, mercado) con lo que se predijo y lo que pasó."""
    df = add_cards(df).sort_values("Date").reset_index(drop=True)
    test_from = pd.Timestamp(test_from) if test_from is not None else df["Date"].max() - pd.Timedelta(days=300)
    test = df[df["Date"] >= test_from]
    out = []
    for _, week in test.groupby(test["Date"].dt.to_period("W-MON")):
        start = week["Date"].min()
        train = df[(df["Date"] < start) & (df["Date"] >= start - pd.Timedelta(days=window_days))]
        pred = LeaguePredictor(train, league, ref_date=start, **kw)
        for _, m in week.iterrows():
            p = pred.predict(m["HomeTeam"], m["AwayTeam"], m.to_dict(), m["Date"])
            actual = {}
            for scope, (hc, ac) in SCOPE_COLS.items():
                if hc in m and pd.notna(m.get(hc)) and pd.notna(m.get(ac)):
                    actual.update({k: bool(v) for k, v in market_outcomes([m[hc]], [m[ac]], scope).items()})
            r = p.rows[p.rows["id"].isin(actual.keys())].copy()
            r["hit"] = r["id"].map(actual)
            r["picked"] = r["id"].isin(set(p.picks["id"]) if not p.picks.empty else set())
            r["Date"], r["HomeTeam"], r["AwayTeam"], r["league"] = m["Date"], p.home, p.away, league
            out.append(r)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def blend(res: pd.DataFrame, m: float, h2h_weight: float = config.H2H_WEIGHT) -> np.ndarray:
    K = res["k_home"] + res["k_away"] + h2h_weight * res["k_h2h"]
    N = res["n_home"] + res["n_away"] + h2h_weight * res["n_h2h"]
    if np.isinf(m):
        return res["p_model"].to_numpy()
    return ((K + m * res["p_model"]) / (N + m)).to_numpy()


def _logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def tune(res: pd.DataFrame, ms=(0, 3, 6, 12, 25, 50, 100, np.inf), h2hs=(0.0, 0.5, 1.0)) -> pd.DataFrame:
    """Compara mezclas modelo/histórico. Menor log-loss = probabilidades más acertadas."""
    y = res["hit"].astype(float).to_numpy()
    rows = []
    for m in ms:
        for w in h2hs if not np.isinf(m) else (0.0,):
            p = blend(res, m, w)
            rows.append({"peso_modelo_m": m, "peso_h2h": w, "log_loss": _logloss(p, y),
                         "brier": float(np.mean((p - y) ** 2))})
    base = res.groupby("id")["hit"].transform("mean").to_numpy()
    rows.append({"peso_modelo_m": "frecuencia de liga", "peso_h2h": np.nan, "log_loss": _logloss(base, y),
                 "brier": float(np.mean((base - y) ** 2))})
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


def calibration(res: pd.DataFrame, col: str = "p_final", bins=(0, .3, .4, .5, .6, .65, .7, .75, .8, .85, .9, 1.01)) -> pd.DataFrame:
    """Probabilidad pronosticada media vs frecuencia real por tramos."""
    r = res.assign(tramo=pd.cut(res[col], bins, right=False))
    g = r.groupby("tramo", observed=True).agg(n=("hit", "size"), prob_media=(col, "mean"), acierto_real=("hit", "mean"))
    return g.reset_index()


def picks_summary(res: pd.DataFrame) -> pd.DataFrame:
    """Rendimiento de los picks que habría propuesto la IA, por familia de mercado.
    `beneficio_justo` = unidades ganadas apostando 1 a la cuota justa (≈0 si está bien calibrado)."""
    p = res[res["picked"]].copy()
    if p.empty:
        return p
    p["fam"] = p["id"].map(lambda i: MARKETS_BY_ID[i].family)
    p["profit"] = np.where(p["hit"], p["cuota_justa"] - 1, -1)
    g = p.groupby("fam").agg(picks=("hit", "size"), prob_media=("p_final", "mean"), acierto=("hit", "mean"),
                             cuota_media=("cuota_justa", "mean"), beneficio_justo=("profit", "sum"))
    tot = pd.DataFrame([{"picks": len(p), "prob_media": p["p_final"].mean(), "acierto": p["hit"].mean(),
                         "cuota_media": p["cuota_justa"].mean(), "beneficio_justo": p["profit"].sum()}], index=["TOTAL"])
    return pd.concat([g.sort_values("picks", ascending=False), tot])


def value_summary(res: pd.DataFrame, min_edge: float = config.VALUE_EDGE) -> pd.DataFrame:
    """Si el histórico trae cuotas de la casa: ROI real apostando cuando p·cuota − 1 > min_edge."""
    v = res[res["cuota_casa"].notna() & (res["valor"] > min_edge)]
    if v.empty:
        return pd.DataFrame()
    profit = np.where(v["hit"], v["cuota_casa"] - 1, -1)
    return pd.DataFrame([{"apuestas": len(v), "acierto": v["hit"].mean(), "cuota_media": v["cuota_casa"].mean(),
                          "beneficio": profit.sum(), "ROI": profit.mean()}])
