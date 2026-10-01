"""Estadísticas del journal: win rate, expectativa, profit factor, desgloses,
filtros avanzados y análisis de confluencias (cuáles aparecen en tus buenas
operaciones y cuáles no aportan)."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Filtros
# --------------------------------------------------------------------------- #
@dataclass
class TradeFilter:
    pairs: list[str] | None = None
    directions: list[str] | None = None
    sessions: list[str] | None = None
    setups: list[str] | None = None
    results: list[str] | None = None
    emotions: list[str] | None = None
    weekdays: list[str] | None = None
    date_from: str | None = None
    date_to: str | None = None
    confluences_all: list[str] | None = None   # deben estar TODAS
    confluences_any: list[str] | None = None   # basta con UNA
    confluences_none: list[str] | None = None  # no debe estar ninguna
    mistakes_any: list[str] | None = None
    followed_plan: bool | None = None
    news_nearby: bool | None = None
    bias_aligned: bool | None = None
    min_confluences: int | None = None
    hour_from: int | None = None
    hour_to: int | None = None


def apply_filter(df: pd.DataFrame, f: TradeFilter) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)

    def isin(col, values):
        return df[col].isin(values) if values else True

    m &= isin("pair", f.pairs)
    m &= isin("direction", f.directions)
    m &= isin("session", f.sessions)
    m &= isin("setup", f.setups)
    m &= isin("result", f.results)
    m &= isin("emotion", f.emotions)
    m &= isin("weekday", f.weekdays)
    if f.date_from:
        m &= df["date"] >= pd.to_datetime(f.date_from)
    if f.date_to:
        m &= df["date"] <= pd.to_datetime(f.date_to)
    if f.confluences_all:
        m &= df["confluences"].apply(lambda c: set(f.confluences_all).issubset(c))
    if f.confluences_any:
        m &= df["confluences"].apply(lambda c: bool(set(f.confluences_any) & set(c)))
    if f.confluences_none:
        m &= df["confluences"].apply(lambda c: not (set(f.confluences_none) & set(c)))
    if f.mistakes_any:
        m &= df["mistakes"].apply(lambda c: bool(set(f.mistakes_any) & set(c)))
    if f.followed_plan is not None:
        m &= df["followed_plan"] == f.followed_plan
    if f.news_nearby is not None:
        m &= df["news_nearby"] == f.news_nearby
    if f.bias_aligned is not None:
        m &= df["bias_aligned"] == f.bias_aligned
    if f.min_confluences:
        m &= df["n_confluences"] >= f.min_confluences
    if f.hour_from is not None:
        m &= df["hour"] >= f.hour_from
    if f.hour_to is not None:
        m &= df["hour"] <= f.hour_to
    return df[m]


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def _streaks(results: list[str]) -> tuple[int, int]:
    best_w = best_l = cur_w = cur_l = 0
    for r in results:
        if r == "Win":
            cur_w, cur_l = cur_w + 1, 0
        elif r == "Loss":
            cur_l, cur_w = cur_l + 1, 0
        best_w, best_l = max(best_w, cur_w), max(best_l, cur_l)
    return best_w, best_l


def summary(df: pd.DataFrame) -> dict:
    """Resumen global: WR, expectativa, profit factor, drawdown, rachas..."""
    n = len(df)
    wins, losses = int(df["is_win"].sum()), int(df["is_loss"].sum())
    be = int((df["result"] == "BE").sum())
    decided = wins + losses
    r = df["r_multiple"].dropna()
    gross_win = r[r > 0].sum()
    gross_loss = -r[r < 0].sum()
    equity = r.cumsum()
    dd = (equity - equity.cummax()).min() if not equity.empty else 0.0
    best_w, best_l = _streaks(df["result"].tolist())
    return {
        "trades": n,
        "wins": wins,
        "losses": losses,
        "breakeven": be,
        "win_rate": round(100 * wins / decided, 1) if decided else None,
        "win_rate_incl_be": round(100 * wins / n, 1) if n else None,
        "total_r": round(float(r.sum()), 2),
        "avg_r": round(float(r.mean()), 2) if not r.empty else None,
        "avg_win_r": round(float(r[r > 0].mean()), 2) if (r > 0).any() else None,
        "avg_loss_r": round(float(r[r < 0].mean()), 2) if (r < 0).any() else None,
        "profit_factor": round(float(gross_win / gross_loss), 2) if gross_loss > 0 else None,
        "expectancy_r": round(float(r.mean()), 3) if not r.empty else None,
        "max_drawdown_r": round(float(dd), 2),
        "best_win_streak": best_w,
        "worst_loss_streak": best_l,
        "total_pnl": round(float(df["pnl"].dropna().sum()), 2) if df["pnl"].notna().any() else None,
        "plan_adherence": round(float(100 * df["followed_plan"].eq(True).sum() / df["followed_plan"].notna().sum()), 1)
        if df["followed_plan"].notna().any() else None,
    }


def _group_stats(g: pd.DataFrame) -> pd.Series:
    wins, losses = g["is_win"].sum(), g["is_loss"].sum()
    decided = wins + losses
    r = g["r_multiple"].dropna()
    return pd.Series({
        "trades": len(g),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": round(100 * wins / decided, 1) if decided else np.nan,
        "total_r": round(r.sum(), 2),
        "expectancy_r": round(r.mean(), 2) if not r.empty else np.nan,
    })


def breakdown(df: pd.DataFrame, column: str, min_trades: int = 1) -> pd.DataFrame:
    """WR y expectativa agrupado por una columna. Si la columna es una lista
    (confluencias, errores) se explota para contar cada etiqueta."""
    if df.empty:
        return pd.DataFrame()
    data = df.explode(column) if column in ("confluences", "mistakes") else df
    data = data[data[column].notna()]
    if data.empty:
        return pd.DataFrame()
    out = data.groupby(column).apply(_group_stats, include_groups=False)
    out = out[out["trades"] >= min_trades]
    return out.sort_values(["expectancy_r", "win_rate"], ascending=False)


def monthly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    d = df.assign(month=df["date"].dt.to_period("M").astype(str))
    return d.groupby("month").apply(_group_stats, include_groups=False)


def equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["r_multiple"].notna()].sort_values(["date", "time", "id"]).reset_index(drop=True)
    d["equity_r"] = d["r_multiple"].cumsum()
    d["trade_n"] = np.arange(1, len(d) + 1)
    return d[["trade_n", "date", "pair", "result", "r_multiple", "equity_r"]]


# --------------------------------------------------------------------------- #
# Confluencias
# --------------------------------------------------------------------------- #
def confluence_lift(df: pd.DataFrame, min_trades: int = 3) -> pd.DataFrame:
    """Para cada confluencia: con qué frecuencia aparece en ganadoras vs
    perdedoras, el WR cuando está presente vs ausente y la diferencia (lift)."""
    decided = df[df["result"].isin(["Win", "Loss"])]
    if decided.empty:
        return pd.DataFrame()
    tags = sorted({t for c in decided["confluences"] for t in c})
    winners, losers = decided[decided["is_win"]], decided[decided["is_loss"]]
    rows = []
    for tag in tags:
        has = decided["confluences"].apply(lambda c: tag in c)
        n_with = int(has.sum())
        if n_with < min_trades:
            continue
        wr_with = decided.loc[has, "is_win"].mean() * 100
        wr_without = decided.loc[~has, "is_win"].mean() * 100 if (~has).any() else np.nan
        rows.append({
            "confluencia": tag,
            "trades_con": n_with,
            "%_en_ganadoras": round(100 * winners["confluences"].apply(lambda c: tag in c).mean(), 1) if len(winners) else np.nan,
            "%_en_perdedoras": round(100 * losers["confluences"].apply(lambda c: tag in c).mean(), 1) if len(losers) else np.nan,
            "wr_con": round(wr_with, 1),
            "wr_sin": round(wr_without, 1) if not np.isnan(wr_without) else np.nan,
            "lift_wr": round(wr_with - wr_without, 1) if not np.isnan(wr_without) else np.nan,
            "expectativa_con_r": round(decided.loc[has, "r_multiple"].mean(), 2),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("lift_wr", ascending=False, na_position="last").reset_index(drop=True)


def confluence_combos(df: pd.DataFrame, size: int = 2, min_trades: int = 3) -> pd.DataFrame:
    """Combinaciones de confluencias (parejas/tríos) y su rendimiento."""
    decided = df[df["result"].isin(["Win", "Loss"])]
    counts: dict[tuple, list] = {}
    for confs, win, r in zip(decided["confluences"], decided["is_win"], decided["r_multiple"]):
        for combo in combinations(sorted(set(confs)), size):
            counts.setdefault(combo, []).append((win, r))
    rows = []
    for combo, vals in counts.items():
        if len(vals) < min_trades:
            continue
        w = [v[0] for v in vals]
        r = [v[1] for v in vals if v[1] is not None and not pd.isna(v[1])]
        rows.append({
            "combinación": " + ".join(combo),
            "trades": len(vals),
            "win_rate": round(100 * sum(w) / len(w), 1),
            "expectativa_r": round(float(np.mean(r)), 2) if r else np.nan,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["win_rate", "trades"], ascending=False).reset_index(drop=True)


def confluence_count_effect(df: pd.DataFrame) -> pd.DataFrame:
    """WR en función del número de confluencias presentes."""
    if df.empty:
        return pd.DataFrame()
    return df.groupby("n_confluences").apply(_group_stats, include_groups=False)


def ideal_checklist(df: pd.DataFrame, min_trades: int = 3, min_lift: float = 10.0, max_items: int = 5) -> list[str]:
    """Confluencias que más mejoran tu WR -> tu checklist de operación ideal."""
    lift = confluence_lift(df, min_trades=min_trades)
    if lift.empty:
        return []
    good = lift[(lift["lift_wr"] >= min_lift) & (lift["expectativa_con_r"] > 0)]
    return good["confluencia"].head(max_items).tolist()
