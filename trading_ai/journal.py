"""Journal de operaciones: almacenamiento en SQLite, importación/exportación CSV
y cálculo de métricas derivadas de cada trade (R múltiplo, resultado, RR planeado)."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import config

COLUMNS = [
    "id", "date", "time", "pair", "direction", "session", "setup", "timeframe",
    "entry", "stop_loss", "take_profit", "exit_price", "risk_pct", "r_multiple",
    "pnl", "result", "confluences", "daily_bias", "followed_plan", "emotion",
    "news_nearby", "mistakes", "notes", "screenshot_url",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    time TEXT,
    pair TEXT NOT NULL,
    direction TEXT NOT NULL,
    session TEXT,
    setup TEXT,
    timeframe TEXT,
    entry REAL,
    stop_loss REAL,
    take_profit REAL,
    exit_price REAL,
    risk_pct REAL,
    r_multiple REAL,
    pnl REAL,
    result TEXT,
    confluences TEXT,
    daily_bias TEXT,
    followed_plan INTEGER,
    emotion TEXT,
    news_nearby INTEGER,
    mistakes TEXT,
    notes TEXT,
    screenshot_url TEXT
);
CREATE TABLE IF NOT EXISTS bias_log (
    date TEXT NOT NULL,
    pair TEXT NOT NULL,
    prob_up REAL NOT NULL,
    technical_score REAL,
    fundamental_score REAL,
    summary TEXT,
    actual_change REAL,
    PRIMARY KEY (date, pair)
);
"""


@dataclass
class Trade:
    date: str
    pair: str
    direction: str
    entry: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    exit_price: float | None = None
    time: str | None = None
    session: str | None = None
    setup: str | None = None
    timeframe: str | None = None
    risk_pct: float | None = None
    r_multiple: float | None = None
    pnl: float | None = None
    result: str | None = None
    confluences: list[str] = field(default_factory=list)
    daily_bias: str | None = None
    followed_plan: bool | None = None
    emotion: str | None = None
    news_nearby: bool | None = None
    mistakes: list[str] = field(default_factory=list)
    notes: str | None = None
    screenshot_url: str | None = None


# --------------------------------------------------------------------------- #
# Cálculos por trade
# --------------------------------------------------------------------------- #
def _num(x) -> float | None:
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return None
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def compute_r_multiple(direction: str, entry, stop_loss, exit_price) -> float | None:
    """R conseguido = beneficio en precio / riesgo inicial en precio."""
    entry, stop_loss, exit_price = _num(entry), _num(stop_loss), _num(exit_price)
    if None in (entry, stop_loss, exit_price):
        return None
    risk = abs(entry - stop_loss)
    if risk == 0:
        return None
    sign = 1 if str(direction).lower().startswith(("l", "c", "b")) else -1  # Long/Compra/Buy
    return round(sign * (exit_price - entry) / risk, 2)


def compute_planned_rr(entry, stop_loss, take_profit) -> float | None:
    entry, stop_loss, take_profit = _num(entry), _num(stop_loss), _num(take_profit)
    if None in (entry, stop_loss, take_profit):
        return None
    risk = abs(entry - stop_loss)
    return round(abs(take_profit - entry) / risk, 2) if risk else None


def classify_result(r: float | None, pnl: float | None = None) -> str | None:
    value = r if r is not None else pnl
    if value is None:
        return None
    threshold = config.BE_THRESHOLD_R if r is not None else 0
    if value > threshold:
        return "Win"
    if value < -threshold:
        return "Loss"
    return "BE"


def _split_tags(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [t.strip() for t in str(value).replace(";", ",").split(",") if t.strip()]


def _to_bool(value) -> bool | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "si", "sí", "yes", "y", "s"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    return None


def normalize_trade(raw: dict) -> dict:
    """Completa campos derivados (R, resultado) y normaliza tipos."""
    t = {k: raw.get(k) for k in COLUMNS if k != "id"}
    t["pair"] = str(t.get("pair") or "").upper().replace("/", "").strip()
    d = str(t.get("direction") or "").strip().lower()
    t["direction"] = "Long" if d.startswith(("l", "c", "b")) else "Short"
    for k in ("entry", "stop_loss", "take_profit", "exit_price", "risk_pct", "r_multiple", "pnl"):
        t[k] = _num(t.get(k))
    if t["r_multiple"] is None:
        t["r_multiple"] = compute_r_multiple(t["direction"], t["entry"], t["stop_loss"], t["exit_price"])
    res = str(t.get("result") or "").strip().lower()
    if res in {"win", "ganada", "w", "tp"}:
        t["result"] = "Win"
    elif res in {"loss", "perdida", "pérdida", "l", "sl"}:
        t["result"] = "Loss"
    elif res in {"be", "breakeven"}:
        t["result"] = "BE"
    else:
        t["result"] = classify_result(t["r_multiple"], t["pnl"])
    t["confluences"] = ", ".join(_split_tags(t.get("confluences")))
    t["mistakes"] = ", ".join(_split_tags(t.get("mistakes")))
    t["followed_plan"] = _to_bool(t.get("followed_plan"))
    t["news_nearby"] = _to_bool(t.get("news_nearby"))
    t["date"] = str(pd.to_datetime(t["date"]).date()) if t.get("date") else None
    for k in ("time", "session", "setup", "timeframe", "daily_bias", "emotion", "notes", "screenshot_url"):
        v = t.get(k)
        t[k] = None if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "" else str(v).strip()
    return t


# --------------------------------------------------------------------------- #
# Persistencia
# --------------------------------------------------------------------------- #
class Journal:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or config.DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # --- trades -------------------------------------------------------------
    def add_trade(self, trade: Trade | dict) -> int:
        data = asdict(trade) if isinstance(trade, Trade) else dict(trade)
        row = normalize_trade(data)
        cols = list(row.keys())
        with self._conn() as c:
            cur = c.execute(
                f"INSERT INTO trades ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                [row[k] for k in cols],
            )
            return int(cur.lastrowid)

    def add_many(self, trades: Iterable[dict]) -> int:
        return sum(1 for t in trades if self.add_trade(t))

    def update_trade(self, trade_id: int, changes: dict) -> None:
        current = self.get_trade(trade_id)
        if current is None:
            raise KeyError(trade_id)
        current.update(changes)
        current.pop("id", None)
        # recalcular derivados si cambian precios
        if any(k in changes for k in ("entry", "stop_loss", "exit_price", "direction")) and "r_multiple" not in changes:
            current["r_multiple"] = None
            current["result"] = changes.get("result")
        row = normalize_trade(current)
        with self._conn() as c:
            c.execute(
                f"UPDATE trades SET {', '.join(f'{k}=?' for k in row)} WHERE id=?",
                [*row.values(), trade_id],
            )

    def delete_trade(self, trade_id: int) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM trades WHERE id=?", (trade_id,))

    def get_trade(self, trade_id: int) -> dict | None:
        df = self.load()
        sub = df[df["id"] == trade_id]
        if sub.empty:
            return None
        rec = sub.iloc[0].to_dict()
        rec["confluences"] = ", ".join(rec["confluences"])
        rec["mistakes"] = ", ".join(rec["mistakes"])
        return rec

    def load(self) -> pd.DataFrame:
        with self._conn() as c:
            df = pd.read_sql_query("SELECT * FROM trades ORDER BY date, time, id", c)
        return prepare_dataframe(df)

    # --- CSV -----------------------------------------------------------------
    def import_csv(self, path_or_buffer) -> int:
        df = pd.read_csv(path_or_buffer)
        df.columns = [_COLUMN_ALIASES.get(c.strip().lower(), c.strip().lower()) for c in df.columns]
        missing = {"date", "pair", "direction"} - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas obligatorias en el CSV: {', '.join(sorted(missing))}")
        return self.add_many(df.to_dict("records"))

    def export_csv(self) -> str:
        df = self.load().copy()
        df["confluences"] = df["confluences"].apply(", ".join)
        df["mistakes"] = df["mistakes"].apply(", ".join)
        return df[COLUMNS].to_csv(index=False)

    # --- histórico de sesgos -------------------------------------------------
    def log_bias(self, date: str, pair: str, prob_up: float, technical: float | None,
                 fundamental: float | None, summary: str = "") -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO bias_log (date, pair, prob_up, technical_score, fundamental_score, summary, actual_change)"
                " VALUES (?, ?, ?, ?, ?, ?, (SELECT actual_change FROM bias_log WHERE date=? AND pair=?))",
                (date, pair, prob_up, technical, fundamental, summary, date, pair),
            )

    def set_bias_outcome(self, date: str, pair: str, actual_change: float) -> None:
        with self._conn() as c:
            c.execute("UPDATE bias_log SET actual_change=? WHERE date=? AND pair=?", (actual_change, date, pair))

    def load_bias_log(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM bias_log ORDER BY date DESC", c)


_COLUMN_ALIASES = {
    "fecha": "date", "hora": "time", "par": "pair", "simbolo": "pair", "symbol": "pair",
    "direccion": "direction", "dirección": "direction", "tipo": "direction", "side": "direction",
    "sesion": "session", "sesión": "session", "entrada": "entry", "sl": "stop_loss",
    "tp": "take_profit", "salida": "exit_price", "exit": "exit_price", "r": "r_multiple",
    "resultado": "result", "confluencias": "confluences", "sesgo": "daily_bias", "bias": "daily_bias",
    "plan": "followed_plan", "emocion": "emotion", "emoción": "emotion", "noticia": "news_nearby",
    "errores": "mistakes", "notas": "notes", "beneficio": "pnl", "profit": "pnl",
}


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte el DataFrame crudo en uno listo para análisis."""
    df = df.copy()
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    df["confluences"] = df["confluences"].apply(_split_tags)
    df["mistakes"] = df["mistakes"].apply(_split_tags)
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.day_name()
    df["hour"] = pd.to_numeric(df["time"].astype(str).str.slice(0, 2), errors="coerce")
    df["n_confluences"] = df["confluences"].apply(len)
    df["planned_rr"] = [compute_planned_rr(e, s, t) for e, s, t in zip(df["entry"], df["stop_loss"], df["take_profit"])]
    df["is_win"] = df["result"] == "Win"
    df["is_loss"] = df["result"] == "Loss"
    df["followed_plan"] = df["followed_plan"].apply(_to_bool)
    df["news_nearby"] = df["news_nearby"].apply(_to_bool)
    bias = df["daily_bias"].fillna("").str.lower()
    dirn = df["direction"].fillna("").str.lower()
    df["bias_aligned"] = np.where(
        bias.str.startswith("neu") | (bias == ""), None,
        ((bias.str.startswith("alc") | bias.str.startswith("bull")) & dirn.str.startswith("l"))
        | ((bias.str.startswith("baj") | bias.str.startswith("bear")) & dirn.str.startswith("s")),
    )
    return df
