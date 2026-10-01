"""Datos de mercado e indicadores técnicos para el sesgo diario de EURUSD y DXY."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Descarga de datos
# --------------------------------------------------------------------------- #
def fetch_ohlc(pair: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Descarga OHLC desde Yahoo Finance. Lanza RuntimeError si no hay datos."""
    import yfinance as yf

    ticker = config.YF_TICKERS.get(pair.upper(), pair)
    data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if data is None or data.empty:
        raise RuntimeError(f"No se pudieron descargar datos de {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    data.index = pd.to_datetime(data.index).tz_localize(None)
    return data


def load_ohlc_csv(buffer) -> pd.DataFrame:
    """Carga OHLC desde CSV (columnas date/open/high/low/close), p.ej. exportado de TradingView."""
    df = pd.read_csv(buffer)
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = next(c for c in df.columns if c in ("date", "time", "datetime", "fecha"))
    idx = pd.to_datetime(df[date_col], unit="s" if pd.api.types.is_numeric_dtype(df[date_col]) else None)
    df = df.set_index(idx)[["open", "high", "low", "close"]].astype(float).sort_index()
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    return df


# --------------------------------------------------------------------------- #
# Indicadores
# --------------------------------------------------------------------------- #
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(s: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(s, 12) - ema(s, 26)
    signal = ema(line, 9)
    return line, signal, line - signal


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def swing_points(df: pd.DataFrame, lookback: int = 3) -> tuple[pd.Series, pd.Series]:
    """Máximos y mínimos de swing (fractales) para leer la estructura."""
    h, l = df["high"], df["low"]
    win = 2 * lookback + 1
    sh = h[(h == h.rolling(win, center=True).max())]
    sl = l[(l == l.rolling(win, center=True).min())]
    return sh, sl


def market_structure(df: pd.DataFrame, lookback: int = 3) -> tuple[float, str]:
    """+1 si hay máximos y mínimos crecientes (HH/HL), -1 si decrecientes (LH/LL)."""
    sh, sl = swing_points(df, lookback)
    if len(sh) < 2 or len(sl) < 2:
        # pocos swings (tendencia muy limpia): compara máximos/mínimos de bloques de 20 velas
        if len(df) < 40:
            return 0.0, "Estructura indefinida"
        cur, prev = df.iloc[-20:], df.iloc[-40:-20]
        if cur["high"].max() > prev["high"].max() and cur["low"].min() > prev["low"].min():
            return 1.0, "Estructura alcista (máximos y mínimos crecientes)"
        if cur["high"].max() < prev["high"].max() and cur["low"].min() < prev["low"].min():
            return -1.0, "Estructura bajista (máximos y mínimos decrecientes)"
        return 0.0, "Estructura en rango"
    hh = sh.iloc[-1] > sh.iloc[-2]
    hl = sl.iloc[-1] > sl.iloc[-2]
    last_close = df["close"].iloc[-1]
    # una ruptura del último swing confirmado (BOS) manda sobre la secuencia de swings anterior
    if last_close > sh.iloc[-1]:
        return (1.0, "BOS alcista: cierre por encima del último máximo con mínimos crecientes") if hl \
            else (0.7, "Ruptura del último máximo (BOS alcista)")
    if last_close < sl.iloc[-1]:
        return (-1.0, "BOS bajista: cierre por debajo del último mínimo con máximos decrecientes") if not hh \
            else (-0.7, "Ruptura del último mínimo (BOS bajista)")
    if hh and hl:
        return 1.0, "Estructura alcista (HH + HL)"
    if not hh and not hl:
        return -1.0, "Estructura bajista (LH + LL)"
    return 0.0, "Estructura en rango / transición"


# --------------------------------------------------------------------------- #
# Sesgo técnico
# --------------------------------------------------------------------------- #
@dataclass
class TechnicalBias:
    pair: str
    score: float                    # -1 (muy bajista) .. +1 (muy alcista)
    signals: list[dict] = field(default_factory=list)
    levels: dict = field(default_factory=dict)
    last_close: float | None = None
    as_of: str | None = None

    @property
    def label(self) -> str:
        return score_label(self.score)


def score_label(score: float) -> str:
    if score > 0.35:
        return "Alcista"
    if score > 0.1:
        return "Ligeramente alcista"
    if score < -0.35:
        return "Bajista"
    if score < -0.1:
        return "Ligeramente bajista"
    return "Neutral"


WEIGHTS = {
    "ema200": 0.15,
    "ema50": 0.12,
    "ema_cross": 0.12,
    "ema20_slope": 0.10,
    "structure": 0.18,
    "macd": 0.10,
    "rsi": 0.08,
    "prev_day": 0.08,
    "weekly": 0.07,
}


def technical_bias(daily: pd.DataFrame, pair: str) -> TechnicalBias:
    """Calcula el sesgo técnico diario a partir de velas diarias."""
    if len(daily) < 60:
        raise ValueError("Se necesitan al menos 60 velas diarias")
    c = daily["close"]
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, min(200, len(c) - 1))
    _, _, hist = macd(c)
    r = rsi(c)
    a = atr(daily)
    last = c.iloc[-1]
    sig: list[dict] = []

    def add(key, value, text):
        sig.append({"señal": key, "valor": round(float(value), 2), "peso": WEIGHTS[key], "lectura": text})

    add("ema200", 1 if last > e200.iloc[-1] else -1,
        f"Precio {'por encima' if last > e200.iloc[-1] else 'por debajo'} de la EMA 200 (tendencia de fondo)")
    add("ema50", 1 if last > e50.iloc[-1] else -1,
        f"Precio {'por encima' if last > e50.iloc[-1] else 'por debajo'} de la EMA 50")
    add("ema_cross", 1 if e20.iloc[-1] > e50.iloc[-1] else -1,
        f"EMA 20 {'sobre' if e20.iloc[-1] > e50.iloc[-1] else 'bajo'} EMA 50")
    slope = (e20.iloc[-1] - e20.iloc[-5]) / (a.iloc[-1] or 1)
    add("ema20_slope", float(np.clip(slope, -1, 1)), f"Pendiente EMA 20 en 5 días: {slope:+.2f} ATR")
    st_val, st_txt = market_structure(daily.tail(120), lookback=5)
    add("structure", st_val, st_txt)
    h = hist.iloc[-1]
    add("macd", (1 if h > 0 else -1) * (1 if abs(h) > abs(hist.iloc[-2]) else 0.5),
        f"Histograma MACD {'positivo' if h > 0 else 'negativo'} y {'creciendo' if abs(h) > abs(hist.iloc[-2]) else 'perdiendo fuerza'}")
    rv = r.iloc[-1]
    if rv > 70:
        rsi_val, rsi_txt = -0.3, f"RSI {rv:.0f}: sobrecompra, riesgo de retroceso"
    elif rv < 30:
        rsi_val, rsi_txt = 0.3, f"RSI {rv:.0f}: sobreventa, riesgo de rebote"
    else:
        rsi_val, rsi_txt = float(np.clip((rv - 50) / 15, -1, 1)), f"RSI {rv:.0f}: momentum {'alcista' if rv > 50 else 'bajista'}"
    add("rsi", rsi_val, rsi_txt)
    prev = daily.iloc[-2]
    if last > prev["high"]:
        pd_val, pd_txt = 1, "Cierre por encima del máximo del día anterior (PDH)"
    elif last < prev["low"]:
        pd_val, pd_txt = -1, "Cierre por debajo del mínimo del día anterior (PDL)"
    else:
        mid = (prev["high"] + prev["low"]) / 2
        pd_val, pd_txt = (0.3 if last > mid else -0.3), "Cierre dentro del rango del día anterior"
    add("prev_day", pd_val, pd_txt)
    weekly = daily.resample("W").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    wv = 1 if weekly["close"].iloc[-1] > ema(weekly["close"], 10).iloc[-1] else -1
    add("weekly", wv, f"Semanal {'por encima' if wv > 0 else 'por debajo'} de la EMA 10 semanal")

    score = sum(s["valor"] * s["peso"] for s in sig) / sum(WEIGHTS.values())
    levels = {
        "PDH": round(float(prev["high"]), 5), "PDL": round(float(prev["low"]), 5),
        "EMA20": round(float(e20.iloc[-1]), 5), "EMA50": round(float(e50.iloc[-1]), 5),
        "EMA200": round(float(e200.iloc[-1]), 5), "ATR14": round(float(a.iloc[-1]), 5),
        "RSI14": round(float(rv), 1),
        "Máx 20d": round(float(daily["high"].tail(20).max()), 5),
        "Mín 20d": round(float(daily["low"].tail(20).min()), 5),
    }
    return TechnicalBias(pair=pair, score=round(float(np.clip(score, -1, 1)), 3), signals=sig, levels=levels,
                         last_close=round(float(last), 5), as_of=str(daily.index[-1].date()))


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 30) -> float | None:
    joined = pd.concat([a.pct_change(), b.pct_change()], axis=1).dropna().tail(window)
    if len(joined) < 10:
        return None
    return round(float(joined.corr().iloc[0, 1]), 2)
