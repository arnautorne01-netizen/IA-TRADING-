"""Genera un journal de ejemplo con patrones realistas para probar la aplicación."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def generate_sample(n: int = 150, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-05")
    rows = []
    for i in range(n):
        day = start + pd.tseries.offsets.BDay(int(i * 0.7))
        pair = rng.choice(["EURUSD", "DXY"], p=[0.75, 0.25])
        direction = rng.choice(["Long", "Short"])
        session = rng.choice(["Londres", "Nueva York", "Overlap Londres-NY", "Asia"], p=[0.4, 0.3, 0.2, 0.1])
        bias = rng.choice(["Alcista", "Bajista", "Neutral"], p=[0.45, 0.45, 0.10])
        aligned = (bias == "Alcista" and direction == "Long") or (bias == "Bajista" and direction == "Short")
        pool = [c for c in config.DEFAULT_CONFLUENCES if c != "Sesgo diario a favor"]
        confs = list(rng.choice(pool, size=rng.integers(1, 6), replace=False))
        if aligned:
            confs.insert(0, "Sesgo diario a favor")
        followed = bool(rng.random() < 0.75)
        emotion = rng.choice(config.EMOTIONS, p=[0.35, 0.2, 0.12, 0.08, 0.08, 0.05, 0.07, 0.05])
        news = bool(rng.random() < 0.15)
        # probabilidad de ganar condicionada a la calidad del proceso
        p = 0.38
        p += 0.15 if aligned else -0.08
        p += 0.10 if "Liquidez barrida" in confs else 0
        p += 0.08 if "FVG" in confs and "Killzone" in confs else 0
        p += 0.06 if "Correlación DXY confirma" in confs else 0
        p += 0.05 if followed else -0.12
        p -= 0.12 if emotion in config.NEGATIVE_EMOTIONS else 0
        p -= 0.10 if news else 0
        p -= 0.10 if session == "Asia" else 0
        win = rng.random() < np.clip(p, 0.05, 0.9)
        entry = round(1.08 + rng.normal(0, 0.02), 5) if pair == "EURUSD" else round(103 + rng.normal(0, 1.5), 3)
        risk = (0.0015 + rng.random() * 0.002) if pair == "EURUSD" else (0.15 + rng.random() * 0.2)
        rr = round(float(rng.choice([1.0, 1.5, 2.0, 2.5, 3.0], p=[0.1, 0.2, 0.35, 0.2, 0.15])), 1)
        sign = 1 if direction == "Long" else -1
        sl = entry - sign * risk
        tp = entry + sign * risk * rr
        if win:
            exit_price = tp if rng.random() < 0.8 else entry + sign * risk * rr * rng.uniform(0.3, 0.9)
        else:
            exit_price = sl if rng.random() < 0.85 else entry - sign * risk * rng.uniform(0.2, 0.9)
        mistakes = []
        if not followed:
            mistakes.append(rng.choice(config.DEFAULT_MISTAKES))
        if emotion == "Venganza":
            mistakes.append("Revenge trading")
        if not aligned and bias != "Neutral":
            mistakes.append("Contra sesgo")
        hour = {"Asia": 2, "Londres": 8, "Overlap Londres-NY": 14, "Nueva York": 16}[session] + int(rng.integers(0, 3))
        rows.append({
            "date": day.date().isoformat(), "time": f"{hour:02d}:{int(rng.integers(0, 60)):02d}",
            "pair": pair, "direction": direction, "session": session,
            "setup": rng.choice(["Barrido + FVG", "Ruptura y retesteo", "Continuación tendencia", "Reversión en OB"]),
            "timeframe": rng.choice(["M5", "M15", "H1"]),
            "entry": entry, "stop_loss": round(sl, 5), "take_profit": round(tp, 5),
            "exit_price": round(float(exit_price), 5), "risk_pct": 1.0,
            "confluences": ", ".join(confs), "daily_bias": bias, "followed_plan": followed,
            "emotion": emotion, "news_nearby": news, "mistakes": ", ".join(dict.fromkeys(mistakes)),
            "notes": "",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    generate_sample().to_csv(config.ROOT / "sample_data" / "sample_trades.csv", index=False)
