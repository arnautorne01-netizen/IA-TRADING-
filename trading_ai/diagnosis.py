"""Motor de diagnóstico: explica por qué cada operación salió bien o mal y
detecta los patrones que separan tus buenas operaciones de las malas.

Funciona sin IA (reglas + estadística de tu propio historial). El coach de
Claude (ai_coach.py) usa este informe como base para un análisis más profundo.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from . import analytics, config

# Factores que se evalúan en cada operación. Cada factor es "negativo" (resta
# probabilidades) o "positivo" (suma). Se usan para comparar ganadoras/perdedoras.
NEGATIVE_FACTORS = {
    "contra_sesgo": "Operación en contra del sesgo diario",
    "sin_plan": "No se siguió el plan de trading",
    "emocion_negativa": "Estado emocional negativo al entrar",
    "noticia_cerca": "Entrada con noticia de alto impacto cerca",
    "pocas_confluencias": "Menos confluencias de las que tienen tus ganadoras",
    "falta_confluencia_clave": "No tenía ninguna de tus 3 confluencias más rentables",
    "rr_bajo": "Ratio riesgo/beneficio planeado bajo (< 1.5)",
    "sesion_debil": "Sesión donde históricamente rindes peor",
    "setup_debil": "Setup con expectativa negativa en tu historial",
    "despues_de_perdida": "Entrada el mismo día tras una pérdida (posible revenge)",
    "sobreoperacion": "Demasiadas operaciones ese día",
    "errores_marcados": "Marcaste errores de ejecución",
}
POSITIVE_FACTORS = {
    "a_favor_sesgo": "A favor del sesgo diario",
    "plan_seguido": "Plan seguido con disciplina",
    "confluencias_clave": "Contenía tus confluencias clave",
    "muchas_confluencias": "Número de confluencias igual o superior al de tus ganadoras",
    "rr_bueno": "Buen ratio riesgo/beneficio planeado (>= 2)",
    "sesion_fuerte": "Sesión donde históricamente rindes mejor",
    "emocion_neutral": "Estado emocional tranquilo/confiado",
}


# Señales de alerta de proceso/conducta (las de confluencias se solapan entre sí y no cuentan aquí)
RED_FLAGS = ["contra_sesgo", "sin_plan", "emocion_negativa", "noticia_cerca", "rr_bajo",
             "setup_debil", "despues_de_perdida", "sobreoperacion", "errores_marcados", "falta_confluencia_clave"]


class Diagnoser:
    def __init__(self, df: pd.DataFrame, min_trades: int = 3):
        self.df = df.sort_values(["date", "time", "id"]).reset_index(drop=True)
        self.min_trades = min_trades
        decided = self.df[self.df["result"].isin(["Win", "Loss"])]
        self.key_confluences = analytics.ideal_checklist(self.df, min_trades=min_trades)
        winners = decided[decided["is_win"]]
        self.winner_median_conf = float(winners["n_confluences"].median()) if len(winners) else 0.0
        self.session_stats = analytics.breakdown(self.df, "session", min_trades=min_trades)
        self.setup_stats = analytics.breakdown(self.df, "setup", min_trades=min_trades)
        overall_wr = decided["is_win"].mean() * 100 if len(decided) else 50
        self.overall_wr = overall_wr
        self.weak_sessions = set(self.session_stats.index[self.session_stats["win_rate"] < overall_wr - 10]) \
            if not self.session_stats.empty else set()
        self.strong_sessions = set(self.session_stats.index[self.session_stats["win_rate"] > overall_wr + 10]) \
            if not self.session_stats.empty else set()
        self.weak_setups = set(self.setup_stats.index[self.setup_stats["expectancy_r"] < 0]) \
            if not self.setup_stats.empty else set()
        self._factors = self._compute_factors()

    # ------------------------------------------------------------------ #
    def _compute_factors(self) -> pd.DataFrame:
        rows = []
        prev_loss_day: dict = {}
        trades_per_day = self.df.groupby(["date"]).size()
        for _, t in self.df.iterrows():
            f: dict[str, bool] = {}
            aligned = t["bias_aligned"]
            f["contra_sesgo"] = aligned is False or aligned == False  # noqa: E712
            f["a_favor_sesgo"] = aligned is True or aligned == True  # noqa: E712
            f["sin_plan"] = t["followed_plan"] is False
            f["plan_seguido"] = t["followed_plan"] is True
            emo = t["emotion"] or ""
            f["emocion_negativa"] = emo in config.NEGATIVE_EMOTIONS
            f["emocion_neutral"] = emo in {"Tranquilo", "Confiado"}
            f["noticia_cerca"] = t["news_nearby"] is True
            confs = set(t["confluences"])
            f["pocas_confluencias"] = self.winner_median_conf > 0 and len(confs) < self.winner_median_conf
            f["muchas_confluencias"] = self.winner_median_conf > 0 and len(confs) >= self.winner_median_conf
            f["falta_confluencia_clave"] = bool(self.key_confluences) and not (set(self.key_confluences[:3]) & confs)
            f["confluencias_clave"] = bool(self.key_confluences) and bool(set(self.key_confluences) & confs)
            rr = t["planned_rr"]
            f["rr_bajo"] = rr is not None and not pd.isna(rr) and rr < 1.5
            f["rr_bueno"] = rr is not None and not pd.isna(rr) and rr >= 2
            f["sesion_debil"] = t["session"] in self.weak_sessions
            f["sesion_fuerte"] = t["session"] in self.strong_sessions
            f["setup_debil"] = t["setup"] in self.weak_setups
            day = t["date"]
            f["despues_de_perdida"] = prev_loss_day.get(day, False)
            f["sobreoperacion"] = trades_per_day.get(day, 0) > 3
            f["errores_marcados"] = len(t["mistakes"]) > 0
            if t["result"] == "Loss":
                prev_loss_day[day] = True
            rows.append(f)
        return pd.DataFrame(rows, index=self.df.index)

    # ------------------------------------------------------------------ #
    def explain_trade(self, idx: int) -> dict:
        """Explicación de una operación concreta (por posición en self.df)."""
        t = self.df.loc[idx]
        f = self._factors.loc[idx]
        neg = [NEGATIVE_FACTORS[k] for k in NEGATIVE_FACTORS if f.get(k)]
        pos = [POSITIVE_FACTORS[k] for k in POSITIVE_FACTORS if f.get(k)]
        if t["mistakes"]:
            neg.append("Errores: " + ", ".join(t["mistakes"]))
        missing = [c for c in self.key_confluences if c not in t["confluences"]]
        if t["result"] == "Loss":
            flags = self.red_flags(idx)
            headline = "Pérdida probablemente evitable" if flags >= 2 else (
                "Pérdida dentro de la estadística (buena ejecución, resultado negativo)" if flags == 0 else
                "Pérdida con un punto débil claro")
        elif t["result"] == "Win":
            flags = self.red_flags(idx)
            headline = "Ganadora de alta calidad" if len(pos) >= 3 and flags == 0 else (
                "Ganadora con malos hábitos: cuidado, el resultado no valida el proceso" if flags >= 2 else
                "Ganadora correcta")
        else:
            headline = "Breakeven"
        return {
            "id": int(t["id"]) if not pd.isna(t["id"]) else None,
            "resultado": t["result"],
            "r": t["r_multiple"],
            "titular": headline,
            "factores_en_contra": neg,
            "factores_a_favor": pos,
            "confluencias_clave_ausentes": missing,
            "quality_score": self.quality_score(idx),
        }

    def red_flags(self, idx: int) -> int:
        f = self._factors.loc[idx]
        flags = [k for k in RED_FLAGS if f.get(k)]
        # "errores marcados" suele duplicar "sin plan": cuentan como una sola alerta
        if "errores_marcados" in flags and "sin_plan" in flags:
            flags.remove("errores_marcados")
        return len(flags)

    def quality_score(self, idx: int) -> int:
        """Nota 0-100 de la calidad del PROCESO (no del resultado)."""
        f = self._factors.loc[idx]
        score = 50 + 8 * sum(bool(f.get(k)) for k in POSITIVE_FACTORS) - 9 * sum(bool(f.get(k)) for k in NEGATIVE_FACTORS)
        return int(np.clip(score, 0, 100))

    def explain_all(self) -> pd.DataFrame:
        rows = [self.explain_trade(i) for i in self.df.index]
        out = pd.DataFrame(rows)
        out.insert(1, "fecha", self.df["date"].dt.date.values)
        out.insert(2, "par", self.df["pair"].values)
        out.insert(3, "dirección", self.df["direction"].values)
        return out

    # ------------------------------------------------------------------ #
    def pattern_report(self) -> dict:
        """Qué tienen en común las buenas, por qué fallan las malas."""
        decided_mask = self.df["result"].isin(["Win", "Loss"])
        wins = self.df["is_win"] & decided_mask
        losses = self.df["is_loss"] & decided_mask
        rows = []
        for key, label in {**NEGATIVE_FACTORS, **POSITIVE_FACTORS}.items():
            col = self._factors[key].astype(bool)
            pw = 100 * col[wins].mean() if wins.any() else np.nan
            pl = 100 * col[losses].mean() if losses.any() else np.nan
            n = int(col[decided_mask].sum())
            wr = 100 * self.df.loc[col & decided_mask, "is_win"].mean() if n else np.nan
            rows.append({"factor": label, "tipo": "negativo" if key in NEGATIVE_FACTORS else "positivo",
                         "%_ganadoras": round(pw, 1), "%_perdedoras": round(pl, 1),
                         "diferencia": round(pl - pw, 1) if not (np.isnan(pw) or np.isnan(pl)) else np.nan,
                         "trades": n, "wr_cuando_ocurre": round(wr, 1) if n else np.nan})
        table = pd.DataFrame(rows)

        loss_causes = table[(table["tipo"] == "negativo") & (table["trades"] > 0)].sort_values("diferencia", ascending=False)
        win_traits = table[(table["tipo"] == "positivo") & (table["trades"] > 0)].sort_values("diferencia")

        mistakes = Counter(m for ms in self.df.loc[losses, "mistakes"] for m in ms)
        loser_quality = [self.quality_score(i) for i in self.df.index[losses]]
        avoidable_idx = [i for i in self.df.index[losses] if self.red_flags(i) >= 2]
        avoidable = len(avoidable_idx)
        r = self.df.loc[self.df.index[losses], "r_multiple"].dropna()
        avoid_r = self.df.loc[avoidable_idx, "r_multiple"].dropna()

        conclusions = []
        for _, row in loss_causes.head(4).iterrows():
            if row["diferencia"] > 10:
                conclusions.append(
                    f"«{row['factor']}» aparece en el {row['%_perdedoras']:.0f}% de tus perdedoras frente al "
                    f"{row['%_ganadoras']:.0f}% de tus ganadoras (WR cuando ocurre: {row['wr_cuando_ocurre']:.0f}%).")
        for _, row in win_traits.head(3).iterrows():
            if row["diferencia"] < -10:
                conclusions.append(
                    f"Tus ganadoras comparten «{row['factor']}» ({row['%_ganadoras']:.0f}% de ganadoras vs "
                    f"{row['%_perdedoras']:.0f}% de perdedoras).")
        if self.key_confluences:
            conclusions.append("Tu checklist de confluencias que más suben tu WR: " + ", ".join(self.key_confluences) + ".")
        if losses.any():
            conclusions.append(
                f"{avoidable} de {int(losses.sum())} pérdidas ({100 * avoidable / losses.sum():.0f}%) tenían 2 o más "
                f"señales de alerta de proceso: eran evitables y te costaron {abs(avoid_r.sum()):.1f}R de {abs(r.sum()):.1f}R perdidos.")

        return {
            "tabla_factores": table,
            "causas_perdidas": loss_causes,
            "rasgos_ganadoras": win_traits,
            "errores_frecuentes_en_perdidas": mistakes.most_common(),
            "checklist_ideal": self.key_confluences,
            "perdidas_evitables": avoidable,
            "r_perdido_evitable": round(float(abs(avoid_r.sum())), 2),
            "calidad_media_perdedoras": round(float(np.mean(loser_quality)), 1) if loser_quality else None,
            "conclusiones": conclusions,
        }

    def what_if(self) -> pd.DataFrame:
        """Simulación: ¿qué WR/expectativa tendrías si eliminases cierto tipo de trade?"""
        base = analytics.summary(self.df)
        rows = [{"escenario": "Histórico real", "trades": base["trades"], "win_rate": base["win_rate"],
                 "total_r": base["total_r"], "expectativa_r": base["expectancy_r"]}]
        for key, label in NEGATIVE_FACTORS.items():
            mask = self._factors[key].astype(bool)
            if not mask.any() or mask.all():
                continue
            s = analytics.summary(self.df[~mask])
            rows.append({"escenario": f"Sin: {label.lower()}", "trades": s["trades"], "win_rate": s["win_rate"],
                         "total_r": s["total_r"], "expectativa_r": s["expectancy_r"]})
        if self.key_confluences:
            mask = self.df["confluences"].apply(lambda c: bool(set(self.key_confluences) & set(c)))
            if mask.any():
                s = analytics.summary(self.df[mask])
                rows.append({"escenario": "Solo con confluencias clave", "trades": s["trades"], "win_rate": s["win_rate"],
                             "total_r": s["total_r"], "expectativa_r": s["expectancy_r"]})
        return pd.DataFrame(rows)
