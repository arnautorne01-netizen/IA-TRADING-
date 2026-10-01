"""Análisis fundamental: calendario económico (noticias de alto impacto),
interpretación de sorpresas (dato real vs previsión) y titulares del día."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests

from . import config

# Peso del EUR en el índice DXY (≈57.6%): una sorpresa del EUR mueve el DXY en sentido contrario
EUR_WEIGHT_IN_DXY = 0.576

# Cada evento: (palabras clave, +1 si un dato MAYOR fortalece la divisa / -1 si la debilita, importancia, explicación)
EVENT_RULES: list[tuple[tuple[str, ...], int, float, str]] = [
    (("non-farm", "nfp", "nóminas"), 1, 1.0, "Empleo creado fuera del sector agrícola. Más empleo = economía fuerte = banco central más duro (hawkish)."),
    (("unemployment rate", "tasa de desempleo", "paro"), -1, 0.8, "Tasa de desempleo. Un paro más alto debilita la divisa (más probabilidad de bajadas de tipos)."),
    (("unemployment claims", "jobless claims", "solicitudes de desempleo"), -1, 0.4, "Peticiones semanales de subsidio. Más peticiones = mercado laboral débil."),
    (("cpi", "ipc", "inflation", "inflación", "hicp"), 1, 1.0, "Inflación. Inflación alta = tipos altos por más tiempo = divisa fuerte."),
    (("pce",), 1, 0.9, "Inflación PCE, la medida preferida de la Fed."),
    (("ppi", "ipp"), 1, 0.5, "Precios a la producción, anticipa la inflación al consumidor."),
    (("federal funds rate", "fomc", "interest rate", "main refinancing", "rate decision", "tipos"), 1, 1.0, "Decisión de tipos. Subida o tono duro (hawkish) = divisa fuerte."),
    (("gdp", "pib"), 1, 0.8, "Crecimiento económico. Más crecimiento = divisa fuerte."),
    (("retail sales", "ventas minoristas"), 1, 0.7, "Consumo. Ventas fuertes = economía robusta."),
    (("ism", "pmi"), 1, 0.7, "Índices de gerentes de compras (>50 expansión). Mejor dato = divisa fuerte."),
    (("jolts", "job openings"), 1, 0.5, "Ofertas de empleo. Más vacantes = mercado laboral tenso."),
    (("adp",), 1, 0.5, "Empleo privado ADP, anticipo de las NFP."),
    (("average hourly earnings", "salarios"), 1, 0.7, "Salarios. Más salarios = más presión inflacionaria."),
    (("consumer confidence", "consumer sentiment", "confianza"), 1, 0.4, "Confianza del consumidor."),
    (("ifo", "zew", "sentix"), 1, 0.5, "Sentimiento económico en Alemania/Eurozona."),
    (("durable goods", "bienes duraderos"), 1, 0.4, "Pedidos de bienes duraderos."),
    (("trade balance", "balanza comercial"), 1, 0.2, "Balanza comercial."),
]

SPEECH_KEYWORDS = ("speaks", "speech", "testimony", "press conference", "minutes", "comparecencia", "habla", "actas")


@dataclass
class EconEvent:
    title: str
    currency: str
    date: pd.Timestamp
    impact: str
    forecast: str | None = None
    previous: str | None = None
    actual: str | None = None

    @property
    def rule(self):
        low = self.title.lower()
        for kws, direction, weight, expl in EVENT_RULES:
            if any(k in low for k in kws):
                return direction, weight, expl
        if any(k in low for k in SPEECH_KEYWORDS):
            return 0, 0.6, "Discurso/comparecencia: el impacto depende del tono (hawkish = divisa fuerte, dovish = débil)."
        return 0, 0.3, "Evento sin regla específica: revisa el contexto."


def parse_number(value) -> float | None:
    """Convierte '3.2%', '250K', '-1.1M', '1.5B' en número."""
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    if not s:
        return None
    m = re.match(r"^([+-]?\d*\.?\d+)\s*([kKmMbBtT%]?)$", s)
    if not m:
        return None
    num = float(m.group(1))
    mult = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}.get(m.group(2).lower(), 1)
    return num * mult


def fetch_calendar(url: str = config.CALENDAR_URL, timeout: int = 15) -> list[EconEvent]:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 trading-ai"})
    resp.raise_for_status()
    return parse_calendar(resp.json())


def parse_calendar(items: list[dict]) -> list[EconEvent]:
    events = []
    for it in items:
        events.append(EconEvent(
            title=it.get("title", ""), currency=it.get("country", ""),
            date=pd.to_datetime(it.get("date"), utc=True), impact=it.get("impact", ""),
            forecast=it.get("forecast") or None, previous=it.get("previous") or None,
            actual=it.get("actual") or None,
        ))
    return events


def relevant_events(events: list[EconEvent], min_impact: str = "High") -> list[EconEvent]:
    order = {"Low": 0, "Medium": 1, "High": 2, "Holiday": -1}
    return [e for e in events if e.currency in ("USD", "EUR") and order.get(e.impact, 0) >= order[min_impact]]


@dataclass
class EventImpact:
    event: EconEvent
    surprise: float | None              # sorpresa normalizada (-1..1); None si aún no hay dato
    currency_effect: float              # +: divisa del evento se fortalece
    eurusd_effect: float                # efecto estimado sobre EURUSD (-1..1)
    dxy_effect: float                   # efecto estimado sobre DXY (-1..1)
    explanation: str
    scenarios: list[str] = field(default_factory=list)


def analyze_event(e: EconEvent) -> EventImpact:
    direction, weight, expl = e.rule
    actual, forecast, previous = parse_number(e.actual), parse_number(e.forecast), parse_number(e.previous)
    base = forecast if forecast is not None else previous
    surprise = None
    effect = 0.0
    if actual is not None and base is not None and direction != 0:
        scale = max(abs(base), 1e-9)
        # sorpresa relativa, saturada: un 10% de desvío ≈ sorpresa máxima
        surprise = float(np.clip((actual - base) / scale * 10, -1, 1))
        if abs(actual - base) < 1e-12:
            surprise = 0.0
        effect = direction * surprise * weight
    if e.currency == "USD":
        eur, dxy = -effect, effect
    else:
        eur, dxy = effect, -effect * EUR_WEIGHT_IN_DXY

    cur = "dólar" if e.currency == "USD" else "euro"
    scenarios = []
    if direction != 0 and base is not None:
        better = "por encima" if direction > 0 else "por debajo"
        worse = "por debajo" if direction > 0 else "por encima"
        up_pair = "EURUSD baja / DXY sube" if e.currency == "USD" else "EURUSD sube / DXY baja"
        down_pair = "EURUSD sube / DXY baja" if e.currency == "USD" else "EURUSD baja / DXY sube"
        scenarios = [
            f"Si sale {better} de {e.forecast or e.previous}: {cur} fuerte → lo lógico es {up_pair}.",
            f"Si sale {worse} de {e.forecast or e.previous}: {cur} débil → lo lógico es {down_pair}.",
        ]
    elif direction == 0:
        scenarios = [f"Tono duro (hawkish) → {cur} fuerte.", f"Tono suave (dovish) → {cur} débil."]

    if surprise is None:
        verdict = "Pendiente de publicación."
    elif abs(effect) < 0.05:
        verdict = "Dato en línea con lo esperado: impacto limitado."
    else:
        strong = effect > 0
        verdict = (f"Dato {'mejor' if strong else 'peor'} de lo esperado ({e.actual} vs {e.forecast or e.previous}) → "
                   f"{cur} {'fuerte' if strong else 'débil'} → lo lógico es que EURUSD "
                   f"{'suba' if eur > 0 else 'baje'} y el DXY {'suba' if dxy > 0 else 'baje'}.")
    return EventImpact(event=e, surprise=surprise, currency_effect=effect, eurusd_effect=round(eur, 3),
                       dxy_effect=round(dxy, 3), explanation=f"{expl} {verdict}", scenarios=scenarios)


def fundamental_score(impacts: list[EventImpact]) -> dict:
    """Agrega las sorpresas publicadas en una puntuación -1..1 para cada par."""
    published = [i for i in impacts if i.surprise is not None]
    if not published:
        return {"EURUSD": 0.0, "DXY": 0.0, "eventos_publicados": 0}
    eur = float(np.clip(sum(i.eurusd_effect for i in published), -1, 1))
    dxy = float(np.clip(sum(i.dxy_effect for i in published), -1, 1))
    return {"EURUSD": round(eur, 3), "DXY": round(dxy, 3), "eventos_publicados": len(published)}


# --------------------------------------------------------------------------- #
# Titulares (sentimiento por palabras clave)
# --------------------------------------------------------------------------- #
USD_POSITIVE = ["hawkish fed", "fed hike", "strong dollar", "dollar rises", "dollar gains", "risk-off",
                "safe haven", "treasury yields rise", "yields jump", "hot inflation", "strong jobs",
                "dólar sube", "fed dura", "aversión al riesgo"]
USD_NEGATIVE = ["dovish fed", "fed cut", "rate cut", "weak dollar", "dollar falls", "dollar slips",
                "yields fall", "weak jobs", "recession fears", "dólar cae", "recorte de tipos"]
EUR_POSITIVE = ["hawkish ecb", "ecb hike", "euro rises", "euro gains", "eurozone growth", "bce dura", "euro sube"]
EUR_NEGATIVE = ["dovish ecb", "ecb cut", "euro falls", "euro slips", "eurozone recession", "euro cae", "bce recorta"]


def fetch_headlines(feeds: list[str] | None = None, limit: int = 40) -> list[dict]:
    import feedparser

    out = []
    for url in feeds or config.NEWS_FEEDS:
        try:
            parsed = feedparser.parse(url)
        except Exception:  # noqa: BLE001 - una fuente caída no debe romper el resto
            continue
        for entry in parsed.entries[:limit]:
            out.append({"title": entry.get("title", ""), "link": entry.get("link", ""),
                        "published": entry.get("published", ""), "source": url})
    return out


def headline_sentiment(headlines: list[dict]) -> dict:
    usd = eur = 0
    hits = []
    for h in headlines:
        t = h["title"].lower()
        du = sum(k in t for k in USD_POSITIVE) - sum(k in t for k in USD_NEGATIVE)
        de = sum(k in t for k in EUR_POSITIVE) - sum(k in t for k in EUR_NEGATIVE)
        if du or de:
            hits.append({**h, "usd": du, "eur": de})
        usd += du
        eur += de
    eurusd = float(np.tanh((eur - usd) / 5))
    return {"EURUSD": round(eurusd, 3), "DXY": round(-eurusd, 3), "titulares_relevantes": hits}
