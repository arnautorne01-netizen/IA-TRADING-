"""Configuración central del asistente de trading."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("TRADING_AI_DB", ROOT / "data" / "journal.db"))

# Modelo de Claude usado por el coach de IA
CLAUDE_MODEL = os.environ.get("TRADING_AI_MODEL", "claude-opus-5-5")

PAIRS = ["EURUSD", "DXY"]

# Tickers de Yahoo Finance
YF_TICKERS = {
    "EURUSD": "EURUSD=X",
    "DXY": "DX-Y.NYB",
}

SESSIONS = ["Asia", "Londres", "Nueva York", "Overlap Londres-NY", "Fuera de sesión"]
DIRECTIONS = ["Long", "Short"]
BIAS_OPTIONS = ["Alcista", "Bajista", "Neutral"]

# Confluencias sugeridas (puedes añadir las tuyas libremente en el journal)
DEFAULT_CONFLUENCES = [
    "Sesgo diario a favor",
    "Sesgo H4 a favor",
    "Liquidez barrida",
    "Order Block",
    "FVG",
    "BOS / CHoCH",
    "Soporte/Resistencia",
    "Zona premium/discount",
    "Killzone",
    "Divergencia RSI",
    "EMA 200 a favor",
    "Noticia a favor",
    "Patrón de velas",
    "Nivel psicológico",
    "PDH/PDL",
    "Asia High/Low",
    "Correlación DXY confirma",
]

DEFAULT_MISTAKES = [
    "Entrada anticipada",
    "Entrada tardía",
    "Mover SL",
    "Cerrar antes de TP",
    "Sobreoperar",
    "Revenge trading",
    "FOMO",
    "Contra sesgo",
    "Operar en noticia",
    "Riesgo excesivo",
    "Sin confirmación",
]

EMOTIONS = ["Tranquilo", "Confiado", "Ansioso", "FOMO", "Frustrado", "Venganza", "Aburrido", "Eufórico"]
NEGATIVE_EMOTIONS = {"Ansioso", "FOMO", "Frustrado", "Venganza", "Aburrido", "Eufórico"}

# Umbral (en R) para considerar un trade como breakeven
BE_THRESHOLD_R = 0.1

# Fuentes RSS de noticias (titulares) — puedes cambiarlas
NEWS_FEEDS = [
    "https://www.fxstreet.com/rss/news",
    "https://www.forexlive.com/feed/news",
    "https://feeds.reuters.com/reuters/businessNews",
]

# Calendario económico (feed JSON público de Forex Factory)
CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
