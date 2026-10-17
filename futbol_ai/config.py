"""Ligas, fuentes de datos y parámetros por defecto."""
from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "futbol"

# Código football-data.co.uk -> (nombre, carpeta del espejo en GitHub o None)
LEAGUES: dict[str, tuple[str, str | None]] = {
    "SP1": ("LaLiga", "la-liga"),
    "E0": ("Premier League", "premier-league"),
    "I1": ("Serie A", "serie-a"),
    "D1": ("Bundesliga", "bundesliga"),
    "F1": ("Ligue 1", "ligue-1"),
    "SP2": ("LaLiga Hypermotion", None),
    "E1": ("Championship", None),
    "I2": ("Serie B", None),
    "D2": ("2. Bundesliga", None),
    "F2": ("Ligue 2", None),
    "N1": ("Eredivisie", None),
    "P1": ("Liga Portugal", None),
    "B1": ("Jupiler Pro League", None),
    "T1": ("Süper Lig", None),
    "SC0": ("Scottish Premiership", None),
    "G1": ("Super League Grecia", None),
}

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
MIRROR_URL = "https://raw.githubusercontent.com/datasets/football-datasets/main/datasets/{slug}/season-{season}.csv"

N_SEASONS = 5            # temporadas completas anteriores + la actual
XI = 0.0019              # decaimiento temporal por día (vida media ~1 año)
PRIOR_SD = 0.30          # desviación del prior de fuerza de cada equipo
PROMOTED_PRIOR = 0.15    # equipos sin histórico: algo peores que la media
BLEND_M = 100            # peso del modelo, en "partidos equivalentes", al mezclar con la frecuencia histórica
H2H_WEIGHT = 0.5         # cada enfrentamiento directo cuenta como medio partido de evidencia
N_RECENT = 10            # últimos partidos en casa del local / fuera del visitante

MIN_PROB = 0.62          # probabilidad mínima para proponer un pick
MIN_ODDS = 1.25          # cuota justa mínima (evita cuotas "regaladas" sin valor)
MAX_ODDS = 2.10          # cuota justa máxima (evita cuotas exageradas)
VALUE_EDGE = 0.03        # ventaja mínima frente a la casa para marcar "valor"
