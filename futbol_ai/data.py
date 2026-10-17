"""Descarga y normalización de resultados históricos y del calendario (football-data.co.uk)."""
from __future__ import annotations

import io
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests

from . import config

STAT_COLS = ["FTHG", "FTAG", "HTHG", "HTAG", "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR", "HF", "AF"]
ODDS_COLS = ["B365H", "B365D", "B365A", "B365>2.5", "B365<2.5", "AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5",
             "PSH", "PSD", "PSA", "MaxH", "MaxD", "MaxA"]


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def current_season_start(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def seasons_to_load(n_seasons: int = config.N_SEASONS, today: date | None = None) -> list[str]:
    """Temporada actual + las `n_seasons` completas anteriores (≈ últimos 5 años)."""
    start = current_season_start(today)
    return [season_code(y) for y in range(start - n_seasons, start + 1)]


def _parse_dates(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    long = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    short = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    return iso.fillna(long).fillna(short)


def normalize(raw: pd.DataFrame, league: str | None = None) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    df = df.dropna(subset=[c for c in ("HomeTeam", "AwayTeam") if c in df.columns])
    if df.empty:
        return pd.DataFrame(columns=["Date", "Div", "HomeTeam", "AwayTeam"])
    df["Date"] = _parse_dates(df["Date"])
    df = df.dropna(subset=["Date"])
    if league is not None:
        df["Div"] = league
    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    for c in STAT_COLS + ODDS_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = ["Date", "Div", "HomeTeam", "AwayTeam", "Referee"] + [c for c in STAT_COLS + ODDS_COLS if c in df.columns]
    if "Time" in df.columns:
        keep.insert(1, "Time")
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


def _get(url: str, timeout: int = 20) -> str | None:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 futbol-ai"})
    except requests.RequestException:
        return None
    if r.status_code != 200 or not r.content.strip():
        return None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return r.content.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def _read_csv_text(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text), on_bad_lines="skip")


def download_season(league: str, season: str, force: bool = False, max_age_h: float = 12) -> pd.DataFrame:
    """Descarga una temporada (con caché local). Prueba football-data.co.uk y luego el espejo de GitHub."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DATA_DIR / f"{league}_{season}.csv"
    is_current = season == season_code(current_season_start())
    if path.exists() and not force:
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if not is_current or age_h < max_age_h:
            return normalize(pd.read_csv(path, encoding="utf-8"), league)

    urls = [config.FOOTBALL_DATA_URL.format(season=season, code=league)]
    slug = config.LEAGUES.get(league, (None, None))[1]
    if slug:
        urls.append(config.MIRROR_URL.format(slug=slug, season=season))
    for url in urls:
        text = _get(url)
        if text is None:
            continue
        try:
            raw = _read_csv_text(text)
        except Exception:
            continue
        if {"HomeTeam", "AwayTeam", "FTHG"}.issubset({str(c).strip().lstrip("﻿") for c in raw.columns}):
            raw.to_csv(path, index=False, encoding="utf-8")
            return normalize(raw, league)
    if path.exists():  # sin red: usar la copia vieja
        return normalize(pd.read_csv(path, encoding="utf-8"), league)
    return normalize(pd.DataFrame(), league)


def load_league(league: str, n_seasons: int = config.N_SEASONS, force: bool = False,
                today: date | None = None) -> pd.DataFrame:
    frames = [download_season(league, s, force=force) for s in seasons_to_load(n_seasons, today)]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return normalize(pd.DataFrame(), league)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["FTHG", "FTAG"]).drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
    return df.sort_values("Date").reset_index(drop=True)


def load_leagues(leagues: list[str], **kw) -> dict[str, pd.DataFrame]:
    return {lg: load_league(lg, **kw) for lg in leagues}


def load_fixtures(leagues: list[str] | None = None, date_from: date | None = None,
                  date_to: date | None = None) -> pd.DataFrame:
    """Próximos partidos (con cuotas) de football-data.co.uk/fixtures.csv."""
    text = _get(config.FIXTURES_URL)
    if text is None:
        return pd.DataFrame(columns=["Date", "Div", "HomeTeam", "AwayTeam"])
    df = normalize(_read_csv_text(text))
    return filter_fixtures(df, leagues, date_from, date_to)


def filter_fixtures(df: pd.DataFrame, leagues=None, date_from=None, date_to=None) -> pd.DataFrame:
    if leagues:
        df = df[df["Div"].isin(leagues)]
    if date_from is not None:
        df = df[df["Date"] >= pd.Timestamp(date_from)]
    if date_to is not None:
        df = df[df["Date"] <= pd.Timestamp(date_to)]
    return df.sort_values(["Date", "Div"]).reset_index(drop=True)


def weekend_range(today: date | None = None) -> tuple[date, date]:
    """Del viernes al lunes del fin de semana en curso (o el próximo)."""
    today = today or date.today()
    wd = today.weekday()  # lunes=0
    fri = today + timedelta(days=4 - wd) if wd <= 4 else today - timedelta(days=wd - 4)
    return max(fri, today), fri + timedelta(days=3)


def parse_fixtures_text(text: str, league: str) -> pd.DataFrame:
    """Partidos escritos a mano, uno por línea: 'Local - Visitante' (fecha opcional al final: '; 2026-10-18')."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        d = pd.Timestamp(datetime.now().date())
        if ";" in line:
            line, ds = line.rsplit(";", 1)
            parsed = _parse_dates(pd.Series([ds.strip()])).iloc[0]
            if pd.notna(parsed):
                d = parsed
        for sep in (" - ", " vs ", " v ", "-"):
            if sep in line:
                h, a = line.split(sep, 1)
                rows.append({"Date": d, "Div": league, "HomeTeam": h.strip(), "AwayTeam": a.strip()})
                break
    return pd.DataFrame(rows, columns=["Date", "Div", "HomeTeam", "AwayTeam"])
