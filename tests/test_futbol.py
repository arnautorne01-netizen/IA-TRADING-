import numpy as np
import pandas as pd
import pytest

from futbol_ai import backtest, data
from futbol_ai.markets import MARKETS, market_outcomes, market_probs
from futbol_ai.model import PoissonTeamModel, overdispersed_matrix, score_matrix
from futbol_ai.predictor import LeaguePredictor, build_combo, select_picks

TEAMS = {"Fuerte": (0.5, -0.4), "Bueno": (0.2, -0.1), "Medio": (0.0, 0.0), "Flojo": (-0.3, 0.3),
         "Malo": (-0.5, 0.4), "Otro": (0.0, 0.1)}


def synthetic_league(seasons=4, seed=0):
    rng = np.random.default_rng(seed)
    rows, d = [], pd.Timestamp("2021-08-15")
    names = list(TEAMS)
    for _ in range(seasons):
        for rnd in range(2):
            for h in names:
                for a in names:
                    if h == a:
                        continue
                    lam = np.exp(0.15 + 0.25 + TEAMS[h][0] + TEAMS[a][1])
                    mu = np.exp(0.15 + TEAMS[a][0] + TEAMS[h][1])
                    hg, ag = rng.poisson(lam), rng.poisson(mu)
                    rows.append({"Date": d, "Div": "XX", "HomeTeam": h, "AwayTeam": a, "FTHG": hg, "FTAG": ag,
                                 "HTHG": rng.binomial(hg, 0.45), "HTAG": rng.binomial(ag, 0.45),
                                 "HC": rng.poisson(5.5), "AC": rng.poisson(4.5), "HY": rng.poisson(2), "AY": rng.poisson(2.2),
                                 "HR": 0, "AR": 0})
                    d += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def league():
    return synthetic_league()


def test_score_matrix_sums_to_one_and_dc_raises_draws():
    m0 = score_matrix(1.4, 1.1, 0.0)
    m1 = score_matrix(1.4, 1.1, -0.1)
    assert m0.sum() == pytest.approx(1)
    assert np.trace(m1) > np.trace(m0)


def test_overdispersion_widens_totals():
    base = overdispersed_matrix(5, 4, 0, 0)
    wide = overdispersed_matrix(5, 4, 0.1, 0.1)
    p_base = market_probs(base, "corners")["C_O11.5"]
    p_wide = market_probs(wide, "corners")["C_O11.5"]
    assert wide.sum() == pytest.approx(1)
    assert p_wide > p_base


def test_model_recovers_team_order(league):
    m = PoissonTeamModel().fit(league.HomeTeam, league.AwayTeam, league.FTHG, league.FTAG, league.Date)
    att = dict(zip(m.teams, m.attack))
    assert att["Fuerte"] > att["Medio"] > att["Malo"]
    assert 0.1 < m.home_adv < 0.45
    lam, mu = m.rates("Fuerte", "Malo")
    assert lam > 2 * mu


def test_fit_is_stable(league):
    # El ajuste converge al mismo sitio desde otro punto de partida → gradiente coherente
    a = PoissonTeamModel().fit(league.HomeTeam, league.AwayTeam, league.FTHG, league.FTAG, league.Date)
    b = PoissonTeamModel(prior_sd=0.3001).fit(league.HomeTeam, league.AwayTeam, league.FTHG, league.FTAG, league.Date)
    assert np.allclose(a.attack, b.attack, atol=1e-3)


def test_market_probs_complementary():
    p = market_probs(score_matrix(1.5, 1.2, -0.05), "ft")
    assert p["1"] + p["X"] + p["2"] == pytest.approx(1)
    assert p["O2.5"] + p["U2.5"] == pytest.approx(1)
    assert p["BTTS_SI"] + p["BTTS_NO"] == pytest.approx(1)
    assert p["1X"] == pytest.approx(p["1"] + p["X"])


def test_market_outcomes():
    o = market_outcomes([2, 0], [1, 0], "ft")
    assert list(o["1"]) == [True, False] and list(o["X"]) == [False, True] and list(o["O2.5"]) == [True, False]


def test_predictor_and_picks(league):
    pr = LeaguePredictor(league, "XX")
    p = pr.predict("fuerte", "Malo")          # nombre aproximado
    assert p.home == "Fuerte"
    r = p.rows.set_index("id")
    assert r.loc["1", "p_final"] > r.loc["2", "p_final"]
    assert set(p.expected) == {"ft", "ht", "corners", "cards"}
    assert (p.rows["p_final"].between(0, 1)).all()
    picks = select_picks(p.rows, 0.6, 1.2, 2.2)
    assert not picks.empty
    assert (picks["p_final"] >= 0.6).all() and picks["family"].is_unique


def test_unknown_team_warns(league):
    p = LeaguePredictor(league, "XX").predict("Recién Ascendido FC", "Medio")
    assert p.warnings


def test_value_with_bookmaker_odds(league):
    p = LeaguePredictor(league, "XX").predict("Fuerte", "Malo", {"B365H": 3.0})
    r = p.rows.set_index("id").loc["1"]
    assert r["cuota_casa"] == 3.0 and r["valor"] == pytest.approx(r["p_final"] * 3 - 1)


def test_combo(league):
    pr = LeaguePredictor(league, "XX")
    preds = [pr.predict(h, a) for h, a in [("Fuerte", "Malo"), ("Bueno", "Flojo"), ("Medio", "Otro")]]
    for p in preds:
        p.picks = select_picks(p.rows, 0.6, 1.1, 2.5)
    c = build_combo(preds, min_odds=1.5)
    assert c and len(c["legs"]) >= 2 and c["cuota_justa"] == pytest.approx(1 / c["prob"])


def test_backtest_calibration(league):
    res = backtest.run(league, "XX", test_from=league.Date.max() - pd.Timedelta(days=20))
    assert not res.empty and res["hit"].notna().all()
    cal = backtest.calibration(res)
    assert cal["n"].sum() == len(res)
    assert not backtest.tune(res).empty


def test_dates_and_seasons():
    s = data._parse_dates(pd.Series(["2024-08-11", "11/08/2024", "11/08/24"]))
    assert s.nunique() == 1
    assert data.seasons_to_load(5, pd.Timestamp("2026-10-17").date()) == ["2122", "2223", "2324", "2425", "2526", "2627"]
    d0, d1 = data.weekend_range(pd.Timestamp("2026-10-14").date())
    assert d0.weekday() == 4 and (d1 - d0).days == 3


def test_parse_manual_fixtures():
    f = data.parse_fixtures_text("Real Madrid - Barcelona; 2026-10-18\nBetis vs Sevilla", "SP1")
    assert list(f.HomeTeam) == ["Real Madrid", "Betis"] and f.Date.iloc[0] == pd.Timestamp("2026-10-18")


def test_every_market_has_scope():
    assert all(m.scope in ("ft", "ht", "corners", "cards") for m in MARKETS)
