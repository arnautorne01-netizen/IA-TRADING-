import numpy as np
import pandas as pd
import pytest

from trading_ai import ai_coach, analytics, bias, diagnosis, fundamentals, technical
from trading_ai.analytics import TradeFilter
from trading_ai.journal import Journal, compute_r_multiple, normalize_trade
from trading_ai.sample import generate_sample


@pytest.fixture
def journal(tmp_path):
    j = Journal(tmp_path / "t.db")
    j.add_many(generate_sample(120).to_dict("records"))
    return j


# --- journal -----------------------------------------------------------------
def test_r_multiple_long_and_short():
    assert compute_r_multiple("Long", 1.1000, 1.0950, 1.1100) == 2.0
    assert compute_r_multiple("Short", 1.1000, 1.1050, 1.1050) == -1.0
    assert compute_r_multiple("Long", 1.1, 1.1, 1.2) is None


def test_normalize_classifies_result_and_tags():
    t = normalize_trade({"date": "2026-03-02", "pair": "eur/usd", "direction": "compra", "entry": 1.1,
                         "stop_loss": 1.095, "exit_price": 1.1005, "confluences": "FVG; OB"})
    assert t["pair"] == "EURUSD" and t["direction"] == "Long"
    assert t["result"] == "BE"  # 0.1R -> dentro del umbral de breakeven
    assert t["confluences"] == "FVG, OB"


def test_csv_roundtrip_with_spanish_headers(tmp_path):
    j = Journal(tmp_path / "x.db")
    csv = tmp_path / "t.csv"
    csv.write_text("fecha,par,dirección,r,confluencias,sesgo\n2026-01-02,EURUSD,Short,2,\"FVG, Killzone\",Bajista\n")
    assert j.import_csv(csv) == 1
    df = j.load()
    assert df.iloc[0]["result"] == "Win"
    assert df.iloc[0]["bias_aligned"] == True  # noqa: E712
    assert "FVG" in j.export_csv()


def test_update_and_delete(journal):
    tid = int(journal.load()["id"].iloc[0])
    journal.update_trade(tid, {"notes": "revisada"})
    assert journal.get_trade(tid)["notes"] == "revisada"
    journal.delete_trade(tid)
    assert journal.get_trade(tid) is None


# --- analytics ---------------------------------------------------------------
def test_summary_consistency(journal):
    df = journal.load()
    s = analytics.summary(df)
    assert s["wins"] + s["losses"] + s["breakeven"] == s["trades"]
    assert 0 <= s["win_rate"] <= 100
    assert s["total_r"] == pytest.approx(df["r_multiple"].sum(), abs=0.05)


def test_filters(journal):
    df = journal.load()
    sub = analytics.apply_filter(df, TradeFilter(pairs=["EURUSD"], confluences_all=["FVG"], followed_plan=True))
    assert (sub["pair"] == "EURUSD").all()
    assert sub["confluences"].apply(lambda c: "FVG" in c).all()
    assert (sub["followed_plan"] == True).all()  # noqa: E712
    none = analytics.apply_filter(df, TradeFilter(confluences_none=["FVG"]))
    assert not none["confluences"].apply(lambda c: "FVG" in c).any()


def test_confluence_lift_detects_edge(journal):
    df = journal.load()
    lift = analytics.confluence_lift(df)
    top = lift.head(4)["confluencia"].tolist()
    # el generador de ejemplo da ventaja a estas confluencias
    assert {"Liquidez barrida", "Sesgo diario a favor"} & set(top)


def test_breakdown_explodes_lists(journal):
    b = analytics.breakdown(journal.load(), "confluences")
    assert "FVG" in b.index


# --- diagnosis ---------------------------------------------------------------
def test_pattern_report(journal):
    d = diagnosis.Diagnoser(journal.load())
    rep = d.pattern_report()
    assert rep["conclusiones"]
    causes = rep["causas_perdidas"]["factor"].head(5).tolist()
    assert any("noticia" in c or "plan" in c or "sesgo" in c or "confluencias" in c for c in causes)
    exp = d.explain_trade(0)
    assert 0 <= exp["quality_score"] <= 100
    assert len(d.what_if()) > 1


# --- technical ---------------------------------------------------------------
def _trend(n=300, drift=0.0005, seed=1):
    rng = np.random.default_rng(seed)
    close = 1.05 * np.exp(np.cumsum(drift + rng.normal(0, 0.003, n)))
    idx = pd.bdate_range("2025-01-01", periods=n)
    op = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": op, "high": np.maximum(op, close) * 1.002,
                         "low": np.minimum(op, close) * 0.998, "close": close}, index=idx)


def test_technical_bias_direction():
    up = technical.technical_bias(_trend(drift=0.002), "EURUSD")
    down = technical.technical_bias(_trend(drift=-0.002), "EURUSD")
    assert up.score > 0.3 and down.score < -0.3
    assert up.label.endswith("lcista")


# --- fundamentals ------------------------------------------------------------
def test_parse_number():
    assert fundamentals.parse_number("250K") == 250_000
    assert fundamentals.parse_number("-0.3%") == -0.3
    assert fundamentals.parse_number("") is None


def test_hot_us_cpi_is_bearish_eurusd():
    ev = fundamentals.EconEvent("CPI m/m", "USD", pd.Timestamp.now(tz="UTC"), "High", "0.2%", "0.2%", "0.5%")
    imp = fundamentals.analyze_event(ev)
    assert imp.eurusd_effect < 0 < imp.dxy_effect
    assert "baje" in imp.explanation


def test_higher_us_unemployment_is_bullish_eurusd():
    ev = fundamentals.EconEvent("Unemployment Rate", "USD", pd.Timestamp.now(tz="UTC"), "High", "4.1%", "4.1%", "4.4%")
    assert fundamentals.analyze_event(ev).eurusd_effect > 0


def test_pending_event_has_scenarios():
    ev = fundamentals.EconEvent("Non-Farm Employment Change", "USD", pd.Timestamp.now(tz="UTC"), "High", "150K", "120K")
    imp = fundamentals.analyze_event(ev)
    assert imp.surprise is None and len(imp.scenarios) == 2


# --- bias --------------------------------------------------------------------
def test_combine_is_consistent_and_bounded():
    up = technical.technical_bias(_trend(drift=0.002), "EURUSD")
    dxy_down = technical.technical_bias(_trend(drift=-0.002, seed=3), "DXY")
    res = bias.combine({"EURUSD": up, "DXY": dxy_down}, {"EURUSD": 0.5, "DXY": -0.5}, {"EURUSD": 0, "DXY": 0})
    assert res["EURUSD"].prob_up > 0.6 and res["DXY"].prob_up < 0.4
    assert 0.2 <= res["DXY"].prob_up <= 0.8
    with_ai = bias.combine({"EURUSD": up, "DXY": dxy_down}, {}, {}, ai_prob={"EURUSD": 0.3})
    assert with_ai["EURUSD"].prob_up < res["EURUSD"].prob_up


def test_bias_log_evaluation(journal):
    journal.log_bias("2026-01-02", "EURUSD", 0.65, 0.4, 0.1)
    journal.log_bias("2026-01-05", "EURUSD", 0.35, -0.4, 0.0)
    journal.set_bias_outcome("2026-01-02", "EURUSD", 0.003)
    journal.set_bias_outcome("2026-01-05", "EURUSD", 0.002)
    ev = bias.evaluate_bias_log(journal.load_bias_log())
    assert ev["evaluados"] == 2 and ev["acierto"] == 50.0


# --- ai coach helpers --------------------------------------------------------
def test_extract_json_from_brief():
    text = 'Análisis...\n```json\n{"EURUSD": {"prob_up": 0.58, "sesgo": "alcista"}, "DXY": {"prob_up": 0.42}}\n```'
    data = ai_coach.extract_json(text)
    assert data["EURUSD"]["prob_up"] == 0.58
    assert "```" not in ai_coach.strip_json_block(text)


def test_load_ohlc_csv(tmp_path):
    path = tmp_path / "eur.csv"
    _trend().rename_axis("date").to_csv(path)
    df = technical.load_ohlc_csv(path)
    assert list(df.columns) == ["open", "high", "low", "close"] and len(df) == 300
    # formato TradingView (time en segundos unix)
    tv = _trend().reset_index(drop=True)
    tv.insert(0, "time", [int(t.timestamp()) for t in pd.bdate_range("2025-01-01", periods=300)])
    tv.to_csv(tmp_path / "tv.csv", index=False)
    assert technical.load_ohlc_csv(tmp_path / "tv.csv").index[0] == pd.Timestamp("2025-01-01")


def test_structure_reads_trend():
    assert technical.market_structure(_trend(drift=0.002).tail(120), 5)[0] > 0
    assert technical.market_structure(_trend(drift=-0.002).tail(120), 5)[0] < 0
