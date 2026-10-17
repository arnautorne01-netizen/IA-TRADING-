"""IA de pronósticos de fútbol.

Ejecuta:  streamlit run futbol_app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from futbol_ai import backtest, config, data, report
from futbol_ai.predictor import LeaguePredictor, build_combo, predict_fixtures, select_picks

st.set_page_config(page_title="IA Fútbol · Pronósticos", page_icon="⚽", layout="wide")

LEAGUE_OPTS = {f"{name} ({code})": code for code, (name, _) in config.LEAGUES.items()}


@st.cache_data(ttl=6 * 3600, show_spinner="Descargando histórico de los últimos 5 años...")
def load_hist(leagues: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    return data.load_leagues(list(leagues))


@st.cache_data(ttl=3 * 3600, show_spinner="Buscando partidos...")
def load_fx(leagues: tuple[str, ...], d0, d1) -> pd.DataFrame:
    return data.load_fixtures(list(leagues), d0, d1)


@st.cache_resource(show_spinner="Ajustando el modelo...")
def predictor(league: str, n_rows: int) -> LeaguePredictor:
    return LeaguePredictor(load_hist((league,))[league], league)


def fmt_picks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df[["Mercado", "p_final", "cuota_justa", "cuota_minima", "cuota_casa", "local_casa", "visit_fuera", "h2h",
              "confianza"]].copy()
    out.columns = ["Mercado", "Prob.", "Cuota justa", "Cuota mínima", "Casa", "Local en casa", "Visit. fuera", "H2H", "Confianza"]
    return out


def show_match(p, min_prob, cmin, cmax):
    p.picks = select_picks(p.rows, min_prob, cmin, cmax)
    lg = config.LEAGUES.get(p.league, (p.league,))[0]
    with st.container(border=True):
        st.subheader(f"{p.home} vs {p.away}")
        st.caption(f"{lg} · {report._fecha(p.date)}")
        cols = st.columns(len(p.expected) + 1)
        for c, (scope, (h, a)) in zip(cols, p.expected.items()):
            c.metric(f"{report.SCOPE_NAMES[scope]} esperados", f"{h:.2f} – {a:.2f}", f"total {h + a:.2f}", delta_color="off")
        cols[-1].markdown("**Marcadores probables**  \n" + "  \n".join(f"{s}: {q:.0%}" for s, q in p.top_scores[:3]))
        for w in p.warnings:
            st.warning(w)
        if p.picks.empty:
            st.info("Sin picks: ningún mercado supera el umbral con una cuota razonable.")
        else:
            st.dataframe(fmt_picks(p.picks), hide_index=True, width="stretch",
                         column_config={"Prob.": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                                        "Cuota justa": st.column_config.NumberColumn(format="%.2f"),
                                        "Cuota mínima": st.column_config.NumberColumn(format="%.2f"),
                                        "Casa": st.column_config.NumberColumn(format="%.2f")})
        with st.expander("Lo que más ha ocurrido en el histórico"):
            mf = report.most_frequent(p, 8)[["Mercado", "p_hist", "local_casa", "visit_fuera", "h2h"]]
            st.dataframe(mf.rename(columns={"p_hist": "Frecuencia", "local_casa": "Local en casa",
                                            "visit_fuera": "Visit. fuera", "h2h": "H2H"}),
                         hide_index=True, width="stretch")
        with st.expander("Todos los mercados"):
            allm = p.rows[["Mercado", "p_model", "p_hist", "p_final", "cuota_justa", "local_casa", "visit_fuera", "h2h"]]
            st.dataframe(allm.sort_values("p_final", ascending=False), hide_index=True, width="stretch")


st.title("⚽ IA de pronósticos de fútbol")
st.caption("Modelo Dixon-Coles con 5 temporadas (goles, 1ª parte, córners y tarjetas) + frecuencias de los últimos "
           "partidos y enfrentamientos directos. Cuota justa = 1/probabilidad; apuesta solo si la casa paga la cuota mínima.")

with st.sidebar:
    st.header("Ajustes")
    sel = st.multiselect("Ligas", list(LEAGUE_OPTS), default=list(LEAGUE_OPTS)[:5])
    leagues = tuple(LEAGUE_OPTS[s] for s in sel)
    min_prob = st.slider("Probabilidad mínima del pick", 0.50, 0.90, config.MIN_PROB, 0.01)
    cmin, cmax = st.slider("Rango de cuota justa", 1.05, 3.50, (config.MIN_ODDS, config.MAX_ODDS), 0.05)
    if st.button("Forzar actualización de datos"):
        st.cache_data.clear(), st.cache_resource.clear()
        for lg in leagues:
            for s in data.seasons_to_load():
                data.download_season(lg, s, force=True)
        st.success("Datos actualizados")

tab_w, tab_m, tab_b = st.tabs(["Fin de semana", "Partidos a mano", "Fiabilidad (backtest)"])

with tab_w:
    d0, d1 = data.weekend_range()
    c1, c2 = st.columns(2)
    d0 = c1.date_input("Desde", d0)
    d1 = c2.date_input("Hasta", d1)
    fx = load_fx(leagues, d0, d1) if leagues else pd.DataFrame()
    if fx.empty:
        st.info("No se han encontrado partidos en football-data.co.uk para esas fechas/ligas. "
                "Puedes escribirlos en la pestaña «Partidos a mano».")
    else:
        hist = load_hist(tuple(sorted(set(fx["Div"]))))
        preds = predict_fixtures(hist, fx)
        for p in preds:
            p.picks = select_picks(p.rows, min_prob, cmin, cmax)
        combo = build_combo(preds)
        if combo:
            st.success(f"**Combinada sugerida** · probabilidad {combo['prob']:.0%} · cuota justa {combo['cuota_justa']:.2f}  \n"
                       + "  \n".join(f"- {l['Partido']}: {l['Mercado']} ({l['p']:.0%})" for l in combo["legs"]))
        st.download_button("Descargar informe HTML", report.to_html(preds, combo), "pronosticos.html", "text/html")
        for p in preds:
            show_match(p, min_prob, cmin, cmax)

with tab_m:
    lg = st.selectbox("Liga", list(LEAGUE_OPTS), key="lg_manual")
    code = LEAGUE_OPTS[lg]
    hist = load_hist((code,))[code]
    if hist.empty:
        st.error("No hay datos de esa liga (¿sin conexión?).")
    else:
        pr = predictor(code, len(hist))
        st.caption("Equipos con nombre del histórico: " + ", ".join(pr.current_teams))
        txt = st.text_area("Un partido por línea: «Local - Visitante»", "Real Madrid - Barcelona" if code == "SP1" else "")
        preds = [pr.predict(r.HomeTeam, r.AwayTeam, None, r.Date) for r in data.parse_fixtures_text(txt, code).itertuples()]
        for p in preds:
            show_match(p, min_prob, cmin, cmax)
        combo = build_combo(preds)
        if combo:
            st.success(f"**Combinada** · probabilidad {combo['prob']:.0%} · cuota justa {combo['cuota_justa']:.2f}")

with tab_b:
    st.write("Simula las últimas jornadas como si fueran futuras: cada semana el modelo solo ve los partidos anteriores. "
             "Si está bien calibrado, cuando dice 75% debe acertar ~75%.")
    lg = st.selectbox("Liga", list(LEAGUE_OPTS), key="lg_bt")
    code = LEAGUE_OPTS[lg]
    desde = st.date_input("Probar desde", pd.Timestamp.today() - pd.Timedelta(days=365))
    if st.button("Ejecutar backtest"):
        with st.spinner("Calculando (puede tardar ~30 s por temporada)..."):
            res = backtest.run(load_hist((code,))[code], code, test_from=desde)
        if res.empty:
            st.error("No hay partidos en ese periodo.")
        else:
            st.subheader("Picks que habría dado la IA")
            st.dataframe(backtest.picks_summary(res), width="stretch")
            st.subheader("Calibración")
            cal = backtest.calibration(res)
            st.dataframe(cal.astype({"tramo": str}), hide_index=True, width="stretch")
            v = backtest.value_summary(res)
            if not v.empty:
                st.subheader("Apostando solo con valor frente a Bet365/media del mercado")
                st.dataframe(v, hide_index=True)
