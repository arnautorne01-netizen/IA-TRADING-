"""IA de Trading — EURUSD & DXY.

Ejecuta:  streamlit run app.py
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from trading_ai import ai_coach, analytics, bias, config, diagnosis, fundamentals, technical
from trading_ai.analytics import TradeFilter
from trading_ai.journal import Journal

st.set_page_config(page_title="IA Trading EURUSD/DXY", page_icon="📈", layout="wide")


@st.cache_resource
def get_journal() -> Journal:
    return Journal()


journal = get_journal()


def load_df() -> pd.DataFrame:
    return journal.load()


def pct(v) -> str:
    return "—" if v is None or pd.isna(v) else f"{v:.1f}%"


def num(v, suffix="") -> str:
    return "—" if v is None or pd.isna(v) else f"{v:+.2f}{suffix}"


def coach_box(fn, *args, spinner="La IA está analizando...", **kwargs):
    if not ai_coach.is_available():
        st.info("Configura `ANTHROPIC_API_KEY` para activar el coach de IA (Claude). "
                "Todo lo demás funciona sin ella.")
        return None
    with st.spinner(spinner):
        try:
            return fn(*args, **kwargs)
        except ai_coach.CoachUnavailable as e:
            st.error(str(e))
            return None


# --------------------------------------------------------------------------- #
# Barra lateral
# --------------------------------------------------------------------------- #
st.sidebar.title("📈 IA Trading")
st.sidebar.caption("EURUSD · DXY")
page = st.sidebar.radio("Sección", [
    "📊 Journal y Win Rate",
    "➕ Registrar operaciones",
    "🔍 Filtros avanzados",
    "🧠 Buenas vs malas",
    "🌍 Sesgo diario y noticias",
    "💬 Pregunta a la IA",
])
st.sidebar.divider()
st.sidebar.write("Coach IA:", "🟢 activo" if ai_coach.is_available() else "⚪ sin API key")

df_all = load_df()

# --------------------------------------------------------------------------- #
# 1. Journal / Dashboard
# --------------------------------------------------------------------------- #
if page == "📊 Journal y Win Rate":
    st.title("📊 Journal y estadísticas")
    if df_all.empty:
        st.warning("Aún no hay operaciones. Ve a «Registrar operaciones» para añadir o importar un CSV "
                   "(puedes cargar `sample_data/sample_trades.csv` para probar).")
        st.stop()

    pairs = st.multiselect("Par", config.PAIRS, default=[p for p in config.PAIRS if p in df_all["pair"].unique()])
    df = df_all[df_all["pair"].isin(pairs)] if pairs else df_all
    s = analytics.summary(df)

    c = st.columns(6)
    c[0].metric("Operaciones", s["trades"])
    c[1].metric("Win Rate", pct(s["win_rate"]), help="Ganadas / (ganadas + perdidas), sin contar BE")
    c[2].metric("R total", num(s["total_r"], "R"))
    c[3].metric("Expectativa", num(s["expectancy_r"], "R"), help="R medio por operación")
    c[4].metric("Profit factor", "—" if s["profit_factor"] is None else f"{s['profit_factor']:.2f}")
    c[5].metric("Máx. drawdown", num(s["max_drawdown_r"], "R"))
    c = st.columns(6)
    c[0].metric("Ganadas / Perdidas / BE", f"{s['wins']} / {s['losses']} / {s['breakeven']}")
    c[1].metric("R medio ganadora", num(s["avg_win_r"], "R"))
    c[2].metric("R medio perdedora", num(s["avg_loss_r"], "R"))
    c[3].metric("Mejor racha", f"{s['best_win_streak']} W")
    c[4].metric("Peor racha", f"{s['worst_loss_streak']} L")
    c[5].metric("Plan cumplido", pct(s["plan_adherence"]))

    eq = analytics.equity_curve(df)
    if not eq.empty:
        fig = px.line(eq, x="trade_n", y="equity_r", markers=True, hover_data=["date", "pair", "result", "r_multiple"],
                      labels={"trade_n": "Operación nº", "equity_r": "R acumulado"}, title="Curva de capital (en R)")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Por mes")
        st.dataframe(analytics.monthly(df), use_container_width=True)
        st.subheader("Por sesión")
        st.dataframe(analytics.breakdown(df, "session"), use_container_width=True)
        st.subheader("Por setup")
        st.dataframe(analytics.breakdown(df, "setup"), use_container_width=True)
    with col2:
        st.subheader("Por día de la semana")
        st.dataframe(analytics.breakdown(df, "weekday"), use_container_width=True)
        st.subheader("Por dirección y par")
        st.dataframe(pd.concat([analytics.breakdown(df, "direction"), analytics.breakdown(df, "pair")]),
                     use_container_width=True)
        st.subheader("Por emoción")
        st.dataframe(analytics.breakdown(df, "emotion"), use_container_width=True)

    st.subheader("A favor vs en contra del sesgo diario")
    al = df.assign(sesgo=df["bias_aligned"].map({True: "A favor", False: "En contra"}).fillna("Neutral/sin dato"))
    st.dataframe(analytics.breakdown(al, "sesgo"), use_container_width=True)

    st.subheader("Todas las operaciones")
    show = df.copy()
    show["confluences"] = show["confluences"].apply(", ".join)
    show["mistakes"] = show["mistakes"].apply(", ".join)
    st.dataframe(show.drop(columns=["is_win", "is_loss"]).sort_values("date", ascending=False),
                 use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# 2. Registrar
# --------------------------------------------------------------------------- #
elif page == "➕ Registrar operaciones":
    st.title("➕ Registrar operaciones")
    tab_new, tab_import, tab_manage = st.tabs(["Nueva operación", "Importar / exportar CSV", "Editar / borrar"])

    known_conf = sorted(set(config.DEFAULT_CONFLUENCES) | {c for cs in df_all["confluences"] for c in cs}) if not df_all.empty else config.DEFAULT_CONFLUENCES
    known_mist = sorted(set(config.DEFAULT_MISTAKES) | {c for cs in df_all["mistakes"] for c in cs}) if not df_all.empty else config.DEFAULT_MISTAKES

    with tab_new:
        with st.form("new_trade", clear_on_submit=True):
            c = st.columns(4)
            d = c[0].date_input("Fecha", value=date.today())
            t = c[1].time_input("Hora de entrada")
            pair = c[2].selectbox("Par", config.PAIRS)
            direction = c[3].selectbox("Dirección", config.DIRECTIONS)
            c = st.columns(4)
            entry = c[0].number_input("Entrada", format="%.5f", value=0.0)
            sl = c[1].number_input("Stop loss", format="%.5f", value=0.0)
            tp = c[2].number_input("Take profit", format="%.5f", value=0.0)
            exit_p = c[3].number_input("Precio de salida", format="%.5f", value=0.0)
            c = st.columns(4)
            r_manual = c[0].number_input("R conseguido (opcional, si no pones precios)", value=0.0, step=0.1)
            pnl = c[1].number_input("Beneficio en € / $ (opcional)", value=0.0)
            risk = c[2].number_input("Riesgo %", value=1.0, step=0.25)
            session = c[3].selectbox("Sesión", config.SESSIONS)
            c = st.columns(3)
            setup = c[0].text_input("Setup / modelo de entrada", placeholder="Ej: Barrido + FVG")
            tf = c[1].selectbox("Temporalidad de entrada", ["M1", "M5", "M15", "M30", "H1", "H4"])
            daily_bias = c[2].selectbox("Sesgo diario que tenías", config.BIAS_OPTIONS)
            confs = st.multiselect("Confluencias presentes", known_conf)
            extra = st.text_input("Otras confluencias (separadas por comas)")
            c = st.columns(3)
            followed = c[0].radio("¿Seguiste el plan?", ["Sí", "No"], horizontal=True)
            emotion = c[1].selectbox("Emoción al entrar", config.EMOTIONS)
            news = c[2].radio("¿Noticia de alto impacto cerca?", ["No", "Sí"], horizontal=True)
            mistakes = st.multiselect("Errores cometidos", known_mist)
            notes = st.text_area("Notas (qué viste, por qué entraste, cómo gestionaste)")
            shot = st.text_input("Enlace a captura (TradingView, etc.)")
            if st.form_submit_button("Guardar operación", type="primary"):
                has_prices = entry and sl and exit_p
                all_conf = confs + [x.strip() for x in extra.split(",") if x.strip()]
                tid = journal.add_trade({
                    "date": d.isoformat(), "time": t.strftime("%H:%M"), "pair": pair, "direction": direction,
                    "entry": entry or None, "stop_loss": sl or None, "take_profit": tp or None,
                    "exit_price": exit_p or None, "r_multiple": None if has_prices else (r_manual or 0.0),
                    "pnl": pnl or None, "risk_pct": risk, "session": session, "setup": setup, "timeframe": tf,
                    "daily_bias": daily_bias, "confluences": all_conf, "followed_plan": followed == "Sí",
                    "emotion": emotion, "news_nearby": news == "Sí", "mistakes": mistakes, "notes": notes,
                    "screenshot_url": shot,
                })
                st.success(f"Operación #{tid} guardada.")

    with tab_import:
        st.markdown("""El CSV necesita al menos `date, pair, direction` y idealmente `entry, stop_loss, take_profit,
exit_price` (o `r_multiple`). Columnas opcionales: `time, session, setup, timeframe, confluences` (separadas por comas),
`daily_bias, followed_plan, emotion, news_nearby, mistakes, notes, pnl`. También acepta nombres en español
(`fecha, par, dirección, entrada, sl, tp, salida, confluencias, sesgo, notas`...).""")
        up = st.file_uploader("Subir CSV", type="csv")
        if up is not None and st.button("Importar"):
            try:
                n = journal.import_csv(up)
                st.success(f"{n} operaciones importadas.")
            except ValueError as e:
                st.error(str(e))
        if st.button("Cargar datos de ejemplo (150 operaciones ficticias)"):
            n = journal.import_csv(config.ROOT / "sample_data" / "sample_trades.csv")
            st.success(f"{n} operaciones de ejemplo cargadas.")
        if not df_all.empty:
            st.download_button("Descargar journal en CSV", journal.export_csv(), "journal.csv", "text/csv")

    with tab_manage:
        if df_all.empty:
            st.info("No hay operaciones.")
        else:
            tid = st.selectbox("Operación", df_all["id"].tolist()[::-1],
                               format_func=lambda i: (lambda r: f"#{i} · {r['date'].date()} · {r['pair']} {r['direction']} · "
                                                      f"{r['result']} {num(r['r_multiple'], 'R')}")(df_all.set_index('id').loc[i]))
            row = journal.get_trade(int(tid))
            c = st.columns(3)
            new_exit = c[0].number_input("Precio de salida", value=float(row["exit_price"] or 0.0), format="%.5f")
            new_r = c[1].number_input("R (si no hay precios)", value=float(row["r_multiple"] or 0.0), step=0.1)
            new_notes = c[2].text_input("Notas", value=row["notes"] or "")
            new_conf = st.text_input("Confluencias (separadas por comas)", value=row["confluences"])
            new_mist = st.text_input("Errores (separados por comas)", value=row["mistakes"])
            cc = st.columns(2)
            if cc[0].button("Guardar cambios"):
                changes = {"notes": new_notes, "confluences": new_conf, "mistakes": new_mist}
                if row["entry"] and row["stop_loss"] and new_exit:
                    changes["exit_price"] = new_exit
                else:
                    changes["r_multiple"] = new_r
                    changes["result"] = None
                journal.update_trade(int(tid), changes)
                st.success("Actualizada.")
                st.rerun()
            if cc[1].button("🗑️ Borrar operación"):
                journal.delete_trade(int(tid))
                st.success("Borrada.")
                st.rerun()

# --------------------------------------------------------------------------- #
# 3. Filtros
# --------------------------------------------------------------------------- #
elif page == "🔍 Filtros avanzados":
    st.title("🔍 Filtra tus operaciones")
    if df_all.empty:
        st.warning("No hay operaciones todavía.")
        st.stop()
    all_conf = sorted({c for cs in df_all["confluences"] for c in cs})
    all_mist = sorted({c for cs in df_all["mistakes"] for c in cs})

    def opts(col):
        return sorted(df_all[col].dropna().unique().tolist())

    with st.expander("Filtros", expanded=True):
        c = st.columns(4)
        f = TradeFilter(
            pairs=c[0].multiselect("Par", opts("pair")),
            directions=c[1].multiselect("Dirección", opts("direction")),
            sessions=c[2].multiselect("Sesión", opts("session")),
            setups=c[3].multiselect("Setup", opts("setup")),
        )
        c = st.columns(4)
        f.results = c[0].multiselect("Resultado", ["Win", "Loss", "BE"])
        f.emotions = c[1].multiselect("Emoción", opts("emotion"))
        f.weekdays = c[2].multiselect("Día", opts("weekday"))
        dr = c[3].date_input("Rango de fechas", value=(df_all["date"].min().date(), df_all["date"].max().date()))
        if isinstance(dr, tuple) and len(dr) == 2:
            f.date_from, f.date_to = dr[0].isoformat(), dr[1].isoformat()
        c = st.columns(3)
        f.confluences_all = c[0].multiselect("Con TODAS estas confluencias", all_conf)
        f.confluences_any = c[1].multiselect("Con ALGUNA de estas", all_conf)
        f.confluences_none = c[2].multiselect("SIN ninguna de estas", all_conf)
        c = st.columns(5)
        tri = {"Todos": None, "Sí": True, "No": False}
        f.followed_plan = tri[c[0].selectbox("Plan seguido", list(tri))]
        f.news_nearby = tri[c[1].selectbox("Noticia cerca", list(tri))]
        f.bias_aligned = tri[c[2].selectbox("A favor del sesgo", list(tri))]
        f.min_confluences = c[3].number_input("Mín. confluencias", 0, 20, 0) or None
        f.mistakes_any = c[4].multiselect("Con estos errores", all_mist)
        hr = st.slider("Hora de entrada", 0, 23, (0, 23))
        if hr != (0, 23):
            f.hour_from, f.hour_to = hr

    sub = analytics.apply_filter(df_all, f)
    s, base = analytics.summary(sub), analytics.summary(df_all)
    c = st.columns(5)
    c[0].metric("Operaciones", s["trades"], delta=s["trades"] - base["trades"])
    c[1].metric("Win Rate", pct(s["win_rate"]),
                delta=None if s["win_rate"] is None or base["win_rate"] is None else f"{s['win_rate'] - base['win_rate']:+.1f} pts")
    c[2].metric("R total", num(s["total_r"], "R"))
    c[3].metric("Expectativa", num(s["expectancy_r"], "R"),
                delta=None if s["expectancy_r"] is None or base["expectancy_r"] is None else f"{s['expectancy_r'] - base['expectancy_r']:+.2f}R")
    c[4].metric("Profit factor", "—" if s["profit_factor"] is None else f"{s['profit_factor']:.2f}")
    if len(sub) and len(sub) < 20:
        st.caption("⚠️ Muestra pequeña (<20 operaciones): las conclusiones son poco fiables.")

    group = st.selectbox("Agrupar el resultado filtrado por", ["confluences", "session", "setup", "hour", "weekday",
                                                                 "emotion", "mistakes", "timeframe", "n_confluences"])
    st.dataframe(analytics.breakdown(sub, group), use_container_width=True)
    show = sub.copy()
    show["confluences"] = show["confluences"].apply(", ".join)
    show["mistakes"] = show["mistakes"].apply(", ".join)
    st.dataframe(show.drop(columns=["is_win", "is_loss"]), use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# 4. Buenas vs malas
# --------------------------------------------------------------------------- #
elif page == "🧠 Buenas vs malas":
    st.title("🧠 ¿Por qué ganan tus buenas y fallan tus malas?")
    decided = df_all[df_all["result"].isin(["Win", "Loss"])]
    if len(decided) < 5:
        st.warning("Necesitas al menos 5 operaciones cerradas (ganadas o perdidas) para el análisis.")
        st.stop()
    min_n = st.sidebar.slider("Mínimo de operaciones por confluencia", 2, 15, 3)
    diag = diagnosis.Diagnoser(df_all, min_trades=min_n)
    rep = diag.pattern_report()

    st.subheader("Conclusiones")
    for line in rep["conclusiones"]:
        st.markdown(f"- {line}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("❌ Por qué fallan tus malas")
        st.caption("Factores que aparecen mucho más en perdedoras que en ganadoras")
        st.dataframe(rep["causas_perdidas"][["factor", "%_perdedoras", "%_ganadoras", "wr_cuando_ocurre", "trades"]],
                     hide_index=True, use_container_width=True)
        if rep["errores_frecuentes_en_perdidas"]:
            st.markdown("**Errores más repetidos en pérdidas:** " +
                        ", ".join(f"{m} ({n})" for m, n in rep["errores_frecuentes_en_perdidas"][:6]))
    with c2:
        st.subheader("✅ Qué tienen en común tus buenas")
        st.caption("Rasgos mucho más frecuentes en ganadoras")
        st.dataframe(rep["rasgos_ganadoras"][["factor", "%_ganadoras", "%_perdedoras", "wr_cuando_ocurre", "trades"]],
                     hide_index=True, use_container_width=True)
        if rep["checklist_ideal"]:
            st.success("**Checklist A+ (confluencias que más suben tu WR):**\n\n" +
                       "\n".join(f"- [ ] {c}" for c in rep["checklist_ideal"]))

    st.subheader("🧩 Confluencias que más ves en tus operaciones buenas")
    lift = analytics.confluence_lift(df_all, min_trades=min_n)
    if not lift.empty:
        fig = go.Figure()
        fig.add_bar(y=lift["confluencia"], x=lift["%_en_ganadoras"], name="% en ganadoras", orientation="h")
        fig.add_bar(y=lift["confluencia"], x=lift["%_en_perdedoras"], name="% en perdedoras", orientation="h")
        fig.update_layout(barmode="group", height=max(300, 28 * len(lift)), yaxis={"autorange": "reversed"})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(lift, hide_index=True, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Mejores combinaciones de 2 confluencias**")
        st.dataframe(analytics.confluence_combos(df_all, 2, min_n).head(15), hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**WR según número de confluencias**")
        st.dataframe(analytics.confluence_count_effect(df_all), use_container_width=True)

    st.subheader("🔮 ¿Y si hubieras evitado...?")
    st.dataframe(diag.what_if(), hide_index=True, use_container_width=True)

    st.subheader("🔎 Diagnóstico operación por operación")
    expl = diag.explain_all()
    only = st.radio("Mostrar", ["Perdedoras", "Ganadoras", "Todas"], horizontal=True)
    if only != "Todas":
        expl = expl[expl["resultado"] == ("Loss" if only == "Perdedoras" else "Win")]
    view = expl.assign(
        factores_en_contra=expl["factores_en_contra"].apply("; ".join),
        factores_a_favor=expl["factores_a_favor"].apply("; ".join),
        confluencias_clave_ausentes=expl["confluencias_clave_ausentes"].apply(", ".join),
    )
    st.dataframe(view.sort_values("fecha", ascending=False), hide_index=True, use_container_width=True)

    st.subheader("🤖 Coach IA")
    focus = st.text_input("¿Algo concreto en lo que quieras que se centre? (opcional)")
    if st.button("Analizar mi journal con IA", type="primary"):
        bd = {k: analytics.breakdown(df_all, k) for k in ("session", "setup", "pair", "direction", "emotion", "weekday", "hour")}
        out = coach_box(ai_coach.coach_journal, df_all, analytics.summary(df_all), rep, lift, bd, focus)
        if out:
            st.markdown(out)
    pick = st.selectbox("Revisar una operación concreta con IA", expl["id"].dropna().astype(int).tolist())
    if st.button("Revisar operación"):
        idx = diag.df.index[diag.df["id"] == pick][0]
        out = coach_box(ai_coach.review_trade, journal.get_trade(pick), diag.explain_trade(idx),
                        analytics.summary(df_all), rep["checklist_ideal"])
        if out:
            st.markdown(out)

# --------------------------------------------------------------------------- #
# 5. Sesgo diario
# --------------------------------------------------------------------------- #
elif page == "🌍 Sesgo diario y noticias":
    st.title("🌍 Sesgo diario EURUSD / DXY")
    st.caption("Las probabilidades son estimaciones heurísticas (limitadas a 20-80%), no garantías.")

    @st.cache_data(ttl=900, show_spinner="Descargando precios...")
    def get_prices(pair):
        return technical.fetch_ohlc(pair)

    @st.cache_data(ttl=900, show_spinner="Descargando calendario...")
    def get_calendar():
        return fundamentals.fetch_calendar()

    @st.cache_data(ttl=900, show_spinner="Leyendo titulares...")
    def get_headlines():
        return fundamentals.fetch_headlines()

    # --- Técnico ---
    tech: dict[str, technical.TechnicalBias | None] = {}
    prices: dict[str, pd.DataFrame] = {}
    cols = st.columns(2)
    for col, pair in zip(cols, config.PAIRS):
        with col:
            st.subheader(pair)
            try:
                prices[pair] = get_prices(pair)
            except Exception as e:  # noqa: BLE001
                st.warning(f"No se pudieron descargar precios de {pair} ({e}). Puedes subir un CSV diario (date,open,high,low,close).")
                up = st.file_uploader(f"CSV diario {pair}", type="csv", key=f"csv_{pair}")
                if up is not None:
                    prices[pair] = technical.load_ohlc_csv(up)
            if pair in prices:
                try:
                    tb = technical.technical_bias(prices[pair], pair)
                except ValueError as e:
                    st.error(str(e))
                    tb = None
                tech[pair] = tb
                if tb:
                    st.metric("Sesgo técnico", tb.label, f"{tb.score:+.2f}")
                    d = prices[pair].tail(90)
                    fig = go.Figure(go.Candlestick(x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"]))
                    for k in ("EMA20", "EMA50", "PDH", "PDL"):
                        fig.add_hline(y=tb.levels[k], line_dash="dot", annotation_text=k)
                    fig.update_layout(height=320, xaxis_rangeslider_visible=False, margin=dict(t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True, key=f"candles_{pair}")
                    st.dataframe(pd.DataFrame(tb.signals), hide_index=True, use_container_width=True)
                    st.json(tb.levels, expanded=False)
    if "EURUSD" in prices and "DXY" in prices:
        corr = technical.rolling_correlation(prices["EURUSD"]["close"], prices["DXY"]["close"])
        if corr is not None:
            st.caption(f"Correlación EURUSD-DXY últimos 30 días: {corr:+.2f}")

    # --- Fundamental ---
    st.header("📰 Noticias de alto impacto (USD / EUR)")
    impacts: list[fundamentals.EventImpact] = []
    try:
        events = fundamentals.relevant_events(get_calendar(), st.selectbox("Impacto mínimo", ["High", "Medium"]))
    except Exception as e:  # noqa: BLE001
        st.warning(f"No se pudo descargar el calendario ({e}). Puedes añadir eventos manualmente abajo.")
        events = []
    with st.expander("Añadir / corregir un dato publicado manualmente"):
        c = st.columns(5)
        m_title = c[0].text_input("Evento", placeholder="CPI m/m")
        m_cur = c[1].selectbox("Divisa", ["USD", "EUR"])
        m_fc = c[2].text_input("Previsión", placeholder="0.3%")
        m_prev = c[3].text_input("Anterior", placeholder="0.2%")
        m_act = c[4].text_input("Real", placeholder="0.5%")
        if m_title:
            events = [e for e in events if e.title != m_title] + [fundamentals.EconEvent(
                m_title, m_cur, pd.Timestamp.now(tz="UTC"), "High", m_fc or None, m_prev or None, m_act or None)]
    now = pd.Timestamp.now(tz="UTC")
    for e in sorted(events, key=lambda x: x.date):
        imp = fundamentals.analyze_event(e)
        impacts.append(imp)
        when = e.date.tz_convert("Europe/Madrid").strftime("%a %d %H:%M")
        icon = "🟢" if imp.surprise is not None else ("⏳" if e.date > now else "❔")
        with st.expander(f"{icon} {when} · {e.currency} · {e.title} — prev. {e.forecast or '—'} · ant. {e.previous or '—'} · real {e.actual or '—'}"):
            st.write(imp.explanation)
            for sc in imp.scenarios:
                st.markdown(f"- {sc}")
            if imp.surprise is not None:
                st.write(f"Efecto estimado → EURUSD {imp.eurusd_effect:+.2f} · DXY {imp.dxy_effect:+.2f}")
            if st.button("Explicar con IA", key=f"ai_{e.title}_{e.date}"):
                out = coach_box(ai_coach.explain_news, {"evento": e.title, "divisa": e.currency, "fecha": str(e.date),
                                                         "prevision": e.forecast, "anterior": e.previous, "real": e.actual})
                if out:
                    st.markdown(out)
    fund = fundamentals.fundamental_score(impacts)

    try:
        heads = get_headlines()
    except Exception:  # noqa: BLE001
        heads = []
    news = fundamentals.headline_sentiment(heads)
    with st.expander(f"Titulares relevantes ({len(news['titulares_relevantes'])} de {len(heads)})"):
        for h in news["titulares_relevantes"][:25]:
            st.markdown(f"- [{h['title']}]({h['link']})")

    # --- Combinado ---
    st.header("🎯 Sesgo del día y probabilidad")
    if "ai_probs" not in st.session_state:
        st.session_state.ai_probs = None
    result = bias.combine(tech, fund, news, ai_prob=st.session_state.ai_probs)
    cols = st.columns(2)
    for col, pair in zip(cols, config.PAIRS):
        b = result[pair]
        with col:
            st.subheader(f"{pair}: {b.label}")
            fig = go.Figure(go.Indicator(mode="gauge+number", value=round(100 * b.prob_up),
                                         number={"suffix": "% ↑"}, gauge={"axis": {"range": [0, 100]},
                                                                           "bar": {"color": "#2e7d32" if b.prob_up >= 0.5 else "#c62828"},
                                                                           "threshold": {"value": 50, "line": {"width": 2}}}))
            fig.update_layout(height=220, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True, key=f"gauge_{pair}")
            st.write(f"Probabilidad aprox.: **{b.prob_up:.0%} subida / {b.prob_down:.0%} bajada**")
            for r in b.reasons:
                st.markdown(f"- {r}")

    if st.button("🌐 Briefing del día con IA (busca noticias del mundo)", type="primary"):
        tech_summary = {p: {"score": t.score, "label": t.label, "levels": t.levels, "close": t.last_close,
                            "signals": [s["lectura"] for s in t.signals]} for p, t in tech.items() if t}
        ev = [{"titulo": i.event.title, "divisa": i.event.currency, "fecha": str(i.event.date), "prevision": i.event.forecast,
               "anterior": i.event.previous, "real": i.event.actual} for i in impacts]
        out = coach_box(ai_coach.market_brief, tech_summary, ev, [h["title"] for h in heads],
                        {p: {"prob_up": b.prob_up, "label": b.label} for p, b in result.items()},
                        spinner="Buscando noticias y analizando el mercado...")
        if out:
            st.session_state.ai_brief = out["markdown"]
            st.session_state.ai_probs = out["prob_up"] or None
            st.rerun()
    if st.session_state.get("ai_brief"):
        st.markdown(st.session_state.ai_brief)

    st.divider()
    today = date.today().isoformat()
    if st.button("💾 Guardar sesgo de hoy en el histórico"):
        for pair, b in result.items():
            journal.log_bias(today, pair, b.prob_up, b.technical, b.fundamental, b.as_text())
        st.success("Guardado. Cuando cierre el día podrás medir el acierto.")
    log = journal.load_bias_log()
    if not log.empty:
        # rellenar el movimiento real de días pasados si tenemos precios
        for _, row in log[log["actual_change"].isna() & (log["date"] < today)].iterrows():
            px_ = prices.get(row["pair"])
            if px_ is not None and pd.Timestamp(row["date"]) in px_.index:
                bar = px_.loc[pd.Timestamp(row["date"])]
                journal.set_bias_outcome(row["date"], row["pair"], float(bar["close"] - bar["open"]))
        log = journal.load_bias_log()
        ev = bias.evaluate_bias_log(log)
        st.subheader("📈 Fiabilidad del sesgo")
        st.write(f"Días evaluados: {ev['evaluados']} · Acierto direccional: {pct(ev['acierto'])} · Brier score: {ev['brier'] or '—'} (menor es mejor, 0.25 = azar)")
        st.dataframe(log, hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------- #
# 6. Chat
# --------------------------------------------------------------------------- #
elif page == "💬 Pregunta a la IA":
    st.title("💬 Pregunta lo que quieras sobre tu trading")
    st.caption("Ej: «¿Qué WR tengo en EURUSD en Londres cuando hay FVG y liquidez barrida?», "
               "«¿Pierdo más los lunes?», «¿Qué debería dejar de hacer?»")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for role, msg in st.session_state.chat:
        st.chat_message(role).markdown(msg)
    q = st.chat_input("Escribe tu pregunta")
    if q:
        st.session_state.chat.append(("user", q))
        st.chat_message("user").markdown(q)
        if df_all.empty:
            ans = "Todavía no tienes operaciones registradas."
        else:
            ans = coach_box(ai_coach.ask, q, df_all, analytics.summary(df_all)) or "_(coach IA no disponible)_"
        st.session_state.chat.append(("assistant", ans))
        st.chat_message("assistant").markdown(ans)
