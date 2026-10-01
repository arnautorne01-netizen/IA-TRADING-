# 📈 IA de Trading — EURUSD & DXY

Asistente personal de trading que:

1. **Lleva tu journal** y calcula tu **win rate**, R total, expectativa, profit factor, drawdown, rachas y % de cumplimiento del plan.
2. **Analiza todas tus operaciones buenas y malas** y te dice:
   - **por qué fallan las malas** (contra sesgo, sin plan, emociones, noticias, pocas confluencias, R:R bajo, sesión débil, revenge trading, sobreoperar...),
   - **por qué funcionan las buenas** y qué tienen en común,
   - **qué confluencias ves más en tus operaciones buenas**, el WR con y sin cada una (*lift*) y las mejores combinaciones,
   - tu **checklist A+** generado con tus propios datos,
   - simulaciones del tipo *«¿qué WR tendría si no hubiera operado contra sesgo?»*.
3. **Calcula el sesgo diario** de EURUSD y DXY con análisis técnico (EMAs 20/50/200, estructura HH/HL/BOS, MACD, RSI, PDH/PDL, semanal) y fundamental (calendario de alto impacto, sorpresa del dato frente a la previsión, titulares).
4. **Explica las noticias de alto impacto**: qué son, qué ha salido y **por qué lo lógico es que suba o baje** cada par, con escenarios antes de que se publiquen.
5. **Da un % aproximado de que suba o baje** cada día, combinando técnico, fundamental y (opcionalmente) un briefing de la IA que busca en internet lo que pasa en el mundo.
6. **Filtra tus operaciones** por par, sesión, setup, confluencias (todas / alguna / ninguna), emoción, errores, hora, día, sesgo, noticias...
7. **Coach con IA (Claude)** que redacta informes de tu journal, revisa operaciones concretas y responde preguntas libres sobre tu trading.

> ⚠️ Las probabilidades son estimaciones heurísticas (limitadas al 20-80% a propósito), no garantías ni consejo financiero. La app guarda un histórico del sesgo diario y mide su acierto real para que sepas cuánto fiarte.

## Instalación

```bash
pip install -r requirements.txt
cp .env.example .env            # opcional: pon tu ANTHROPIC_API_KEY para activar el coach IA
export ANTHROPIC_API_KEY=sk-ant-...   # (o usa `ant auth login`)
streamlit run app.py
```

Sin clave de API todo funciona igual (estadística + motor de reglas); solo se desactivan los botones de IA.

## Uso rápido

- **Registrar operaciones** → formulario, o importa un CSV. Puedes cargar `sample_data/sample_trades.csv` (150 operaciones ficticias) para probar.
- **Journal y Win Rate** → métricas, curva de capital en R y desgloses por mes, sesión, setup, día, dirección, emoción y sesgo.
- **Filtros avanzados** → combina filtros y compara contra tu media.
- **Buenas vs malas** → diagnóstico completo, confluencias, simulaciones y análisis con IA.
- **Sesgo diario y noticias** → sesgo técnico, calendario, titulares, % de subida/bajada y briefing con IA.
- **Pregunta a la IA** → chat sobre tus operaciones.

### Formato del CSV

Obligatorias: `date, pair, direction`. Recomendadas: `entry, stop_loss, take_profit, exit_price` (o directamente `r_multiple`).
Opcionales: `time, session, setup, timeframe, confluences` (separadas por comas), `daily_bias` (Alcista/Bajista/Neutral),
`followed_plan` (sí/no), `emotion, news_nearby, mistakes, notes, pnl`. También acepta cabeceras en español
(`fecha, par, dirección, entrada, sl, tp, salida, r, confluencias, sesgo, errores, notas`).

**Consejo:** cuanto más constante seas etiquetando confluencias, emociones y errores, mejor te dirá la IA qué te funciona.

### Línea de comandos

```bash
python -m trading_ai.cli import mis_operaciones.csv
python -m trading_ai.cli stats
python -m trading_ai.cli diagnose [--ai]
python -m trading_ai.cli bias [--ai]      # sesgo del día de EURUSD y DXY
```

## Fuentes de datos

| Dato | Fuente | Si falla |
|---|---|---|
| Precios diarios | Yahoo Finance (`EURUSD=X`, `DX-Y.NYB`) vía `yfinance` | Subir CSV diario (p. ej. exportado de TradingView) |
| Calendario económico | Feed JSON público de Forex Factory | Añadir el dato a mano (previsión / real) |
| Titulares | RSS (FXStreet, ForexLive, Reuters; editable en `trading_ai/config.py`) | Se ignora |
| Contexto mundial | Claude con búsqueda web | Solo modelo de reglas |

## Cómo se calcula

- **Resultado de cada trade**: R = (salida − entrada) / (entrada − SL), con el signo según la dirección. Un resultado entre −0.1R y +0.1R cuenta como BE.
- **Win rate**: ganadas / (ganadas + perdidas), sin contar los BE.
- **Lift de una confluencia**: el WR con esa confluencia menos el WR sin ella.
- **Pérdida evitable**: una pérdida con 2 o más señales de alerta de proceso.
- **Sesgo**: 55% técnico (cada par se ajusta con el inverso del otro, por la correlación EURUSD/DXY), 30% sorpresas de datos publicados y 15% titulares. Se convierte en probabilidad con una logística acotada al 20-80% y, si pides el briefing de IA, se mezcla con la probabilidad que estima Claude.

## Estructura

```
app.py                     Dashboard Streamlit
trading_ai/journal.py      Base de datos SQLite, CSV, cálculo de R
trading_ai/analytics.py    Métricas, desgloses, filtros, confluencias
trading_ai/diagnosis.py    Por qué ganan/pierden tus operaciones
trading_ai/technical.py    Precios e indicadores, sesgo técnico
trading_ai/fundamentals.py Calendario, sorpresas, titulares
trading_ai/bias.py         Sesgo combinado y probabilidades
trading_ai/ai_coach.py     Coach con Claude
trading_ai/cli.py          Línea de comandos
tests/                     Tests (pytest)
```
