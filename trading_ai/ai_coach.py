"""Coach de trading con Claude.

- Analiza tu journal y te explica por qué fallan tus malas y por qué funcionan tus buenas.
- Revisa operaciones individuales.
- Resume el contexto mundial del día (con búsqueda web) y estima el sesgo de EURUSD/DXY.
- Responde preguntas libres sobre tus operaciones.

Requiere ANTHROPIC_API_KEY (o un perfil de `ant auth login`). Sin credenciales,
el resto de la aplicación sigue funcionando con el motor de reglas.
"""
from __future__ import annotations

import json
import os
import re
from datetime import date

import pandas as pd

from . import config

SYSTEM_COACH = """Eres un coach de trading profesional especializado en forex (EURUSD y el índice del dólar DXY).
Hablas en español, de forma directa, concreta y honesta. Tu trabajo es ayudar al trader a mejorar su proceso,
no a validar resultados: una ganadora con mala ejecución es un mal trade, y una perdedora bien ejecutada es un buen trade.
Basa tus conclusiones en los datos que recibes (estadísticas, factores, confluencias); cita números concretos
y señala cuando la muestra es demasiado pequeña para sacar conclusiones fiables.
Nunca prometas resultados ni des certezas: habla en probabilidades."""

SYSTEM_MARKET = """Eres un analista macro y técnico de forex que prepara el briefing diario de EURUSD y DXY para un trader.
Hablas en español. Usa la búsqueda web para encontrar lo que ha pasado HOY y en las últimas 24-48 horas: datos
macro publicados (dato real vs previsión), bancos centrales (Fed, BCE), discursos, geopolítica, aversión al riesgo,
rendimientos de bonos. Combínalo con el análisis técnico que te paso.
Para cada noticia de alto impacto explica en 2-3 frases qué es, qué salió y por qué lo lógico es que el par suba o baje.
Recuerda: EURUSD y DXY tienen correlación inversa fuerte (el euro pesa ~57.6% en el DXY).
Da probabilidades aproximadas y honestas (rara vez por encima del 70%): el mercado es incierto."""


class CoachUnavailable(RuntimeError):
    pass


def _has_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.environ.get("ANTHROPIC_PROFILE"))


def is_available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return _has_credentials()


def _client():
    try:
        import anthropic
    except ImportError as e:
        raise CoachUnavailable("Instala el paquete 'anthropic' (pip install anthropic)") from e
    return anthropic.Anthropic()


def _call(system: str, prompt: str, web_search: bool = False, effort: str = "high",
          max_continuations: int = 5) -> str:
    """Llama a Claude (streaming) y devuelve el texto final. Gestiona pause_turn y rechazos."""
    import anthropic

    client = _client()
    messages: list[dict] = [{"role": "user", "content": prompt}]
    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 8}] if web_search else []
    texts: list[str] = []
    try:
        for _ in range(max_continuations + 1):
            kwargs = dict(
                model=config.CLAUDE_MODEL,
                max_tokens=64000,
                system=system,
                messages=messages,
                thinking={"type": "adaptive"},
                output_config={"effort": effort},
                # si los clasificadores de seguridad rechazan la petición, el servidor reintenta en otro modelo
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            if tools:
                kwargs["tools"] = tools
            with client.beta.messages.stream(**kwargs) as stream:
                response = stream.get_final_message()
            if response.stop_reason == "refusal":
                raise CoachUnavailable("Claude rechazó la petición. Reformula la pregunta.")
            texts += [b.text for b in response.content if b.type == "text"]
            if response.stop_reason != "pause_turn":
                break
            # la búsqueda web se pausó: se reenvía el turno para que el servidor continúe
            messages = [messages[0], {"role": "assistant", "content": response.content}]
    except anthropic.AuthenticationError as e:
        raise CoachUnavailable("Clave de API de Anthropic no válida.") from e
    except anthropic.RateLimitError as e:
        raise CoachUnavailable("Límite de peticiones alcanzado, inténtalo en un minuto.") from e
    except anthropic.APIConnectionError as e:
        raise CoachUnavailable("No hay conexión con la API de Anthropic.") from e
    except anthropic.APIStatusError as e:
        raise CoachUnavailable(f"Error de la API de Anthropic ({e.status_code}): {e.message}") from e
    return "".join(texts).strip()


def _df_to_compact_csv(df: pd.DataFrame, max_rows: int = 400) -> str:
    cols = ["date", "time", "pair", "direction", "session", "setup", "timeframe", "r_multiple", "result",
            "planned_rr", "confluences", "daily_bias", "followed_plan", "emotion", "news_nearby", "mistakes", "notes"]
    d = df[[c for c in cols if c in df.columns]].tail(max_rows).copy()
    d["date"] = d["date"].dt.strftime("%Y-%m-%d")
    for c in ("confluences", "mistakes"):
        d[c] = d[c].apply(lambda v: "|".join(v))
    return d.to_csv(index=False)


def _table(df: pd.DataFrame) -> str:
    return "(sin datos)" if df is None or len(df) == 0 else df.to_csv()


def extract_json(text: str) -> dict | None:
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    raw = m.group(1) if m else None
    if raw is None:
        m = re.search(r"(\{[^{}]*\"EURUSD\".*\})", text, re.S)
        raw = m.group(1) if m else None
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def strip_json_block(text: str) -> str:
    return re.sub(r"```json\s*\{.*?\}\s*```", "", text, flags=re.S).strip()


# --------------------------------------------------------------------------- #
# Funciones del coach
# --------------------------------------------------------------------------- #
def coach_journal(df: pd.DataFrame, summary: dict, patterns: dict, lift: pd.DataFrame,
                  breakdowns: dict[str, pd.DataFrame], focus: str = "") -> str:
    prompt = f"""Analiza mi journal de trading completo.

## Estadísticas globales
{json.dumps(summary, ensure_ascii=False, default=str)}

## Factores: % en ganadoras vs perdedoras (calculado por el motor de reglas)
{_table(patterns.get('tabla_factores'))}

## Conclusiones preliminares del motor de reglas
{chr(10).join('- ' + c for c in patterns.get('conclusiones', []))}

## Confluencias (lift = WR con la confluencia - WR sin ella)
{_table(lift)}

## Desgloses
{chr(10).join(f'### Por {k}{chr(10)}{_table(v)}' for k, v in breakdowns.items())}

## Operaciones (CSV)
{_df_to_compact_csv(df)}

{('## Enfoque pedido por el trader' + chr(10) + focus) if focus else ''}

Quiero un informe con estas secciones:
1. **Diagnóstico general** (3-4 frases con los números clave).
2. **Por qué fallan mis operaciones malas**: causas ordenadas por impacto en R, con ejemplos (fecha/par).
3. **Por qué funcionan mis operaciones buenas**: qué tienen en común (confluencias, sesión, sesgo, estado emocional).
4. **Las confluencias que más veo en mis buenas operaciones** y cuáles no aportan nada.
5. **Patrones ocultos** que detectes en los datos y que las reglas no recogen (notas, horas, días, combinaciones).
6. **Mi checklist de operación A+** (5-8 puntos concretos y medibles).
7. **3 reglas nuevas** para mi plan de trading, con el impacto estimado si las hubiera aplicado.
"""
    return _call(SYSTEM_COACH, prompt, effort="high")


def review_trade(trade: dict, explanation: dict, summary: dict, checklist: list[str]) -> str:
    prompt = f"""Revisa esta operación concreta.

Operación: {json.dumps(trade, ensure_ascii=False, default=str)}
Diagnóstico del motor de reglas: {json.dumps(explanation, ensure_ascii=False, default=str)}
Mis estadísticas globales: {json.dumps(summary, ensure_ascii=False, default=str)}
Mi checklist de confluencias ganadoras: {checklist}

Dime: (1) por qué salió así, (2) si la ejecución fue buena independientemente del resultado,
(3) qué haría un trader profesional distinto, (4) la lección en una frase."""
    return _call(SYSTEM_COACH, prompt, effort="medium")


def market_brief(tech_summary: dict, events: list[dict], headlines: list[str], rule_bias: dict,
                 today: str | None = None) -> dict:
    """Briefing diario con búsqueda web. Devuelve {'markdown': str, 'prob_up': {'EURUSD': p, 'DXY': p}, 'raw': dict|None}."""
    today = today or date.today().isoformat()
    prompt = f"""Fecha de hoy: {today}.

## Análisis técnico calculado (puntuación -1 bajista .. +1 alcista)
{json.dumps(tech_summary, ensure_ascii=False, default=str)}

## Calendario económico de alto impacto (USD/EUR) de esta semana
{json.dumps(events, ensure_ascii=False, default=str)}

## Titulares recientes recogidos por RSS
{chr(10).join('- ' + h for h in headlines[:30]) or '(no disponibles)'}

## Sesgo calculado por el modelo de reglas
{json.dumps(rule_bias, ensure_ascii=False, default=str)}

Busca en la web lo que ha pasado hoy y ayer en el mundo que afecte al euro y al dólar, y redacta el briefing:
1. **Resumen del día** (qué mueve el mercado hoy, en 4-5 frases).
2. **Noticias de alto impacto**: para cada una, resumen breve + dato real vs previsión si ya salió +
   "por este motivo lo lógico es que EURUSD suba/baje y el DXY suba/baje". Si aún no ha salido, los escenarios.
3. **Sesgo técnico** de EURUSD y DXY en 2-3 frases con niveles clave.
4. **Sesgo final y probabilidad**: % aproximado de que EURUSD y DXY cierren hoy al alza vs a la baja, y qué
   invalidaría el escenario.
5. **Riesgos/horas a evitar** hoy (noticias, discursos).

Termina SIEMPRE con un bloque JSON exactamente con este formato:
```json
{{"EURUSD": {{"prob_up": 0.55, "sesgo": "alcista"}}, "DXY": {{"prob_up": 0.45, "sesgo": "bajista"}}}}
```"""
    text = _call(SYSTEM_MARKET, prompt, web_search=True, effort="high")
    data = extract_json(text)
    probs = {}
    if data:
        for pair in ("EURUSD", "DXY"):
            try:
                probs[pair] = float(data[pair]["prob_up"])
            except (KeyError, TypeError, ValueError):
                pass
    return {"markdown": strip_json_block(text), "prob_up": probs, "raw": data}


def explain_news(event: dict) -> str:
    """Explicación rápida de una noticia de alto impacto concreta (con búsqueda web)."""
    prompt = f"""Noticia de alto impacto: {json.dumps(event, ensure_ascii=False, default=str)}
Busca el resultado y la reacción del mercado si ya se publicó. Explícame en un párrafo corto qué es,
qué ha salido frente a lo esperado y por qué lo lógico es que EURUSD y DXY suban o bajen por ese motivo."""
    return _call(SYSTEM_MARKET, prompt, web_search=True, effort="medium")


def ask(question: str, df: pd.DataFrame, summary: dict) -> str:
    prompt = f"""Mis estadísticas: {json.dumps(summary, ensure_ascii=False, default=str)}

Mis operaciones (CSV):
{_df_to_compact_csv(df)}

Pregunta: {question}"""
    return _call(SYSTEM_COACH, prompt, effort="medium")
