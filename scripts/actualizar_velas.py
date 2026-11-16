"""Actualiza las velas de la bitácora con Twelve Data (EURUSD y GBPUSD).

Uso: python3 actualizar_velas.py <carpeta_entrada> <carpeta_salida>

- <carpeta_entrada> tiene los documentos actuales de la colección "market" de la bitácora
  (EURUSD_m5.json, EURUSD_m15.json, EURUSD_m30.json, EURUSD_h1.json y lo mismo con GBPUSD),
  tal como los guarda ArtifactData con out_dir: {"t":[...],"o":[...],"h":[...],"l":[...],"c":[...],...}.
- Descarga de Twelve Data las velas de 5 minutos que faltan desde la última guardada (1 petición por par;
  la clave la añade el proxy del entorno, no hace falta pasarla) y con ellas rehace las velas de 15 y 30
  minutos y de 1 hora. Escribe los documentos actualizados en <carpeta_salida>.
- Si el mercado de forex está cerrado (de viernes 17:00 a domingo 17:00 de Nueva York) no hace nada y
  sale con código 3.
"""
import datetime as dt
import json
import os
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

PARES = {"EURUSD": "EUR/USD", "GBPUSD": "GBP/USD"}
MAXIMO = {"m5": 5000, "m15": 5000, "m30": 5000, "h1": 5000}
PASO = {"m15": 900, "m30": 1800, "h1": 3600}
NY = ZoneInfo("America/New_York")


def mercado_abierto(ahora):
    ny = ahora.astimezone(NY)
    d, h = ny.weekday(), ny.hour  # lunes = 0
    if d == 5:
        return False
    if d == 4 and h >= 17:
        return False
    if d == 6 and h < 17:
        return False
    return True


def cargar(carpeta, nombre):
    ruta = os.path.join(carpeta, nombre + ".json")
    if not os.path.exists(ruta):
        return {}
    d = json.load(open(ruta))
    return {t: (o, h, l, c) for t, o, h, l, c in zip(d["t"], d["o"], d["h"], d["l"], d["c"])}


def guardar(carpeta, nombre, velas, maximo):
    ts = sorted(velas)[-maximo:]
    doc = {"t": ts, "o": [velas[t][0] for t in ts], "h": [velas[t][1] for t in ts],
           "l": [velas[t][2] for t in ts], "c": [velas[t][3] for t in ts], "source": "Twelve Data",
           "updated": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")}
    json.dump(doc, open(os.path.join(carpeta, nombre + ".json"), "w"))
    return ts[-1] if ts else None


def descargar_5m(simbolo, desde_ts):
    faltan = int((time.time() - desde_ts) // 300) + 3 if desde_ts else 5000
    n = max(12, min(5000, faltan))
    url = (f"https://api.twelvedata.com/time_series?symbol={simbolo}&interval=5min"
           f"&outputsize={n}&timezone=UTC")
    salida = subprocess.run(["curl", "-sS", "-m", "60", url], capture_output=True, text=True).stdout
    d = json.loads(salida)
    if d.get("status") != "ok":
        raise RuntimeError(f"{simbolo}: {d.get('message', d)}")
    velas = {}
    for x in d["values"]:
        t = int(dt.datetime.strptime(x["datetime"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp())
        velas[t] = tuple(round(float(x[k]), 5) for k in ("open", "high", "low", "close"))
    return velas


def agrupar(velas5, paso):
    out = {}
    for t in sorted(velas5):
        o, h, l, c = velas5[t]
        k = t - t % paso
        if k not in out:
            out[k] = [o, h, l, c]
        else:
            v = out[k]
            v[1], v[2], v[3] = max(v[1], h), min(v[2], l), c
    return {k: tuple(v) for k, v in out.items()}


def main():
    entrada, salida = sys.argv[1], sys.argv[2]
    os.makedirs(salida, exist_ok=True)
    ahora = dt.datetime.now(dt.timezone.utc)
    if not mercado_abierto(ahora):
        print("Mercado de forex cerrado: no hay velas nuevas.")
        sys.exit(3)
    resumen = []
    for par, simbolo in PARES.items():
        m5 = cargar(entrada, f"{par}_m5")
        nuevas = descargar_5m(simbolo, max(m5) if m5 else None)
        # solo velas de días de mercado (fuera las cotizaciones de fin de semana)
        nuevas = {t: v for t, v in nuevas.items() if mercado_abierto(dt.datetime.fromtimestamp(t, dt.timezone.utc))}
        if not nuevas:
            resumen.append(f"{par}: sin velas nuevas")
            continue
        m5.update(nuevas)
        ultima = guardar(salida, f"{par}_m5", m5, MAXIMO["m5"])
        desde = min(nuevas)
        for tf, paso in PASO.items():
            base = cargar(entrada, f"{par}_{tf}")
            inicio = desde - desde % paso
            tramo = {t: v for t, v in m5.items() if t >= inicio}
            base.update(agrupar(tramo, paso))
            guardar(salida, f"{par}_{tf}", base, MAXIMO[tf])
        hora = dt.datetime.fromtimestamp(ultima, dt.timezone.utc).astimezone(ZoneInfo("Europe/Madrid")).strftime("%a %d %H:%M")
        resumen.append(f"{par}: {len(nuevas)} velas de 5 min, hasta {hora} (Madrid)")
    print(" · ".join(resumen))


if __name__ == "__main__":
    main()
