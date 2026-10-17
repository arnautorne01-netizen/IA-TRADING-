"""Monta web/futbol.html a partir de la plantilla, el motor, los datos y los partidos publicados.

Para actualizar los partidos del día: edita fixtures.json (sube "version") y ejecuta  python web/futbol_src/build.py
"""
from pathlib import Path

src = Path(__file__).parent
page = (src / "page.html").read_text(encoding="utf-8")
for marker, name in (("/*ENGINE*/", "engine.js"), ("/*DATA*/", "futbol_data.json"), ("/*FIXTURES*/", "fixtures.json")):
    page = page.replace(marker, (src / name).read_text(encoding="utf-8"))
(src.parent / "futbol.html").write_text(page, encoding="utf-8")
print("web/futbol.html actualizado")
