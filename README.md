# Foco Vial

TP4 de Ciencia de Datos Aplicada (ITBA) · Matías Chiffone Guarde, María Candelaria Terán Ortiz y Olivia Vella Delgado.

Estima si un siniestro vial que ya ocurrió en CABA terminó leve o grave/mortal, con el Random Forest del TP3. Es una herramienta académica de apoyo: no predice accidentes ni reemplaza la atención de emergencias.

**Página:** https://chiffoneg.github.io/foco-vial/ · **API:** https://COMPLETAR.onrender.com/docs

## Cómo funciona

Página web (`docs/`, GitHub Pages) → API FastAPI (`api/`, Render) → modelo del TP3 (`modelo/`).

**Entrada** (`POST /predecir`): fecha, hora, comuna, tipo de vía, modo de la víctima, contraparte y coordenadas opcionales.
**Salida:** puntaje de gravedad, clasificación (leve o grave/mortal), factores asociados y advertencias. Los datos inválidos devuelven error 422 con el motivo.

## Correrlo en la computadora

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload          # API en http://127.0.0.1:8000/docs
cd docs && python -m http.server 8080  # página en http://localhost:8080 (otra terminal)
python -m pytest -q                    # 12 pruebas
```
