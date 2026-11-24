"""Pruebas reales sobre la API y el modelo, sin reentrenar."""
from datetime import date
import json
from pathlib import Path

from fastapi.testclient import TestClient
from modelo.predictor import cargar_modelo, preparar_features
from api.main import app

cliente = TestClient(app)
CASO = {
    "fecha": "2025-03-15", "hora": 3, "comuna": 14,
    "tipo_de_via": "AUTOPISTA", "modo_victima": "PEATON", "contraparte": "AUTO",
    "latitud": -34.60, "longitud": -58.43,
}
REFERENCIAS = Path(__file__).parent / "casos_referencia.json"


def test_health_carga_modelo():
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json()["estado"] == "ok"


def test_info_incluye_categorias_y_metricas():
    r = cliente.get("/info")
    assert r.status_code == 200
    d = r.json()
    assert len(d["predictoras"]) == 16
    assert "AUTOPISTA" in d["opciones"]["tipo_de_via"]
    assert "PEATON" in d["opciones"]["modo_victima"]
    assert "PR-AUC" in d["metricas_test_2025"]


def test_prediccion_valida_sin_reentrenamiento():
    r = cliente.post("/predecir", json=CASO)
    assert r.status_code == 200, r.text
    d = r.json()
    assert 0 <= d["puntaje_gravedad"] <= 1
    assert d["clasificacion"] in ("LEVE", "GRAVE_O_MORTAL")
    assert (d["puntaje_gravedad"] >= d["umbral"]) == (d["clasificacion"] == "GRAVE_O_MORTAL")
    assert "no" in d["nota_puntaje"].lower()


def test_la_api_reproduce_el_pipeline_guardado_en_cinco_casos():
    # Casos sintéticos de referencia calculados mediante el .joblib, NO son registros etiquetados de 2025.
    casos = json.loads(REFERENCIAS.read_text(encoding="utf-8"))
    assert len(casos) == 5
    modelo = cargar_modelo()
    for caso in casos:
        entrada = caso["entrada"]
        r = cliente.post("/predecir", json=entrada)
        assert r.status_code == 200, (entrada, r.text)
        observado = r.json()["puntaje_gravedad"]
        esperado = float(caso["puntaje_esperado"])
        directo = float(modelo.predict_proba(preparar_features(entrada))[0, 1])
        assert abs(directo - esperado) < 1e-9
        assert abs(observado - esperado) < 1e-6


def test_reproduccion_de_variables_temporales_y_comuna():
    x = preparar_features(CASO).iloc[0]
    assert x["hora"] == 3
    assert x["dia_semana"] == 5  # 15 de marzo de 2025 fue sábado
    assert x["fin_de_semana"] == 1
    assert x["mes"] == 3
    assert x["estacion"] == "OTONO"
    assert x["franja_horaria"] == "MADRUGADA"
    assert x["comuna"] == "14.0"  # Importante: no convertir a cadena '14'
    assert x["usuario_vulnerable"] == 1


def test_comuna_incorrecta():
    r = cliente.post("/predecir", json={**CASO, "comuna": 20})
    assert r.status_code == 422


def test_hora_incorrecta():
    r = cliente.post("/predecir", json={**CASO, "hora": 24})
    assert r.status_code == 422


def test_categoria_no_conocida():
    r = cliente.post("/predecir", json={**CASO, "contraparte": "TRINEO"})
    assert r.status_code == 422
    assert "contraparte" in str(r.json())


def test_sin_dato_muestra_advertencia():
    r = cliente.post("/predecir", json={**CASO, "modo_victima": "SIN_DATO"})
    assert r.status_code == 200
    assert any("SIN_DATO" in s for s in r.json()["advertencias"])


def test_coordenadas_fuera_de_caba_no_interrumpen_la_evaluacion():
    r = cliente.post("/predecir", json={**CASO, "latitud": -40.0, "longitud": -60.0})
    assert r.status_code == 200
    assert any("Coordenadas fuera" in s for s in r.json()["advertencias"])


def test_campos_no_permitidos():
    r = cliente.post("/predecir", json={**CASO, "gravedad_siniestro": "GRAVE"})
    assert r.status_code == 422  # ¡Nunca pedir el target como predictor!


def test_cors_permite_la_pagina_web():
    # La página publicada en GitHub Pages está en otro dominio: el navegador exige este permiso.
    r = cliente.options("/predecir", headers={
        "Origin": "https://ejemplo.github.io",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") in ("*", "https://ejemplo.github.io")


def test_info_publica_vias_por_comuna():
    vias = cliente.get("/info").json()["vias_por_comuna"]
    assert "AV. GRAL. PAZ" in vias["12"]          # la General Paz limita con la comuna 12
    assert "AV. GRAL. PAZ" not in vias["1"]       # y no pasa por la comuna 1
    assert "AUTOPISTA" not in vias["11"]


def test_combinacion_inexistente_muestra_advertencia():
    r = cliente.post("/predecir", json={**CASO, "comuna": 1, "tipo_de_via": "AV. GRAL. PAZ"})
    assert r.status_code == 200
    assert any("no tiene siniestros registrados" in s for s in r.json()["advertencias"])


def test_combinacion_valida_no_advierte():
    r = cliente.post("/predecir", json={**CASO, "comuna": 12, "tipo_de_via": "AV. GRAL. PAZ"})
    assert not any("no tiene siniestros registrados" in s for s in r.json()["advertencias"])


def test_factores_con_porcentajes_y_sin_jerga():
    d = cliente.post("/predecir", json=CASO).json()   # peatón, autopista, 3 h
    texto = " ".join(d["asociaciones_generales"])
    assert "Víctima peatón" in texto and "Autopista" in texto and "Madrugada" in texto
    assert "%" in texto and "EDA" not in texto and "TP3" not in texto


def test_percentil_del_puntaje():
    alto = cliente.post("/predecir", json=CASO).json()
    bajo = cliente.post("/predecir", json={**CASO, "modo_victima": "SIN_DATO", "contraparte": "SIN_DATO",
                                              "tipo_de_via": "CALLE", "hora": 13}).json()
    assert 0 <= bajo["percentil_2025"] < alto["percentil_2025"] <= 100
