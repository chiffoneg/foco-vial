"""Inferencia del clasificador del TP3, separada de la API y de la interfaz.

El puntaje estima severidad CONDICIONADA a un siniestro registrado.
NO estima la probabilidad de que ocurra un siniestro.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "pipeline_siniestros_grave_mortal.joblib"
META_FILE = BASE_DIR / "metadata.json"


@lru_cache(maxsize=1)
def cargar_modelo():
    if not MODEL_FILE.is_file():
        raise FileNotFoundError("No se encontró el archivo .joblib del TP3")
    modelo = joblib.load(MODEL_FILE)
    if not hasattr(modelo, "predict_proba") or not hasattr(modelo, "feature_names_in_"):
        raise ValueError("El archivo del modelo no contiene un pipeline válido")
    if list(modelo.classes_) != [0, 1]:
        raise ValueError("Las clases del modelo no coinciden con el TP3")
    return modelo


@lru_cache(maxsize=1)
def cargar_metadata() -> dict:
    return json.loads(META_FILE.read_text(encoding="utf-8")) if META_FILE.exists() else {}


def opciones_categoricas() -> dict[str, list[str]]:
    """Opciones aprendidas por el OneHotEncoder del modelo efectivamente cargado."""
    prep = cargar_modelo().named_steps["prep"]
    encoder = prep.named_transformers_["cat"].named_steps["onehot"]
    nombres = list(prep.transformers_[1][2])
    return {col: [str(v) for v in valores] for col, valores in zip(nombres, encoder.categories_)}


def preparar_features(entrada: dict) -> pd.DataFrame:
    """Reproduce las 16 columnas predictoras del TP3.

    El TP3 representa comuna como cadena de float (ej. '14.0').
    Los valores SIN_DATO se mantienen porque están en el modelo heredado;
    la API deja constancia del sesgo administrativo en las advertencias.
    """
    fecha = date.fromisoformat(str(entrada["fecha"]))
    hora = int(entrada["hora"])
    mes = fecha.month
    dia_semana = fecha.weekday()  # lunes = 0
    franja = (
        "MADRUGADA" if hora <= 5 else
        "MANANA" if hora <= 11 else
        "TARDE" if hora <= 17 else "NOCHE"
    )
    estacion = {
        12: "VERANO", 1: "VERANO", 2: "VERANO",
        3: "OTONO", 4: "OTONO", 5: "OTONO",
        6: "INVIERNO", 7: "INVIERNO", 8: "INVIERNO",
        9: "PRIMAVERA", 10: "PRIMAVERA", 11: "PRIMAVERA",
    }[mes]
    comuna = entrada.get("comuna")
    comuna_str = f"{float(comuna):.1f}" if comuna is not None else "SIN_DATO"
    modo = entrada["modo_victima"]
    contraparte = entrada["contraparte"]
    tipo = entrada["tipo_de_via"]
    lat, lon = entrada.get("latitud"), entrada.get("longitud")
    lat = np.nan if lat is None else float(lat)
    lon = np.nan if lon is None else float(lon)

    registro = {
        "hora": hora,
        "dia_semana": dia_semana,
        "fin_de_semana": int(dia_semana >= 5),
        "hora_sin": math.sin(2 * math.pi * hora / 24),
        "hora_cos": math.cos(2 * math.pi * hora / 24),
        "mes": mes,
        "estacion": estacion,
        "franja_horaria": franja,
        "comuna": comuna_str,
        "tipo_de_via_siniestro": tipo,
        "latitud_siniestro": lat,
        "longitud_siniestro": lon,
        "modo_desplazamiento_victima": modo,
        "contraparte_siniestro": contraparte,
        "usuario_vulnerable": int(modo in ["PEATON", "MOTO", "BICICLETA", "MONOPATIN"]),
        "contraparte_pesada": int(contraparte in ["CAMION", "TRANSPORTE PUBLICO", "TREN"]),
    }
    esperadas = list(cargar_modelo().feature_names_in_)
    if set(registro) != set(esperadas):
        raise RuntimeError("Las variables derivadas no coinciden con las del TP3")
    return pd.DataFrame([registro], columns=esperadas)


def evaluar(entrada: dict) -> dict:
    """Ejecuta el pipeline ya entrenado, sin reentrenar ni modificar el umbral."""
    entrada = entrada.copy()
    advertencias = []
    if entrada.get("comuna") is None:
        advertencias.append("Falta la comuna. En el TP3 la completitud del registro dependía de la severidad.")
    if any(entrada.get(c) == "SIN_DATO" for c in ["modo_victima", "contraparte", "tipo_de_via"]):
        advertencias.append("Hay campos SIN_DATO: pueden reflejar cómo se registró el hecho, no el riesgo vial.")
    lat, lon = entrada.get("latitud"), entrada.get("longitud")
    if (lat is None) != (lon is None):
        entrada["latitud"] = entrada["longitud"] = None
        advertencias.append("Se recibió una sola coordenada: se omitió la ubicación y se usó la imputación del pipeline.")
    elif lat is not None and lon is not None and not (-34.71 <= lat <= -34.52 and -58.54 <= lon <= -58.33):
        entrada["latitud"] = entrada["longitud"] = None
        advertencias.append("Coordenadas fuera del rango aproximado de CABA: se omiten y el pipeline imputa la ubicación.")

    X = preparar_features(entrada)
    modelo = cargar_modelo()
    scores = modelo.predict_proba(X)[0]
    idx = list(modelo.classes_).index(1)
    score = float(scores[idx])
    meta = cargar_metadata()
    umbral = float(meta.get("umbral_clasificacion", 0.5))
    clase = "GRAVE_O_MORTAL" if score >= umbral else "LEVE"

    # Asociaciones GENERALES respaldadas por los análisis TP2/TP3; no son explicaciones locales del Random Forest.
    asociaciones = []
    if entrada["modo_victima"] == "PEATON":
        asociaciones.append("Peatones: mayor severidad observada en el EDA histórico.")
    if entrada["modo_victima"] == "MOTO":
        asociaciones.append("Motocicletas: asociación positiva con la severidad en el TP3.")
    if entrada["tipo_de_via"] == "AUTOPISTA":
        asociaciones.append("Autopistas: mayor proporción histórica de graves o mortales que en calles.")
    if entrada["contraparte"] == "OBJETO FIJO":
        asociaciones.append("Objeto fijo: asociación positiva en la Regresión Logística del TP3.")
    return {
        "puntaje_gravedad": round(score, 6),
        "umbral": umbral,
        "clasificacion": clase,
        "etiqueta": "Alerta de gravedad" if clase == "GRAVE_O_MORTAL" else "Clasificado como leve",
        "asociaciones_generales": asociaciones,
        "advertencias": advertencias,
        "modelo": meta.get("modelo", "Random Forest"),
        "version_modelo": meta.get("version_modelo", "tp3"),
        "alcance": "Clasificación retrospectiva condicional a un siniestro ocurrido; no predice accidentes ni establece causas.",
        "nota_puntaje": "Puntaje del clasificador sin calibrar: no equivale a una probabilidad real de lesión grave.",
    }
