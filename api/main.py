"""Servicio FastAPI. Desde la carpeta raíz: uvicorn api.main:app --reload."""
from __future__ import annotations

from datetime import date
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from modelo.predictor import cargar_modelo, cargar_metadata, opciones_categoricas, evaluar

app = FastAPI(
    title="Foco Vial | API de severidad vial",
    version="1.0.0",
    description=(
        "API educativa: estima severidad condicional a un siniestro ya ocurrido. "
        "No predice la ocurrencia de accidentes ni determina atención de emergencias."
    ),
)

# CORS: permite que la página web (publicada en GitHub Pages, otro dominio) consuma la API.
# Por defecto acepta cualquier origen: la API es pública, no usa cookies ni guarda datos.
# Para restringirla, definir CORS_ORIGINS="https://usuario.github.io" en Render.
_origenes = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origenes,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class EntradaSiniestro(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fecha: date = Field(description="Fecha del hecho AAAA-MM-DD")
    hora: int = Field(ge=0, le=23, description="Hora local, entre 0 y 23")
    comuna: Optional[int] = Field(default=None, ge=1, le=15)
    tipo_de_via: str = Field(min_length=1)
    modo_victima: str = Field(min_length=1)
    contraparte: str = Field(min_length=1)
    latitud: Optional[float] = Field(default=None, ge=-90, le=90)
    longitud: Optional[float] = Field(default=None, ge=-180, le=180)

    @field_validator("fecha")
    @classmethod
    def comprobar_fecha(cls, valor):
        if not 2019 <= valor.year <= date.today().year or valor > date.today():
            raise ValueError("La fecha debe estar entre 2019 y hoy")
        return valor

    @field_validator("tipo_de_via", "modo_victima", "contraparte")
    @classmethod
    def normalizar(cls, valor):
        return valor.strip().upper()


@app.get("/health")
def health():
    try:
        cargar_modelo()
        return {"estado": "ok", "modelo_cargado": True, "modelo": cargar_metadata().get("version_modelo", "tp3")}
    except Exception:
        return JSONResponse(status_code=503, content={"estado": "no_disponible", "modelo_cargado": False})


@app.get("/info")
def info():
    try:
        modelo = cargar_modelo()
        meta = cargar_metadata()
        opciones = opciones_categoricas()
    except Exception:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    return {
        "proyecto": meta.get("proyecto", "Siniestros viales CABA"),
        "modelo": meta.get("modelo", "Random Forest"),
        "version_modelo": meta.get("version_modelo", "tp3"),
        "predictoras": list(modelo.feature_names_in_),
        "opciones": {
            "tipo_de_via": opciones["tipo_de_via_siniestro"],
            "modo_victima": opciones["modo_desplazamiento_victima"],
            "contraparte": opciones["contraparte_siniestro"],
        },
        "umbral": float(meta.get("umbral_clasificacion", 0.5)),
        "metricas_test_2025": meta.get("metricas_test", {}),
        "matriz_confusion_test_2025": meta.get("matriz_confusion_test_2025", {}),
        "test_n": meta.get("n_test", 11750),
        "disclaimer": "Evaluación retrospectiva; puntajes no calibrados y no aptos para triage de emergencias.",
    }


@app.post("/predecir")
def predecir(entrada: EntradaSiniestro):
    try:
        valores = entrada.model_dump(mode="json")
        disponibles = opciones_categoricas()
        mapa = {
            "tipo_de_via": "tipo_de_via_siniestro",
            "modo_victima": "modo_desplazamiento_victima",
            "contraparte": "contraparte_siniestro",
        }
        errores = []
        for campo, columna in mapa.items():
            permitidos = disponibles[columna]
            if valores[campo] not in permitidos:
                errores.append({"campo": campo, "valor": valores[campo], "valores_validos": permitidos})
        if errores:
            raise HTTPException(
                status_code=422,
                detail={"mensaje": "Hay categorías desconocidas", "errores": errores},
            )
        return evaluar(valores)
    except HTTPException:
        raise
    except (FileNotFoundError, RuntimeError, ValueError):
        raise HTTPException(status_code=503, detail="Modelo no disponible o incompatibilidad con el TP3")
