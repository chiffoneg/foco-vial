"""Foco Vial: interfaz Streamlit que consume únicamente la API FastAPI.

Aplicación experimental académica. No realiza triage ni predice ocurrencia de choques.
"""
from __future__ import annotations

from datetime import date
import html
import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Foco Vial | Severidad de siniestros en CABA",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://127.0.0.1:8000")).rstrip("/")
except Exception:
    API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.markdown(
    """<style>
    :root {--accent:#F59A3B;--ink:#101B2A;--ink2:#19293C;--muted:#8192A8;--line:#273A51;--paper:#DCE5EE;}
    .stApp {background: #0E1928; color: #E6EDF5;}
    [data-testid="stSidebar"] {background:#132235; border-right:1px solid #2A3B50;}
    [data-testid="stHeader"] {background:#0E1928;}
    [data-testid="stTabs"] button {font-weight:600;}
    .block-container {max-width:1280px;padding-top:2rem;padding-bottom:3rem;}
    .fv-overline {text-transform:uppercase;letter-spacing:.18em;color:#F59A3B;font-size:.75rem;font-weight:700;}
    .fv-hero {background:radial-gradient(ellipse at top right,#243A53,#152438 55%,#101D2E);border:1px solid #30445C;border-radius:20px;padding:28px 32px 22px;margin-bottom:18px;}
    .fv-brand {display:flex;align-items:center;gap:14px;margin-bottom:11px;}
    .fv-mark {width:48px;height:48px;border-radius:15px;background:#F59A3B;color:#102037;display:flex;align-items:center;justify-content:center;font-size:29px;font-weight:900;}
    .fv-title {margin:0!important;font-size:2.6rem!important;line-height:1.15!important;letter-spacing:-.035em;color:#F3F6FC!important;}
    .fv-subtitle {color:#C1CFDF;font-size:1.02rem;line-height:1.6;margin:8px 0 0;max-width:750px;}
    .fv-foot {color:#8EA0B7;font-size:.76rem;margin-top:18px;letter-spacing:.025em;}
    .fv-panel {border:1px solid #30445C;background:#17283C;padding:19px 21px;border-radius:14px;margin:8px 0;}
    .fv-panel h4 {color:#F5F7FA;margin:0 0 7px;font-size:1rem;}
    .fv-panel p {color:#BAC8D8;margin:0;font-size:.9rem;line-height:1.55;}
    .fv-mini {display:flex;flex-wrap:wrap;gap:9px;margin:8px 0 16px;}
    .fv-chip {border:1px solid #3B4D61;border-radius:99px;padding:5px 11px;color:#BAC9D9;font-size:.79rem;}
    .fv-label {color:#A1B3C9;text-transform:uppercase;font-weight:700;letter-spacing:.1em;font-size:.7rem;}
    .fv-reading {font-size:.9rem;line-height:1.6;color:#BDD0E3;}
    .fv-result {border-radius:15px;padding:20px 22px;margin:16px 0 12px;border:1px solid #46617B;background:#1A2E43;}
    .fv-result strong {font-size:1.3rem;color:#FBFCFD;}
    .fv-result p {margin:7px 0 0;color:#C0D1E1;line-height:1.5;}
    .fv-result-alert {border-color:#C37B35;background:#3A2A21;}
    .fv-step {color:#F59A3B;font-size:.75rem;font-weight:700;letter-spacing:.11em;}
    [data-testid="stMetric"] {border:1px solid #2D4259;border-radius:13px;background:#17283C;padding:15px 17px;}
    [data-testid="stMetricLabel"] {color:#B4C5D8;}
    [data-testid="stMetricValue"] {color:#F5F7FA;}
    [data-testid="stForm"] {background:#15263A;border-radius:16px;padding:16px;border:1px solid #2D435A;}
    [data-testid="stDataFrame"] {border-radius:10px;}
    </style>""",
    unsafe_allow_html=True,
)

st.markdown(
    """<div class="fv-hero">
        <div class="fv-overline">CIENCIA DE DATOS APLICADA · ITBA · ENTREGA FINAL</div>
        <div class="fv-brand"><div class="fv-mark">⌾</div><h1 class="fv-title">Foco Vial</h1></div>
        <div class="fv-subtitle">Una mirada basada en datos sobre la <b>severidad de los siniestros viales</b> en la Ciudad de Buenos Aires. Del análisis histórico a una herramienta interactiva.</div>
        <div class="fv-mini"><span class="fv-chip">CABA · 2019–2025</span><span class="fv-chip">Random Forest del TP3</span><span class="fv-chip">API + Interfaz</span></div>
        <div class="fv-foot">Proyecto académico · Matías Chiffone Guarde · María Candelaria Terán Ortiz · Olivia Vella Delgado</div>
       </div>""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=45, show_spinner=False)
def obtener_info(url: str) -> dict:
    respuesta = requests.get(f"{url}/info", timeout=90)
    respuesta.raise_for_status()
    return respuesta.json()


try:
    with st.spinner("Conectando con la API de Foco Vial. Si estuvo inactiva, puede tardar hasta un minuto en despertar..."):
        info = obtener_info(API_URL)
    st.sidebar.success("Servicio conectado · API activa")
except requests.RequestException:
    st.sidebar.error("Servicio desconectado")
    st.error("No pudimos conectar con la API de Foco Vial.")
    st.info("Si la API está publicada en Render, puede estar despertando después de un rato sin uso. Esperá unos segundos y reintentá.")
    if st.button("Reintentar conexión", type="primary"):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"API configurada: {API_URL}")
    st.caption("En una computadora local: iniciá la API con `uvicorn api.main:app --reload`.")
    st.stop()

st.sidebar.markdown("### 🔎 Foco Vial")
st.sidebar.markdown("**Modelo:** Random Forest")
st.sidebar.caption("Entrenamiento: 2019–2024  |  Evaluación: 2025")
st.sidebar.caption("Pipeline del TP3 · 16 variables · umbral 0,50")
st.sidebar.divider()
st.sidebar.markdown("**Qué hace la herramienta**")
st.sidebar.caption("Estima un puntaje de severidad *condicionado a que un siniestro ocurrió*. No predice dónde ocurrirán accidentes.")
st.sidebar.markdown("**Límite importante**")
st.sidebar.caption("No reemplaza la atención de emergencias ni proporciona probabilidades calibradas.")
st.sidebar.divider()
st.sidebar.caption("Servicio: FastAPI · Interfaz: Streamlit")


def preparar_ejemplo(identificador: str) -> None:
    ejemplos = {
        "peaton": (date(2025, 3, 15), 3, 14, "AUTOPISTA", "PEATON", "AUTO"),
        "auto": (date(2025, 7, 8), 15, 6, "CALLE", "AUTO", "AUTO"),
        "faltante": (date(2025, 8, 11), 13, "SIN_DATO", "CALLE", "SIN_DATO", "SIN_DATO"),
    }
    fecha, hora, comuna, via, modo, contra = ejemplos[identificador]
    st.session_state.update(
        fecha=fecha, hora=hora, comuna=comuna, tipo=via, modo=modo,
        contra=contra, usar_coordenadas=False, ultima_respuesta=None,
    )


def inicializar() -> None:
    inicial = {
        "fecha": date(2025, 3, 15), "hora": 3, "comuna": 14,
        "tipo": "CALLE", "modo": "PEATON", "contra": "AUTO",
        "usar_coordenadas": False, "lat": -34.60, "lon": -58.43,
        "ultima_respuesta": None,
    }
    for clave, valor in inicial.items():
        st.session_state.setdefault(clave, valor)


inicializar()
# Todos los datos corresponden a las tablas y resultados efectivamente documentados en los TP2/TP3.
metricas = info.get("metricas_test_2025", {})
matriz = info.get("matriz_confusion_test_2025", {})

st.markdown('<div class="fv-overline">01 / EN CIFRAS</div>', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Siniestros preparados", "65.738", help="Dataset limpio TP2, período 2019–2025")
k2.metric("Graves o mortales", "5,6%", help="Aproximación sobre el dataset preparado")
k3.metric("Recall en test 2025", f"{metricas.get('Recall', 0.5843):.1%}")
k4.metric("PR-AUC frente a prevalencia", "2,3×", help="PR-AUC 0,153 frente a prevalencia 0,066 en 2025")

st.markdown(" ")
tab_analizar, tab_evidencia, tab_sistema = st.tabs([
    "🔍 Evaluar un siniestro", "📊 Qué dicen los datos", "⚙️ Cómo funciona Foco Vial"
])

with tab_analizar:
    st.markdown("### Evaluación exploratoria de un hecho")
    st.markdown('<div class="fv-reading">Ingresamos las características conocidas de un siniestro y consultamos el modelo entrenado. El resultado es un <b>puntaje orientativo no calibrado</b>, no una predicción de accidentes futuros.</div>', unsafe_allow_html=True)
    st.markdown("#### Probá un caso de ejemplo")
    a, b, c = st.columns(3)
    a.button("🚶 Peatón en autopista", on_click=preparar_ejemplo, args=("peaton",), width="stretch")
    b.button("🚘 Auto en calle", on_click=preparar_ejemplo, args=("auto",), width="stretch")
    c.button("◌ Registro incompleto", on_click=preparar_ejemplo, args=("faltante",), width="stretch")
    st.caption("Casos ilustrativos armados por el equipo, no son siniestros reales del dataset.")

    with st.form("formulario_foco_vial"):
        st.markdown('<div class="fv-step">DATOS DEL SINIESTRO</div>', unsafe_allow_html=True)
        izquierda, derecha = st.columns(2, gap="large")
        with izquierda:
            fecha = st.date_input("Fecha del hecho", min_value=date(2019, 1, 1), max_value=date.today(), key="fecha")
            hora = st.slider("Hora del hecho", min_value=0, max_value=23, key="hora")
            comuna = st.selectbox("Comuna", ["SIN_DATO"] + list(range(1, 16)), key="comuna")
            tipo = st.selectbox("Tipo de vía", info["opciones"]["tipo_de_via"], key="tipo")
        with derecha:
            modo = st.selectbox("Modo de desplazamiento de la víctima", info["opciones"]["modo_victima"], key="modo")
            contraparte = st.selectbox("Contraparte del siniestro", info["opciones"]["contraparte"], key="contra")
            coordenadas = st.checkbox("Agregar coordenadas de CABA (opcional)", key="usar_coordenadas")
            if coordenadas:
                latitud = st.number_input("Latitud", min_value=-90.0, max_value=90.0, format="%.6f", key="lat")
                longitud = st.number_input("Longitud", min_value=-180.0, max_value=180.0, format="%.6f", key="lon")
            else:
                latitud, longitud = None, None
        st.caption("La API calcula automáticamente las variables temporales y derivadas. No se ingresa la gravedad ni el número de víctimas.")
        enviado = st.form_submit_button("Analizar siniestro", type="primary", width="stretch")

    if enviado:
        payload = {
            "fecha": fecha.isoformat(), "hora": int(hora),
            "comuna": None if comuna == "SIN_DATO" else int(comuna),
            "tipo_de_via": tipo, "modo_victima": modo, "contraparte": contraparte,
            "latitud": latitud, "longitud": longitud,
        }
        try:
            with st.spinner("Consultando el modelo a través de la API..."):
                respuesta = requests.post(f"{API_URL}/predecir", json=payload, timeout=90)
            if respuesta.status_code == 422:
                st.session_state["ultima_respuesta"] = None
                st.error("Hay datos que la API no puede procesar. Revisá los campos ingresados.")
                st.json(respuesta.json().get("detail"))
            else:
                respuesta.raise_for_status()
                st.session_state["ultima_respuesta"] = respuesta.json()
        except requests.RequestException:
            st.session_state["ultima_respuesta"] = None
            st.error("No se obtuvo respuesta del servicio. Verificá que la API siga activa.")

    salida = st.session_state["ultima_respuesta"]
    if salida:
        st.markdown("### Resultado de la consulta")
        alerta = salida["clasificacion"] == "GRAVE_O_MORTAL"
        estilo = "fv-result fv-result-alert" if alerta else "fv-result"
        texto = "Señal de gravedad en el clasificador" if alerta else "Clasificado como leve en el modelo"
        explicacion = (
            "La señal supera el umbral del TP3. No confirma una lesión grave en este caso."
            if alerta else "El puntaje queda por debajo del umbral. No descarta lesiones graves."
        )
        st.markdown(
            f'<div class="{estilo}"><div class="fv-label">CLASIFICACIÓN DEL MODELO</div><strong>{html.escape(texto)}</strong><p>{html.escape(explicacion)}</p></div>',
            unsafe_allow_html=True,
        )
        s1, s2, s3 = st.columns([1, 1, 1])
        s1.metric("Puntaje orientativo", f"{salida['puntaje_gravedad']:.1%}")
        s2.metric("Umbral de clasificación", f"{salida['umbral']:.2f}")
        s3.metric("Versión", "Random Forest TP3")
        st.progress(float(max(0, min(1, salida["puntaje_gravedad"]))))
        st.caption("La barra refleja el puntaje del modelo, no una probabilidad calibrada ni una escala de riesgo clínico.")
        if salida.get("advertencias"):
            st.markdown("#### Aspectos a considerar")
            for adv in salida["advertencias"]:
                st.warning(adv)
        if salida.get("asociaciones_generales"):
            st.markdown("#### Contexto histórico relacionado")
            for item in salida["asociaciones_generales"]:
                st.markdown(f"- {item}")
            st.caption("Estos patrones descriptivos no son una explicación individual de la decisión del Random Forest.")
        with st.expander("Ver respuesta técnica de la API"):
            st.json(salida)

with tab_evidencia:
    st.markdown("### De la exploración al modelo")
    st.markdown("El TP2 identificó asociaciones descriptivas. El TP3 comprobó hasta dónde puede generalizar un clasificador sobre un año posterior.")
    st.markdown('<div class="fv-overline">02 / HALLAZGOS HISTÓRICOS</div>', unsafe_allow_html=True)
    st.caption("Porcentajes de siniestros graves o mortales dentro de cada grupo, según el EDA del TP2. No son probabilidades ajustadas por exposición.")
    e1, e2 = st.columns(2, gap="large")
    with e1:
        st.markdown("**¿Cuándo fue más frecuente la gravedad?**")
        st.bar_chart(pd.DataFrame({"% grave/mortal": [11.0, 4.4]}, index=["Madrugada", "Tarde"]), horizontal=True)
        st.caption("Madrugada 11% frente a tarde 4,4%.")
    with e2:
        st.markdown("**¿Qué tipo de vía aparece asociado?**")
        st.bar_chart(pd.DataFrame({"% grave/mortal": [13.0, 4.4]}, index=["Autopista", "Calle"]), horizontal=True)
        st.caption("Autopista 13% frente a calle 4,4%.")
    st.markdown('<div class="fv-overline">03 / VALIDACIÓN TEMPORAL</div>', unsafe_allow_html=True)
    st.markdown("**Entrenamiento: 2019–2024 · Evaluación: 2025**")
    m1, m2, m3, m4 = st.columns(4)
    for c, nombre in zip((m1, m2, m3, m4), ("Precision", "Recall", "F1", "PR-AUC")):
        valor = metricas.get(nombre)
        if isinstance(valor, (float, int)):
            c.metric(nombre, f"{valor:.3f}")
    st.caption("PR-AUC del baseline en 2025: 0,066. Random Forest: 0,153. La precisión de 0,130 indica muchas falsas alarmas.")
    if matriz:
        st.markdown("#### Matriz de confusión del test 2025")
        data = pd.DataFrame(
            [[matriz["TN"], matriz["FP"]], [matriz["FN"], matriz["TP"]]],
            index=["Real: leve", "Real: grave o mortal"],
            columns=["Predijo leve", "Predijo grave/mortal"],
        )
        st.dataframe(data, width="stretch")
        lado1, lado2 = st.columns(2)
        lado1.markdown('<div class="fv-panel"><h4>454 detectados</h4><p>De 777 siniestros realmente graves o mortales en el test de 2025.</p></div>', unsafe_allow_html=True)
        lado2.markdown('<div class="fv-panel"><h4>3.027 falsas alertas</h4><p>La herramienta requiere interpretación humana y no es apta para decidir emergencias.</p></div>', unsafe_allow_html=True)
    st.markdown("#### Un hallazgo que cambia cómo interpretamos el modelo")
    st.warning("Los datos faltantes no se distribuyen al azar. En el TP2, el modo de víctima faltaba en 17,5% de los leves, 0,9% de los graves y 0% de los mortales. El modelo heredado aún conserva algunas categorías SIN_DATO.")

with tab_sistema:
    st.markdown("### Una aplicación, tres componentes")
    st.markdown("La interfaz **no abre el modelo directamente**: se comunica por HTTP con una API que valida la entrada y ejecuta el pipeline guardado del TP3.")
    a, b, c = st.columns(3, gap="medium")
    with a:
        st.markdown('<div class="fv-panel"><div class="fv-step">01 · INTERFAZ</div><h4>Streamlit</h4><p>Recibe fecha, hora, comuna, vía, participante y contraparte. Presenta el resultado en lenguaje claro.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="fv-panel"><div class="fv-step">02 · SERVICIO</div><h4>FastAPI</h4><p>Valida valores, arma variables derivadas y comunica la solicitud al modelo.</p></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="fv-panel"><div class="fv-step">03 · MODELO</div><h4>Random Forest</h4><p>Pipeline persistido, con 16 predictoras, clasificación binaria y umbral 0,50 del TP3.</p></div>', unsafe_allow_html=True)
    st.markdown("#### Flujo de una consulta")
    st.code("Usuario → Streamlit → POST /predecir → FastAPI → Pipeline TP3 → Respuesta JSON → Interfaz", language="text")
    st.markdown("#### Servicio conectado")
    x, y = st.columns(2)
    x.markdown("**GET `/health`** · verifica disponibilidad")
    x.markdown("**GET `/info`** · métricas y categorías válidas")
    x.markdown("**POST `/predecir`** · evalúa un siniestro")
    y.caption("El modelo se carga una sola vez, ya entrenado. Las transformaciones se ejecutan de forma reproducible.")
    with st.expander("Metadatos del modelo cargado"):
        st.json({"modelo": info["modelo"], "version": info["version_modelo"], "variables": info["predictoras"], "umbral": info["umbral"]})
    st.markdown("#### Alcance y límites")
    st.markdown("Esta herramienta es una **demostración académica de clasificación retrospectiva**. No predice si ocurrirá un choque, no establece causas y no sustituye una decisión operativa. La calibración, los umbrales y el tratamiento de faltantes son próximos pasos.")

st.divider()
st.caption("FOCO VIAL · Trabajo práctico 4 · Ciencia de Datos Aplicada · ITBA · Datos abiertos GCBA / OMSV")
