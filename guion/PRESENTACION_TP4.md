# Presentación final TP4 · propuesta de 8–9 minutos

> La consigna pide **7–10 minutos** con demostración en vivo. Mantener la demo funcional por encima de agregar más diapositivas.

| Tiempo | Bloque | Qué mostramos |
|---|---|---|
| 0:00–1:05 | Recap | Problema: clasificar severidad de un siniestro ocurrido. Dataset GCBA, 65.738 casos, target grave/mortal vs leve. |
| 1:05–2:15 | Modelo TP3 | Entrenamiento 2019–24, test 2025. Random Forest. PR-AUC 0,153 y recall 0,584, con las limitaciones de falsos positivos. |
| 2:15–3:05 | Integración | Diagrama: página web (GitHub Pages) → API FastAPI (Render) → variables → modelo .joblib → JSON → página. Aclarar que no se reentrena al cargar un caso. |
| 3:05–6:20 | DEMO en vivo | indicador "API conectada", formulario, ejemplo peatón/autopista, auto/calle y registro `SIN_DATO`, leer puntaje, umbral y advertencias, y abrir "Ver lo que viajó entre la página y la API". Resultados esperados: peatón/autopista 0,71 (alerta), auto/calle 0,17 (leve), registro incompleto 0,05 (leve con advertencias). |
| 6:20–7:15 | Evidencia de funcionamiento | Mostrar test `pytest`, 5 casos consistentes API/modelo y gestión de errores con comuna 20. |
| 7:15–8:45 | Cierre | Qué aporta y límites: puntajes sin calibrar, 13% de precisión, datos faltantes, riesgo de registrar un atajo. Próximos pasos: modelo v2, exposición vial, variables espaciales y evaluación operativa. |

## Mini guion de la demostración

"En el TP3 dejamos entrenado el Random Forest y lo guardamos. En esta entrega lo conectamos a un sistema. Primero comprobamos que la API responde. Después ingresamos fecha, hora, comuna, vía y participantes. La API reconstruye las 16 variables del modelo, consulta el archivo guardado y devuelve un puntaje y una clasificación. Si faltan datos, mostramos una advertencia: sabemos del TP2 que la completitud se relaciona con la gravedad registrada. Finalmente, probamos un caso inválido para demostrar el manejo de errores."

## Preparación antes de entrar al aula

- Abrir la página publicada (github.io) 5 minutos antes, para despertar la API en Render, y correr los 3 casos.
- Tener la versión local lista en la computadora por si falla internet.
- Guardar una captura o video de respaldo, por si falla internet o la demo en vivo.
- No decir que la herramienta predice dónde habrá choques; clasifica retrospectivamente un siniestro.
- No decir que los puntajes son probabilidades calibradas ni que los factores son causas.
- Aclarar que el modelo es el del TP3 (verificado) y que los cambios pedidos en la devolución quedan para el reentrenamiento.
