# Contexto 25: Integración de CD11 — Aspectos Prácticos y Evaluación de Modelos
**Fecha de registro:** 2026-11-12

## 1. Instrucciones
* Integrar exhaustivamente la cátedra **Aspectos Prácticos y Evaluación** (93 diapositivas) en el curso de Ciencia de Datos (Doctorado UCN), correspondiente a la **Unidad XI (CD11)**, unidad de cierre del bloque de modelos supervisados clásicos (Fase 1).
* Fuentes académicas: CMU 15-388/688 (Zico Kolter), Harvard CS109 (Pavlos Protopapas et al.) y CMU 11-785 (Bhiksha Raj et al.).
* Ubicación curricular: Sucede a Modelos No Lineales y SVM (CD10 / Unidad X) y antecede a Aproximación Universal en Redes Neuronales (CD13 / Unidad I de Fase 2). Integra las lecciones transversales de depuración, datos faltantes, métricas de evaluación médica y escalamiento computacional.
* Requisitos de implementación técnica:
  1. **Módulo CLI en Python puro (`aspectos_practicos.py`):**
     - Sin dependencias de terceros (solo biblioteca estándar de Python 3).
     - Módulo 11.1: Debugging en Ciencia de Datos y la Ley de Kolter (Zico Kolter: *"Lo que distingue a los expertos del resto es lo que hacen cuando lo primero no funciona"*; el proxy humano: inspección visual a ciegas de entradas; estudio de Google USENIX FAST 2007 en 100k discos duros: >56% de discos fallados con 0 en SMART crítico y >36% en cero absoluto; Backblaze 2025: 76.7% con alguna señal, pero 1 de 4 falla sin avisar; acciones ante problemas imposibles: cambiar entradas vs cambiar salidas / cuantificar riesgo relativo con tasa base 1.36% anual; acciones ante problemas factibles: diagnósticos de sesgo/varianza, optimización y función de pérdida; matriz 2x2 de Kolter y sospecha ante éxito inmediato por data leakage).
     - Módulo 11.2: Datos Faltantes (Missing Data) y Métodos de Imputación (vicios de `dropna` y `fillna(mean)`; variable indicadora de ausencia $X_{j,\text{miss}} \in \{0, 1\}$; taxonomía de Rubin 1976: MCAR, MAR, MNAR e incomprobabilidad empírica; 5 métodos de imputación con datos exactos: media global $\bar{x}_2 = 3.91$, kNN $k=2$ con predicciones $3.15$ y $5.10$, regresión determinista $\hat{x}_2 = 1.12 + 0.62 x_1$ dando $3.29$ y $5.15$, e imputación con incertidumbre estocástica $\hat{y} + \mathcal{N}(0, 0.11)$ con $\sigma = 0.33$; imputación encadenada multivariada MICE).
     - Módulo 11.3: Evaluación de Clasificadores y Caso de Triage en Pandemia (caso clínico 251 pacientes: 134 urgencia Alta, 117 urgencia Baja; matrices de confusión de Regresión Logística $[58, 59; 37, 97]$ y kNN $[55, 62; 33, 101]$; cálculo exacto de métricas: exactitud $0.62$, sensibilidad/recall $0.72$ vs $0.75$, especificidad $0.50$ vs $0.47$, precision $0.62$, F1-score $0.67$ vs $0.68$; perilla de umbral de decisión: bajo $[55, 62; 35, 99]$, medio $[58, 59; 37, 97]$, alto $[61, 56; 39, 95]$; curva ROC e historia en el radar de Chain Home Bawdsey; optimización contextual para Brasil $\text{TPR}+\text{FPR}\le 0.5 \implies (0.15, 0.35)$ Logística umbral alto, Alemania $\text{TPR}\ge 0.85 \implies (0.75, 0.86)$ Logística umbral bajo, e India $\text{TPR}+\text{FPR}\le 1.0 \implies (0.38, 0.62)$ kNN umbral medio).
     - Módulo 11.4: Ética, Receta de ML y Big Data COST (intención vs impacto en cuadrante de negligencia; auditoría por subgrupos y minimización del riesgo en minorías; receta de 7 pasos; prevención de data leakage; aumento de datos en audio, imágenes, series temporales y tablas; los 4 puntos de inflexión de cómputo: RAM, disco, clúster pequeño, clúster tolerante a fallas; *Scalability! But at what COST?* de McSherry et al. 2015: PageRank Twitter 128 núcleos Spark $857\text{s}$ vs 1 hilo laptop $275\text{s}$; DuckDB y «Big Data is Dead» de MotherDuck 2023).
     - Menú interactivo con navegación completa y suite `--auto` que valida numéricamente cada cálculo y ejemplo.
  2. **Integración en Orquestador Maestro (`perceptron_xor.py`):**
     - Registrar CD11 en `UNIDADES_CATEDRA` como Unidad XI (Num 11, ID 'cd11').
     - Incorporar opción en `menu_fase1()` y función `launch_practical_module()`.
     - Agregar flags CLI `--practica`, `--evaluacion`, `--roc`, `-c11`, `--cd11`.
     - Incluir verificación automática de CD11 en la suite máster `--auto`.
     - Actualizar banners y métricas: 19 Unidades de Cátedra (1200 Diapositivas) • 77 Laboratorios Interactivos.
  3. **Suite Web Interactiva (`web/index.html`):**
     - Agregar Labs 73, 74, 75 y 76 (elevando el total de 73 a 77 laboratorios interactivos, paneles `#panel-73` a `#panel-76`).
     - Estilo visual de Unidad XI en tonos Ámbar Cálido / Dorado Ocre (`#d97706`, `#f59e0b`, `#fbbf24`).
     - Visualizadores Canvas y tableros interactivos:
       * Lab 73: Matriz Factible/Imposible de Kolter y el Proxy de Predicción Manual con Telemetría de Discos Duros.
       * Lab 74: Simulador de Mecanismos de Ausencia (MCAR, MAR, MNAR) y Comparador de 5 Métodos de Imputación.
       * Lab 75: Matriz de Confusión Interactiva, Triage de 251 Pacientes y Curva ROC Multipaís (Brasil, Alemania, India).
       * Lab 76: Auditoría Ética por Subgrupos, Receta de 7 Pasos y Métricas Big Data COST (Laptop vs. Clúster) + Autoevaluación XI.
     - Quizzes 73 a 76 con retroalimentación explicativa paso a paso.
     - Actualización de metadatos en `topicMetadata`, `ORDERED_UNITS` y `ORDERED_TABS`.
  4. **Documentación Teórica (`INFORME.md`):**
     - Insertar sección completa para CD11 (Aspectos Prácticos y Evaluación) con derivaciones matemáticas, matrices de confusión, análisis ROC y referencias históricas (Rubin 1976, Pinheiro et al. 2007, McSherry et al. 2015, Tigani 2023).
     - Actualizar Conclusiones y Referencias.
  5. **Documentación del Repositorio (`README.md`):**
     - Actualizar métricas globales: 19 Unidades Académicas • 1200 Diapositivas • 77 Laboratorios Interactivos.
     - Detallar CD11 en listas temáticas, comandos y estructura de archivos.

## 2. Plan de Trabajo
* **Objetivo:** Implementar, integrar y verificar con exactitud matemática la Unidad XI (CD11: Aspectos Prácticos y Evaluación) en la suite CLI, orquestador maestro, laboratorio web y documentación técnica.
* **Fases:**
  1. *Fase 1:* Redacción del presente documento de contexto (`contexto/context25.md`).
  2. *Fase 2:* Codificación de `aspectos_practicos.py` en Python puro con suite `--auto`.
  3. *Fase 3:* Conexión en `perceptron_xor.py` con launchers, flags CLI y tests integrados.
  4. *Fase 4:* Extensión de `web/index.html` con Labs 73 a 76, paneles `#panel-73` a `#panel-76`, CSS y quizzes.
  5. *Fase 5:* Actualización de `INFORME.md` y `README.md`.
  6. *Fase 6:* Verificación exhaustiva de código y validación web con Node.js.

## 3. Referencias
* Documento PDF cátedra *Aspectos Prácticos y Evaluación* (Ciencia de Datos — Doctorado UCN, 93 diapositivas).
* CMU 15-388/688: Practical Data Science (Z. Kolter).
* Harvard CS109: Data Science (P. Protopapas et al.).
* CMU 11-785: Deep Learning (B. Raj et al.).
