# Contexto 21 — Integración de la Unidad CD09: Ensamblaje de Modelos y Random Forests (59 Diapositivas)
**Fecha de registro:** 2026-11-11

## 1. Instrucciones y Directrices
* **Archivo fuente cargado:** Presentación de cátedra Cátedra UCN: `Ensamblaje de modelos y random forests` (59 diapositivas / 240 páginas de progresión), basada en Harvard CS109 (P. Protopapas et al.) y T. Hastie, R. Tibshirani y J. Friedman (*The Elements of Statistical Learning*, cap. 15).
* **Ubicación en el plan de estudios:** Novena unidad en orden cronológico de cátedra (culminación de la familia de reflejos y modelos basados en árboles):
  $$\text{CD01} \to \text{CD02} \to \text{CD03} \to \text{CD04} \to \text{CD05} \to \text{CD06} \to \text{CD07} \to \text{CD08} \to \mathbf{\text{CD09 (Ensambles y Random Forests)}} \to \text{CD13} \to \dots \to \text{CD18}$$
* **Compatibilidad técnica:**
  - Implementación en Python estándar puro sin dependencias externas obligatorias para CLI (`ensambles_random_forest.py`).
  - Integración en el orquestador maestro `perceptron_xor.py` (Fase 1, opción `[9]`, banderas CLI `--ensemble`, `--rf`, `-c9`, `--cd09`, `--forest`, `--boosting`, `--adaboost`, suite `--auto`).
  - Suite web interactiva en `web/index.html` con renderizado HTML5 Canvas nativo y Quizzes 57 a 60 (Labs 09.1 a 09.4).
  - Documentación formal exhaustiva en `INFORME.md` (Sección 0.12).
* **Totales del proyecto actualizados:** 15 Unidades Académicas • 939 Diapositivas • 61 Laboratorios Interactivos.

---

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Implementar, simular, verificar y documentar exhaustivamente todas las formulaciones matemáticas, derivaciones teóricas, trazas numéricas a mano, simulaciones estadísticas y experimentos de las 59 diapositivas de la Unidad CD09 (Ensamblaje de Modelos y Random Forests).
* **Fases e hitos principales:**
  1. **Análisis exhaustivo del documento:** 4 Módulos temáticos extraídos de las 59 diapositivas:
     - Módulo 09.1: Sabiduría de las Multitudes, Buey de Plymouth (Galton 1907) y Aritmética del Voto Independiente (Condorcet).
     - Módulo 09.2: Bootstrap, Algoritmo de Bagging (Breiman 1996), Validación Out-of-Bag (OOB) e Importancia MDI.
     - Módulo 09.3: El Límite del Bagging (Correlación de Árboles), Derivación ESL 15.1, Piso Infranqueable $\rho \sigma^2$ y Random Forests (Breiman 2001).
     - Módulo 09.4: Boosting Aditivo (Gradient Boosting sobre Residuos, AdaBoost a Mano en 13 Puntos), Deducción de Tasa Óptima $\lambda^{(i)}$, Voting y Stacking.
  2. **Módulo computacional Python (`ensambles_random_forest.py`):**
     - Módulo 09.1: Simulación de Galton ($N=787$, mediana $1207$ lbs vs $1198$ lbs real, error $0.751\%$); cálculo del voto por mayoría para $p=0.7$ ($B=3 \to 0.784, B=5 \to 0.837, B=15 \to 0.950, B=101 \to \approx 1$) y $p=0.55$ ($B=101 \to 0.844$); demostración de falla con copias idénticas.
     - Módulo 09.2: Probabilidad bootstrap $1-(1-1/N)^N \to 1-e^{-1} \approx 0.632$ y OOB $(1-1/N)^N \to e^{-1} \approx 0.368$ ($N=10 \to 0.349$); simulación de bagging en regresión sobre 30 puntos reduciendo MSE de $1.36$ a $1.04$.
     - Módulo 09.3: Formulación de varianza de Hastie ESL 15.1 $\rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$; verificación del piso $0.600$ para $\rho=0.6$; contraste $B=10 \to 0.64$, $B=20 \to 0.62$ vs bajar $\rho=0.3 \to 0.37$; simulación Monte Carlo con componentes común y propia coincidiendo con $0.6040$; algoritmo de Random Forest con sorteo de $m$ predictores.
     - Módulo 09.4: Gradient Boosting en 10 puntos (MSE $3.09 \to 1.16 \to 0.90 \to 0.55$) y traza a mano 2 puntos ($(1,2), (2,4) \to r=(-1,1)$); AdaBoost en 13 puntos (3 rondas: $\lambda_1 = 1.24, \lambda_2 = 1.20, \lambda_3 = 1.15 \to 0$ errores); derivación analítica de tasa óptima $\lambda = \frac{1}{2}\ln \frac{1-\varepsilon}{\varepsilon}$ con mínimo en $\varepsilon=0.3 \to \lambda=0.424$; Hard vs Soft Voting ($(0.45, 0.45, 0.90) \to -1$ vs $+1$); Stacking con metamodelo fuera de muestra.
     - Suite automatizada `--auto` con assertions estrictas probando cada valor teórico al 100%.
  3. **Integración en el orquestador central (`perceptron_xor.py`):**
     - Añadir CD09 a `UNIDADES_CATEDRA` en índice 8 (num 9), actualizando el catálogo a 15 unidades.
     - Menú Fase 1 actualizado a 9 unidades (`[1-9, T, V]`), opción `[9]` ejecutando `launch_ensemble_module()`.
     - Flags CLI: `--ensemble`, `--rf`, `-c9`, `--cd09`, `--forest`, `--boosting`, `--adaboost`.
     - Suite unificada `--auto` incorporando las pruebas de `ensambles_random_forest.py`.
  4. **Suite Web Interactiva (`web/index.html`):**
     - Estilos CSS terracota/ámbar (`#ea580c`, `#f97316`, `.active-unit-09`, `.tab-btn.unit09-tab.active`, `.unit-pill.u09`).
     - Botón `unitBtn09` en selector de Fase 1.
     - Navegación `#unit09Tabs` (Labs 57 a 60).
     - Paneles `#panel-57` a `#panel-60` con controles interactivos, 4 Canvas visualizadores (`drawCd09Crowds`, `drawCd09Bagging`, `drawCd09ForestVariance`, `drawCd09Boosting`) y Quizzes oficiales 57 a 60.
     - Actualizado `topicMetadata` (longitud 61), `ORDERED_UNITS`, `ORDERED_TABS`, `filterPhase`, `selectUnit`, `switchTab` y `quizData`.
     - Validación sintáctica mediante Node.js aprobada al 100%.
  5. **Documentación formal:**
     - `INFORME.md`: Sección 0.12 con desarrollos matemáticos exhaustivos, derivaciones de ESL 15.1 y $\lambda^*$, trazas completas a mano, actualización de conclusiones y 7 nuevas referencias bibliográficas canónicas.
     - `README.md`: Resumen global actualizado a 15 unidades, 939 diapositivas, 61 laboratorios interactivos; sección temática para CD09; comandos CLI y diagrama de estructura de archivos.
     - `contexto/context21.md`: Registro estructurado de la sesión conforme al skill `almacenamiento`.
