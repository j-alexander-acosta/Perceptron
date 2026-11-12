# Contexto 22 — Integración de la Unidad CD10: Modelos No Lineales y Redes Neuronales (65 Diapositivas)
**Fecha de registro:** 2026-11-12

## 1. Instrucciones y Directrices
* **Archivo fuente cargado:** Presentación de cátedra UCN: `Modelos No Lineales y Redes Neuronales` (65 diapositivas / 304 páginas de progresión), basada en Stanford CS221 (P. Liang et al.), Harvard CS109 (P. Protopapas et al.), CMU 11-785 (B. Raj et al.) y C. Bishop (*Pattern Recognition and Machine Learning*, cap. 7).
* **Ubicación en el plan de estudios:** Décima unidad en orden cronológico de cátedra (puente fundamental entre los modelos de reflejos lineales y el conexionismo neuronal profundo):
  $$\text{CD01} \to \dots \to \text{CD09} \to \mathbf{\text{CD10 (Modelos No Lineales y Redes Neuronales)}} \to \text{CD13} \to \dots \to \text{CD18}$$
* **Compatibilidad técnica:**
  - Implementación en Python estándar puro sin dependencias externas obligatorias para CLI (`modelos_no_lineales.py`).
  - Integración en el orquestador maestro `perceptron_xor.py` (Fase 1, opción `[10]`, banderas CLI `--nonlinear`, `--knn`, `--svm`, `--kernel`, `-c10`, `--cd10`, suite `--auto`).
  - Suite web interactiva en `web/index.html` con renderizado HTML5 Canvas nativo y Quizzes 61 a 64 (Labs 10.1 a 10.4).
  - Documentación formal exhaustiva en `INFORME.md` (Sección 0.13).
  - Actualización de totales del repositorio en `README.md`: 16 Unidades Académicas • 1004 Diapositivas • 65 Laboratorios Interactivos.

---

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Implementar, simular, verificar y documentar exhaustivamente todas las formulaciones matemáticas, derivaciones teóricas, trazas numéricas a mano, simulaciones estadísticas y experimentos de las 65 diapositivas de la Unidad CD10 (Modelos No Lineales y Redes Neuronales).
* **Fases e hitos principales:**
  1. **Análisis exhaustivo del documento:** 4 Módulos temáticos extraídos de las 65 diapositivas:
     - Módulo 10.1: kNN en Regresión y Clasificación, Voto Mayoritario, Métrica Euclidiana, Estandarización y la Maldición de la Dimensionalidad (Diaps. 5–15).
     - Módulo 10.2: Características No Lineales Diseñadas a Mano (Polinomiales, A Trozos $B^d$, Periódicas), Proyección al Hiperplano y Límites Combinatorios (Diaps. 16–28).
     - Módulo 10.3: Máquinas de Vectores de Soporte (SVM): Margen Máximo Duro, Libertad de Escala, Holguras y Margen Suave ($C$), Dual de Lagrange y Equivalencia con Pérdida Hinge $+ L_2$ (Diaps. 29–40).
     - Módulo 10.4: El Truco del Kernel (Polinomial a Mano, Catálogo Mercer, RBF Gaussiano con $\gamma$, Grilla Exponencial $C \times \gamma$) y Emergencia de las Redes Neuronales (El Problema XOR, Matriz $V$, Activaciones y Redes Multicapa) (Diaps. 41–65).
  2. **Módulo computacional Python (`modelos_no_lineales.py`):**
     - Módulo 10.1: Simulación de 9 puntos de entrenamiento y consulta $x_q = 2.2$; cálculo a mano de $1$-NN ($\hat{y}=3.8$) y $3$-NN ($\hat{y}=4.0$); evaluación con 4 puntos de prueba (RMSE train vs test para $k=1,3,9$: $0.00/0.23$, $0.31/0.19$, $0.50/0.38$); kNN de clasificación binaria con $k=5$ (3 vs 2 $\implies \hat{P}=0.6$); cálculo de la maldición de la dimensionalidad ($p=10 \to 79\%$, $p=100 \to 98\%$).
     - Módulo 10.2: Predictores polinomiales ($\phi(x) = [1, x, x^2]$), constantes a trozos ($\phi(2.3) = [0,0,1,0,0]$ con costo $B^d$), periódicos ($\phi(x) = [1, x, x^2, \cos(3x)]$); clasificador circular cuadrático $(x_1-1)^2 + (x_2-1)^2 \le 2$ y proyección 3D a paraboloide; explosión de dimensiones $\binom{J+d}{d}$ ($J=100, d=5 \to 96\,560\,646$).
     - Módulo 10.3: Distancia con signo $d(x) = \frac{w\cdot x + b}{\|w\|}$; cálculo a mano en 3 puntos para $3x_1 + 4x_2 - 5 = 0$ ($d=1.6, 2.0, -1.6 \implies \text{margen}=1.6$); margen duro en 8 puntos ($w=[0.5, 0.5], b=-1$, ancho $2\sqrt{2} \approx 2.83$); colapso ante ruido ($0.51$); holguras $\xi_n$ y margen suave con $C=0.1$ (ancho $3.41$, 6 SV, 1 error) vs $C=100$ (ancho $0.51$, 3 SV, 0 errores); conexión con pérdida hinge $+ L_2$ ($\lambda = 1/(2C)$).
     - Módulo 10.4: Dual de Lagrange y pesos $\alpha_n$; verificación algebraica y numérica del kernel polinomial cuadrático $(1 + x\cdot z)^2 = 36$ vs $\phi(x)\cdot \phi(z) = 36$; catálogo de kernels; radio RBF $\sigma = \sqrt{1/(2\gamma)}$ ($\gamma=0.1 \to 2.24$, $\gamma=1 \to 0.71$, $\gamma=5 \to 0.32$, $\gamma=30 \to 0.13$); experimento de sobreajuste en 4000 puntos ($92.9\%$ vs $76.9\%$ test); problema XOR de colisión vehicular, descomposición en $h_1, h_2$, matriz $V = [[-1, 1, -1], [-1, -1, 1]]$ y vector $w=[1, 1]$; gradiente de función escalón vs activaciones suaves (Sigmoide, ReLU, $\tanh$, Softplus); redes de 2 y más capas; cuadro comparativo de las 4 vías no lineales.
     - Suite automatizada `--auto` con assertions estrictas probando cada valor teórico al 100%.
  3. **Integración en el orquestador central (`perceptron_xor.py`):**
     - Añadir CD10 a `UNIDADES_CATEDRA` en índice 9 (num 10), ampliando el catálogo a 16 unidades.
     - Menú Fase 1 actualizado a 10 unidades (`[1-10, T, V]`), opción `[10]` ejecutando `launch_nonlinear_module()`.
     - Flags CLI: `--nonlinear`, `--knn`, `--svm`, `--kernel`, `-c10`, `--cd10`.
     - Suite unificada `--auto` incorporando las pruebas de `modelos_no_lineales.py`.
  4. **Suite Web Interactiva (`web/index.html`):**
     - Estilos CSS cian/azul petróleo moderno (`#0891b2`, `#06b6d4`, `.active-unit-10`, `.tab-btn.unit10-tab.active`, `.unit-pill.u10`).
     - Botón `unitBtn10` en selector de Fase 1.
     - Navegación `#unit10Tabs` (Labs 61 a 64).
     - Paneles `#panel-61` a `#panel-64` con controles interactivos, 4 Canvas visualizadores (`drawCd10Knn`, `drawCd10Features`, `drawCd10SvmMargin`, `drawCd10KernelXor`) y Quizzes oficiales 61 a 64.
     - Actualizado `topicMetadata` (longitud 65), `ORDERED_UNITS`, `ORDERED_TABS`, `filterPhase`, `selectUnit`, `switchTab` y `quizData`.
  5. **Documentación formal:**
     - `INFORME.md`: Sección 0.13 con desarrollos matemáticos exhaustivos, derivaciones de distancia normalizada, dualidad de Lagrange, verificación del kernel polinomial, equivalencia hinge-$L_2$, descomposición matricial de XOR, actualización de conclusiones y referencias bibliográficas canónicas (Vapnik 1995, Cover 1965, Mercer 1909, Hsu et al. 2003, Bishop 2006).
     - `README.md`: Resumen global actualizado a 16 unidades, 1004 diapositivas, 65 laboratorios interactivos; sección temática para CD10; comandos CLI y diagrama de estructura de archivos.
     - `contexto/context22.md`: Registro estructurado de la sesión conforme al skill `almacenamiento`.

## 3. Referencias
* **Path del plan original:** `Modelos No Lineales y Redes Neuronales (Cátedra UCN / Stanford CS221 / Harvard CS109 / CMU 11-785 / Bishop PRML Cap. 7)`
