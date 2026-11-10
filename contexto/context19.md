# Contexto 19 — Integración de la Unidad CD07: Reducción de Dimensionalidad (73 Diapositivas)
**Fecha de registro:** 2026-11-10

## 1. Instrucciones y Directrices
* **Archivo fuente cargado:** `/Users/alexanderacosta/.gemini/antigravity/brain/c8f17cb7-96ad-4898-8b55-e7d88fe53835/.user_uploaded/media_1794330734917_7608804a.pdf` (`CD07-Reducción-de-Dimensionalidad.pdf`).
* **Ubicación en el plan de estudios:** Séptima unidad en orden estricto de cátedra:
  $$\text{CD01} \to \text{CD02} \to \text{CD03} \to \text{CD04} \to \text{CD05} \to \text{CD06} \to \mathbf{\text{CD07 (Reducción de Dimensionalidad)}} \to \text{CD13} \to \dots \to \text{CD18}$$
* **Compatibilidad técnica:** Implementación en Python estándar puro sin dependencias externas obligatorias para CLI (`reduccion_dimensionalidad.py`), integración en `perceptron_xor.py` (opción `[7]`, banderas CLI `--cd07`, `--pca`, `--lda`, `--tsne`, `--umap`, suite automática `--auto`), suite web interactiva en `web/index.html` con renderizado Canvas nativo, y documentación formal en `INFORME.md` (Sección 0.10).
* **Totales del proyecto actualizados:** 13 Unidades Académicas • 812 Diapositivas • 53 Laboratorios Interactivos.

---

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Implementar, simular, verificar y documentar exhaustivamente todas las formulaciones matemáticas, trazas numéricas a mano, teoremas y experimentos controlados de las 73 diapositivas (321 páginas de transición) de la Unidad CD07 (Reducción de Dimensionalidad).
* **Fases e hitos principales:**
  1. **Análisis exhaustivo del documento:** 4 Módulos temáticos extraídos de las 73 diapositivas (Maldición de la Dimensionalidad, PCA Paso a Paso & SVD, Fisher LDA & Criterio de Fisher, Manifold Learning con t-SNE, Student-t, UMAP & HDBSCAN).
  2. **Módulo computacional Python (`reduccion_dimensionalidad.py`):** Implementación rigurosa de la razón de volumen $P_d = \frac{\pi^{d/2}}{\Gamma(d/2+1)2^d}$ en $d \in \{1, 2, 3, 6, 10, 15, 100\}$, simulación Monte Carlo de $N=1\,000$ puntos, concentración de distancias de Beyer et al., traza manual exacta de PCA sobre 6 puntos en 2D $((6,4), (5,5), (5,2), (3,4), (3,1), (2,2))$ con centrado $\bar{x}=(4,3)$, covarianza $S$, autovalores $\lambda_1=3, \lambda_2=1$, autovectores normalizados a $45^\circ$ y $135^\circ$, proyecciones $z_i$, reconstrucción con error total exacto $\text{Loss}^* = N\lambda_2 = 6.0$, varianza explicada $75\%$ y $25\%$, descomposición SVD de Eckart-Young-Mirsky; Criterio de Fisher sobre 8 puntos de dos clases ($m_1=(2,4), m_2=(5,5)$) con matriz intra-clase $S_W = \text{diag}(4, 36)$, vector óptimo $w^* \propto S_W^{-1}(m_2-m_1) \propto [27, 1]^T$ ($J=2.28$ vs $J=0.03$ de PCA, $82\times$ mejor separación); probabilidades de vecindad gaussiana $p_{j|i}$ con $\sigma$ adaptativo por perplejidad, solución del Crowding Problem mediante distribución t de Student de colas pesadas ($810\times$ más densa a distancia 3) y reglas de interpretación de mapas 2D de Wattenberg et al. con pipeline moderno UMAP $\to$ HDBSCAN.
  3. **Integración en orquestador central (`perceptron_xor.py`):** Inserción en `UNIDADES_CATEDRA` (índice 6, num 7), actualización de menú Fase 1 (opción `[7]`), menú Fase 2 (desplazamiento a índices 7..12), banderas CLI (`--cd07`, `--dim`, `--pca`, `-c7`, `--lda`, `--tsne`, `--umap`), suite automática `--auto` (13/13 suites aprobadas al 100%).
  4. **Suite Web Interactiva (`web/index.html`):** Inserción de estilos CSS dorados/ámbar (`#eab308`), botón `unitBtn07`, navegación `#unit07Tabs` (Labs 49 a 52), paneles `#panel-49` a `#panel-52`, Canvas interactivos y Quizzes oficiales 49 a 52 con validación sintáctica Node.js aprobada al 100%.
  5. **Documentación formal:** Redacción de la Sección 0.10 en `INFORME.md`, actualización de totales en `README.md` e `INFORME.md`, agregado de referencias canónicas (Fisher 1936, Turk & Pentland 1991, Beyer et al. 1999, van der Maaten & Hinton 2008, Wattenberg et al. 2016, McInnes et al. 2018) y registro en `contexto/context19.md`.

---

## 3. Síntesis Teórica y Numérica Implementada

### 3.1. Módulo 1: La Maldición de la Dimensionalidad (Diaps. 1–11)
* **Razón analítica de volumen Bola en Hipercubo (Diaps. 8–10):**
  $$P_d = \frac{\text{Vol}(B_d(1))}{\text{Vol}(C_d(2))} = \frac{\pi^{d/2}}{\Gamma(d/2 + 1) \cdot 2^d}$$
  * $d=1$: $P_1 = 1.0$ ($100\%$ de coincidencia).
  * $d=2$: $P_2 = \pi/4 \approx 0.7854$ ($78.5\%$).
  * $d=3$: $P_3 = \pi/6 \approx 0.5236$ ($52.4\%$; casi la mitad del cubo son esquinas).
  * $d=6$: $P_6 \approx 0.0807$ ($8.07\%$).
  * $d=10$: $P_{10} \approx \mathbf{0.00249}$ (**$0.25\%$**; solo 1 de cada 401 puntos cae en el centro).
  * $d=15$: $P_{15} \approx 1.16 \times 10^{-5}$.
  * $d=100$: $P_{100} \approx \mathbf{1.87 \times 10^{-70}}$ (prácticamente 0; el cubo es un erizo de $2^{100}$ esquinas vacías).
* **Simulación Monte Carlo con $N=1\,000$ puntos (Diap. 11):** En $d=2$, caen $\approx 785$ puntos dentro del círculo; en $d=10$, caen en promedio **$2.5$ puntos** (apenas 2 o 3 puntos en todo el conjunto).
* **Teorema de concentración de distancias (Beyer et al., 1999):** $\lim_{d \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} = 0$.

### 3.2. Módulo 2: PCA Paso a Paso a Mano y SVD (Diaps. 12–37)
* **Formulación dual:** Minimizar error cuadrático $\sum \|\hat{x}_i - x_i\|^2 \iff$ Maximizar varianza proyectada $v^T S v$ sujeto a $\|v\|=1$ por el Teorema de Pitágoras ($\text{tr}(S) = v^T S v + \text{Error}$).
* **Ejemplo 6 puntos en 2D a mano (Diaps. 25–32):** $\{(6,4), (5,5), (5,2), (3,4), (3,1), (2,2)\}$.
  * Media: $\bar{x} = (4.0, 3.0)$.
  * Matriz de covarianza: $S = \begin{bmatrix} 2.0 & 1.0 \\ 1.0 & 2.0 \end{bmatrix}$.
  * Autovalores: $\lambda_1 = 3.0$ ($75\%$), $\lambda_2 = 1.0$ ($25\%$).
  * Autovectores: $v^{(1)} = \frac{1}{\sqrt{2}}[1, 1]^T$ ($45^\circ$), $v^{(2)} = \frac{1}{\sqrt{2}}[1, -1]^T$ ($135^\circ$).
  * Proyecciones $z$: $[3/\sqrt{2}, 3/\sqrt{2}, 0, 0, -3/\sqrt{2}, -3/\sqrt{2}]$.
  * Pérdida total de compresión: $\text{Loss}^* = \sum \|\hat{x}_i - x_i\|^2 = \mathbf{6.0} = N \cdot \lambda_2 = 6 \times 1.0 = \mathbf{6.0}$.
* **Teorema de Eckart-Young-Mirsky y SVD:** $\tilde{X} = U \Sigma V^T \implies S = V(\Sigma^2 / N)V^T \implies \lambda_j = s_j^2 / N$.

### 3.3. Módulo 3: Fisher LDA y Criterio de Fisher (Diaps. 38–49)
* **Criterio de Fisher:** $J(w) = \frac{(w^T(m_2-m_1))^2}{w^T S_W w} \implies w^* \propto S_W^{-1}(m_2 - m_1)$.
* **Ejemplo trabajado 8 puntos (Diaps. 42–44):** $m_1 = (2,4), m_2 = (5,5) \implies m_2 - m_1 = (3, 1)$, $S_W = \text{diag}(4, 36)$.
  * Vector óptimo: $w^* \propto [3/4, 1/36]^T \propto \mathbf{[27, 1]^T} \implies w^* \approx (0.9993, 0.0370)$.
  * Comparación de separabilidad: $J(w^*_{\text{Fisher}}) = \mathbf{2.28}$ vs $J(v_{\text{PCA}}) = \mathbf{0.03}$ (**$82\times$ mejor separación supervisada**).
* **Segundo caso de Fisher (Diap. 49):** $m_2 - m_1 = (2,2), S_W = \text{diag}(2,8) \implies w^* \propto (4, 1)$.
* **Límite dimensional:** Rango máximo de $C-1$ dimensiones proyectadas para $C$ clases.

### 3.4. Módulo 4: Manifold Learning — t-SNE, UMAP y Pipeline HDBSCAN (Diaps. 50–73)
* **Probabilidades gaussianas con $\sigma$ adaptativo (Diaps. 56–58):** En $\{0, 1, 2, 8\}$, para $x=1$ con $\text{Perp}=2$, $\sigma = 1.0$; para el punto aislado $x=8$, $\sigma$ se expande a **$2.26$**.
* **Colas pesadas de la t de Student contra el Crowding Problem (Diap. 60):** En baja dimensión, $q_{ij} \propto (1 + d^2)^{-1}$. A $d=3$, Student-t vale $0.100$ vs $0.000123$ de la gaussiana (**$810\times$ más densa en la cola**).
* **Reglas de interpretación de Wattenberg et al. (2016) y Pipeline Moderno:** Jamás aplicar clustering sobre proyecciones 2D; reducir a 10D con UMAP y alimentar a HDBSCAN.

---

## 4. Entregables Generados y Modificados
1. [`reduccion_dimensionalidad.py`](../reduccion_dimensionalidad.py): Módulo autónomo en Python estándar puro con los 4 módulos interactivos y suite de tests unitarios (`--auto`).
2. [`perceptron_xor.py`](../perceptron_xor.py): Integración de CD07 en menú Fase 1 (opción `[7]`), flags CLI, y suite unificada de 13 unidades.
3. [`web/index.html`](../web/index.html): Botón `unitBtn07`, barra de pestañas `#unit07Tabs`, paneles `#panel-49` a `#panel-52`, 4 Canvas interactivos, simulación de distancias y Quizzes 49 a 52 (validación Node.js limpia).
4. [`README.md`](../README.md): Actualización del resumen a 13 unidades, 812 diapositivas y 53 laboratorios; sección detallada CD07.
5. [`INFORME.md`](../INFORME.md): Incorporación de la Sección 0.10 con desarrollos matemáticos exhaustivos, actualización de conclusiones y 6 nuevas referencias canónicas.
6. [`contexto/context19.md`](context19.md): Este registro de preservación de contexto y memoria de proyecto.

---

## 5. Referencias
* **Path del plan original:** `/Users/alexanderacosta/.gemini/antigravity/brain/c8f17cb7-96ad-4898-8b55-e7d88fe53835/.user_uploaded/media_1794330734917_7608804a.pdf` (`CD07-Reducción-de-Dimensionalidad.pdf`).
