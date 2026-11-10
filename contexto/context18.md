# Contexto 18 — Integración de la Unidad CD06: Clustering — Agrupamiento No Supervisado (57 Diapositivas)
**Fecha de registro:** 2026-11-10

## 1. Instrucciones y Directrices
* **Archivo fuente cargado:** `/Users/alexanderacosta/.gemini/antigravity/brain/c8f17cb7-96ad-4898-8b55-e7d88fe53835/.user_uploaded/media_1794312880174_425df60f.pdf` (`CD06-Clustering.pdf`).
* **Ubicación en el plan de estudios:** Sexta unidad en orden estricto de cátedra:
  $$\text{CD01} \to \text{CD02} \to \text{CD03} \to \text{CD04} \to \text{CD05} \to \mathbf{\text{CD06 (Clustering)}} \to \text{CD13} \to \dots \to \text{CD18}$$
* **Compatibilidad técnica:** Implementación en Python estándar puro sin dependencias externas obligatorias para CLI (`clustering.py`), integración en `perceptron_xor.py` (opción `[6]`, suite automática y banderas CLI), suite web interactiva en `web/index.html` con renderizado Canvas nativo, y documentación formal en `INFORME.md` (Sección 0.9).
* **Totales del proyecto actualizados:** 12 Unidades Académicas • 739 Diapositivas • 49 Laboratorios Interactivos.

---

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Implementar, simular, verificar y documentar exhaustivamente todas las formulaciones matemáticas, trazas numéricas a mano, teoremas y experimentos controlados de las 57 diapositivas (246 páginas de transición) de la Unidad CD06 (Clustering).
* **Fases e hitos principales:**
  1. **Análisis exhaustivo del documento:** 4 Módulos temáticos extraídos de las 57 diapositivas (K-Means & Lloyd, Rand Index & ARI, Jerárquico Aglomerativo & GMM, Densidad DBSCAN & HDBSCAN).
  2. **Módulo computacional Python (`clustering.py`):** Implementación rigurosa de las trazas a mano 1D y 2D, simulación de 210 puntos en 3 grupos con K-Means++ vs trampa de mínimo local, curva oficial del codo, sensibilidad a outliers, conteo combinatorial de pares $\binom{n}{2}$, índice de Rand crudo vs ARI de Hubert-Arabie, dendrogramas Single vs Complete Linkage demostrando la imposibilidad de $K=3$ por empate, DBSCAN paso a paso demostrando colapso de minPts=3 a 4, experimento de las Dos Lunas (K-Means vs DBSCAN) y dilema de densidades heterogéneas vs HDBSCAN.
  3. **Integración en orquestador central (`perceptron_xor.py`):** Inclusión en `UNIDADES_CATEDRA` (índice 5), actualización de menú Fase 1, Fase 2, banderas CLI (`--cd06`, `--cluster`, `-c6`, `--kmeans`, `--dbscan`), suite automática `--auto` (12/12 suites aprobadas al 100%).
  4. **Suite Web Interactiva (`web/index.html`):** Inserción de estilos CSS verde lima (`#84cc16`), botón `unitBtn06`, navegación `#unit06Tabs` (Labs 45 a 48), paneles `#panel-45` a `#panel-48`, Canvas interactivos y Quizzes oficiales 45 a 48 con validación sintáctica Node.js aprobada.
  5. **Documentación formal:** Redacción de la Sección 0.9 en `INFORME.md`, actualización de totales en `README.md` e `INFORME.md`, agregado de referencias canónicas (Lloyd 1982, Rand 1971, Hubert & Arabie 1985, Ester et al. 1996, Arthur & Vassilvitskii 2007, Campello et al. 2013) y registro en `contexto/context18.md`.

---

## 3. Síntesis Teórica y Numérica Implementada

### 3.1. Módulo 1: K-Means, Minimización Alternada de Lloyd y Codo (Diaps. 1–21)
* **Doble naturaleza de K-Means (Diap. 4):**
  $$\text{Loss}_{\text{kmeans}}(z, \mu) = \frac{1}{n}\sum_{i=1}^n \|\phi(x_i) - \mu_{z_i}\|_2^2$$
  Lloyd (1982) alterna optimización por bloques: asignación $z_i \leftarrow \arg\min_k \|\phi(x_i) - \mu_k\|^2$ y recentrado $\mu_k \leftarrow \frac{1}{|C_k|}\sum_{i \in C_k}\phi(x_i)$.
* **Teorema de convergencia:** Cada subpaso reduce o mantiene la pérdida monótonamente en espacio de estados finito $K^n$, garantizando convergencia a un mínimo local en pasos finitos. No garantiza el óptimo global.
* **Traza 1D exacta (Diap. 12):** $\{0, 2, 10, 12\}$ con $\mu=[0, 2]$. Iter 1: $z=[1,2,2,2], \mu=[0, 8]$. Iter 2: $z=[1,1,2,2], \mu=[1, 11]$. Converge en 2 pasos con $\text{Loss} = \mathbf{4.0}$.
* **Paso 2D a mano exacto (Diap. 15):** $(0,0), (2,0), (6,4), (8,4)$ con $\mu_1=(0,0), \mu_2=(2,0)$. Iter 1: $\mu_1=(0,0), \mu_2=(16/3, 8/3) \approx (5.33, 2.67)$ con $\text{Loss}=26.67$. Iter 2: $\mu_1=(1,0), \mu_2=(7,4)$ con $\text{Loss} = \mathbf{4.0}$.
* **Simulación de 210 puntos en 3 grupos (Diap. 18):** Mínimo local severo ($\text{Loss} \approx 1\,450$) frente a K-Means++ que alcanza el óptimo global ($\text{Loss} \approx 328$).
* **Curva oficial del codo (Diap. 21):** Para $K=1..8$: $[1863.2, 955.1, \mathbf{107.5}, 90.1, 77.9, 66.1, 58.4, 50.7]$. Quiebre neto en $K=3$ ($\Delta = 847.6$ vs $\Delta = 17.4$ para $K=4$).
* **Sensibilidad a Outliers:** Con $\{0, 2, 10, 12, 30\}$, el punto $30$ secuestra $\mu_2=30.0$, colapsando $\mu_1=6.0$ ($\text{Loss}=104.0$).

### 3.2. Módulo 2: Evaluación de Clusters: Rand Index (RI), Rand Ajustado (ARI) y Ruido (Diaps. 22–30)
* **Matriz de contingencia de pares $\binom{n}{2}$ (Diap. 24):** Conteo de $a$ (acuerdo positivo), $b$ (acuerdo negativo), $c$ (falso negativo), $d$ (falso positivo).
* **Rand Index:** $\text{RI} = \frac{a + b}{\binom{n}{2}}$.
* **Demostración de inflación de base (Diaps. 25–26):** En $n=6$, dar 1 solo cluster produce $\text{RI} = 6/15 = \mathbf{0.40}$; dar clusters individuales da $\text{RI} = 9/15 = \mathbf{0.60}$. En ambos casos el $\text{ARI} = \mathbf{0.0000}$.
* **Rand Ajustado de Hubert-Arabie (1985, Diap. 27):** Descuenta el valor esperado bajo asignación aleatoria. Para 6 puntos con 1 error: $a=4, b=6, c=2, d=3 \implies \text{RI} = 10/15 \approx \mathbf{0.67}$, corregido a $\text{ARI} = \mathbf{0.3243}$.

### 3.3. Módulo 3: Agrupamiento Jerárquico: Dendrogramas, Linkage y GMM (Diaps. 31–37)
* **Matriz de distancias 1D sobre $\{1, 2, 5, 9, 12\}$ (Diaps. 31–33):**
  * Single Linkage ($d_{\min}$): fusiones en $1.0, 3.0, 3.0, 4.0$. El empate en $h=3.0$ entre $\{1,2\}\text{ con }5$ y $9\text{ con }12$ hace **imposible obtener $K=3$ clusters** (salta de 4 a 2).
  * Complete Linkage ($d_{\max}$): fusiones en $1.0, 3.0, 4.0, 11.0$, aislando $K=3$ para $h \in [3.0, 4.0)$.
* **Gaussian Mixture Models (GMM, Diaps. 36–37):** Soft clustering mediante responsabilidades continuas $\gamma_{ik} = P(z_i = k \mid x_i) \in [0, 1]$ calculadas por el algoritmo Expectation-Maximization (EM).

### 3.4. Módulo 4: Basado en Densidad: DBSCAN, Dos Lunas, Densidades Heterogéneas y HDBSCAN (Diaps. 38–57)
* **DBSCAN (Ester et al., 1996):** Clasificación en Core ($|N_\varepsilon(p)| \ge \text{minPts}$), Border y Noise con alcanzabilidad y conectividad transitiva de densidad.
* **Ejemplo interactivo de 10 puntos (Diaps. 43–48):** Con $\varepsilon=1.05, \text{minPts}=3 \implies 7$ núcleos, 2 fronteras, 1 ruido. Con $\text{minPts}=4$, el arco entero colapsa a 5 puntos de ruido al tener $\le 3$ vecinos por punto.
* **Experimento de las Dos Lunas (Diap. 38):** K-Means falla cometiendo 31–33 errores ($\text{ARI} \approx 0.490$) al trazar una frontera lineal euclidiana; DBSCAN alcanza 0 errores, 0 ruido y $\text{ARI} = \mathbf{1.0000}$.
* **Dilema de densidades heterogéneas y HDBSCAN (Diaps. 50–57):** Un único $\varepsilon$ no puede reconciliar densidades dispares ($\varepsilon=0.25$ desintegra el disperso como ruido; $\varepsilon=0.60$ funde los dos compactos). HDBSCAN (Campello et al., 2013) utiliza la distancia de alcanzabilidad mutua $d_{\text{mreach}}$ y el árbol condensado por persistencia ($\lambda = 1/\varepsilon$).

---

## 4. Estado de Verificación del Sistema
1. **CLI Standalone (`clustering.py --auto`):**
   * Módulo 1 (K-Means 1D y 2D a mano, 210 pts, codo en K=3 y outliers): 100% aprobado.
   * Módulo 2 (Rand Index, patologías de inflación de base y ARI Hubert-Arabie): 100% aprobado.
   * Módulo 3 (Dendrogramas, Single vs Complete y empate K=3): 100% aprobado.
   * Módulo 4 (DBSCAN 10 pts, colapso minPts=4, Dos Lunas y densidades heterogéneas): 100% aprobado.
2. **Orquestador Central (`perceptron_xor.py --auto`):**
   * Verificación integral ejecutada de punta a punta: las 12 Unidades Académicas (CD01, CD02, CD03, CD04, CD05, CD06, CD13, CD14, CD15, CD16, CD17, CD18) compilan y pasan todos sus tests automáticos (100% de éxito).
3. **Suite Web Interactiva (`web/index.html`):**
   * 14,657 líneas de código HTML/CSS/JS.
   * Validación de sintaxis JavaScript aprobada mediante `node --check` (0 errores).
   * 49 Laboratorios Interactivos distribuidos en 12 Unidades temáticas con selectores visuales, canvas dinámicos y cuestionarios de autoevaluación (Quizzes 1 a 48).
4. **Documentación:**
   * `INFORME.md`: Sección 0.9 completa con desarrollos analíticos rigurosos y 27 referencias bibliográficas canónicas.
   * `README.md`: Actualizado con la descripción temática completa de CD06, comandos de ejecución y estructura de archivos.
   * `contexto/context18.md`: Registro completo del estado del proyecto.
