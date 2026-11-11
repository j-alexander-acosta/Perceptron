# Contexto 20 — Integración de la Unidad CD08: Modelos de Árboles de Clasificación y Regresión (68 Diapositivas)
**Fecha de registro:** 2026-11-11

## 1. Instrucciones y Directrices
* **Archivo fuente cargado:** `/Users/alexanderacosta/.gemini/antigravity/brain/29ebebe9-b2a4-4d43-8309-7fbcb91ee63b/.user_uploaded/media_1794419540062_1fe1822f.pdf` (`CD08-Árboles-de-Clasificación-y-Regresión.pdf`).
* **Ubicación en el plan de estudios:** Octava unidad en orden estricto de cátedra (culminación de la Fase 1):
  $$\text{CD01} \to \text{CD02} \to \text{CD03} \to \text{CD04} \to \text{CD05} \to \text{CD06} \to \text{CD07} \to \mathbf{\text{CD08 (Árboles CART/ID3)}} \to \text{CD13} \to \dots \to \text{CD18}$$
* **Compatibilidad técnica:** Implementación en Python estándar puro sin dependencias externas obligatorias para CLI (`arboles_decision.py`), integración en `perceptron_xor.py` (opción `[8]`, banderas CLI `--cd08`, `--trees`, `--cart`, `--arboles`, suite automática `--auto`), suite web interactiva en `web/index.html` con renderizado Canvas nativo y Quizzes 53 a 56, y documentación formal en `INFORME.md` (Sección 0.11).
* **Totales del proyecto actualizados:** 14 Unidades Académicas • 880 Diapositivas • 57 Laboratorios Interactivos.

---

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Implementar, simular, verificar y documentar exhaustivamente todas las formulaciones matemáticas, trazas numéricas a mano, teoremas y experimentos controlados de las 68 diapositivas (305 páginas de transición) de la Unidad CD08 (Modelos de Árboles de Clasificación y Regresión).
* **Fases e hitos principales:**
  1. **Análisis exhaustivo del documento:** 4 Módulos temáticos extraídos de las 68 diapositivas:
     - Módulo 08.1: Particionado Ortogonal, Complejidad Booleana $2^{2^n}$, NP-Completitud (Hyafil & Rivest, 1976) y Trampa Categórica Nominal.
     - Módulo 08.2: Criterios de Impureza (Insensibilidad del Error vs Concavidad Estricta de Gini y Entropía) y Algoritmo ID3 a Mano (Dataset 10 días "¿Salgo a correr?").
     - Módulo 08.3: Control de Complejidad, Miopía del Early Stopping y Poda CART por Costo-Complejidad (Weakest Link Pruning y árbol de 100 puntos).
     - Módulo 08.4: CART para Regresión Continua (Reducción de Varianza MSE, dataset 6 puntos a mano) y Transición a Ensambles (Bagging/Random Forest/Boosting).
  2. **Módulo computacional Python (`arboles_decision.py`):**
     - Módulo 08.1: Simulación de frontera oblicua $x_2 = 0.8 x_1$ vs particionado ortogonal escalonado ($K \in \{1, 2, 4, 8\}$); explosión de funciones booleanas $2^{2^n}$ ($n=1\to 4, n=2\to 16, n=3\to 256, n=4\to 65\,536, n=7\to 3.4028\times 10^{38}$); demostración de la trampa del orden categórico sobre {Azul, Rojo, Amarillo}.
     - Módulo 08.2: Cálculo analítico de Error de Clasificación, Impureza de Gini y Entropía de Shannon; prueba de insensibilidad del error en el Caso 1 ($14A, 5B \to 6A, 0B$ y $8A, 5B$, $\Delta E=0.000, \Delta I_G=+0.0639, \Delta H=+0.1738$) y Caso 2 ($9A, 3B \to 6A, 0B$ y $3A, 3B$, $\Delta E=0.000, \Delta I_G=+0.1250$); ejecución paso a paso de ID3 en 10 días ($H(S)=1.0$, $IG(\text{Clima})=0.60$ gana como raíz; en rama Soleado, $IG(\text{Viento})=1.0$ partición perfecta Suave $\to$ Sí, Fuerte $\to$ No).
     - Módulo 08.3: Demostración de la miopía de parada temprana por ganancia mínima (falla en paridad/XOR); cálculo del eslabón más débil $g(t) = \frac{R(t) - R(T_t)}{|T_t| - 1}$ en el árbol de 100 puntos de la cátedra ($T_0: 3\text{ hojas}, R=0.14$; $x_2<3$ colapsa con $g=0.040 \to T_1: 2\text{ hojas}, R=0.18$; raíz colapsa con $g=0.320 \to T_2: 1\text{ hoja}, R=0.50$; para $\alpha=0.10 \implies C(T_1)=0.38 < C(T_0)=0.44 < C(T_2)=0.60$).
     - Módulo 08.4: Formulación continua CART con $\hat{y}_m = \bar{y}_m$ y minimización de varianza cuadrática; evaluación de cortes en el dataset de 6 puntos ($x=[1..6], y=[2.0, 3.0, 2.5, 8.0, 9.0, 8.5]$, $x<3.5$ produce $\text{MSE}=0.167$ vs $x<2.5$ con $4.667$ y $x<4.5$ con $3.875$); progresión de profundidad (Prof 1 $\to$ MSE 0.167, Prof 2 $\to$ 0.062, Prof 3 $\to$ 0.000 memorización total); análisis de sesgo-varianza y puente a ensambles.
     - Suite `--auto` con 10 tests de aserción formal aprobados al 100%.
  3. **Integración en orquestador central (`perceptron_xor.py`):**
     - Añadido `CD08` a `UNIDADES_CATEDRA` en índice 7 (num 8), actualizando la lista a 14 unidades.
     - Menú Fase 1 actualizado a 8 unidades (`[1-8, T, V]`), con opción 8 ejecutando `launch_trees_module()`.
     - Menú Fase 2 reindexado a índices 8..13.
     - Flags CLI: `--trees`, `-c8`, `--cd08`, `--cart`, `--arboles`, `--tree`.
     - Suite `--auto` actualizada para correr los tests de `arboles_decision.py` como paso 8 de 14.
  4. **Suite Web Interactiva (`web/index.html`):**
     - Estilos CSS esmeralda / verde bosque (`#10b981`, `.active-unit-08`, `.tab-btn.unit08-tab.active`, `.unit-pill.u08`).
     - Botón `unitBtn08` en selector de Fase 1.
     - Navegación `#unit08Tabs` (Labs 53 a 56).
     - Paneles `#panel-53` a `#panel-56` con controles interactivos, 4 Canvas visualizadores (`drawCd08Partitions`, `drawCd08Impurity`, `drawCd08Pruning`, `drawCd08Regression`) y Quizzes oficiales 53 a 56.
     - Actualizado `topicMetadata` (longitud 57), `ORDERED_UNITS`, `ORDERED_TABS`, `filterPhase`, `selectUnit`, `switchTab` y `quizData`.
     - Validación sintáctica mediante Node.js aprobada al 100% (833.958 caracteres, 0 errores sintácticos).
  5. **Documentación formal:**
     - `INFORME.md`: Sección 0.11 redactada con formulaciones analíticas, demostraciones de concavidad/insensibilidad, trazas completas a mano, actualización de conclusiones y agregado de 4 referencias bibliográficas canónicas (Breiman et al. 1984, Quinlan 1986, Hyafil & Rivest 1976, Hastie et al. 2009).
     - `README.md`: Resumen global actualizado a 14 unidades, 880 diapositivas, 57 laboratorios interactivos; sección temática para CD08; comandos CLI y diagrama de estructura de archivos.
     - `contexto/context20.md`: Registro estructurado de la sesión conforme al skill `almacenamiento`.

---

## 3. Síntesis Teórica y Numérica Implementada

### 3.1. Módulo 1: Particionado Ortogonal, Complejidad Booleana y NP-Completitud (Diaps. 1–19)
* **Particionado Ortogonal de CART (Diaps. 3–5):**
  Los árboles de decisión solo producen cortes axiales $x_j \le t$. Para aproximar una frontera lineal oblicua simple ($x_2 = 0.8 x_1$), el árbol requiere una escalera de hiperplanos ortogonales.
* **Explosión Combinatoria $2^{2^n}$ (Diaps. 6–10):**
  Para $n$ variables binarias hay $2^n$ filas en la tabla de verdad y $2^{2^n}$ funciones booleanas posibles:
  $$n=1 \to 4, \quad n=2 \to 16, \quad n=3 \to 256, \quad n=4 \to 65\,536, \quad n=7 \to \mathbf{3.4028 \times 10^{38}}$$
* **Teorema de Hyafil & Rivest (1976, Diap. 11):**
  Construir el árbol binario óptimo es **NP-completo**, obligando a usar algoritmos voraces (*greedy heuristics*).
* **Trampa del Orden Categórico (Diaps. 17–18):**
  Ordenar nominales $\{1, 2, 3\}$ impide aislar $\{C_1, C_3\}$ de $\{C_2\}$. CART evalúa las $2^{K-1}-1$ particiones binarias disjuntas.

### 3.2. Módulo 2: Criterios de Impureza y Algoritmo ID3 a Mano (Diaps. 20–39)
* **Insensibilidad del Error de Clasificación:**
  * Caso 1 (Diap. 24): Padre $14A, 5B$ ($N=19, E=5/19$) dividido en $6A, 0B$ (puro) y $8A, 5B$.
    $$E_{\text{pond}} = \frac{6}{19}(0) + \frac{13}{19}\left(\frac{5}{13}\right) = \frac{5}{19} \implies \mathbf{\Delta E = 0.000}$$
    $$\mathbf{\Delta I_G = +0.0639}, \quad \mathbf{\Delta H = +0.1738\text{ bit}}$$
  * Caso 2 (Diap. 26): Padre $9A, 3B$ ($N=12$) dividido en $6A, 0B$ y $3A, 3B$. $\mathbf{\Delta E = 0.000}$, mientras $\mathbf{\Delta I_G = +0.1250}$.
  * Causa: concavidad estricta (desigualdad de Jensen) de Gini y Entropía frente a la linealidad a trozos del error.
* **Trazado de ID3 en 10 Días (Harvard CS109, Diaps. 31–39):**
  $H(S) = 1.000\text{ bit}$. Raíz: Clima $IG = \mathbf{0.600}$ (ganador) vs Viento $0.029$ y Temp $0.029$. En rama Soleado (2 Sí, 2 No): Viento genera $IG = \mathbf{1.000\text{ bit}}$ dividiendo perfectamente en Suave $\to$ Sí y Fuerte $\to$ No.

### 3.3. Módulo 3: Control de Complejidad y Poda CART (Diaps. 40–54)
* **Miopía de Parada Temprana:** Parar por $\Delta I < \varepsilon$ se detiene prematuramente en problemas tipo XOR/Paridad con $50\%$ de error porque el primer corte no produce ganancia inmediata.
* **Weakest Link Pruning:** Minimiza $C_\alpha(T) = R(T) + \alpha |T|$ colapsando el nodo interno con menor eslabón:
  $$g(t) = \frac{R(t) - R(T_t)}{|T_t| - 1}$$
* **Árbol de 100 Puntos (Diap. 48):**
  * $T_0$ ($3$ hojas, $R=0.14$).
  * Nodo $x_2 < 3$ con $g(t) = 0.040 \implies T_1$ ($2$ hojas, $R=0.18$).
  * Raíz con $g=0.320 \implies T_2$ ($1$ hoja, $R=0.50$).
  * Para $\alpha=0.10 \implies C(T_1) = 0.38 < C(T_0) = 0.44 < C(T_2) = 0.60$.

### 3.4. Módulo 4: CART para Regresión y Ensambles (Diaps. 55–68)
* Predicción en hoja: media muestral local $\hat{y}_m = \bar{y}_m$.
* Criterio voraz: minimización de la varianza ponderada (suma de errores cuadráticos).
* **Dataset de 6 Puntos (Diap. 57):** $x=[1..6], y=[2.0, 3.0, 2.5, 8.0, 9.0, 8.5]$.
  * $x < 2.5 \implies \text{MSE} = 4.667$.
  * $x < 3.5 \implies \text{MSE} = \mathbf{0.167}$ (**¡Partición Óptima Global!**).
  * $x < 4.5 \implies \text{MSE} = 3.875$.
* Progresión de profundidad: Prof 1 ($\text{MSE}=0.167$), Prof 2 ($0.062$), Prof 3 ($\text{MSE}=0.000$, memorización).
* Conexión con ensambles: Bagging / Random Forest para reducir varianza, Boosting para reducir sesgo.

---

## 4. Entregables Generados y Modificados
1. [`arboles_decision.py`](file:///Users/alexanderacosta/Documents/Doctorado%20UCN/Semestre%202%202026/Ciencia%20de%20Datos/Perceptron/arboles_decision.py): Módulo autónomo en Python estándar puro con los 4 submódulos interactivos y suite de tests unitarios (`--auto`).
2. [`perceptron_xor.py`](file:///Users/alexanderacosta/Documents/Doctorado%20UCN/Semestre%202%202026/Ciencia%20de%20Datos/Perceptron/perceptron_xor.py): Integración de CD08 en menú Fase 1 (opción `[8]`), flags CLI, y suite unificada de 14 unidades.
3. [`web/index.html`](file:///Users/alexanderacosta/Documents/Doctorado%20UCN/Semestre%202%202026/Ciencia%20de%20Datos/Perceptron/web/index.html): Botón `unitBtn08`, navegación `#unit08Tabs`, paneles `#panel-53` a `#panel-56`, 4 Canvas interactivos, simulación y Quizzes 53 a 56 (validación sintáctica Node.js aprobada).
4. [`README.md`](file:///Users/alexanderacosta/Documents/Doctorado%20UCN/Semestre%202%202026/Ciencia%20de%20Datos/Perceptron/README.md): Actualización del resumen a 14 unidades, 880 diapositivas y 57 laboratorios; sección detallada CD08.
5. [`INFORME.md`](file:///Users/alexanderacosta/Documents/Doctorado%20UCN/Semestre%202%202026/Ciencia%20de%20Datos/Perceptron/INFORME.md): Incorporación de la Sección 0.11 con desarrollos matemáticos exhaustivos, actualización de conclusiones y 4 nuevas referencias canónicas.
6. [`contexto/context20.md`](file:///Users/alexanderacosta/Documents/Doctorado%20UCN/Semestre%202%202026/Ciencia%20de%20Datos/Perceptron/contexto/context20.md): Registro histórico del plan de trabajo y contexto técnico.

---

## 5. Referencias
* **Path del plan original:** `/Users/alexanderacosta/.gemini/antigravity/brain/29ebebe9-b2a4-4d43-8309-7fbcb91ee63b/.user_uploaded/media_1794419540062_1fe1822f.pdf` (`CD08-Árboles-de-Clasificación-y-Regresión.pdf`).
