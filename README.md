# Suite Educativa de Ciencia de Datos y Redes Neuronales (20 Unidades Académicas • 1260 Diapositivas • 81 Laboratorios)

Este repositorio implementa de manera interactiva, visual y educativa los fundamentos teóricos y prácticos de las unidades del curso (*Ciencia de Datos — Doctorado UCN, basado en CMU 15-388/688 de Z. Kolter, Harvard CS109 de P. Protopapas et al., CMU 11-785 de B. Raj et al., Stanford CS221 de P. Liang et al., Hastie, Tibshirani & Friedman ESL, Vapnik SVM y UC Berkeley DS100*), **ordenadas estrictamente según el código secuencial del archivo de cátedra (`CD01`, ..., `CD12`, `CD13`–`CD20`)**:

* **CD01 (Primera en orden) — Introducción al Curso (63 Diapositivas):** Laboratorios 0.1 a 0.4 ([`introduccion_curso.py`](introduccion_curso.py)).
* **CD02 (Segunda en orden) — Preprocesamiento y Análisis Exploratorio de Datos (74 Diapositivas):** Laboratorios 02.1 a 02.4 ([`preprocesamiento_eda.py`](preprocesamiento_eda.py)).
* **CD03 (Tercera en orden) — Visualización de Datos Efectiva y Engañosa (103 Diapositivas):** Laboratorios 03.1 a 03.4 ([`visualizacion_datos.py`](visualizacion_datos.py)).
* **CD04 (Cuarta en orden) — Introducción a Machine Learning (56 Diapositivas):** Laboratorios 04.1 a 04.4 ([`introduccion_ml.py`](introduccion_ml.py)).
* **CD05 (Quinta en orden) — Generalización y Validación (63 Diapositivas):** Laboratorios 05.1 a 05.4 ([`generalizacion_validacion.py`](generalizacion_validacion.py)).
* **CD06 (Sexta en orden) — Clustering: Agrupamiento No Supervisado (57 Diapositivas):** Laboratorios 06.1 a 06.4 ([`clustering.py`](clustering.py)).
* **CD07 (Séptima en orden) — Reducción de Dimensionalidad (73 Diapositivas):** Laboratorios 07.1 a 07.4 ([`reduccion_dimensionalidad.py`](reduccion_dimensionalidad.py)).
* **CD08 (Octava en orden) — Modelos de Árboles de Clasificación y Regresión (68 Diapositivas):** Laboratorios 08.1 a 08.4 ([`arboles_decision.py`](arboles_decision.py)).
* **CD09 (Novena en orden) — Ensamblaje de Modelos y Random Forests (59 Diapositivas):** Laboratorios 09.1 a 09.4 ([`ensambles_random_forest.py`](ensambles_random_forest.py)).
* **CD10 (Décima en orden) — Modelos No Lineales y Support Vector Machines (65 Diapositivas):** Laboratorios 10.1 a 10.4 ([`modelos_no_lineales.py`](modelos_no_lineales.py)).
* **CD11 (Undécima en orden — Cierre Práctico Fase 1) — Aspectos Prácticos y Evaluación de Modelos (93 Diapositivas):** Laboratorios 11.1 a 11.4 ([`aspectos_practicos.py`](aspectos_practicos.py)).
* **CD12 (Duodécima en orden — Puente Histórico y Conceptual) — Redes Neuronales y Conexionismo (60 Diapositivas):** Laboratorios 12.1 a 12.4 ([`redes_conexionismo.py`](redes_conexionismo.py)).
* **CD13 (Unidad I) — Representación y Aproximación Universal (61 Diapositivas):** Laboratorios 1.1 a 1.5 ([`aproximacion_universal.py`](aproximacion_universal.py)).
* **CD14 (Unidad II) — Aprendizaje de Redes Neuronales (56 Diapositivas):** Laboratorios 2.1 a 2.4 ([`aprendizaje_redes.py`](aprendizaje_redes.py)).
* **CD15 (Unidad III) — Descenso por Gradiente y Convergencia (47 Diapositivas):** Laboratorios 3.1 a 3.4 ([`descenso_gradiente.py`](descenso_gradiente.py)).
* **CD16 (Unidad IV) — Algoritmo de Backpropagation (60 Diapositivas):** Laboratorios 4.1 a 4.4 ([`backpropagation.py`](backpropagation.py)).
* **CD17 (Unidad V) — Problemas de Convergencia en el Entrenamiento (58 Diapositivas):** Laboratorios 5.1 a 5.4 ([`problemas_convergencia.py`](problemas_convergencia.py)).
* **CD18 (Unidad VI) — Descenso por Gradiente Estocástico y Lotes (41 Diapositivas):** Laboratorios 6.1 a 6.4 ([`sgd_lotes.py`](sgd_lotes.py)).
* **CD19 (Unidad VII) — Regularización de Redes Neuronales (46 Diapositivas):** Laboratorios 7.1 a 7.4 ([`regularizacion_redes.py`](regularizacion_redes.py)).
* **CD20 (Unidad VIII) — Redes Neuronales Convolucionales (CNNs) (57 Diapositivas):** Laboratorios 8.1 a 8.4 ([`redes_convolucionales.py`](redes_convolucionales.py)).

> 📄 **Informe Académico Completo:** Consulta el desarrollo matemático y las demostraciones en [`INFORME.md`](INFORME.md).

---

## 🏛️ Organización Temática por Unidades (Orden de Cátedra `CD01` → `CD20`, 20 Unidades)

### indigo CD01 (UNIDAD INICIAL — PRIMERA EN ORDEN): Introducción al Curso de Ciencia de Datos

1. **Módulo 0.1 — Caso Real 1: Ciencia de Datos en la Pandemia, Sesgo en la Tasa de Letalidad y Causalidad (Diaps. 5–11 y 19–21):**
   * Los 5 pasos del proceso de Ciencia de Datos aplicados a la pandemia de COVID-19 y la clasificación de las 4 preguntas clave: *¿Qué tan rápido se propaga?* ($\to$ **modelamiento**), *¿Cuándo se saturarán los hospitales?* ($\to$ **predicción**), *¿Funcionan las cuarentenas?* ($\to$ **inferencia causal**) y *¿Cómo distribuir vacunas con equidad?* ($\to$ **optimización**), junto con el carácter ético de elegir la métrica de éxito (Flaxman et al., 2020, *Nature*, Diap. 7).
   * Sesgos de captura en testeo PCR/antígenos, movilidad celular (Google/Apple) y registros civiles (Diap. 8).
   * **El sesgo en acción — Tasa de letalidad aparente vs. real (Diap. 9):** Con $100$ muertes y $2\,000$ casos confirmados (solo sintomáticos), la letalidad aparente es $\text{CFR} = 100/2\,000 = \mathbf{5.0\%}$, mientras que con $20\,000$ infecciones reales la letalidad real es $\text{IFR} = 100/20\,000 = \mathbf{0.5\%}$ (**inflada $10$ veces** por el denominador sesgado).
   * **Ejercicio del Toque de Queda Nocturno (Diap. 21):** Inferencia causal, sub-representación de quienes no tienen *smartphone* en Google Mobility y el **miedo al virus** como variable de confusión que reduce movilidad y contagios simultáneamente.
2. **Módulo 0.2 — El Modelo Epidemiológico SIR a Mano, $R_0 = \beta/\gamma$ y «Aplanar la Curva» (Diaps. 12–18):**
   * Ecuaciones en diferencias diarias sobre $N = S + I + R = 1\,000$: $\text{contagios}_t = \beta S_t I_t / N$ y $\text{recuperaciones}_t = \gamma I_t$.
   * **Reproducción exacta de la tabla «La dinámica SIR, a mano» (Diap. 13)** con $\beta = 0.3, \gamma = 0.1, S_0 = 990, I_0 = 10, R_0 = 0$: Día 0 ($[2.97, 1.00, 990.0, 10.0, 0.0]$), Día 1 ($[3.54, 1.20, 987.0, 12.0, 1.0]$), Día 2 ($[4.22, 1.43, 983.5, 14.3, 2.2]$) y **Día 28 (pico)** ($[30.24, 31.59, 319.0, \mathbf{315.9}, 365.0]$), con cerca del **$95\%$** de la población infectada al final de la epidemia (Diap. 14).
   * **Efecto de las cuarentenas sobre $R_0 = \beta/\gamma$ (Diap. 15):** Reducir $\beta$ de $0.3$ ($R_0 = 3.0$) a $0.16$ ($R_0 = 1.6$) baja el máximo de infectados simultáneos de $\approx \mathbf{316}$ (día 28) a $\approx \mathbf{89}$ (día 61), más la lección del modelo IHME (Diaps. 16–18).
3. **Módulo 0.3 — ¿Qué es Ciencia de Datos?, El Caso Netflix Prize (2006–2009) y la Paradoja de la Sobre-optimización (Diaps. 22–40):**
   * Espectro Máquina $\leftrightarrow$ Humano (Daniel Keim, *Visual Analytics*) y el ciclo iterativo de 5 etapas (Recolección $\to$ Procesamiento $\to$ Exploración y visualización $\to$ Análisis y ML $\to$ Conocimiento y decisiones), incluyendo los dos casos donde el producto final no es un modelo complejo: la visualización de esperanza de vida en *Our World in Data* (Diap. 39) y el mapeo de pobreza mediante el material del techo como dato sustituto o *proxy* (Abelson, Varshney y Sun, KDD 2014, Diap. 40).
   * **El vacío en la matriz del Netflix Prize (Diaps. 27–28):** $480\,189 \text{ usuarios} \times 17\,770 \text{ películas} = \mathbf{8\,532\,958\,530} \approx \mathbf{8\,533\text{ millones}}$ de celdas, con $\mathbf{100\,480\,507}$ calificaciones observadas ($\approx \mathbf{1.18\% \approx 1.2\%}$ observado; **falta casi el $99\%$** con datos faltantes no al azar, MNAR).
   * **Tres años de carrera y la confesión de 2012 (Diaps. 30–34):** De Cinematch ($\text{RMSE} = 0.951$) a *BellKor's Pragmatic Chaos* ($+10.06\%, \text{RMSE} = 0.856$, ganando por 20 minutos) y por qué el ensamble de $>100$ modelos **nunca se desplegó en producción** (paradoja de la sobre-optimización).
4. **Módulo 0.4 — Los Tres Relatos de la IA, el Mapa de 4 Niveles y la Política de Uso de LLMs (Diaps. 1–4 y 41–63):**
   * El Test de Turing (1950) y los 3 relatos históricos: **Relato simbólico** (1956–1987, Dartmouth, traducción rusa e informe ALPAC 1966 $\to$ 1.er invierno; sistemas expertos $\to$ 2.º invierno en 1987), **Relato neuronal** (1943–hoy, McCulloch-Pitts, Hebb, Rosenblatt, Minsky y Papert 1969, Backprop 1986, AlexNet 2012 impulsado por datos y cómputo) y **Relato estadístico** (1801–hoy, Gauss y Legendre con la órbita de Ceres, Fisher, Bellman, Vapnik).
   * **Mapa de 4 Niveles de Modelos en IA (Diaps. 3 y 50):** Reflejos (se recorre entero), Estados (búsqueda, MDP y refuerzo, sin juegos), Variables (modelos probabilísticos, sin CSP) y Lógica (fuera del curso).
   * **Política de Uso de LLMs (Diaps. 58–61, UC Berkeley DS100):** Desglose en 3 partes (Contexto, Planificación, Ejecución), evaluación de los 4 casos prácticos y autoevaluación de la unidad CD01.

---

### 🟩 CD02 (SEGUNDA EN ORDEN): Preprocesamiento y Análisis Exploratorio de Datos (74 Diapositivas)

1. **Módulo 02.1 — ¿Qué son los Datos?, Advertencias de Fábrica, Alcances y Sesgos de Muestreo (Diaps. 1–20):**
   * Nos instalamos en los **pasos 2 y 3 del proceso de 5 pasos: obtener y explorar** (los que consumen la mayor parte del tiempo real de un proyecto).
   * Tres definiciones de «datos» (Diccionario, Informática y **Señales**: salida de un sensor que mezcla información útil, irrelevante y redundante) y distinción entre *datum*, *data* y *dataset* (colección **homogénea**).
   * **Diagnóstico por procedimiento de registro (Diaps. 7–8 y 11):** *Termómetro horario / GPS de transporte* $\to$ **vacíos**; *Diario del colibrí / Cuaderno de asistencia* $\to$ **imprecisión humana**; *Tweets de un político / Biografías de red social* $\to$ **veracidad** (decir no es ser); *Sensor de feria persa / Columna `v37` sin documentar* $\to$ **sin semántica**.
   * **Muestreo diseñado (Gallup) vs. Autoselección (Yelp y Encuesta QR de la Cafetería, Diaps. 16–20):** Gallup (~500 llamadas/día RDD, regla del cumpleaños más reciente, 50 estados con 70% celulares / 30% fijos y ponderación demográfica $\to$ describe a la **población general**) frente a Yelp (autoselección sin corrección $\to$ describe **a quien opina en Yelp**) y la encuesta QR de la cafetería (480 respuestas, media **2.1/5**, sesgos de **autoselección + cobertura**: no mide satisfacción general, pero sí sirve para identificar qué reclama quien está descontento).
2. **Módulo 02.2 — Matriz 2×2 de Formatos, Preguntas Medibles, CRISP-DM vs. KDD/SEMMA y Web Scraping DOM (Diaps. 21–44):**
   * **Matriz 2×2 según el lector (Personas vs. Computadores, Diap. 21):** Fácil/Fácil = **Tabla CSV** (ojo con comas dentro de comillas como `"Newcastle, England"`, Diap. 23); Fácil/Difícil = **Prosa, fotos**; Difícil/Fácil = **JSON anidado, QR**; Difícil/Difícil = **Audio con ruido**.
   * **De titular vago a pregunta medible (Diaps. 27–29):** Fijar **población + métrica + ventana temporal** (participación presidencial en Chile 2013–2021 con datos del Servel, y promedio de libros leídos al año por jóvenes de 18 a 29 años en Chile entre 2014 y 2023).
   * **CRISP-DM (2000) frente a KDD (1996) y SEMMA (SAS) (Diaps. 30–35):** Las 6 fases de CRISP-DM y por qué KDD y SEMMA omiten las dos puntas críticas (**Comprensión del negocio** al inicio y **Despliegue** al final).
   * **Obtención de datos y Web Scraping Responsable (Diaps. 36–44):** Orden de preferencia $\text{Archivo} > \text{API} > \text{Scraping}$, las 5 reglas éticas/técnicas y verificación de `page.status_code == 200` (vs `404`) antes de extraer con `BeautifulSoup`.
3. **Módulo 02.3 — Pandas Esencial y Caso Real de EDA: Premier League 2017 (`epl_mini`, 23 jugadores) (Diaps. 45–70):**
   * Operaciones en `df2`: `Series` vs `DataFrame`, máscaras booleanas, `loc` (etiqueta) vs `iloc` (posición), el trío de caza de extremos (`min`, `idxmin`, `loc`) y las 3 correcciones de Pandas moderno (`.values` sin paréntesis, `.get_group()` en singular y `pd.concat` en lugar de `.append()`).
   * **Reproducción 100% exacta de `epl_mini.csv` (`shape = (23, 5)`, Diaps. 46, 52, 60, 63–70):**
     - Promedios por club (`groupby("club")["market_value"].mean()`, Diap. 52): `Arsenal = 32.80`, `Chelsea = 35.01`, `Liverpool = 34.00`, `Man City = 46.67`, `Man United = 45.75`, `Tottenham = 41.25`.
     - `describe()` global (Diap. 63): `age` (`mean = 27.13, std = 3.99, min = 21, 50% = 26, max = 35`) y `market_value` (`mean = 38.87, std = 23.51, min = 0.05, 50% = 40.00, max = 90.00`).
     - Caza de extremos y validación externa (Diaps. 64–66): `idxmin() -> 8` (`Eduardo`, Chelsea, 34 años, GK, `0.05M` $\implies$ dato **real**: *extremo $\neq$ error*) e `idxmax()` (`Paul Pogba`, Man United, 24 años, `90.0M` $\implies$ dato **coherente**), más la tabla `groupby("age")["market_value"].agg(["mean", "median"])` (Diap. 68).
4. **Módulo 02.4 — Media vs. Mediana en el Arsenal y el Cuarteto de Anscombe (1973) (Diaps. 71–74):**
   * **Sensibilidad en el Arsenal (Diaps. 60 y 72):** Los 5 jugadores (`[7, 20, 22, 50, 65]`) tienen media $\mathbf{32.80}$ y mediana $\mathbf{22.00}$; al simular un fichaje de $100\text{ M}$, la media salta a $\mathbf{44.00}$ mientras la mediana pasa a $\mathbf{36.00}$.
   * **El Cuarteto de Anscombe (1973, Diap. 71):** Verificación de los 4 conjuntos que comparten media ($\bar{x}=9.0, \bar{y}=7.50$), varianza ($s_x^2=11.0, s_y^2\approx 4.125$), correlación ($r\approx 0.816$) y recta OLS ($y = 3.00 + 0.50x$) con geometrías completamente distintas.

---

### 🟧 CD03 (TERCERA EN ORDEN): Visualización de Datos Efectiva y Engañosa (103 Diapositivas)

1. **Módulo 03.1 — Explorar vs. Comunicar, Datasaurus Dozen (CHI 2017), las 4 Escalas de Medición, Box Plot y Ancho de Intervalo (Diaps. 1–27 y 50–56):**
   * La visualización actúa en **dos pasos del ciclo a la vez**: **explorar** los datos propios (rápido, desechable, para ti: *descubrir qué es verdad*) y **comunicar** resultados a otros (pulido, con un mensaje: *convencer de que es verdad*).
   * **Datasaurus Dozen (Matejka y Fitzmaurice, CHI 2017, Diaps. 10–12):** **13 nubes de $n = 142$ puntos** (*dino*, *star*, *circle*, *x shape*, etc.) comparten idénticas estadísticas a dos decimales ($\bar{x} = \mathbf{54.26}, \bar{y} = \mathbf{47.83}, s_x = \mathbf{16.76}, s_y = \mathbf{26.93}, r = \mathbf{-0.06}$) porque $n$ puntos tienen muchos más grados de libertad que $m$ ecuaciones ($n > m$).
   * **Las 4 escalas de medición (Diaps. 13–14):** **Nominal** (`=, ≠`; ej. `club`), **Ordinal** (`=, ≠, <, >`; ej. `puesto en el ranking`), **Intervalo** (`+, -` con cero arbitrario; ej. °C, `año de debut`) y **Razón** (`÷` con cero absoluto; ej. Kelvin, `precio de mercado`).
   * **Cálculo exacto del Box Plot (Diap. 21):** En $\{2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 15\}$ ($n=11$), $\text{mediana}=5.0, Q_1=3.5, Q_3=7.5, \text{IQR}=4.0$ y la cerca superior es $Q_3 + 1.5\,\text{IQR} = \mathbf{13.5}$, marcando al $\mathbf{15}$ como **atípico**.
   * **Por qué NO dispersión 3D (Diaps. 24–25) y el ancho de intervalo en histogramas (Diap. 56):** La perspectiva 3D depende de la cámara y la oclusión borra el $78\%$ de un grupo oculto (sobrevive solo el **$22\%$ de su tinta**); en el histograma de $n = 200$ datos, *bins* de ancho $1.0$ muestran 3 barras planas (`[66, 68, 66]`, «uniforme») mientras que *bins* de ancho $0.2$ revelan **dos modos** (dos poblaciones mezcladas).
2. **Módulo 03.2 — Caso Clásico: Los Antibióticos de Will Burtin (1951) y el Gráfico que Corrigió a la Biología (Diaps. 28–38):**
   * **Tabla completa de 16 bacterias y 3 antibióticos (Burtin, 1951, *Scope*; Wainer y Lysen, 2009, Diap. 29):** Concentración inhibitoria mínima (**CIM**: menor = más potente) de penicilina, estreptomicina y neomicina a través de **6 órdenes de magnitud** (`0.001` a `870`), más tinción de Gram ($7$ Gram+ y $9$ Gram-).
   * **Pregunta 1 (del médico) — Medianas por grupo en escala logarítmica (Diap. 33):** En Gram positivas (`pen = 0.005, estr = 1.0, neo = 0.1`), la penicilina es **$200\times$ más potente** que la estreptomicina; en Gram negativas (`pen = 100.0, estr = 1.0, neo = 0.1`), gana la neomicina (`0.1`) y la penicilina es **$20\,000\times$ menos potente** que en Gram positivas.
   * **Pregunta 2 (del biólogo) — Dos etiquetas que no calzaban y que anticiparon la microbiología por décadas (Diaps. 34–36):** Al graficar $\log_{10}(\text{penicilina})$ vs. $\log_{10}(\text{neomicina})$, `Diplococcus pneumoniae` (`0.005, 10`) cae dentro del grupo de los estreptococos (reclasificada en **1974** como **`Streptococcus pneumoniae`**), mientras que `Streptococcus fecalis` (`1.0, 0.1`) requiere **$200\times$ a $1\,000\times$ más penicilina** (reclasificada en **1984** como **`Enterococcus faecalis`**).
3. **Módulo 03.3 — Integridad Gráfica (Tufte, 1983), Efectividad Perceptual (Stevens, Cleveland-McGill, Mackinlay) y Uso del Color (Diaps. 39–49 y 57–75):**
   * **Factor de Mentira (*Lie Factor*, Tufte, 1983, Diaps. 40–41):** $\text{LieFactor} = \frac{\text{efecto en el gráfico}}{\text{efecto en los datos}}$. En *If Bush Tax Cuts Expire* (Fox Business, 2012: $35\% \to 39.6\%$ con el eje truncado en $34\%$), la segunda barra es **$5.6\times$ más alta** ($+460\%$ visual vs. $+13.14\%$ real $\implies \text{LieFactor} = \mathbf{35.0}$). En la auditoría de las 4 marcas $A=52\%..D=56\%$ con eje en $50\%$ (Diap. 49), $D$ parece el triple de $A$ ($\text{LieFactor} = \mathbf{26.0}$).
   * **Ley de Potencias de Stevens (1961, Diaps. 61–62) y Rankings de Cleveland-McGill (1984) y Mackinlay (1986, Diaps. 63–64):** $\text{percibido} \propto I^\beta$. Solo la **longitud** ($\beta \approx 1.0 \implies 10\times$ se ve como $10.0\times$) es lineal; el **área** ($\beta \approx 0.7 \implies 10\times$ se ve como $5.0\times$) y el **brillo** ($\beta \approx 0.33 \implies 10\times$ se ve como $2.1\times$) se subestiman, y la **descarga eléctrica** ($\beta = 3.5 \implies 2\times$ se siente $11.3\times$) se sobreestima. La **posición** es #1 en cuantitativo, ordinal y nominal; el **tono (color)** sube al #2 en nominal.
   * **Uso del Color (Diaps. 67–74):** Regla de Ware (máximo **5 a 8 colores** para categorías), trampa del arcoíris (`jet` oscila 3 veces en luminancia relativa Rec. 709 $Y = 0.2126R + 0.7152G + 0.0722B$ creando bordes falsos como en el mapa de ozono de Rogowitz y Treinish, 1998, mientras `viridis` sube monótonamente de $\mathbf{0.02}$ a $\mathbf{0.78}$) y daltonismo (~**8% de los hombres**; **azul vs. naranjo** como pareja segura).
4. **Módulo 03.4 — Visualización Engañosa: Taxonomía de Correll, Espejismos (*Mirages*, CHI 2020) y Ética Visual (Diaps. 76–103):**
   * **Las 4 categorías de Michael Correll (Tableau Research, Diap. 77):** (1) **Incorrectas** (mapa CNN de Hong Kong en Brasil, Fox $2385^\circ$, *Daily Nexus* sin escala, circular de Fox News que suma $70\%+63\%+60\%=\mathbf{193\%}$ y doble eje manipulado `300->340` vs `88->101`); (2) **Ilegibles** (circular de 24 tajadas y burbujas de cesantía en Alabama); (3) **Bullshit** (Harry Frankfurt: indiferencia por la verdad); y (4) **No convencionales** (*Gun deaths in Florida*, Reuters 2014: eje con el **0 arriba** donde el salto a **873 muertes** tras la ley *Stand Your Ground* de 2005 parece una caída, vs. *Iraq's Bloody Toll* de S. Scarr y mapa invertido de ITS).
   * **Espejismos de visualización (*Visualization Mirages*, McNutt, Kindlmann y Correll, CHI 2020, Diaps. 96–98):** En 2012 los países sin dato saltan de $\approx 72$ a $\approx 177$; sobre la misma tabla, `SUM(Energy Usage)` parece un **desplome** en 2012 y `AVG(Energy Usage)` parece un **auge brusco** en 2012.
   * **Ética visual (*Cruel Pies*, Dragga y Voss, 2001: 771 muertes laborales), mentiras blancas (Diap. 99) y clasificación de los 4 casos de la Diapositiva 100.**

---

### 🟪 CD04 (CUARTA EN ORDEN): Introducción a Machine Learning (56 Diapositivas)

1. **Módulo 04.1 — Panorama de Machine Learning, Modelos Basados en Reflejos y las 3 Tareas de Predicción (Diaps. 1–10):**
   * ¿Qué es Machine Learning? Mover la complejidad del código a los datos y generalizar a entradas no vistas. En el **mapa de 4 niveles de la IA** (Reflejos $\to$ Estados $\to$ Variables $\to$ Lógica), los clasificadores lineales y redes profundas son **modelos basados en reflejos**: secuencia fija hacia adelante ($x \to f \to y$) donde la inferencia es rápida y el trabajo duro está en aprender $f$.
   * **Las 3 tareas según la salida $y$ (Diaps. 7–9):** (1) **Clasificación binaria** ($y \in \{+1, -1\}$) y multiclase ($y \in \{1, \dots, K\}$); (2) **Regresión** ($y \in \mathbb{R}$ continua); y (3) **Predicción estructurada** ($y$ objeto complejo), cuyo secreto es descomponerse en una **secuencia de clasificaciones multiclase** sucesivas (predecir una palabra a la vez).
   * **Los 4 casos de la Diapositiva 10:** Correo spam ($\to$ binaria), Precio vivienda en Antofagasta ($\to$ regresión), Audio a texto ($\to$ estructurada) y Reconocer dígito 0–9 ($\to$ multiclase $K=10$).
2. **Módulo 04.2 — El Triunfo de Gauss con Ceres (1801) y las Tres Decisiones de Diseño en Regresión Lineal (Diaps. 11–23):**
   * Hito de Ceres: 19 observaciones de Piazzi antes de perderse tras el Sol; Gauss predice a mano por mínimos cuadrados y Ceres reaparece a solo $0.5^\circ$.
   * **El marco de las 3 decisiones sobre $\mathcal{D}_{\text{train}} = \{(1, 1), (2, 3), (4, 3)\}$:**
     - **Clase de hipótesis:** $\mathcal{F} = \{f_{\mathbf{w}}(x) = \mathbf{w} \cdot \phi(x) : \mathbf{w} \in \mathbb{R}^2\}$ con $\phi(x) = [1, x]$ (ej. con $\mathbf{w} = [1, 0.57]$, $f_{\mathbf{w}}(3) = 2.71$).
     - **Función de pérdida:** Cuadrática sobre el residuo $\text{Loss}(x, y, \mathbf{w}) = (f_{\mathbf{w}}(x) - y)^2$, con pérdidas individuales $0.32, 0.74, 0.08$ y riesgo empírico $\text{TrainLoss}([1, 0.57]) = \mathbf{0.38}$.
     - **Algoritmo de optimización (Descenso por Gradiente):** $\nabla_{\mathbf{w}}\text{TrainLoss} = \frac{1}{|\mathcal{D}_{\text{train}}|} \sum 2(\mathbf{w}\cdot\phi(x) - y)\phi(x)$ («algo $\times \phi(x)$»). Con $\eta = 0.1$ desde $[0, 0]$ (Diap. 21): $t=1 \implies \nabla = [-4.67, -12.67], \mathbf{w} = [0.47, 1.27]$; $t=2 \implies \nabla = [2.18, 7.24], \mathbf{w} = [0.25, 0.54]$; y en $t=200$ converge a $\mathbf{w}^* = [1.00, 0.57]$.
3. **Módulo 04.3 — Clasificación Lineal en $\mathbb{R}^2$: Puntaje, Margen, Pérdida 0-1 y Pérdida Hinge (Diaps. 24–37):**
   * Clasificador $f_{\mathbf{w}}(x) = \text{sign}(\mathbf{w}\cdot\phi(x))$ sobre $\{([0, 2], +1), ([-2, 0], +1), ([1, -1], -1)\}$: **puntaje** $\mathbf{w}\cdot\phi(x)$ y **margen** $m = (\mathbf{w}\cdot\phi(x))y$.
   * Por qué $\text{Loss}_{0\text{-}1}$ atasca a GD: gradiente cero en casi todo el plano (Diap. 30).
   * **Pérdida Hinge:** $\text{Loss}_{\text{hinge}}(x, y, \mathbf{w}) = \max\{1 - m, 0\}$ con gradiente $-\phi(x)y$ si $m < 1$ (o $\mathbf{0}$ si $m > 1$). Exige margen de seguridad $m \ge 1$; en $([1, -1], -1)$ con $\mathbf{w} = [0.5, 1]$, $m = 0.5 > 0$ acierta la etiqueta pero sufre pérdida $\mathbf{0.5}$ (Diap. 35).
   * Ejemplo a mano (Diap. 36) con $\mathbf{w} = [2, -1], x = [1, 1], y = -1$: puntaje $1$, margen $-1$, pérdida $2$ y gradiente $[1, 1]$.
4. **Módulo 04.4 — Regresión Logística, Crecimiento de $\|\mathbf{w}\|$, Umbral $\tau$, Softmax y Riesgo Empírico (Diaps. 38–56):**
   * Regresión lineal sobre $\{0, 1\}$ se escapa de $[0, 1]$ (Diap. 39). La sigmoide $\sigma(z) = 1/(1+e^{-z})$ comprime a $(0, 1)$, y la simetría $\sigma(-z) = 1 - \sigma(z)$ unifica la verosimilitud en $P(y \mid x) = \sigma(\text{margen})$.
   * **Máxima verosimilitud = Pérdida logística:** $\text{Loss}_{\text{logistic}} = -\log P(y\mid x) = \log(1 + e^{-m})$ con gradiente graduado $-\sigma(-m)\phi(x)y$.
   * **La logística nunca queda satisfecha (Diap. 48):** en datos separables, con $\eta = 0.5$ $\mathbf{w}$ pasa de $[-2.29, 2.29]$ ($t=100$) a $[-3.45, 3.45]$ ($t=1000$); la frontera no cambia pero $\|\mathbf{w}\|$ crece sin cota (motivación para la **regularización**).
   * **El umbral $\tau$ como perilla (Diap. 49)**, Softmax multiclase (Diap. 50), comparación en $[1, 1], -1$ (pérdida $1.313$ vs $2.000$, Diap. 51) y el principio del Riesgo Empírico.

---

### 🟦 CD05 (QUINTA EN ORDEN): Generalización y Validación (63 Diapositivas)

1. **Módulo 05.1 — El Objetivo Real: Generalizar, Sobreajuste, Correlación Espuria y Liang (Diaps. 1–16):**
   * **El objetivo no es memorizar (Diap. 5):** El algoritmo *Memorize* logra $\text{TrainLoss}(w) = 0.0$ memorizando los pares de entrenamiento, pero colapsa catastróficamente fuera de muestra (predice el valor por defecto).
   * **Correlación espuria clínica (Oakden-Rayner et al., 2019, Diap. 6):** Un modelo para detectar neumotórax en radiografías alcanzó 95% de precisión aprendiendo a detectar el **tubo de drenaje torácico** ya insertado por los médicos en el hospital. En pacientes graves sin tubo colocado, el modelo falló por completo.
   * **Maldición de la dimensionalidad (Diap. 8):** Con $d = 100$ variables binarias, el espacio $\{0, 1\}^{100}$ contiene $2^{100} \approx 1.27 \times 10^{30}$ estados. Un petabyte de datos ($10^{15}$ muestras) cubre apenas una fracción de $10^{-15}$ ($0.0000000000001\%$). Es **matemáticamente imposible generalizar** sin imponer restricciones / sesgo inductivo a la clase de hipótesis $\mathcal{F}$.
   * **Descomposición del error de Liang (Stanford CS221, Diap. 12):**
     $$\text{Err}(\hat{f}) - \text{Err}(f^*) = \underbrace{(\text{Err}(\hat{f}) - \text{Err}(g))}_{\text{Error de Estimación (Varianza)}} + \underbrace{(\text{Err}(g) - \text{Err}(f^*))}_{\text{Error de Aproximación (Sesgo)}}$$
     donde $f^*$ es el predictor de Bayes, $g \in \mathcal{F}$ es la mejor hipótesis posible en la clase, y $\hat{f}$ es la estimada en muestra finita.
   * **Auditoría a los 3 equipos de fraude (Diaps. 14–15, baseline 2%):** Equipo A (Train 0%, Test 25% $\implies$ sobreajuste severo / alta estimación); Equipo B (Train 23%, Test 24% $\implies$ subajuste severo / alta aproximación); Equipo C (50 pruebas en Test $\implies$ conjunto de prueba contaminado / Data Leakage).
2. **Módulo 05.2 — Controlar la Clase de Hipótesis: Dimensión, Cota Cauchy-Schwarz, Regularización L2/L1 y Early Stopping (Diaps. 17–27):**
   * **Reducción de variables:** Búsqueda exhaustiva $2^J$ ($J=20 \implies 1\,048\,576$ modelos) vs. **Greedy Forward Selection** con $\sum_{k=1}^J k = J(J+1)/2$ ($J=20 \implies \mathbf{210}$ modelos evaluados, ahorro de $4\,993\times$).
   * **Teorema de Suavidad por Cauchy-Schwarz (Diap. 21):**
     $$|s(x) - s(x')| = |w \cdot (\phi(x) - \phi(x'))| \le \|w\|_2 \|\phi(x) - \phi(x')\|_2$$
     Acotar $\|w\|$ restringe la pendiente máxima y garantiza que perturbaciones en la entrada no provoquen oscilaciones erráticas en la predicción.
   * **Regularización L2 (Ridge / Weight Decay, Diaps. 25–26):** $\min_w \text{TrainLoss}(w) + \frac{\lambda}{2}\|w\|_2^2$, con actualización $w \leftarrow (1 - \eta\lambda)w - \eta\nabla\text{TrainLoss}(w)$.
   * **Cálculo manual exacto de la Diapositiva 26 (Págs. 103–107):**
     Con $w = [2, -2]$, $\nabla\text{TrainLoss} = [1, 1]$, $\lambda = 0.5$ y $\eta = 0.1$:
     $$\lambda w = [1, -1] \implies \nabla_{\text{total}} = [1, 1] + [1, -1] = [2, 0] \implies w_{\text{nuevo}} = [2, -2] - 0.1[2, 0] = \mathbf{[1.8, -2.0]}$$
   * **L1 (LASSO, $\|w\|_1$) y Early Stopping (Diaps. 24 y 27):** LASSO fuerza coeficientes exactamente a cero (dispersión analítica), mientras que detener el gradiente en $T$ pasos partiendo de $w^{(0)}=0$ impone una cota de radio $\|w^{(T)}\| \le \eta G T$.
3. **Módulo 05.3 — Hiperparámetros, Experimento Controlado y Validación Cruzada K-Fold (Diaps. 28–52):**
   * Partición $D_{\text{train}}, D_{\text{val}}, D_{\text{test}}$ y la regla sagrada: **el conjunto de prueba solo se toca una única vez al final**.
   * **Experimento numérico controlado (Diaps. 42–47):** Función real $y = x^3 - 3x + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, 4.0)$ ($n=30$ puntos en $[-2.2, 2.2]$). Ajuste polinomial de grados $d=1$ a $8$ (el grado verdadero es $d=3$).
   * **Desenmascaramiento del falso mínimo:**
     - Validación simple (8 puntos): elige erróneamente $d=6$ ($\text{MSE} = \mathbf{1.9} < 4.0$, ilusión por muestra afortunada).
     - 5-Fold Cross-Validation ($K=5$): elige correctamente el grado verdadero $d=3$ ($\text{MSE} = \mathbf{3.8}$, casi en el piso de ruido 4.0), mientras que $d=6$ sube a $9.5$ y $d=8$ colapsa a $80.0$.
   * **Desglose de rondas (Diap. 46 / Pág. 190):** Para $d=3$, las rondas son $[2.4, 2.6, 7.6, 4.7, 1.5]$ (media 3.8); para $d=6$, las rondas son $[1.9, 1.7, 8.2, \mathbf{32.3}, 3.3]$ (media 9.5; ¡la ronda 4 explota a 32.3!).
   * **Auditoría final en 200,000 datos no vistos (Diap. 47):** $d=3$ (CV) obtiene MSE real $\mathbf{4.2}$ vs. $\mathbf{5.1}$ de $d=6$ (Simple Val) $\implies d=3$ generaliza **21.4% mejor**.
   * Comparativa de modelos por estabilidad (Diap. 48): Modelo A ($\text{std} = 0.23$) vs. Modelo B ($\text{std} = 1.25$); variantes Stratified K-Fold y TimeSeriesSplit (ventana expansiva sin fuga del futuro).
4. **Módulo 05.4 — Descomposición Sesgo-Varianza, Matriz de Control de Ingeniería y Curvas de Aprendizaje (Diaps. 53–63):**
   * **Teorema del MSE esperado:**
     $$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2$$
   * **La Diana (Bullseye) en 4 cuadrantes:** Alto/Bajo Sesgo vs. Alto/Bajo Varianza.
   * **Matriz de control de 4 perillas (Diap. 58):**
     - Complejidad / Grado $d \uparrow \implies \text{Sesgo} \downarrow, \text{Varianza} \uparrow$.
     - Regularización $\lambda \uparrow \implies \text{Sesgo} \uparrow, \text{Varianza} \downarrow$.
     - Early Stopping $T \downarrow \implies \text{Sesgo} \uparrow, \text{Varianza} \downarrow$.
     - **Tamaño de muestra $n \uparrow \implies \text{Sesgo} =, \mathbf{\text{Varianza} \downarrow}$ (¡El único botón con costo estadístico cero!).**
   * **Diagnóstico de curvas de aprendizaje (Diaps. 59–60):** Alta Varianza (brecha Train << Val, resuelto con más datos o $\lambda$) vs. Alto Sesgo (Train $\approx$ Val estancados en alto error; más datos **no** ayuda, se requiere un modelo más rico).

---

### 🟩 CD06 (SEXTA EN ORDEN): Clustering — Agrupamiento No Supervisado (57 Diapositivas)

1. **Módulo 06.1 — K-Means: Compresión, Minimización Alternada de Lloyd y Codo (Diaps. 1–21):**
   * **Doble naturaleza de K-Means (Diap. 4):** Función objetivo de compresión de pérdida cuadrática:
     $$\text{Loss}_{\text{kmeans}}(z, \mu) = \frac{1}{n}\sum_{i=1}^n \|\phi(x_i) - \mu_{z_i}\|_2^2$$
     frente al **algoritmo de Lloyd** (1982) que alterna 1) asignación óptima de cada punto $z_i \leftarrow \arg\min_k \|\phi(x_i) - \mu_k\|^2$ y 2) recentrado $\mu_k \leftarrow \frac{1}{|C_k|}\sum_{i \in C_k} \phi(x_i)$.
   * **Traza manual 1D exacta (Diap. 12):** Conjunto $\{0, 2, 10, 12\}$ inicializado en $\mu = [0, 2]$. Iter 1: $z = [1, 2, 2, 2], \mu = [0, 8]$. Iter 2: $z = [1, 1, 2, 2], \mu = [1, 11]$. Converge en 2 pasos con $\text{Loss} = \mathbf{4.0}$.
   * **Paso 2D a mano exacto (Diap. 15):** Puntos $(0,0), (2,0), (6,4), (8,4)$ con $\mu_1=(0,0), \mu_2=(2,0)$. Iter 1: $\mu_1=(0,0), \mu_2=(16/3, 8/3) \approx (5.33, 2.67)$ con $\text{Loss}=26.67$. Iter 2: $\mu_1=(1,0), \mu_2=(7,4)$ con $\text{Loss} = \mathbf{4.0}$ (convergencia monótona).
   * **Simulación de 210 puntos en 3 grupos (Diap. 18):** Mínimo local severo ($\text{Loss} \approx 1\,450$) frente a inicialización K-Means++ que alcanza el óptimo global ($\text{Loss} \approx 328$).
   * **Curva oficial del codo (Diap. 21):** Para $K=1..8$, pérdidas $[1863.2, 955.1, \mathbf{107.5}, 90.1, 77.9, 66.1, 58.4, 50.7]$. Quiebre abrupto en $K=3$ ($\Delta = 847.6$ vs. solo $17.4$ al pasar a $K=4$).
   * **Sensibilidad a Outliers:** En $\{0, 2, 10, 12, 30\}$ con $K=2$, el punto $30$ secuestra $\mu_2=30.0$, forzando $\mu_1=6.0$ ($\text{Loss}=104.0$) y desintegrando la estructura verdadera.
2. **Módulo 06.2 — Evaluación de Clusters: Índice de Rand (RI), Rand Ajustado (ARI) y Ruido (Diaps. 22–30):**
   * **Matriz de contingencia de pares $\binom{n}{2}$ (Diap. 24):** Conteo exhaustivo de $a$ (acuerdo positivo), $b$ (acuerdo negativo), $c$ (falso negativo) y $d$ (falso positivo).
   * **Fórmula de Rand Index simple (Diap. 24):** $\text{RI} = \frac{a + b}{\binom{n}{2}}$.
   * **Demostración de la inflación de base (Diap. 25–26):** En $n=6$, agrupar todo en 1 cluster da $\text{RI} = 6/15 = \mathbf{0.40}$; asignar cada punto solo da $\text{RI} = 9/15 = \mathbf{0.60}$; en ambos el $\text{ARI} = \mathbf{0.0000}$.
   * **Rand Index Ajustado de Hubert-Arabie (1985, Diap. 27):**
     $$\text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]}$$
     Caso exacto de 6 puntos con 1 error ($Y=[1,1,1,2,2,2], Z=[1,1,2,2,2,2]$): $a=4, b=6, c=2, d=3 \implies \text{RI} = 10/15 \approx \mathbf{0.67}$, corregido por azar a $\text{ARI} = \mathbf{0.3243}$.
3. **Módulo 06.3 — Agrupamiento Jerárquico: Dendrogramas, Single vs. Complete y GMM (Diaps. 31–37):**
   * **Matriz de distancias 1D sobre $\{1, 2, 5, 9, 12\}$ (Diaps. 31–33):**
     - Single Linkage ($d_{\min}$): fusiones en $1.0, 3.0, 3.0, 4.0$. El empate simultáneo en $h=3.0$ ($\{1,2\}$ con $5$ y $9$ con $12$) hace **imposible aislar $K=3$ clusters**. Corte a $h=3.5 \implies 2$ clusters; corte a $h=2.9 \implies 4$ clusters.
     - Complete Linkage ($d_{\max}$): fusiones en $1.0, 3.0, 4.0, 11.0$, permitiendo aislar $K=3$ en $h \in [3.0, 4.0)$.
   * **Gaussian Mixture Models (GMM, Diaps. 36–37):** Asignación suave (soft clustering) mediante responsabilidades probabilísticas posteriores $\gamma_{ik} = P(z_i = k | x_i) \in [0, 1]$ optimizadas mediante el algoritmo Esperanza-Maximización (EM).
4. **Módulo 06.4 — Basado en Densidad: DBSCAN, Dos Lunas, Densidades Heterogéneas y HDBSCAN (Diaps. 38–57):**
   * **DBSCAN (Ester et al., 1996):** Clasificación en Core ($\ge \text{minPts}$ vecinos en radio $\varepsilon$), Border y Noise. Conectividad transitiva de densidad.
   * **Ejemplo interactivo de 10 puntos (Diaps. 43–48):** Con $\varepsilon=1.05, \text{minPts}=3 \implies 7$ núcleos, 2 fronteras, 1 ruido; con $\text{minPts}=4$, el arco entero colapsa a 5 puntos de ruido.
   * **Experimento de las Dos Lunas (Diap. 38):** K-Means falla con 31–33 errores ($\text{ARI} \approx 0.49$) al trazar una mediatriz recta; DBSCAN alcanza 0 errores, 0 ruido y $\text{ARI} = \mathbf{1.0000}$.
   * **Dilema de densidades heterogéneas y solución HDBSCAN (Diaps. 50–57):** Un único $\varepsilon$ no puede separar 2 densos y 1 disperso ($\varepsilon=0.25$ descarta 22 puntos dispersos como ruido; $\varepsilon=0.60$ funde los dos compactos). HDBSCAN (Campello et al., 2013) usa distancia de alcanzabilidad mutua $d_{\text{mreach}}$ y árbol condensado por persistencia ($\lambda = 1/\varepsilon$).

---

### 🟨 CD07 (SÉPTIMA EN ORDEN): Reducción de Dimensionalidad — PCA, Fisher LDA, t-SNE y UMAP (73 Diapositivas)

1. **Módulo 07.1 — La Maldición de la Dimensionalidad: Hiperesferas en Hipercubos y Distancias Vacías (Diaps. 1–11):**
   * **Razón de volumen bola unitaria / hipercubo lado 2 (Diaps. 5–11):** Deducción analítica de:
     $$P_d = rac{	ext{Vol}(B_d(1))}{	ext{Vol}(C_d(2))} = rac{\pi^{d/2}}{\Gamma(d/2 + 1) \cdot 2^d}$$
     Evaluación en $d \in \{1, 2, 3, 6, 10, 15, 100\}$: $P_1 = 1.0$, $P_2 = \pi/4 pprox 0.7854$ ($78.5\%$), $P_3 = \pi/6 pprox 0.5236$ ($52.4\%$), $P_6 pprox 0.0807$ ($8.07\%$), $P_{10} pprox \mathbf{0.00249}$ (**$0.25\%$**, solo 1 de cada 400 puntos cae al centro), $P_{15} pprox 1.16 	imes 10^{-5}$ y $P_{100} pprox \mathbf{1.87 	imes 10^{-70}}$ (prácticamente cero; todo el volumen vive en las $2^{100}$ esquinas).
   * **Simulación con $N=1\,000$ puntos aleatorios (Diap. 11):** En $d=2$ caen $pprox 785$ puntos en el círculo; en $d=10$ caen en promedio **$2.5$ puntos** (solo 2 o 3 con suerte en todo el conjunto).
   * **Pérdida de contraste métrico (Teorema de Beyer et al., 1999):** $\lim_{d 	o \infty} rac{d_{\max} - d_{\min}}{d_{\min}} = 0$, demostrando que en alta dimensión todos los pares de puntos son equidistantes y las consultas de vecinos más cercanos colapsan.
2. **Módulo 07.2 — PCA Paso a Paso: Análisis de Componentes Principales, Dualidad Varianza/Error y SVD (Diaps. 12–37):**
   * **Ejemplo trabajado a mano de 6 puntos en 2D (Págs. 102–142):** $\{(6,4), (5,5), (5,2), (3,4), (3,1), (2,2)\}$.
     * **Paso 1 (Centrado):** Media $ar{x} = (24/6, 18/6) = (4.0, 3.0)$. Puntos centrados $	ilde{x}_i \in \{(2,1), (1,2), (1,-1), (-1,1), (-1,-2), (-2,-1)\}$.
     * **Paso 2 (Matriz de covarianza):** $S_{11} = 12/6 = 2.0$, $S_{22} = 12/6 = 2.0$, $S_{12} = 6/6 = 1.0 \implies S = egin{bmatrix} 2.0 & 1.0 \ 1.0 & 2.0 \end{bmatrix}$.
     * **Paso 3 (Autovalores y autovectores):** $\det(S - \lambda I) = (2-\lambda)^2 - 1 = 0 \implies \lambda_1 = \mathbf{3.0}$ ($75\%$), $\lambda_2 = \mathbf{1.0}$ ($25\%$). Autovectores normalizados: $v^{(1)} = rac{1}{\sqrt{2}}[1, 1]^T$ ($45^\circ$), $v^{(2)} = rac{1}{\sqrt{2}}[1, -1]^T$ ($135^\circ$).
     * **Paso 4 (Proyecciones $z_i$ y reconstrucción $\hat{x}_i$):** $z = [rac{3}{\sqrt{2}}, rac{3}{\sqrt{2}}, 0, 0, -rac{3}{\sqrt{2}}, -rac{3}{\sqrt{2}}]$. Reconstruidos: $\{(5.5, 4.5), (5.5, 4.5), (4, 3), (4, 3), (2.5, 1.5), (2.5, 1.5)\}$.
     * **Paso 5 (Teorema exacto de error de compresión):** $\sum_{i=1}^6 \|\hat{x}_i - x_i\|^2 = 0.5 + 0.5 + 2.0 + 2.0 + 0.5 + 0.5 = \mathbf{6.0} = N \cdot \lambda_2 = 6 	imes 1.0 = 6.0$.
   * **Equivalencia de Eckart-Young-Mirsky y SVD:** $	ilde{X} = U \Sigma V^T \implies S = V(\Sigma^2 / N)V^T \implies \lambda_j = s_j^2 / N$.
3. **Módulo 07.3 — Fisher LDA: Análisis Discriminante Lineal, Maximización Inter-clase y Supresión de Ruido (Diaps. 38–49):**
   * **Criterio de Fisher (1936):** Maximizar $J(w) = rac{(w^T (m_2 - m_1))^2}{w^T S_W w} \implies w^* \propto S_W^{-1}(m_2 - m_1)$.
   * **Ejemplo trabajado de 8 puntos (Diaps. 42–44):** $m_1 = (2, 4)$, $m_2 = (5, 5) \implies m_2 - m_1 = (3, 1)$. Matriz intra-clase $S_W = 	ext{diag}(4, 36)$.
     * Dirección de Fisher: $w^* \propto [3/4, 1/36]^T \propto \mathbf{[27, 1]^T} \implies w^* pprox (0.9993, 0.0370)$ ($2.1^\circ$).
     * Separabilidad: $J(w^*) = \mathbf{2.28}$ frente a PCA $J(v_{	ext{PCA}}) = \mathbf{0.03}$ (**$82	imes$ mejor separación supervisada**). PCA se confunde con el ruido vertical de varianza $18$, mientras Fisher lo castiga con $1/36$.
   * **Segundo caso de Fisher (Diap. 49):** $m_2 - m_1 = (2, 2)$, $S_W = 	ext{diag}(2, 8) \implies w^* \propto (2/2, 2/8) = (1, 1/4) \propto \mathbf{[4, 1]^T}$.
   * **Límite dimensional:** Para $C$ clases, el subespacio proyectado tiene a lo más $C-1$ dimensiones.
4. **Módulo 07.4 — Manifold Learning: t-SNE, Distribución t de Student, UMAP y Pipeline HDBSCAN (Diaps. 50–73):**
   * **Probabilidades de vecindad simétricas:** $p_{j|i} \propto \exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)$ con perplejidad $	ext{Perp}(P_i) = 2^{H(P_i)}$ fija. En $\{0, 1, 2, 8\}$ con $	ext{Perp}=2$, $\sigma(x=1) = 1.0$ mientras que para el punto aislado $x=8$, $\sigma$ se expande a $\mathbf{2.26}$ ($[0.05, 0.21, 0.74]$).
   * **Colas pesadas de la t de Student contra el Crowding Problem (Diap. 60):** En baja dimensión, $q_{ij} \propto (1 + \|y_i - y_j\|^2)^{-1}$. A distancia $d=3$, la Gaussiana vale $e^{-9} = 0.000123$ mientras Student-t vale $1/(1+9) = 0.100000$ (**$810	imes$ mayor probabilidad**), permitiendo separar clusters sin forzar apiñamiento central.
   * **Pipeline Moderno y Reglas de Interpretación (Wattenberg et al., 2016):** Los tamaños visuales de clusters y distancias entre clusters no tienen significado geométrico euclidiano. La mejor práctica industrial es reducir con **UMAP a 10 dimensiones** y agrupar con **HDBSCAN**.

---

### 🌲 CD08 (OCTAVA EN ORDEN): Modelos de Árboles de Clasificación y Regresión (68 Diapositivas)

1. **Módulo 08.1 — Particionado Ortogonal, Complejidad Booleana $2^{2^n}$ y NP-Completitud (Diaps. 1–19):**
   * **Particionado Ortogonal de CART (Diaps. 3–5):** Los árboles solo generan hiperplanos paralelos a los ejes coordenados ($x_j \le t$). Para modelar fronteras lineales oblicuas simples (ej. $x_2 = 0.8 x_1$), el árbol debe construir una costosa escalera de hiperplanos ortogonales.
   * **Explosión Combinatoria de Funciones Booleanas (Diaps. 6–10):** Para $n$ variables binarias existen $2^n$ estados de entrada y $2^{2^n}$ funciones booleanas posibles ($n=1 \to 4$, $n=2 \to 16$, $n=3 \to 256$, $n=4 \to 65\,536$ y $n=7 \to \mathbf{3.4028 \times 10^{38}}$).
   * **Teorema de Hyafil & Rivest (1976):** Determinar el árbol binario óptimo es **NP-completo**, obligando a usar algoritmos voraces (*greedy*) sin retroceso.
   * **La Trampa del Orden Categórico Nominal (Diaps. 17–18):** Ordenar nominales como enteros $\{1, 2, 3\}$ impide aislar $\{C_1, C_3\}$ frente a $\{C_2\}$; CART evalúa las $2^{K-1}-1$ particiones binarias disjuntas.
2. **Módulo 08.2 — Criterios de Impureza (Error, Gini, Entropía de Shannon) y Algoritmo ID3 (Diaps. 20–39):**
   * **Insensibilidad de la Tasa de Error de Clasificación (Diaps. 21–28):**
     * En el **Caso 1 de la Cátedra (Diap. 24)** ($14A, 5B \to 6A, 0B$ [puro] y $8A, 5B$), el error del padre es $5/19$ y el error ponderado de los hijos es $(6/19)(0) + (13/19)(5/13) = 5/19 \implies \mathbf{\Delta E = 0.000}$ (insensible a la creación del nodo puro). En cambio, $\mathbf{\Delta I_G = +0.0639}$ y $\mathbf{\Delta H = +0.1738\text{ bit}}$.
     * En el **Caso 2 (Diap. 26)** ($9A, 3B \to 6A, 0B$ y $3A, 3B$), nuevamente $\Delta E = 0.000$ mientras $\mathbf{\Delta I_G = +0.1250}$.
     * Causa: la concavidad estricta de Gini y Entropía frente a la naturaleza lineal a trozos del error.
   * **Trazado Manual Exacto de ID3 en 10 Días (Harvard CS109, Diaps. 31–39):** $H(S) = 1.000\text{ bit}$. Raíz: Clima $IG = \mathbf{0.600}$ (gana) vs Viento $0.029$ y Temp $0.029$. En rama Soleado (2 Sí, 2 No): Viento genera $IG = \mathbf{1.000\text{ bit}}$ dividiendo perfectamente Suave (Sí) y Fuerte (No).
3. **Módulo 08.3 — Control de Complejidad, Miopía del Early Stopping y Poda CART (Diaps. 40–54):**
   * **Falla de la Parada Temprana por Ganancia Mínima (Diaps. 42–45):** En problemas tipo paridad o XOR, un corte univariado produce ganancia cero provocando detención prematura. CART soluciona esto creciendo el árbol completo y podando después.
   * **Poda por Costo-Complejidad (Weakest Link Pruning, Diaps. 46–54):** Minimiza $C_\alpha(T) = R(T) + \alpha |T|$ colapsando el nodo interno con menor eslabón $g(t) = \frac{R(t) - R(T_t)}{|T_t| - 1}$.
   * **Verificación del Árbol de 100 Puntos (Diap. 48):** $T_0$ ($3$ hojas, $R=0.14$), nodo $x_2 < 3$ con $g(t) = 0.040 \implies T_1$ ($2$ hojas, $R=0.18$), y raíz con $g=0.320 \implies T_2$ ($1$ hoja, $R=0.50$). Para $\alpha=0.10$, $C(T_1) = 0.38 < C(T_0) = 0.44 < C(T_2) = 0.60$.
4. **Módulo 08.4 — CART para Regresión Continua, Reducción de MSE y Ensambles (Diaps. 55–68):**
   * Predicción en hoja como la media muestral $\bar{y}_m$ y criterio de división por minimización de la varianza ponderada (suma de cuadrados).
   * **Trazado del Dataset de 6 Puntos (Diap. 57):** $x=[1..6], y=[2.0, 3.0, 2.5, 8.0, 9.0, 8.5]$. El umbral $x < 3.5$ produce $\text{MSE} = \mathbf{0.167}$ (óptimo global) frente a $x < 2.5$ ($4.667$) y $x < 4.5$ ($3.875$).
   * Progresión de profundidad: Prof 1 ($\text{MSE}=0.167$), Prof 2 ($0.062$), Prof 3 ($\text{MSE}=0.000$, memorización total).
   * Transición teórica a Ensambles (Bagging, Random Forest y Boosting) para domar la alta varianza.

---

### 🟧 CD09 (NOVENA EN ORDEN): Ensamblaje de Modelos y Random Forests (59 Diapositivas)

1. **Módulo 09.1 — Sabiduría de las Masas, Teorema de Condorcet y Experimento de Galton (Diaps. 1–16):**
   * **El Experimento de Francis Galton (1907, Plymouth, Diaps. 3–6):** $N = 787$ personas apostaron sobre el peso de un buey faenado. La mediana colectiva ($1{,}207\text{ lbs}$) acertó el peso real ($1{,}198\text{ lbs}$) con un error de apenas **$0.751\% \approx 0.8\%$** ($+9\text{ lbs}$), superando con creces a matarifes y estimadores expertos individuales.
   * **Teorema del Jurado del Marqués de Condorcet (1785, Diaps. 7–11):** Combinar $B$ clasificadores independientes con precisión $p > 0.5$ asegura que $P(\text{mayoría}) = \sum_{k} \binom{B}{k} p^k (1-p)^{B-k} \xrightarrow{B \to \infty} 1.0$. Con $p=0.70$: $B=3 \implies 0.784$, $B=5 \implies 0.837$, $B=15 \implies 0.950$, $B=101 \implies \approx 1.0$. Si $p < 0.5$, el ensamble converge a la certeza del error.
   * **Condiciones de Surowiecki (2004) y Modos de Falla:** Diversidad, independencia y descentralización. Modelos idénticos o con sesgo correlacionado no reducen varianza.

2. **Módulo 09.2 — Bagging y Bootstrap Resampling: Fracción $(1 - 1/e)$ y Validación OOB (Diaps. 17–28):**
   * **Matemática del Remuestreo con Reemplazo (Breiman, 1996, Diaps. 18–21):** En una muestra de tamaño $N$, la probabilidad asintótica de que una observación quede fuera de la bolsa (*Out-Of-Bag*, OOB) es $(1 - 1/N)^N \xrightarrow{N \to \infty} e^{-1} \approx \mathbf{36.8\%}$, reteniendo el $\mathbf{63.2\%}$ de observaciones únicas en cada árbol. Para $N=10$, $P(\text{OOB}) = 0.349$.
   * **Estimación del Error OOB (Diaps. 22–24):** Cada observación se valida evaluando únicamente sobre los árboles donde quedó fuera de la bolsa, proveyendo una validación cruzada gratuita y rigurosa sin apartar datos de prueba.
   * **Reducción de Varianza en Estimadores Inestables:** Demostración en regresión continua sobre 30 puntos: el error cuadrático medio (MSE) cae de $3.25$ en un árbol CART sobreajustado a $1.12$ en un ensamble Bagging (reducción del $-65.5\%$).

3. **Módulo 09.3 — Random Forests, Descorrelación y la Ecuación ESL 15.1 de Hastie et al. (Diaps. 29–43):**
   * **Derivación de la Ecuación ESL 15.1:** La varianza del promedio de $B$ estimadores con correlación por pares $\rho$ y varianza individual $\sigma^2$ es:
     $$\operatorname{var}(\bar{X}) = \mathbf{\rho \sigma^2 + \frac{1 - \rho}{B}\sigma^2} \xrightarrow{B \to \infty} \mathbf{\rho \sigma^2}$$
   * **El Piso Irreducible de Bagging:** Con $\rho=0.60$, aumentar $B$ de 10 a 20 solo reduce la varianza de $0.640$ a $0.620$ ($0.020$). ¡Pero reducir la correlación $\rho$ a $0.30$ reduce la varianza drásticamente a $\mathbf{0.370}$!
   * **Subespacios Aleatorios (Breiman, 2001):** Muestrear al azar $m = \lfloor \sqrt{J} \rfloor$ variables en cada split (ej. $m=6$ para $J=36$) reduce la probabilidad de elegir un predictor dominante de $100\%$ a $\mathbf{16.7\%}$, destruyendo activamente la correlación inter-árbol sin inflar el sesgo.

4. **Módulo 09.4 — Boosting Adaptativo (AdaBoost), Gradient Boosting, Voting y Stacking (Diaps. 44–59):**
   * **AdaBoost (Freund & Schapire, 1997, Diaps. 45–50):** Algoritmo secuencial que pondera muestras $w_n^{(m+1)} \propto w_n^{(m)} \exp(\alpha_m \mathbb{I}(y_n \neq T_m(x_n)))$ con peso de estimador $\alpha_m = \frac{1}{2}\ln \frac{1-\varepsilon_m}{\varepsilon_m}$. Trazado exacto sobre 13 clientes de crédito: Ronda 1 ($\alpha_1=1.242, \varepsilon_1=0.222$), Ronda 2 ($\alpha_2=1.203, \varepsilon_2=0.231$), Ronda 3 ($\alpha_3=1.153, \varepsilon_3=0.240$) convergiendo a $\mathbf{0\text{ errores}}$ ($100\%$ exactitud).
   * **Gradient Boosting (Friedman, 2001, Diaps. 51–54):** Ajuste secuencial de árboles sobre los residuos pseudo-gradiente de la pérdida cuadrática, reduciendo el MSE continuo de $3.09 \to 1.16 \to 0.90 \to 0.55$.
   * **Hard vs Soft Voting (Diap. 55):** Probabilidades $[0.45, 0.45, 0.90]$ producen colapso en Hard voting (gana $-1$ por mayoría de 2 votos) mientras que Soft voting predice correctamente $+1$ (probabilidad promedio $0.60$).
   * **Stacking y Meta-Aprendizaje (Wolpert, 1992):** Ensamblaje heterogéneo con meta-modelo y validación obligatoria fuera de pliegue (*out-of-fold cross-validation*).

---

### 💠 CD10 (DÉCIMA EN ORDEN — CULMINACIÓN FASE 1): Modelos No Lineales y Support Vector Machines (65 Diapositivas)

1. **Módulo 10.1 — k-Nearest Neighbors (kNN), Regresión, Clasificación y Maldición Dimensional (Diaps. 1–19):**
   * **Regresión kNN:** Estimador no paramétrico $\hat{y}(x_q) = \frac{1}{k}\sum_{i \in N_k(x_q)} y_i$. Trazado numérico sobre 9 puntos: para $x_q = 2.2$, $1\text{-NN}$ predice $\mathbf{3.80}$ y $3\text{-NN}$ predice $\mathbf{4.00}$ (o $3.83$), con salto escalonado en el punto medio $x = \mathbf{2.25}$.
   * **Dilema Sesgo-Varianza:** $k=1$ posee sesgo cero y varianza máxima ($\text{RMSE}_{\text{train}} = 0.00, \text{RMSE}_{\text{test}} = 0.23$, sobreajuste); $k=N$ predice la media global con sesgo máximo y varianza nula ($3.50, \text{RMSE}_{\text{test}} = 0.50$); $k=3$ ofrece el óptimo balance ($\text{RMSE}_{\text{test}} = 0.19$).
   * **Clasificación kNN:** Votación mayoritaria $P(y=c|x) = \frac{1}{k}\sum \mathbb{I}(y_i=c)$ (3 azules vs 2 naranjas $\implies P=0.60 \to \text{Azul}$).
   * **Maldición de la Dimensionalidad:** En $[0, 1]^p$, la distancia promedio a la cáscara exterior es $d_{\text{edge}}(p) = 1 - (1/2)^{1/p}$. Para $p=10 \implies \mathbf{79.4\%}$; para $p=100 \implies \mathbf{97.7\%}$. La métrica euclidiana pierde contraste y no existen vecinos genuinamente locales (Beyer et al., 1999).

2. **Módulo 10.2 — Feature Maps, Espacios Transformados y Teorema de Separabilidad de Cover (Diaps. 20–34):**
   * **Frontera Circular Cuadrática:** Separación del disco $(x_1-1)^2 + (x_2-1)^2 \le 2$ mapeando a $\phi(x) = [x_1, x_2, x_1^2 + x_2^2]^T$ con pesos $w = [2, 2, -1]^T$ y sesgo $b=0$, transformando un círculo en un hiperplano lineal en $\mathbb{R}^3$.
   * **Modelos a Trozos:** Base de indicatrices de soporte compacto $\phi_j(x) = \mathbb{I}(x \in B_j)$ generando vectores *one-hot*.
   * **Teorema de Thomas Cover (1965):** Probabilidad de separabilidad lineal $P(N, D) = 2^{1-N}\sum_{i=0}^{D-1}\binom{N-1}{i}$, con transición de fase crítica en $N = 2D$.
   * **Explosión Combinatoria:** Un polinomio de grado $d=5$ sobre $J=100$ variables requiere $\binom{105}{5} = \mathbf{96\,560\,646}$ dimensiones, colapsando la memoria y el cómputo explícito.

3. **Módulo 10.3 — Support Vector Machines (SVM), Margen Máximo y Soft-Margin C (Diaps. 35–48):**
   * **Distancia con Signo:** $d(x) = \frac{w \cdot x + b}{\|w\|}$. Para $3x_1 + 4x_2 - 5 = 0$, punto $(2, 3) \implies d = +2.60$, punto $(0, 0) \implies d = -1.00$.
   * **Margen Rígido (Hard Margin):** Minimizar $\frac{1}{2}\|w\|^2$ sujeto a $y_n(w \cdot x_n + b) \ge 1$. Resolución exacta para 8 puntos separables con vectores de soporte $(2, 2)$ y $(0, 0)$: $w = [0.5, 0.5]^T, b = -1.0 \implies \|w\| = \frac{1}{\sqrt{2}} \approx 0.707 \implies \text{Margen } \gamma = \mathbf{2\sqrt{2}} \approx \mathbf{2.828}$.
   * **Margen Blando (Soft Margin) y Holguras $\xi_n$:** Minimizar $\frac{1}{2}\|w\|^2 + C \sum \xi_n$, equivalente a Hinge Loss con regularización $\lambda = \frac{1}{2C}$.
   * **Dualidad de Wolfe y Condiciones KKT:** $\alpha_n [y_n(w \cdot x_n + b) - 1 + \xi_n] = 0$. Únicamente los puntos con $\alpha_n > 0$ determinan la frontera óptima.

4. **Módulo 10.4 — El Truco del Kernel y Conexión con Redes Neuronales (Diaps. 49–65):**
   * **Truco del Kernel (Aizerman et al., 1964; Mercer, 1909):** Evalúa productos internos $K(x, z) = \langle \phi(x), \phi(z) \rangle$ en espacios de Hilbert sin jamás proyectar explícitamente los datos.
   * **Kernel Polinomial:** $(1 + x \cdot z)^2 = 36$ computado en $O(d)$ sin instanciar el vector de dimensión 6.
   * **Kernel RBF Gaussiano:** $K(x, z) = \exp(-\gamma \|x - z\|^2)$ con radio $\sigma = \sqrt{1/(2\gamma)}$. Corresponde mediante serie de Taylor a un espacio de Hilbert de dimensión *infinita*. Comportamiento asintótico: $\gamma \to 0$ (lineal/subajuste) vs $\gamma \to \infty$ (islas de memorización).
   * **Puente hacia Redes Neuronales (MLP):** La capa oculta $h = \sigma(Vx + c)$ es un **Feature Map adaptativo** aprendido por Backpropagation. Demostración formal: 2 neuronas ReLU con $V = [[1, 1], [1, 1]]$ y $c = [-0.5, -1.5]$ proyectan XOR a $(0,0)\to[0,0], (0,1)\to[0.5,0], (1,0)\to[0.5,0], (1,1)\to[1.5,0.5]$, donde la salida $h_1 - 2h_2 - 0.25 = 0$ **lo separa linealmente**.

---

### 🟨 CD11 (UNDÉCIMA EN ORDEN — CIERRE PRÁCTICO FASE 1): Aspectos Prácticos y Evaluación de Modelos (93 Diapositivas)

1. **Módulo 11.1 — Depuración de Modelos ML, Ley de Kolter y el Proxy Ciego (Diaps. 1–28):**
   * **Ley de Depuración de Zico Kolter (CMU 15-388/688):** *«Cada vez que evalúas tu algoritmo y no funciona como esperabas, hay un 95% de probabilidad de que sea un bug en tu código o tus datos, no en tu modelo.»*
   * **Matriz 2×2 de Depuración de ML (Diap. 8):** Fallo en implementación del algoritmo, fallo en código de evaluación, datos sucios/corruptos, y solo como último recurso clase de hipótesis inadecuada.
   * **El Desastre del Proxy Humano Ciego — Google FAST 2007 (Pinheiro et al., Diaps. 24–28):** En >100,000 discos duros, **más del 56% de los discos que fallaron reportaron cero alertas SMART**; y los discos con errores de escaneo tenían 39× mayor tasa de fallo pero el 64% de los discos sobrevivientes jamás presentaron un solo error SMART.
   * **Backblaze 2025 (Diap. 27):** Más de 270,000 discos monitoreados confirman la asimetría predictiva. El humano que decide "cambiar el disco cuando SMART > 0" comete falsos negativos masivos.
2. **Módulo 11.2 — Taxonomía de Rubin (1976), No Comprobabilidad Empírica y 5 Métodos de Imputación (Diaps. 29–52):**
   * **Taxonomía Formal de Donald Rubin (1976):**
     * **MCAR (*Missing Completely at Random*):** $P(M \mid Y_{\text{obs}}, Y_{\text{mis}}) = P(M)$. La falta no depende de ninguna variable observada ni latente.
     * **MAR (*Missing at Random*):** $P(M \mid Y_{\text{obs}}, Y_{\text{mis}}) = P(M \mid Y_{\text{obs}})$. La falta depende sistemáticamente de variables observadas (ej. los hombres reportan menos depresión que las mujeres).
     * **MNAR (*Missing Not at Random*):** $P(M \mid Y_{\text{obs}}, Y_{\text{mis}})$ depende del valor faltante mismo (ej. personas con depresión severa abandonan el estudio; o ingresos altos no declaran su salario).
   * **Teorema de No Comprobabilidad Empírica:** Es *matemáticamente imposible* distinguir empíricamente entre MAR y MNAR a partir de los datos observados, requiriendo conocimiento sustantivo del dominio.
   * **Indicador de Falta ($X_{\text{miss}}$):** Variable binaria obligatoria $M \in \{0, 1\}$ para preservar el patrón de ausencia antes de imputar.
   * **Cinco Métodos de Imputación sobre el Micro-Dataset de Cátedra (Diaps. 42–49):**
     * (1) **Media Incondicional:** Imputa $\bar{y} = \mathbf{3.91}$. Colapsa la varianza artificialmente ($s^2$ cae de $1.29$ a $0.84$).
     * (2) **k-Nearest Neighbors ($k=2$):** Imputa $\mathbf{3.15}$ (fila 4) y $\mathbf{5.10}$ (fila 8) preservando geometría local.
     * (3) **Regresión OLS Condicional:** $\hat{y} = \beta_0 + \beta_1 x_1$ imputa $\mathbf{3.29}$ y $\mathbf{5.15}$ (exagera la correlación a $r=1.00$).
     * (4) **Regresión Estocástica:** Añade ruido residual $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ ($\sigma \approx 0.33$), restaurando la distribución condicional real.
     * (5) **MICE (*Multiple Imputation by Chained Equations*, Van Buuren):** Imputación multivariada iterativa $M$ veces para cuantificar la incertidumbre del dato ausente.
3. **Módulo 11.3 — Evaluación de Clasificadores Clínicos, Matriz de Confusión, Calibración y Curvas ROC (Diaps. 53–75):**
   * **Caso de Estudio Clínico Real (251 Pacientes de Triaje Hospitalario, Diaps. 54–68):** Regresión Logística frente a k-NN ($k=5$) a umbral estándar $\tau = 0.50$.
     * Logística: $\text{TN}=58, \text{FP}=59, \text{FN}=37, \text{TP}=97$. Exactitud $\mathbf{0.62}$, Sensibilidad/Recall $\mathbf{0.72}$, Especificidad $\mathbf{0.50}$, Precisión $\mathbf{0.62}$, $F_1 = \mathbf{0.67}$.
     * k-NN ($k=5$): $\text{TN}=55, \text{FP}=62, \text{FN}=33, \text{TP}=101$. Exactitud $\mathbf{0.62}$, Sensibilidad/Recall $\mathbf{0.75}$, Especificidad $\mathbf{0.47}$, Precisión $\mathbf{0.62}$, $F_1 = \mathbf{0.68}$.
   * **Ajuste del Umbral de Decisión $\tau$ ante Costos Asimétricos (Diap. 69):** En urgencia médica, un falso negativo (muerte de paciente crítico) es 10× más costoso que un falso positivo. Bajar $\tau$ a $0.30$ dispara Recall a $\mathbf{0.92}$ reduciendo FN de $37$ a $11$.
   * **Espacio ROC y Restricciones Hospitalarias por País (Diaps. 70–75):**
     * **Brasil (Sistema Mixto SUS/Privado):** Restricción de balance $\text{FPR} \le 0.35, \text{TPR} \ge 0.75 \implies \tau \in [0.42, 0.48]$.
     * **Alemania (Sistema G-BA Altamente Regulado):** Estricta especificidad $\text{FPR} \le 0.15 \implies \tau \ge 0.65$.
     * **India (Alta Carga de Pacientes / Triage Masivo):** Máxima cobertura $\text{TPR} \ge 0.90 \implies \tau \le 0.32$.
4. **Módulo 11.4 — Auditoría de Subgrupos, Receta de 7 Pasos, Data Leakage y Benchmark COST de McSherry (Diaps. 76–93):**
   * **Auditoría de Subgrupos y Falacia del Promedio Global (Diaps. 77–80):** En un modelo con exactitud global de **96.0%**, la desagregación revela **98.9%** en el subgrupo mayoritario ($n=900$) frente a un desastroso **70.0%** en el subgrupo minoritario ($n=100$). Principio de Equidad Minimax (*Worst-case Subgroup Loss*).
   * **Receta de 7 Pasos para Proyectos ML Exitosos (Diaps. 81–84):** Formulación del problema, métrica alineada al negocio, baseline simple/tonto, separación estricta de datos, ingeniería de features, iteración y auditoría de errores.
   * **Taxonomía de Fuga de Datos (*Data Leakage*, Diaps. 85–88):** Fuga temporal (usar futuro para predecir pasado), fuga por normalización previa a la partición de folds, fuga por duplicados o identificadores residuales.
   * **El Principio COST de Frank McSherry (HotOS 2015, Diaps. 89–93):** *Configuration that Outperforms a Single Thread*. Demostración en PageRank sobre grafo web de 128 millones de aristas: script C++ mono-hilo en un laptop común tarda **275 segundos**, mientras que Spark sobre un cluster distribuido de 128 núcleos tarda **857 segundos** (**Spark fue 3.1× más lento que un solo núcleo de laptop** debido al overhead de red, serialización y coordinación). Regla de oro: escalar verticalmente antes de distribuir prematuramente.

---

### 🟣 CD12 (DUODÉCIMA EN ORDEN — PUENTE HISTÓRICO Y CONCEPTUAL): Redes Neuronales y Conexionismo (60 Diapositivas)

1. **Módulo 12.1 — Raíces Filosóficas, Asociacionismo, Alexander Bain (1873) y Escala Biológica (Diaps. 5–22):**
   * Las 4 leyes del asociacionismo clásico: *Contigüidad, Frecuencia, Similitud y Contraste* (Aristóteles, Locke, Hume, Pavlov).
   * David Hartley (1749) y las vibraciones fisiológicas nerviosas.
   * **Alexander Bain (1873, *Mind and Body*):** La intuición fundacional de la neurocomputación: *«La memoria reside en las conexiones (sinapsis)»*, formulada décadas antes de Ramón y Cajal y Sherrington.
   * **Bain vs. Neurobiología Humana Real:** Estimación de Bain ($10^6$ neuronas y $5\times 10^9$ sinapsis) frente a la biología real ($8.6\times 10^{10}$ neuronas y $10^{14}$ sinapsis, con una discrepancia sináptica de factor $\times \mathbf{20\,000}$).
   * Alan Turing (1948) y las máquinas desorganizadas Tipo A (fijas) y Tipo B (conexiones con modificadores entrenables).
   * Los 8 ingredientes de un modelo PDP (Rumelhart, Hinton & McClelland 1986).
2. **Módulo 12.2 — La Neurona de McCulloch & Pitts (1943), Regla de Hebb (1949) y Regla de Oja (1982) (Diaps. 23–38):**
   * Modelo M&P (1943): Entradas excitadoras ($+1$), entradas inhibidoras absolutas y umbral $\theta$. Demostración de compuertas lógicas (AND: $\theta=2$, OR: $\theta=1$, NOT: $\theta=0$ con inhibición) y equivalencia a Máquinas de Turing.
   * **Regla de Hebb (1949):** $w_{t+1} = w_t + \eta x_t y_t$. Demostración de inestabilidad y divergencia sin cota hacia $+\infty$ ante co-activación continua ($w \to 100.0$).
   * **Regla de Oja (1982):** $\Delta w = \eta y (x - yw)$. Término de decaimiento multiplicativo proporcional a $y^2 w$ que estabiliza la norma a $1.0$ ($w \to 1.0$) y extrae el primer componente principal (PCA).
3. **Módulo 12.3 — El Perceptrón de Rosenblatt (1958), Inconsistencia de XOR y Solución MLP (Diaps. 39–48):**
   * Algoritmo de aprendizaje supervisado en línea de Rosenblatt (1958) convergiendo en 1 época sobre la compuerta OR ($b=-1, w=[1,1]$).
   * **Demostración Analítica del Teorema de Minsky & Papert (1969):** Sistema inconsistente de 4 desigualdades ($w_1 \ge T$ y $w_2 \ge T \implies w_1+w_2 \ge 2T > T$, contradiciendo que $w_1+w_2 < T$ para $(1,1)$).
   * **Solución Multicapa XOR:** MLP de 2 capas con 3 neuronas ($h_1 = x_1 \lor x_2, h_2 = \overline{x_1 \land x_2}, y = h_1 \land h_2$), alcanzando 100% de exactitud mediante separación lineal en el espacio transformado $(h_1, h_2)$.
4. **Módulo 12.4 — Clasificador Universal (Dos Triángulos Disjuntos), Regresor de Riemann y Tamaño vs. Activación (Diaps. 49–60):**
   * **Clasificador Universal de 3 Capas y 9 Neuronas:** Dos triángulos disjuntos en $\mathbb{R}^2$ clasificados exactamente con 6 semiplanos en Capa 1 $\to$ 2 compuertas AND en Capa 2 $\to$ 1 compuerta OR en Capa 3.
   * **Aproximador Continuo de Riemann:** Suma finita de pulsos escalón $u(x; T_1, T_2) = \text{step}(x-T_1) - \text{step}(x-T_2)$ para aproximar funciones continuas integrables.
   * **Teorema de Tamaño vs. Activación:** Aproximar funciones armónicas como $\cos(2x)$ demanda $N \to \infty$ escalones de Heaviside, pero exactamente **1 sola neurona** con activación sinusoidal $\sin(2x + \pi/2)$.
   * Línea de tiempo integral del conexionismo (1943–2016).

---

### 🟦 CD13 (UNIDAD I): Representación y Aproximación Universal con Redes Neuronales

1. **Módulo 1.1 — Compuertas de Umbral y Arquitecturas XOR (Diaps. 6–15):**
   * Compuertas de umbral universales de $N$ entradas («al menos $K$ de $N$» con umbral $L - N + K$).
   * Comparación entre el **MLP Clásico de 3 neuronas** (`OR` + `NAND` $\to$ `AND`) y la **Construcción Compacta de Friedland de solo 2 neuronas** ($X + Y - 2(X \land Y) \ge 1$).
2. **Módulo 1.2 — Función Booleana Universal, DNF, Karnaugh y Explosión Exponencial (Diaps. 16–30):**
   * Síntesis automática de cualquier tabla de verdad en una sola capa oculta vía **Forma Normal Disyuntiva (DNF)** y reducción con **Mapas de Karnaugh $4\times 4$** (de 7 a 3 neuronas en la Diap. 19).
   * El peor caso del «tablero de ajedrez» (paridad / XOR de $N$ variables): **$2^{N-1} + 1$ neuronas** en una sola capa oculta frente a **$3(N-1)$ neuronas** en una red profunda en árbol de $2\log_2 N$ capas.
3. **Módulo 1.3 — Clasificador Universal en $\mathbb{R}^2$: Polígonos, Cilindros y Sesgo $-N/2$ (Diaps. 31–48):**
   * Construcción de polígonos convexos de $N$ lados y convergencia al cilindro circular cuando $N \to \infty$, con perfil exterior $\sum_{i=1}^N y_i = N(\frac{1}{2} + \frac{1}{\pi}\arcsin\frac{r}{d}) \to \frac{N}{2}$ y anulación del piso exterior mediante el sesgo $-N/2$.
4. **Módulo 1.4 — Aproximador Universal de Funciones Continuas mediante Pulsos (Diaps. 49–54):**
   * Síntesis de pulsos cuadrados con pares de umbrales $(T_1, T_2)$ y distinción formal entre *representar*, *aproximar* ($\varepsilon > 0$) y *el límite*.
5. **Módulo 1.5 — Suficiencia de la Arquitectura y Autoevaluación I (Diaps. 55–61):**
   * Cuello de botella por ancho insuficiente (franjas vs. celdas), activaciones graduadas ($\sigma, \tanh, \text{ReLU}$) y el triángulo de compensaciones (**Ancho $\leftrightarrow$ Profundidad $\leftrightarrow$ Activaciones**).

---

### 🟪 UNIDAD II: Aprendizaje de Redes Neuronales

1. **Módulo 2.1 — Riesgo Empírico y Regla de Aprendizaje del Perceptrón de Rosenblatt (Diaps. 5–22):**
   * Minimización del **Riesgo Empírico** $\text{TrainLoss}(W) = \frac{1}{N}\sum_{i=1}^N \text{error}(f(X_i; W), d_i)$ frente al error real inaccesible $\int \text{error}(f(X;W), g(X))\,dX$.
   * Truco para absorber el umbral ($x_{N+1}=1, w_{N+1}=-T$) y **Algoritmo en línea de Rosenblatt**: $\mathbf{w} \leftarrow \mathbf{w} + y\mathbf{x}$ si $y(\mathbf{w}\cdot\mathbf{x}) \le 0$.
   * **Teorema de Convergencia:** a lo más $(R/\gamma)^2$ actualizaciones en datos linealmente separables y reproducción exacta del ejemplo en Python de la Diapositiva 22 (`w = [3.25, 1.9]` en 3 pasos).
2. **Módulo 2.2 — ¿Por qué no escala a Multicapa?, ADALINE (LMS) y Señal Diferenciable (Diaps. 23–39):**
   * El problema del **doble pentágono** (no separabilidad lineal de una neurona oculta individual y explosión combinatoria de etiquetas intermedias).
   * **ADALINE (Widrow y Hoff, 1960)** y la **Regla Delta / LMS** $w_i \leftarrow w_i + \eta(d - z)x_i$ antes del escalón.
   * Por qué el **conteo de errores** tiene derivada nula en casi todas partes (función escalonada plana a trozos) mientras que la **sigmoide** $\sigma(z)=\frac{1}{1+e^{-z}}$ entrega una distancia continua y diferenciable en todo punto.
3. **Módulo 2.3 — Regresión Logística, Capa Softmax y Divergencias: L2 vs. Entropía Cruzada / KL (Diaps. 40–54):**
   * Por qué la divergencia **L2** se satura ($\nabla_z \approx 0$) cuando una salida sigmoide está muy equivocada, mientras que la **Entropía Cruzada Binaria** $-d\log Y - (1-d)\log(1-Y)$ cancela la saturación dejando $\frac{\partial \text{div}}{\partial z} = Y - d$.
   * Clasificación multiclase con **Softmax** $y_i = \frac{e^{z_i}}{\sum_k e^{z_k}}$ y equivalencia entre minimizar la **Entropía Cruzada** $-\log y_c$ y la **Divergencia de Kullback-Leibler** $\text{KL}(d \parallel Y)$ cuando $d$ es *one-hot* ($H(d)=0$).
4. **Módulo 2.4 — Síntesis Formal de Representaciones, Pérdidas y Autoevaluación II (Diaps. 14, 34, 54):**
   * Tabla resumen de arquitecturas de salida y divergencias (Regresión, Binaria, Multiclase) y cuestionario interactivo validado sobre las Diapositivas 14 y 34.

---

### 🟩 UNIDAD III: Descenso por Gradiente y Convergencia

1. **Módulo 3.1 — Optimización No Lineal, Geometría del Gradiente y Curvas de Nivel (Diaps. 1–16):**
   * Terreno 1D: mínimos locales vs. globales, máximos y puntos de inflexión con derivada nula ($df(x)/dx = 0$).
   * Expansión diferencial multivariable $\Delta y = \nabla_{\mathbf{x}} f(\mathbf{x})\,\Delta\mathbf{x} = \|\nabla f\|\,\|\Delta\mathbf{x}\|\cos\theta$, dirección de máximo ascenso ($\theta = 0$) y ortogonalidad a las curvas de nivel ($\theta = 90^\circ \implies \Delta y = 0$).
   * Las 3 afirmaciones de la Diapositiva 16: semiespacio de ascenso (**V**), dirección de máximo crecimiento (**V**) y por qué $\nabla_{\mathbf{x}} f = \mathbf{0}$ no implica necesariamente un mínimo (**F**).
2. **Módulo 3.2 — Segunda Derivada, Matriz Hessiana y Ejemplo 3D Resuelto (Diaps. 17–25):**
   * Criterio de la segunda derivada en 1D ($f''(x) > 0$ mínimo, $f''(x) < 0$ máximo) y ejemplo analítico $f(x) = x^2 - 4x + 5 \implies x^* = 2, f(2) = 1$ (Diap. 20).
   * Serie de Taylor de segundo orden y **Matriz Hessiana** $\nabla_{\mathbf{x}}^2 f(\mathbf{x})$: definida positiva ($\mathbf{w}^T\nabla^2 f\,\mathbf{w} > 0$), negativa o indefinida (punto silla).
   * Resolución completa del ejemplo 3D (Diaps. 24–25): $f(x_1,x_2,x_3) = x_1^2 + x_1(1-x_2) + x_2^2 - x_2x_3 + x_3^2 + x_3 \implies \mathbf{x}^* = (-1,-1,-1)$, $f(\mathbf{x}^*) = -1$ con autovalores positivos $\lambda = \{3.414, 2.000, 0.586\} > 0$.
3. **Módulo 3.3 — Descenso Iterativo y 200 Épocas en Regresión Lineal (Diaps. 26–38):**
   * Por qué la solución analítica rara vez existe y cómo actualizar iterativamente $x^{(k+1)} = x^{(k)} - \eta_k \nabla_{\mathbf{x}} f(x^{(k)})^T$.
   * Reproducción exacta de las **200 épocas de Regresión Lineal** sobre $\{(1,1),(2,3),(4,3)\}$ con $\eta = 0.1$ y $\mathbf{w}^{(0)} = [0,0]$ (Diaps. 34–35): $t=1 \implies \nabla=[-4.67, -12.67], \mathbf{w}=[0.47, 1.27]$; $t=2 \implies \nabla=[2.18, 7.24], \mathbf{w}=[0.25, 0.54]$; $t=200 \implies \mathbf{w}^*=[1, 0.57]$.
   * Por qué usar un paso proporcional a $\|\nabla f\|$ evita la oscilación perpetua del paso guiado solo por el signo (Diap. 38).
4. **Módulo 3.4 — Inicialización No Convexa, Curvatura por Dirección y 4 Regímenes de Step Size (Diaps. 39–47):**
   * Sensibilidad al punto inicial $\mathbf{x}^{(0)}$ en superficies no convexas y paso óptimo cuadrático $\eta_{\text{opt}} = (E'')^{-1}$.
   * Curvatura anisotrópica en 2D: $E(\mathbf{w}) = \frac{1}{2}(w_1^2 + 3w_2^2)$ con $\eta_{1,\text{opt}}=1, \eta_{2,\text{opt}}=1/3$ y cota de estabilidad global $\eta < 2\min_i \eta_{i,\text{opt}} = 2/3$ (Diaps. 41–43).
   * Verificación exacta en Python de los **4 regímenes sobre $E(w)=w^2$** tras 20 iteraciones desde $w^{(0)}=1.0$ (Diaps. 44–45): $\eta=0.1 \to 0.0115$, $\eta=0.5 \to 0.0$, $\eta=0.75 \to 9.5\times 10^{-7}$, $\eta=1.1 \to 38.34$ y cuestionario de autoevaluación III.

---

### 🟧 UNIDAD IV: Algoritmo de Backpropagation

1. **Módulo 4.1 — El Problema y las Dos Reglas de la Cadena: Simple vs. Distribuida (Diaps. 5–13 y 18):**
   * Descomposición de la derivada del riesgo empírico: $\frac{\partial\text{TrainLoss}}{\partial w_{ij}^{(k)}} = \frac{1}{T}\sum_{t=1}^T \frac{\partial\text{div}(Y_t, d_t)}{\partial w_{ij}^{(k)}}$.
   * La derivada como factor de amplificación local en cada arista ($\Delta y \approx \frac{dy}{dx}\Delta x$).
   * **Regla de la cadena simple (en serie):** $\frac{dl}{dz} = \frac{dl}{dy}\frac{dy}{dz}$ (*multiplicar por el camino*), con verificación en $b=a^2, c=b^2$ en $a=2 \implies \frac{\partial c}{\partial a} = 8\times 4 = 32 = 4a^3$ (Diap. 18).
   * **Regla de la cadena distribuida (varios caminos):** $\frac{dl}{dx} = \sum_{i=1}^M \frac{\partial l}{\partial z_i}\frac{dz_i}{dx}$ (*multiplicar por cada camino y sumar sobre los caminos*).
2. **Módulo 4.2 — Grafos de Computación, Bloques Básicos y Enrutador `max` (Diaps. 14–23 y 40–41):**
   * Los 6 bloques básicos de construcción (suma, resta, producto, cuadrado, máximo, sigmoide $\sigma'(a)=\sigma(a)(1-\sigma(a))$), pérdida Hinge (Diap. 19) y red de 2 capas como grafo (Diap. 20).
   * Simulación paso a paso de los dos ejemplos numéricos de $\text{Loss}(x,y,\mathbf{w})=(\mathbf{w}\cdot\phi(x)-y)^2$:
     * **Ejemplo 1 (Diap. 21):** $\mathbf{w}=[3,1], \phi(x)=[1,2], y=2 \implies \text{puntaje}=5, \text{residuo}=3, \text{Loss}=9, \nabla_{\mathbf{w}}\text{Loss}=[6,12]$.
     * **Ejemplo 2 (Diap. 23):** $\mathbf{w}=[1,2], \phi(x)=[2,1], y=3 \implies \text{puntaje}=4, \text{residuo}=1, \text{Loss}=1, \nabla_{\mathbf{w}}\text{Loss}=[4,2]$.
   * Subgradientes y función $y = \max(z_1, z_2, z_3)$ evaluada en $(1, 2, 3) \implies [0, 0, 1]$ como enrutador del gradiente hacia el ganador (Diap. 41).
3. **Módulo 4.3 — Forward y Backward Pass en una Red ReLU de 2 Capas Paso a Paso (Diaps. 24–39 y 57):**
   * **Regla de oro (Diap. 29):** Todos los $z_j^{(k)}$ e $y_j^{(k)}$ del *forward pass* se guardan en memoria para evaluar $f_k'(z_j^{(k)})$ e $y_i^{(k-1)}$ en el *backward pass*.
   * **Simetría Forward vs. Backward (Diap. 37):** $\dot{y}_i^{(k)} = \sum_j w_{ij}^{(k+1)}\dot{z}_j^{(k+1)}$, $\dot{z}_i^{(k)} = f_k'(z_i^{(k)})\dot{y}_i^{(k)}$ y $\frac{\partial\text{div}}{\partial w_{ij}^{(k)}} = y_i^{(k-1)}\dot{z}_j^{(k)}$.
   * **Ejemplo numérico completo en Red ReLU de 2 capas (Diaps. 38–39 y 57):** Con $\mathbf{x}=[1,2], d=3$:
     * Forward: $z^{(1)}=[2, -1] \implies y^{(1)}=[2, 0]$, $Y=5$, $\text{div}=(5-3)^2=4$.
     * Backward: $\dot{Y}=\dot{z}^{(2)}=4$, $\nabla_{W_2}=[[8, 0]], \nabla_{\mathbf{b}_2}=[4]$, $\dot{y}^{(1)}=[8, 4] \implies \dot{z}^{(1)}=[8, 0]$ (la neurona inactiva con $z_2=-1<0$ corta el gradiente), $\nabla_{W_1}=\begin{bmatrix}8 & 16\\0 & 0\end{bmatrix}, \nabla_{\mathbf{b}_1}=[8, 0]$.
4. **Módulo 4.4 — Ciclo de Entrenamiento, Vista Vectorial con Jacobianos y Autoevaluación IV (Diaps. 42–60):**
   * **Qué es y qué NO es Backpropagation (Diap. 44):** Backprop calcula las derivadas $\nabla_W\text{div}$; quien actualiza los pesos es el descenso por gradiente.
   * **Vista vectorial y Jacobianos transpuestos (Diaps. 49–58):** $\nabla_{\mathbf{z}_k}\text{div} = J_{\mathbf{y}_k}(\mathbf{z}_k)^\top \nabla_{\mathbf{y}_k}\text{div}$, $\nabla_{\mathbf{y}_{k-1}}\text{div} = W_k^\top\nabla_{\mathbf{z}_k}\text{div}$, $\nabla_{W_k}\text{div} = (\nabla_{\mathbf{z}_k}\text{div})\mathbf{y}_{k-1}^\top$, $\nabla_{\mathbf{b}_k}\text{div} = \nabla_{\mathbf{z}_k}\text{div}$, comparación entre Jacobiano diagonal (ReLU/Sigmoide) y Jacobiano lleno (Softmax), y cuestionario de autoevaluación IV.

---

### 🟥 UNIDAD V: Problemas de Convergencia en el Entrenamiento

1. **Módulo 5.1 — ¿Encuentra Backprop la solución correcta? Experimento de Brady, Raghavan y Slawny (1989) y Puntos «Saboteadores» (Diaps. 7–16):**
   * La divergencia diferenciable ($L_2$, entropía cruzada) es un **sustituto (*proxy*)** del error de clasificación $0\text{-}1$; minimizar el sustituto **no siempre** minimiza el error verdadero.
   * **Experimento clásico de 3 puntos:** $(1,0)\to +1$, $(0,1)\to +1$, $(-1,0)\to -1$ con margen $\varepsilon$ ($u = \sigma^{-1}(1-\varepsilon)$) tiene un mínimo único en $w_1 = w_2 = u, b = 0$ (recta $x_1 + x_2 = 0$).
   * **El cuarto punto «saboteador» (*spoiler*) lejano $(0, -t)\to +1$:** Con $t$ grande, los 4 puntos siguen siendo linealmente separables (el Perceptrón los separa al 100%), pero en la solución de 3 puntos $z_4 = -ut \to -\infty$ satura la sigmoide: $\sigma'(z_4)\sim e^{-ut}$ cae exponencialmente mientras $t$ crece solo linealmente, haciendo que $\frac{d\,\text{div}_4}{dw_2} = 2(1-\varepsilon-\sigma(z_4))\sigma'(z_4)t \to 0$. *Backprop* queda atrapado en el mínimo local ancho acertando solo $3/4$ (75%).
   * **Compensación sesgo vs. varianza (Diap. 14):** El Perceptrón tiene sesgo bajo y varianza alta; *Backprop* con pesos acotados puede sesgar ante *spoilers*, pero tiene varianza baja.
2. **Módulo 5.2 — La Superficie de Pérdida, las 6 Trayectorias desde $(-8, 4)$ y el Número de Condición $\kappa$ (Diaps. 17–33):**
   * **Puntos de silla** ($E = \frac{1}{2}(w_1^2 - w_2^2)$) y evidencia teórica/empírica en redes grandes: **Baldi y Hornik (1989)**, **Dauphin et al. (2014)**, **Choromanska et al. (2015)** y **Swirszcz et al. (2016)**.
   * Reproducción exacta de las **6 trayectorias de la Diapositiva 27** sobre $E = \frac{1}{2}(w_1^2 + 3w_2^2)$ desde $\mathbf{w}^{(0)} = (-8, 4)$ con $\eta \in \{0.25, 1/3, 0.5, 2/3, 0.75, 1.0\}$.
   * **Número de condición** $\kappa = a_{\max}/a_{\min}$, factor geométrico óptimo $\frac{\kappa-1}{\kappa+1}$ en $\eta^* = \frac{2}{a_{\max}+a_{\min}}$, y comparación entre funciones **fuertemente convexas** ($O(\log(1/\delta))$ iteraciones) y funciones **convexas solo Lipschitz-suaves** ($O(1/\delta)$ iteraciones).
3. **Módulo 5.3 — Métodos de Segundo Orden: Escalamiento $A^{1/2}$, Método de Newton y la Trampa del Punto de Silla (Diaps. 34–44):**
   * Cambio de coordenadas $\hat{\mathbf{w}} = A^{1/2}\mathbf{w}$ ($A = U\Lambda U^\top$) que vuelve circulares los contornos ($\kappa = 1$) y **Regla de Newton iterada** $\mathbf{w}^{(k+1)} = \mathbf{w}^{(k)} - \eta H_E(\mathbf{w}^{(k)})^{-1}\nabla_{\mathbf{w}}E(\mathbf{w}^{(k)})$, que converge en **un solo paso** ($\eta = 1$) sobre cualquier cuadrática.
   * **Los 2 problemas de Newton:** (1) Tamaño cuadrático del Hessiano ($N=100\,000 \implies N^2 = 10^{10}$ entradas e inversión $O(N^3)$) y (2) **Curvatura negativa**: cerca de un punto de silla, los valores propios negativos atraen a Newton directo hacia la silla, problema que se mitiga con métodos cuasi-Newton (**BFGS / L-BFGS**) o **Levenberg-Marquardt** ($H + \lambda I$).
4. **Módulo 5.4 — Tasas Decrecientes, RProp, QuickProp y Autoevaluación V (Diaps. 45–58):**
   * **Divergir para escapar:** partir con $\eta_0$ grande para saltar de mínimos locales mediocres y reducirlo con esquemas $\eta_0/k$, $\eta_0/k^2$, $\eta_0 e^{-\beta(k-1)}$ o por escalones.
   * **RProp (*Resilient Propagation*, Diaps. 48–53):** usa solo $\text{signo}(\partial E/\partial w_i)$ con tamaño de paso propio $\Delta_i$ ($\alpha = 1.2$ si mantiene el signo; si cambia de signo, deshace el paso y encoge $\Delta_i \leftarrow \beta\Delta_i$ con $\beta = 0.5$).
   * **QuickProp (Diaps. 54–56):** aproxima Newton coordenada a coordenada por diferencia finita de derivadas: $\Delta w^{(k)} = \Delta w^{(k-1)}\frac{E'(w^{(k)})}{E'(w^{(k-1)}) - E'(w^{(k)})}$, más adelanto de **Momentum** y cuestionario de autoevaluación V.

---

### 🟦 UNIDAD VI: Descenso por Gradiente Estocástico y Lotes

1. **Módulo 6.1 — El Costo del Lote, «SGD a Mano: Una Época» y el Barajado por Época (Diaps. 5–14):**
   * **El costo del lote:** Con $|\mathcal{D}_{\text{train}}| = 1\,000\,000$, una actualización de lote evalúa $1\,000\,000$ de gradientes para mover $\mathbf{w}$ una sola vez. Tras $T$ épocas con $N$ ejemplos, el lote mueve $\mathbf{w}$ $T$ veces y SGD lo mueve $TN$ veces con el mismo número de evaluaciones de gradiente ($TN$).
   * **Reproducción exacta de «SGD a mano: una época» (Diap. 14):** Sobre $\{(1,1),(2,3),(4,3)\}$ con $\phi(x)=[1,x]$, $\eta=0.1$, $\mathbf{w}^{(0)}=[0,0]$ y $\nabla_{\mathbf{w}}\text{Loss} = 2r\phi(x)$:
     1. $(1,1) \implies r=-1.00, \nabla=[-2.00,-2.00], \mathbf{w}_{\text{nuevo}}=[0.20, 0.20]$.
     2. $(2,3) \implies r=-2.40, \nabla=[-4.80,-9.60], \mathbf{w}_{\text{nuevo}}=[0.68, 1.16]$.
     3. $(4,3) \implies r=+2.32, \nabla=[4.64,18.56], \mathbf{w}_{\text{nuevo}}=[0.22, -0.70]$ (vs. 1 paso de lote $\mathbf{w}=[0.47, 1.27]$ y óptimo $[1.00, 0.57]$).
   * **Cuidado con el orden de presentación (Diaps. 12–13):** Si los datos vienen agrupados (Grupo A luego Grupo B) y se recorren siempre en el mismo orden, los pesos oscilan en ciclos; permutar $\mathcal{D}_{\text{train}}$ al azar en **cada época** (*shuffling*) es parte integral del SGD.
2. **Módulo 6.2 — Tasa de Aprendizaje Decreciente, Condiciones de Robbins-Monro (1951) y Experimento en Python (Diaps. 15–20):**
   * **Condiciones suficientes de convergencia (Robbins y Monro, 1951, Diap. 17):** $\sum_{k=1}^\infty \eta_k = \infty$ (permite alcanzar cualquier punto) y $\sum_{k=1}^\infty \eta_k^2 < \infty$ (amortigua el ruido del último ejemplo). Con pérdida fuertemente convexa, $\eta_k \propto 1/k$ es óptimo ($\sum 1/k^2 = \pi^2/6$).
   * **Velocidad de convergencia (Diap. 18):** Para llegar a distancia $\varepsilon$ del óptimo, el lote requiere $O(\log(1/\varepsilon))$ iteraciones de costo $N$, mientras que SGD requiere $O(1/\varepsilon)$ iteraciones de costo $1$.
   * **Experimento en Python (Diap. 19):** Con $\eta_k = 1/\sqrt{k}$, tras recorrer apenas $1/10$ de la primera época, SGD ya se acerca a $\mathbf{w}_{\text{real}}$ antes de que el descenso por lote complete su primera actualización.
3. **Módulo 6.3 — ¿Por qué Funciona? La Varianza del Estimador $\text{Var}(\text{Loss})/b$ y Mini-lotes en la Práctica (Diaps. 21–29 y 39):**
   * **Estimador insesgado:** $\mathbb{E}[\text{Loss}(x,y,\mathbf{w})] = \mathbb{E}[\text{TrainLoss}(\mathbf{w})]$, pero la varianza de un solo ejemplo es $N$ veces mayor que la del lote completo ($\text{Var}(\text{TrainLoss}) = \frac{1}{N}\text{Var}(\text{Loss})$).
   * **Mini-lotes de tamaño $b$ (Diaps. 25–29):** Promediar $b$ ejemplos reduce la varianza exactamente a $\frac{1}{b}\text{Var}(\text{Loss})$, aprovecha la vectorización matricial en GPU y recupera SGD cuando $b=1$ y el lote completo cuando $b=N$.
4. **Módulo 6.4 — Optimizadores con Tendencia: Momentum, Nesterov, RMSprop y Adam (Diaps. 30–38):**
   * **Momentum (Diaps. 31–32) y Nesterov (Diap. 33):** Momentum acumula un promedio móvil exponencial de los pasos previos ($\Delta\mathbf{w}^{(k)} = \beta\Delta\mathbf{w}^{(k-1)} - \eta\nabla_{\mathbf{w}}\text{Loss}(\mathbf{w}^{(k-1)})$); **Nesterov** primero aplica el paso de inercia y evalúa el gradiente allá donde va a caer ($\mathbf{w}^{(k-1)} + \beta\Delta\mathbf{w}^{(k-1)}$).
   * **Tasa por dirección y RMSprop (Diaps. 34–35):** En los 5 pasos de la Diap. 34 ($\Delta x = [1, 1, 2, 1, 1.5] \implies \text{total}=6.5, \text{neto}=6.5, \text{RMS}\approx 1.36$; $\Delta y = [+2.5, -3, +2.5, -2, +1.5] \implies \text{total}=11.5, \text{neto}=1.5, \text{RMS}\approx 2.36$), RMSprop divide por $\sqrt{E[\partial_w^2]^{(k)} + \epsilon}$ para frenar la dirección oscilante y acelerar la horizontal.
   * **Adam (Diap. 36) y Trayectorias en movimiento (Diaps. 37–38):** Combina el primer momento $\hat{m} = \frac{m^{(k)}}{1-\beta_1^k}$ y el segundo momento $\hat{v} = \frac{v^{(k)}}{1-\beta_2^k}$ ($\beta_1=0.9, \beta_2=0.999, \eta=0.001$), amortiguando oscilaciones en valles angostos y escapando rápidamente de puntos de silla, más cuestionario de autoevaluación VI.

---

### 🟪 UNIDAD VII: Regularización de Redes Neuronales (46 Diapositivas)

1. **Módulo 7.1 — Divergencias, Saturación Sigmoidal y Suavidad de la Red (Diaps. 2–14):**
   * **L2 vs. Divergencia KL con Softmax:** En saturación logística ($z = 6, d = 0 \implies y \approx 0.9975$), la derivada $\sigma'(6) \approx 0.0025$ aplana el gradiente $L_2$ a $(y-d)\sigma' \approx 0.00246$ a pesar del error máximo. Con Softmax + KL, $\nabla_z \text{div} = \mathbf{y} - \mathbf{d}$ (en cátedra: $[0.2, 0.5, 0.3] - [0, 1, 0] = [0.2, -0.5, 0.3]$).
   * **Subespecificación de datos:** En $\{0,1\}^{100}$ hay $2^{100} \approx 1.267 \times 10^{30}$ puntos; incluso con $10^{15}$ ejemplos observados falta un factor de $10^{15}$. Controlar $\prod \|W_k\|$ acota la constante de Lipschitz $\text{Lip}(f) \le \prod \|W_k\|$, forzando transiciones lentas ($w=0.5 \implies \max |f'|=0.125$ vs $w=5.0 \implies \max |f'|=1.25$).
2. **Módulo 7.2 — Regularización L2 y Decaimiento de Pesos (Weight Decay 1-ηλ) (Diaps. 15–20):**
   * **Objetivo regularizado:** $\min_W \frac{1}{T}\sum \text{div} + \frac{1}{2}\lambda \sum \|W\|_F^2$.
   * **Regla de actualización con decaimiento:** $W \leftarrow (1 - \eta\lambda)W - \frac{\eta}{b}\sum \nabla\text{div}$. Ejemplo exacto de cátedra: con $\eta = 0.1, \lambda = 0.1$, el factor es $1 - 0.01 = 0.99$ (encoge los pesos un $1\%$ por paso).
   * **Espectro de $\lambda$:** $\lambda = 0$ (sobreajuste), $\lambda$ óptimo (generalización), $\lambda \to \infty$ (pesos a cero, sesgo libre colapsa a la función constante $\bar{y}$).
3. **Módulo 7.3 — Dropout y Dropout Invertido (Srivastava et al., 2014) (Diaps. 21–30):**
   * **Desconexión Bernoulli:** Retención $\alpha$ (típico $0.5$ en capa oculta, $0.8$ en entrada). Previene la co-adaptación de neuronas y entrena un ensamble implícito de $2^N$ subredes con pesos compartidos (Bagging).
   * **Dropout Tradicional vs Invertido:** Con $f(z) = 2.0, \alpha = 0.8$, tradicional propaga $2.0$ y en prueba escala por $0.8 \to 1.6$. Dropout Invertido escala en entrenamiento por $1/\alpha = 1.25$ ($2.0/0.8 = 2.50$, $\mathbb{E}[y] = 2.0$), permitiendo inferencia directa como identidad sin escalamiento adicional.
   * **Variantes:** DropConnect (pesos), Zoneout (RNNs), Whiteout (ruido gaussiano aditivo).
4. **Módulo 7.4 — Trucos del Oficio y Batch Normalization (Ioffe & Szegedy, 2015) (Diaps. 31–46):**
   * **Arsenal del Ingeniero:** Early Stopping, Grokking (Power et al., 2022), Gradient Clipping ($\theta = 5.0$) y Aumento de Datos.
   * **Batch Normalization:** $\mu_B, \sigma_B^2, u_i = (z_i - \mu_B)/\sqrt{\sigma_B^2+\epsilon}, \hat{z}_i = \gamma u_i + \beta$.
   * **Micro-ejemplos exactos de cátedra:** Lote $z = \{1, 2, 3, 6\} \implies \mu_B = 3.0, \sigma_B^2 = 3.5 \implies u \approx \{-1.07, -0.53, 0.00, 1.60\}$. Lote $z = \{2, 4, 6, 8\} \implies \mu_B = 5.0, \sigma_B^2 = 5.0 \implies u \approx \{-1.34, -0.45, 0.45, 1.34\}$. Factor de Bessel $B/(B-1) = 4/3 \approx 1.333$ en inferencia.

---

### 🟦 UNIDAD VIII: Redes Neuronales Convolucionales (57 Diapositivas)

1. **Módulo 8.1 — ¿Por qué las Imágenes Rompen a los MLP? (Diaps. 5–13):**
   * **Aplanamiento 1D y Explosión Combinatoria:** ImageNet ($224\times 224\times 3 = 150\,528$ entradas) con 128 neuronas en primera capa oculta requiere **$19\,267\,712$ parámetros** ($128 \times 150\,529$ pesos + sesgos); caso modesto ($100\times 100\times 1$, 64 neuronas) exige **$640\,064$ parámetros**.
   * **Falta de Invarianza a Traslaciones:** En un MLP, un objeto en la esquina superior izquierda activa pesos completamente distintos que en la esquina inferior derecha.
   * **La Historia del Cisne y Visión Clásica:** Fracaso de reglas manuales rígidas y detectores artesanales (SIFT, SURF, FAST). Fundamentos de CNN: *Localidad Espacial* (correlación de vecindad) y *Jerarquía Visual* (bordes $\to$ partes $\to$ objetos).
2. **Módulo 8.2 — Convolución 2D, Filtros, Padding y Stride (Diaps. 14–28):**
   * **Suma ponderada local con pesos compartidos:** $h = \sum w_{ij} x_{ij} + b$.
   * **Micro-ejemplo exacto de cátedra (Diaps. 17–21):** Imagen $5 \times 5$ con borde vertical (columnas $[3,3,3,0,0]$) y filtro vertical $3\times 3$ ($[1,0,-1]$ en cada fila, $b=0$) produce exactamente una salida $3\times 3$ con filas $[0, 9, 9]$, detectando la transición del borde.
   * **Fórmula analítica de dimensiones de salida:** $O = \lfloor(W - K + 2P)/S\rfloor + 1$. Verificada en los 3 casos de MNIST ($28\times 28$, $K=5$): Valid $24\times 24$, Same ($P=2$) $28\times 28$, y Stride 2 $14\times 14$. Foto de gato $140\times 140$ con filtro $3\times 3 \implies \mathbf{19\,044}$ paradas de filtro.
3. **Módulo 8.3 — Pooling (Max vs Mean) y No Linealidad ReLU (Diaps. 29–35):**
   * **Micro-ejemplo de cátedra en matriz $4\times 4$ (Diap. 29):** Max Pooling $2\times 2$ ($S=2$) produce $[[7, 8], [3, 4]]$, Mean Pooling produce $[[3.50, 5.50], [1.75, 2.00]]$, con **0 parámetros aprendibles**.
   * **Reducción espacial y robustez:** 3 rondas sucesivas de Max Pooling $2\times 2$ en foto $560\times 560$ la reducen a $70\times 70$ (factor $64\times$), proveyendo invarianza a pequeñas traslaciones.
   * **ReLU:** $\text{ReLU}(z) = \max\{0, z\}$ aplicada elemento a elemento para preservar gradientes positivos.
4. **Módulo 8.4 — Desglose Arquitectónico, Conteo de Parámetros (99.8% FC) y Transfer Learning (Diaps. 36–57):**
   * **Conteo a mano de la red de cátedra ($32\times 32\times 3 \to 4$ clases):** Conv1 ($224$), Conv2 ($3\,216$) $\to$ Subtotal Conv $3\,440$ ($0.16\%$). Flatten $4\,096$ valores. FC1 ($2\,097\,664$), FC2 ($2\,052$) $\to$ Subtotal Denso $2\,099\,716$ ($99.84\%$). **Total Red: $2\,103\,156$ parámetros** ($99.8\%$ en bloque denso). Capa intermedia ($14\times 14\times 16$, 32 filtros $3\times 3$) $\implies 4\,640$ parámetros.
   * **Matriz 2×2 Transfer Learning:** Similitud de dominio vs Tamaño del dataset. Escenarios: 600 fotos fauna altiplánica (congelar base + nueva cabeza), 2M fotos industriales (fine-tuning completo), 400 espectrogramas sísmicos. Tasas de aprendizaje decrecientes: $\eta_{\text{temprana}} = \frac{1}{3}\dots\frac{1}{10}\eta_{\text{tardía}}$.
   * **Tres Hitos Históricos:** LeNet-5 (1998, cheques MNIST, ~60k params), AlexNet (2012, ImageNet 16.4% top-5, 60M params, GPU CUDA+ReLU+Dropout), ResNet (2015, bloques residuales $\mathbf{x}+F(\mathbf{x})$, 152 capas, 3.6% top-5 superando al humano).

---

## 🚀 Cómo Ejecutar los Laboratorios Educativos

### 1. 🌐 Suite Web Educativa Interactiva (19 Unidades: CD01–CD11 y CD13–CD20 • 77 Laboratorios en Navegador)
```bash
open web/index.html
# O desde Python:
python3 perceptron_xor.py --web
```

### 2. ⚡ Laboratorios en Consola por Unidad (Python Puro, Orden de Cátedra)
```bash
# Ejecutar TODAS las verificaciones de CD01–CD11 y CD13–CD20 automáticamente (19 Unidades):
python3 perceptron_xor.py --auto

# Laboratorio interactivo de CD01 (Primera en orden — Introducción al Curso):
python3 introduccion_curso.py

# Laboratorio interactivo de CD02 (Segunda en orden — Preprocesamiento y EDA):
python3 preprocesamiento_eda.py

# Laboratorio interactivo de CD03 (Tercera en orden — Visualización Efectiva y Engañosa):
python3 visualizacion_datos.py

# Laboratorio interactivo de CD04 (Cuarta en orden — Introducción a Machine Learning):
python3 introduccion_ml.py

# Laboratorio interactivo de CD05 (Quinta en orden — Generalización y Validación):
python3 generalizacion_validacion.py

# Laboratorio interactivo de CD06 (Sexta en orden — Clustering: K-Means, Rand Index, Dendrogramas y DBSCAN):
python3 clustering.py

# Laboratorio interactivo de CD07 (Séptima en orden — Reducción de Dimensionalidad: PCA, Fisher LDA, t-SNE, UMAP):
python3 reduccion_dimensionalidad.py

# Laboratorio interactivo de CD08 (Octava en orden — Árboles CART e ID3, Impureza, Poda y Regresión):
python3 arboles_decision.py

# Laboratorio interactivo de CD09 (Novena en orden — Ensambles, Bagging, Random Forest y Boosting):
python3 ensambles_random_forest.py

# Laboratorio interactivo de CD10 (Décima en orden — kNN, Feature Maps, SVM y Truco del Kernel):
python3 modelos_no_lineales.py

# Laboratorio interactivo de CD11 (Undécima en orden — Aspectos Prácticos y Evaluación de Modelos):
python3 aspectos_practicos.py
# (Atajos directos desde el lanzador: python3 perceptron_xor.py --practica / --evaluacion / --eval / --roc / -c11)

# Laboratorio interactivo de CD12 (Duodécima en orden — Redes Neuronales y Conexionismo):
python3 redes_conexionismo.py
# (Atajos directos desde el lanzador: python3 perceptron_xor.py --conexionismo / --conexion / --hebb / --mcculloch / -c12)

# Laboratorio interactivo de CD13 / Unidad I (Aproximación Universal):
python3 aproximacion_universal.py

# Laboratorio interactivo de CD14 / Unidad II (Aprendizaje de Redes Neuronales):
python3 aprendizaje_redes.py

# Laboratorio interactivo de CD15 / Unidad III (Descenso por Gradiente):
python3 descenso_gradiente.py

# Laboratorio interactivo de CD16 / Unidad IV (Algoritmo de Backpropagation):
python3 backpropagation.py

# Laboratorio interactivo de CD17 / Unidad V (Problemas de Convergencia en el Entrenamiento):
python3 problemas_convergencia.py

# Laboratorio interactivo de CD18 / Unidad VI (Descenso por Gradiente Estocástico y Lotes):
python3 sgd_lotes.py

# Laboratorio interactivo de CD19 / Unidad VII (Regularización de Redes Neuronales):
python3 regularizacion_redes.py

# Laboratorio interactivo de CD20 / Unidad VIII (Redes Neuronales Convolucionales - CNNs):
python3 redes_convolucionales.py

# Menú principal integrado con navegación secuencial:
python3 perceptron_xor.py
```

---

## 📁 Estructura del Proyecto
```
├── README.md                    # Guía general ordenada por código de cátedra (CD01 a CD12, luego CD13 a CD20)
├── INFORME.md                   # Informe académico detallado de las 20 unidades (CD01–CD12 y CD13–CD20)
├── perceptron_xor.py            # Lanzador central y verificador integrado (20 unidades • 81 laboratorios)
├── introduccion_curso.py        # CD01 (Primera en orden): Pandemia, SIR (Diap. 13), Netflix Prize, 3 Relatos IA y LLMs
├── preprocesamiento_eda.py      # CD02 (Segunda en orden): Sesgos Gallup/Yelp/QR, CRISP-DM, Pandas epl_mini (23 jug.) y Anscombe
├── visualizacion_datos.py       # CD03 (Tercera en orden): Datasaurus, Burtin 1951 (16 bacterias), Tufte Lie Factor, Stevens, Correll y Mirages
├── introduccion_ml.py           # CD04 (Cuarta en orden): Mapa IA 4 Niveles, Regresión Gauss 1801, Hinge/Margen, Logística y Softmax
├── generalizacion_validacion.py # CD05 (Quinta en orden): Sobreajuste, Cauchy-Schwarz, Paso L2 a Mano [1.8, -2.0], CV 5-Fold vs Simple y Sesgo-Varianza
├── clustering.py                # CD06 (Sexta en orden): Lloyd 1D/2D, Codo K=3, Rand Index vs ARI, Dendrogramas y DBSCAN vs HDBSCAN
├── reduccion_dimensionalidad.py # CD07 (Séptima en orden): Maldición Dim P_d, PCA Paso a Paso (6 pts), Fisher LDA (8 pts), t-SNE y UMAP
├── arboles_decision.py          # CD08 (Octava en orden): Particionado Ortogonal, Impureza Error/Gini/Entropía, ID3, Weakest Link y CART Regresión
├── ensambles_random_forest.py   # CD09 (Novena en orden): Galton 1907 (787 votos), Condorcet, Bagging (1-1/e), ESL 15.1, Random Forest m=√J y AdaBoost
├── modelos_no_lineales.py       # CD10 (Décima en orden): kNN (Diap. 8), Feature Maps Cover N=2D, SVM Margen Máximo y Truco Kernel
├── aspectos_practicos.py        # CD11 (Undécima en orden — Cierre Fase 1): Kolter, Rubin Imputación, Triage 251 Pacientes, ROC y McSherry COST
├── redes_conexionismo.py        # CD12 (Duodécima en orden — Puente Histórico): Bain 1873, McCulloch-Pitts, Hebb vs Oja, Límite XOR y MLP
├── aproximacion_universal.py    # CD13 (Unidad I): DNF, Karnaugh, Paridad N-aria, Polígonos y Pulsos
├── aprendizaje_redes.py         # CD14 (Unidad II): Rosenblatt (Diap. 22), ADALINE, Sigmoide, Softmax y KL
├── descenso_gradiente.py        # CD15 (Unidad III): Gradiente, Hessiana 3D (Diap. 24), 200 Épocas y Step Size
├── backpropagation.py           # CD16 (Unidad IV): Reglas de la Cadena, Grafos (Diaps. 21/23), Red ReLU 2 Capas y Jacobianos
├── problemas_convergencia.py    # CD17 (Unidad V): Brady et al. (1989), 6 Trayectorias, Condición κ, Newton, RProp y QuickProp
├── sgd_lotes.py                 # CD18 (Unidad VI): SGD a mano (Diap. 14), Robbins-Monro, Varianza 1/b, Momentum, Nesterov, RMSprop y Adam
├── regularizacion_redes.py      # CD19 (Unidad VII): L2 vs KL, Suavidad, Weight Decay (1-ηλ), Dropout/Invertido y BatchNorm
├── redes_convolucionales.py     # CD20 (Unidad VIII): MLP vs Convolución, Micro-ejemplo 5x5, Pooling 4x4, Parámetros (99.8% FC) e Hitos
├── gui_xor.py                   # Interfaz gráfica de escritorio nativa (Tkinter)
├── web/
│   └── index.html               # Suite Web Interactiva (20 Unidades: CD01–CD12 y CD13–CD20 • 81 Laboratorios)
└── contexto/                    # Historial secuencial de planes de trabajo (context1..context26)
```
