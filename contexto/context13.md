# Contexto 13: Integración de la Unidad CD02 (Segunda en Orden) — Preprocesamiento y Análisis Exploratorio de Datos (74 Diapositivas)
**Fecha de registro:** 2026-11-09

## 1. Instrucciones
* Incorporar la nueva presentación académica de 74 diapositivas (**«CD02-Preprocesamiento-y-EDA.pdf»**, Ciencia de Datos — UCN, basado en material de Harvard CS109 de P. Protopapas, K. Rader et al.) en la **segunda posición** del proyecto (inmediatamente después de `CD01: Introducción al Curso` y antes de `CD13`–`CD18`), respetando estrictamente el orden numérico del nombre de archivo (`CD01`, `CD02`, ..., `CD13`–`CD18`).
* Desarrollar el módulo computacional interactivo y autoverificable en Python puro (`preprocesamiento_eda.py`) que implemente y verifique numéricamente las 4 secciones de la unidad `CD02`:
  1. **¿Qué son los datos?, Advertencias de Fábrica, Alcances y Sesgos de Muestreo (Diaps. 1–20):**
     * Ubicación antes del mapa de la IA y en los **pasos 2 y 3 del proceso de 5 pasos: obtener y explorar** (los que consumen la mayor parte del tiempo real de un proyecto).
     * Las 3 definiciones de «datos» (Diccionario, Informática y Señales) y la distinción entre *datum* (una observación), *data* (múltiples observaciones) y *dataset* (colección **homogénea** con mismo enfoque y estructura).
     * **Advertencias de fábrica en los 4 escenarios y las 4 fuentes de trabajo cotidiano (Diaps. 7–8 y 11):**
       - *Termómetro cada hora* / *GPS de app de transporte* $\to$ **vacíos** (cortes de luz, túneles sin señal).
       - *Diario del colibrí* / *Cuaderno de asistencia a un taller* $\to$ **imprecisión humana**.
       - *Tweets de un político* / *Biografías de una red social* $\to$ **veracidad** (decir no es ser).
       - *Sensor misterioso de feria persa* / *Columna `v37` sin documentar* $\to$ **sin semántica**.
     * **Alcances (Integrales vs. Muestreados), los 5 sesgos de la fuente (Diaps. 13–15) y el contraste Gallup vs. Yelp + Encuesta QR de la Cafetería (Diaps. 16–20):**
       - *Gallup (muestreo diseñado):* ~500 llamadas/día al azar (RDD), regla del cumpleaños más reciente dentro del hogar, cobertura de 50 estados con mezcla **70% celulares / 30% fijos** y **ponderación demográfica** $\to$ describe a la **población general**.
       - *Yelp (autoselección):* el opinante decide, sin corrección posterior, sobrerrepresenta extremos (la maravilla y el desastre) $\to$ describe **a quien opina en Yelp**.
       - *Encuesta QR en la cafetería universitaria (Diap. 20):* 480 respuestas en un mes, promedio **2.1 de 5**; operan **autoselección** y **cobertura**, por lo que no mide satisfacción general pero sí sirve para identificar qué reclama quien está descontento.
  2. **Formatos de Datos, Hacer las Preguntas Correctas, Metodologías CRISP-DM / KDD / SEMMA y Web Scraping Responsable (Diaps. 21–44):**
     * **Matriz 2×2 de dificultad según el lector (Personas vs. Computadores, Diap. 21):** Fácil/Fácil = **Tabla CSV**; Fácil/Difícil = **Prosa, fotos** (papeleta electoral confusa); Difícil/Fácil = **JSON anidado, QR**; Difícil/Difícil = **Audio con ruido**.
     * Comparación de formatos Texto plano vs. CSV/TSV (trampa de la coma dentro de `"Newcastle, England"`) vs. XML vs. JSON (Diaps. 22–26).
     * **Transformación de titulares vagos en preguntas medibles (Diaps. 27–29):** agregar **población, métrica y ventana temporal** (ej. participación electoral presidencial en Chile 2013–2021 con datos del Servel, y promedio de libros leídos al año por jóvenes de 18 a 29 años en Chile entre 2014 y 2023).
     * **Metodologías con nombre: CRISP-DM (2000), KDD (Fayyad et al., 1996) y SEMMA (SAS) (Diaps. 30–35):** las 6 fases de CRISP-DM (Comprensión del negocio $\leftrightarrow$ Comprensión de los datos $\to$ Preparación de los datos $\leftrightarrow$ Modelado $\to$ Evaluación $\to$ Despliegue), por qué KDD y SEMMA omiten las dos puntas críticas (**Comprensión del negocio** y **Despliegue**), y su calce con los 3 hitos del curso (Propuesta = 1+2, Avance = 3+4, Informe final = 5+6).
     * **Obtención de datos y Web Scraping Responsable (Diaps. 36–44):** orden de preferencia $\text{Archivo} > \text{API} > \text{Scraping}$, las 5 reglas del scraping responsable (`robots.txt`, ToS, pausas + `User-Agent`, licencias/privacidad, crédito a la fuente), árbol DOM con `requests` (`status_code == 200` vs `404`) y `BeautifulSoup` (`soup.title.text -> 'Tabla de posiciones'`, `[a["href"] for a in soup.find_all("a")] -> ['/equipo/arsenal', '/equipo/chelsea']`).
  3. **Pandas Esencial y Caso Real de EDA: Premier League (`epl_mini`, 23 jugadores, Cosecha 2017) (Diaps. 45–70):**
     * Verificación exacta de `df2 = DataFrame({"a": [1, 4, 3], "b": [5, 2, 6], "c": [7, 8, 9]})`: indexación por columna, máscara booleana `df2["a"] == 4`, diferencia entre `loc` (etiqueta) e `iloc` (posición), el trío de caza de extremos (`min() -> 2`, `idxmin() -> 1`, `loc`) y el ejercicio `df2.loc[df2["a"] > 2, "c"] -> [8, 9]` (Diap. 54), junto con las 3 correcciones de Pandas moderno (`.values` sin paréntesis, `.get_group()` en singular y `pd.concat` en lugar de `.append()`, Diap. 53).
     * **Reconstrucción matemática 100% exacta del dataset `epl_mini.csv` (23 jugadores, 5 columnas, Diaps. 46, 52, 60, 63–70):**
       - `df.shape = (23, 5)` (`name`, `club`, `age`, `position`, `market_value`).
       - Promedios por club (`groupby("club")["market_value"].mean()`, Diap. 52): `Arsenal = 32.80`, `Chelsea = 35.01`, `Liverpool = 34.00`, `Man City = 46.67`, `Man United = 45.75`, `Tottenham = 41.25`.
       - Resumen global `describe()` (Diap. 63): `age` (`count=23, mean=27.13, std=3.99, min=21.00, 25%=24.00, 50%=26.00, 75%=30.50, max=35.00`) y `market_value` (`count=23, mean=38.87, std=23.51, min=0.05, 25%=21.00, 50%=40.00, 75%=52.50, max=90.00`).
       - Caza de extremos y validación externa (Diaps. 64–66): `idxmin() -> 8` (`Eduardo`, `Chelsea`, `age=34`, `GK`, `market_value=0.05`, tercer arquero veterano en 2017 $\implies$ dato real: *extremo no es sinónimo de error*) e `idxmax()` (`Paul Pogba`, `Man United`, `age=24`, `market_value=90.0`, traspaso récord mundial de 2016 $\implies$ dato coherente).
       - Tabla por edad `groupby("age")["market_value"].agg(["mean", "median"])` (Diap. 68), destacando `age=23` donde `Harry Kane` (`60.0M`) separa la media (`38.33`) de la mediana (`30.00`).
  4. **Media vs. Mediana en el Caso del Arsenal y el Cuarteto de Anscombe (1973) (Diaps. 71–74):**
     * **Caso Arsenal (Diaps. 60 y 72):** los 5 jugadores (`Alexis Sanchez = 65`, `Mesut Ozil = 50`, `Petr Cech = 7`, `Theo Walcott = 20`, `Laurent Koscielny = 22`) tienen media $\mathbf{32.80}$ y mediana $\mathbf{22.00}$; al simular un fichaje de $100\text{ M}$, la media salta a $\mathbf{44.00}$ ($+34.1\%$) mientras la mediana pasa a $\mathbf{36.00}$.
     * **El Cuarteto de Anscombe (1973) (Diap. 71):** verificación de los 4 conjuntos de $n=11$ puntos que comparten idénticas estadísticas descriptivas ($\bar{x}=9.0$, $\bar{y}\approx 7.50$, $s_x^2=11.0$, $s_y^2\approx 4.125$, $r\approx 0.816$, recta $y = 3.00 + 0.500x$) pero exhiben estructuras visuales completamente distintas (relación lineal ruidosa, parábola determinista sin ruido, línea perfecta con un *outlier* vertical y línea vertical con un punto de alto apalancamiento).
* Actualizar `web/index.html` situando **CD02: Preprocesamiento y Análisis Exploratorio de Datos (74 Diaps.)** en **segunda posición** dentro del selector `.unit-switcher` (justo después de `CD01` y antes de `CD13`), incorporando 4 nuevos laboratorios interactivos (`panel-29` a `panel-32`) y 4 nuevas preguntas de autoevaluación (`quiz-item-29` a `quiz-item-32`), alcanzando **8 Unidades Académicas (460 Diapositivas) y 33 Laboratorios Interactivos** ordenados por código de cátedra (`CD01`, `CD02`, `CD13`–`CD18`).
* Integrar `preprocesamiento_eda.py` en segunda posición dentro de `perceptron_xor.py` (`--eda` / `-e` / `--cd02`, menú interactivo y segundo bloque de `--auto`), `README.md` e `INFORME.md`.

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Integrar teórica, computacional y visualmente la segunda unidad del curso **CD02 — Preprocesamiento y Análisis Exploratorio de Datos (74 Diapositivas)** inmediatamente después de `CD01` en toda la suite del proyecto.
* **Fases/Hitos principales:**
  1. Creación del registro estructurado `contexto/context13.md`.
  2. Implementación del módulo `preprocesamiento_eda.py` con verificaciones numéricas exactas de las Diapositivas 6–20, 21–44, 45–70 y 71–74.
  3. Actualización de `web/index.html` ubicando **CD02 — Preprocesamiento y EDA** como segunda unidad en `.unit-switcher` y añadiendo sus 4 laboratorios (`panel-29` a `panel-32`) y 4 reactivos de autoevaluación (`quiz-item-29` a `quiz-item-32`).
  4. Integración en segunda posición dentro de `perceptron_xor.py`, actualización de `README.md` e `INFORME.md`, y verificación automatizada (`python3 perceptron_xor.py --auto`).
* **Entregables:** `contexto/context13.md`, `preprocesamiento_eda.py`, `web/index.html` actualizado (8 unidades ordenadas `CD01`, `CD02`, `CD13`–`CD18`; 33 laboratorios), `perceptron_xor.py`, `README.md` e `INFORME.md`.

## 3. Referencias
* **Path del plan original:** `contexto/context13.md`
* **Material de cátedra:** Presentación *«CD02-Preprocesamiento-y-EDA.pdf»* (Ciencia de Datos — UCN, 74 diapositivas, basado en material de Harvard CS109 de P. Protopapas, K. Rader et al.).
* **Artículos y fuentes fundamentales citados en la unidad CD02:**
  * Anscombe, F. J. (1973). *Graphs in Statistical Analysis*. The American Statistician, 27(1), 17–21.
  * Fayyad, U., Piatetsky-Shapiro, G., & Smyth, P. (1996). *From Data Mining to Knowledge Discovery in Databases*. AI Magazine, 17(3), 37–54.
  * Chapman, P., Clinton, J., Kerber, R., et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS Inc.
  * Wickham, H. (2014). *Tidy Data*. Journal of Statistical Software, 59(10), 1–23.
