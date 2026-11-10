# INFORME TÉCNICO-CIENTÍFICO
## Comprobación Formal y Computacional de la Resolución del Operador XOR Mediante un Perceptrón Multicapa

**Programa:** Doctorado — Universidad Católica del Norte (UCN)  
**Asignatura:** Ciencia de Datos  
**Semestre:** 2 - 2026  
**Autor:** J. Alexander Acosta  
**Fecha:** 26 de octubre de 2026  
**Repositorio:** [https://github.com/j-alexander-acosta/Perceptron.git](https://github.com/j-alexander-acosta/Perceptron.git)  

---

### Resumen

Este informe presenta la fundamentación teórica, deducción matemática e implementación computacional de una red neuronal artificial del tipo **Perceptrón Multicapa (MLP)** de dos capas, diseñada específicamente para resolver el problema de la función lógica **XOR** ($\oplus$). Se valida empírica y formalmente la afirmación teórica: *"El perceptrón multicapa resuelve XOR; componer capas sí resuelve XOR"*, sustentada en la descomposición booleana $X \oplus Y = (X \lor Y) \land (\bar{X} \lor \bar{Y})$. Mediante el uso de neuronas con funciones de activación de escalón unitario con umbral $\theta$, se demuestra cómo la capa oculta efectúa una transformación de representación que proyecta un espacio de entrada no linealmente separable en un espacio intermedio donde la función de salida es linealmente separable. La verificación computacional confirma una exactitud del 100% para todos los estados del espacio discreto $\{0, 1\}^2$.

---

## 0. Unidad Inicial CD01 (Primera en Orden de Cátedra): Introducción al Curso de Ciencia de Datos (63 Diapositivas)

Siguiendo el orden cronológico de las unidades del curso (`CD01`, ..., `CD13`–`CD18`), el módulo [`introduccion_curso.py`](introduccion_curso.py) y los laboratorios `0.1` a `0.4` de [`web/index.html`](web/index.html) abren la suite verificando numérica y conceptualmente las 63 diapositivas de **CD01 — Introducción al Curso** (basado en material de CMU 15-388/688 de Z. Kolter, Harvard CS109 de P. Protopapas et al., Stanford CS221 de P. Liang et al., y UC Berkeley DS100):

### 0.1. Caso Real 1: Ciencia de Datos en la Pandemia, Sesgo en la Tasa de Letalidad y Causalidad (Diapositivas 5–11 y 19–21)
* **Los 5 pasos y las 4 preguntas clave (Diaps. 6–7):**
  1. *¿Qué tan rápido se propaga el virus?* $\to$ **Modelamiento** (modelo compartimental SIR).
  2. *¿Cuándo se saturarán los hospitales?* $\to$ **Predicción** (proyección de curvas UCI a corto plazo).
  3. *¿Funcionan las cuarentenas?* $\to$ **Inferencia causal** (estimación contrafactual de intervenciones no farmacológicas; Flaxman et al., 2020, *Nature*).
  4. *¿Cómo distribuir vacunas con equidad?* $\to$ **Optimización** (asignación sujeta a restricciones y principios éticos).
* **El sesgo en acción — Tasa de letalidad aparente vs. real (Diapositiva 9):**
  $$\text{letalidad aparente (CFR)} = \frac{\text{muertes}}{\text{casos confirmados}}, \qquad \text{letalidad real (IFR)} = \frac{\text{muertes}}{\text{infecciones reales}}$$
  Con $100$ muertes y $2\,000$ casos confirmados (sesgados hacia pacientes sintomáticos), $\text{CFR} = 100/2\,000 = \mathbf{5.0\%}$. Si las infecciones reales fueron $20\,000$, $\text{IFR} = 100/20\,000 = \mathbf{0.5\%}$. El mismo numerador con un denominador sesgado **infla la letalidad aparente $10$ veces** (cambia el orden de magnitud).
* **Evaluación del toque de queda nocturno con datos de Google Mobility (Diapositiva 21):** (1) Es un problema de **inferencia causal**; (2) sub-representa a quienes no poseen *smartphone*; (3) observar que bajaron movilidad y contagios no basta para probar causalidad debido a **variables de confusión** (el miedo al virus reduce ambos simultáneamente).

### 0.2. El Modelo Epidemiológico SIR a Mano, $R_0 = \beta/\gamma$ y «Aplanar la Curva» (Diapositivas 12–18)
* **Ecuaciones en diferencias diarias (Diapositiva 12):** Para una población $N = S_t + I_t + R_t = 1\,000$:
  $$\text{contagios}_t = \frac{\beta S_t I_t}{N}, \qquad \text{recuperaciones}_t = \gamma I_t$$
  $$S_{t+1} = S_t - \text{contagios}_t, \qquad I_{t+1} = I_t + \text{contagios}_t - \text{recuperaciones}_t, \qquad R_{t+1} = R_t + \text{recuperaciones}_t$$
* **Verificación exacta de la tabla «La dinámica SIR, a mano» (Diapositiva 13):** Con $\beta = 0.3$, $\gamma = 0.1$ ($1/\gamma = 10$ días), $S_0 = 990.0$, $I_0 = 10.0$, $R_0 = 0.0$:
  * **Día 0:** $\text{contagios} = 2.97$, $\text{recuperaciones} = 1.00$, $S = 990.0$, $I = 10.0$, $R = 0.0$.
  * **Día 1:** $\text{contagios} = 3.54$, $\text{recuperaciones} = 1.20$, $S = 987.0$, $I = 12.0$, $R = 1.0$.
  * **Día 2:** $\text{contagios} = 4.22$, $\text{recuperaciones} = 1.43$, $S = 983.5$, $I = 14.3$, $R = 2.2$.
  * **Día 28 (pico):** $\text{contagios} = 30.24$, $\text{recuperaciones} = 31.59$, $S = 319.0$, $I = \mathbf{315.9}$, $R = 365.0$, y al día 160 cerca del **$95\%$** ($94.7\%$) de la población ha sido infectada (Diapositiva 14).
* **Efecto de las cuarentenas sobre $R_0 = \beta/\gamma$ (Diapositiva 15):** Reducir $\beta$ de $0.30$ ($R_0 = 3.0$) a $0.16$ ($R_0 = 1.6$) baja el pico de infectados simultáneos de $\approx \mathbf{316}$ (día 28) a $\approx \mathbf{89}$ (día 61).

### 0.3. ¿Qué es Ciencia de Datos?, El Caso Netflix Prize (2006–2009) y la Paradoja de la Sobre-optimización (Diapositivas 22–40)
* **Tamaño del vacío en la matriz de Netflix (Diapositivas 27–28):** $480\,189 \text{ usuarios} \times 17\,770 \text{ películas} = \mathbf{8\,532\,958\,530} \approx \mathbf{8\,533\text{ millones}}$ de celdas, con solo $\mathbf{100\,480\,507}$ calificaciones observadas ($\approx \mathbf{1.18\% \approx 1.2\%}$ observado y **$98.82\%$ faltante no al azar — MNAR**).
* **Evolución del RMSE y la confesión de 2012 (Diapositivas 30–34):** De Cinematch ($\text{RMSE} = 0.951$) a $+8.43\%$ (Año 1), $+9.44\%$ (Año 2) y $+10.06\%$ ($\text{RMSE} = 0.856$, *BellKor's Pragmatic Chaos* en 2009 por 20 minutos). Como revelaron Amatriain y Basilico (2012), el ensamble de $>100$ modelos **nunca se desplegó en producción** porque el costo de ingeniería no justificaba la ganancia offline y porque el paso al *streaming* cambió el problema de predecir estrellas a *ranking*.

### 0.4. Los Tres Relatos de la IA, el Mapa de 4 Niveles y la Política de Uso de LLMs (Diapositivas 1–4 y 41–63)
* **Los 3 relatos históricos de la IA (Diapositivas 41–51):**
  1. **Relato simbólico (1956–1987):** Taller de Dartmouth (1956), fracaso en traducción automática ruso-inglés (*«el vodka es bueno, pero la carne está podrida»* $\to$ Informe ALPAC, 1966, 1.er invierno) y sistemas expertos ($\to$ 2.º invierno en 1987).
  2. **Relato neuronal (1943–hoy):** McCulloch y Pitts (1943) $\to$ Hebb (1949) $\to$ Rosenblatt (1958) $\to$ Minsky y Papert (1969, límite XOR) $\to$ Backpropagation (1986) $\to$ AlexNet (2012), donde el salto entre 1986 y 2012 fue impulsado por **datos y cómputo (GPUs)**.
  3. **Relato estadístico (1801–hoy):** Gauss y Legendre (1801, mínimos cuadrados para la órbita de Ceres) $\to$ Fisher (1936) $\to$ Bellman (1957, MDP) $\to$ Cortes y Vapnik (1995, SVM).
* **Mapa de 4 niveles en IA (Diapositivas 3 y 50):** **Reflejos** (se recorre entero), **Estados** (búsqueda, MDP y refuerzo, sin juegos), **Variables** (modelos probabilísticos, sin CSP) y **Lógica** (fuera del curso).
* **Política de Uso de LLMs (Diapositivas 58–61, UC Berkeley DS100):** Desglose en Contexto, Planificación y Ejecución; casos 1 y 3 son **Apropiados**, mientras que los casos 2 (pedir el EDA completo) y 4 (usar código de visualización que no puedes explicar) son **Inapropiados**.

---

## 0.5. Segunda Unidad CD02 (Segunda en Orden de Cátedra): Preprocesamiento y Análisis Exploratorio de Datos (74 Diapositivas)

Inmediatamente después de `CD01`, el módulo [`preprocesamiento_eda.py`](preprocesamiento_eda.py) y los laboratorios `02.1` a `02.4` de [`web/index.html`](web/index.html) implementan y verifican las 74 diapositivas de **CD02 — Preprocesamiento y Análisis Exploratorio de Datos** (basado en Harvard CS109 de P. Protopapas, K. Rader et al.):

### 0.5.1. ¿Qué son los Datos?, Advertencias de Fábrica, Alcances y Sesgos de Muestreo (Diapositivas 1–20)
* **Tres definiciones de «datos» (Diap. 6):** Diccionario (información factual para razonar o calcular), Informática (información digital transmisible o procesable) y **Señales** (salida de un sensor que mezcla información útil, irrelevante y redundante, y debe procesarse).
* **Diagnóstico por procedimiento de registro (Diaps. 7–8 y 11):**
  * *Termómetro cada hora / GPS de app de transporte* $\implies$ **Vacíos** (cortes de luz, túneles sin señal).
  * *Diario del colibrí / Cuaderno de asistencia a taller* $\implies$ **Imprecisión humana**.
  * *Tweets de un político / Biografías de red social* $\implies$ **Veracidad** (decir no es ser).
  * *Sensor de feria persa / Columna `v37` sin documentar* $\implies$ **Sin semántica**.
* **Gallup vs. Yelp y la Encuesta QR de la Cafetería (Diaps. 16–20):** Gallup utiliza muestreo diseñado (~500 llamadas/día RDD, regla del cumpleaños más reciente en fijos, 50 estados con 70% celulares / 30% fijos y ponderación demográfica) para describir a la **población general**; Yelp sufre de autoselección sin corrección y describe **a quien opina en Yelp**; la encuesta QR de la cafetería (480 respuestas, media **2.1/5**) combina **autoselección** y **cobertura** (no mide satisfacción del estudiantado, pero sí sirve para identificar qué reclama quien está descontento).

### 0.5.2. Matriz 2×2 de Formatos, Preguntas Medibles, CRISP-DM vs. KDD/SEMMA y Web Scraping DOM (Diapositivas 21–44)
* **Matriz 2×2 según el lector (Diap. 21):** Fácil Personas / Fácil Computadores = **Tabla CSV**; Fácil Personas / Difícil Computadores = **Prosa, fotos**; Difícil Personas / Fácil Computadores = **JSON anidado, QR**; Difícil Personas / Difícil Computadores = **Audio con ruido**.
* **Metodologías con nombre (Diaps. 30–35):** Las 6 fases de **CRISP-DM (2000)** (1. Comprensión del negocio $\leftrightarrow$ 2. Comprensión de los datos $\to$ 3. Preparación de los datos $\leftrightarrow$ 4. Modelado $\to$ 5. Evaluación $\to$ 6. Despliegue) frente a **KDD (1996)** y **SEMMA (SAS)**, las cuales omiten las dos puntas del ciclo (**Comprensión del negocio** y **Despliegue**).
* **Obtención de datos y Web Scraping Responsable (Diaps. 36–44):** Orden de preferencia $\text{Archivo} > \text{API} > \text{Scraping}$, respeto de `robots.txt` y términos de servicio, pausas e identificación en `User-Agent`, y verificación de `page.status_code == 200` (vs `404`) antes de extraer del árbol DOM con `BeautifulSoup`.

### 0.5.3. Pandas Esencial y Caso Real Premier League 2017 (`epl_mini`, 23 jugadores) (Diapositivas 45–70)
* **Verificación exacta de `epl_mini.csv` (`shape = (23, 5)`, Diaps. 46, 52, 60, 63–70):**
  * **Promedios por club (`groupby("club")["market_value"].mean()`, Diap. 52):** `Arsenal = 32.80`, `Chelsea = 35.01`, `Liverpool = 34.00`, `Man City = 46.67`, `Man United = 45.75`, `Tottenham = 41.25`.
  * **Resumen global `describe()` (Diap. 63):** `age` ($\text{mean} = 27.13, \text{std} = 3.99, \min = 21.00, Q_1 = 24.00, \text{mediana} = 26.00, Q_3 = 30.50, \max = 35.00$) y `market_value` ($\text{mean} = 38.87, \text{std} = 23.51, \min = 0.05, Q_1 = 21.00, \text{mediana} = 40.00, Q_3 = 52.50, \max = 90.00$).
  * **Caza de extremos y validación externa (Diaps. 64–66):** `idxmin() -> 8` corresponde a `Eduardo` (`Chelsea`, 34 años, `GK`, `0.05M` = 50 mil €; tercer arquero veterano en 2017 sin minutos en liga $\implies$ **dato real**, *extremo $\neq$ error*), mientras que `idxmax()` corresponde a `Paul Pogba` (`Man United`, 24 años, `90.0M`, récord mundial de traspaso en 2016 $\implies$ **dato coherente**).
  * **Resumen por edad (`groupby("age")["market_value"].agg(["mean", "median"])`, Diaps. 67–69):** A los 23 años, `Harry Kane` (`Tottenham`, `60.0M`) separa la media (`38.33M`) de la mediana (`30.00M`).

### 0.5.4. Media vs. Mediana en el Arsenal y el Cuarteto de Anscombe (1973) (Diapositivas 71–74)
* **Sensibilidad en el Arsenal (Diap. 72):** Los 5 jugadores del Arsenal (`[7, 20, 22, 50, 65]`) tienen media $\mathbf{32.80\text{ M}}$ y mediana $\mathbf{22.00\text{ M}}$; si fichan a un jugador de $100\text{ M}$, la media salta a $\mathbf{44.00\text{ M}}$ ($+34.1\%$) mientras la mediana pasa a $\mathbf{36.00\text{ M}}$.
* **El Cuarteto de Anscombe (1973, Diap. 71):** Cuatro conjuntos de $n=11$ observaciones que comparten exactamente $\bar{x}=9.0$, $\bar{y}\approx 7.50$, $s_x^2=11.0$, $s_y^2\approx 4.125$, $r\approx 0.816$ y la recta $\hat{y} = 3.00 + 0.500x$, demostrando que los resúmenes numéricos y los gráficos se necesitan mutuamente.

---

## 0.6. Tercera Unidad CD03 (Tercera en Orden de Cátedra): Visualización de Datos Efectiva y Engañosa (103 Diapositivas)

A continuación de `CD02`, el módulo [`visualizacion_datos.py`](visualizacion_datos.py) y los laboratorios `03.1` a `03.4` de [`web/index.html`](web/index.html) implementan y auditan las 103 diapositivas de **CD03 — Visualización de Datos Efectiva y Engañosa** (basado en CMU 15-388/688 de Z. Kolter y Michael Correll, Tableau Research):

### 0.6.1. Explorar vs. Comunicar, Datasaurus Dozen (CHI 2017), Escalas de Medición y Catálogo de Gráficos (Diapositivas 1–27 y 50–56)
* **Dos propósitos simultáneos (Diaps. 3–6):** La visualización opera en dos pasos del proceso de 5 pasos a la vez: **explorar** los datos propios (rápido, desechable, para ti: *descubrir qué es verdad*) y **comunicar/presentar** resultados a otros (pulido, con un mensaje: *convencer de que es verdad*).
* **Datasaurus Dozen (Matejka y Fitzmaurice, CHI 2017, Diaps. 10–12):** **13 nubes de $n = 142$ puntos** (*dino*, *star*, *circle*, *x shape*, etc.) comparten exactamente las 5 estadísticas hasta el segundo decimal:
  $$\bar{x} = \mathbf{54.26}, \quad \bar{y} = \mathbf{47.83}, \quad s_x = \mathbf{16.76}, \quad s_y = \mathbf{26.93}, \quad r = \mathbf{-0.06}$$
  Porque $m$ estadísticas son $m$ ecuaciones y los $n$ puntos tienen muchos más grados de libertad ($n > m$).
* **Las 4 escalas de medición (Diaps. 13–14):**
  1. **Nominal:** categorías sin orden (`=`, $\neq$; ej. `club del jugador`).
  2. **Ordinal:** categorías con orden pero sin distancia con significado (`=`, $\neq$, `<`, `>`; ej. `puesto en el ranking`).
  3. **Intervalo:** numérico con cero arbitrario (`+`, `-`; ej. °C, `año de debut`).
  4. **Razón:** numérico donde el cero significa ausencia total (todas, incluida `÷`; ej. Kelvin, `precio de mercado`).
* **Cálculo exacto del Box Plot (Diap. 21):** Para $\{2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 15\}$ ($n=11$), se obtiene $\text{mediana} = \mathbf{5.0}$, $Q_1 = \mathbf{3.5}$, $Q_3 = \mathbf{7.5}$, $\text{IQR} = Q_3 - Q_1 = \mathbf{4.0}$ y cerca superior $Q_3 + 1.5\,\text{IQR} = \mathbf{13.5}$, identificando a $\mathbf{15} > 13.5$ como **atípico**.
* **Por qué NO dispersión 3D (Diaps. 24–25) y el ancho de intervalo en histogramas (Diap. 56):** En 3D la cámara elige la conclusión y la oclusión oculta el $78\%$ de un grupo real (sobrevive solo el **$22\%$ de su tinta**). En un histograma sobre $n = 200$ datos, *bins* de ancho $1.0$ muestran 3 barras planas (`[66, 68, 66]`, «uniforme»), mientras que *bins* de ancho $0.2$ revelan **dos modos** (dos poblaciones mezcladas).

### 0.6.2. Caso Clásico: Los Antibióticos de Will Burtin (1951) y el Gráfico que Corrigió a la Biología (Diapositivas 28–38)
* **La tabla de 16 bacterias y 3 antibióticos (Burtin, 1951, *Scope*; Wainer y Lysen, 2009, Diap. 29):** Concentración inhibitoria mínima (**CIM**: menor = más potente) de penicilina, estreptomicina y neomicina a través de **6 órdenes de magnitud** (`0.001` a `870`), más tinción de Gram ($7$ Gram+ y $9$ Gram-).
* **Pregunta 1 (del médico) — Medianas por grupo en escala logarítmica (Diap. 33):**
  * **Gram positivas:** $\text{penicilina} = \mathbf{0.005}$, $\text{estreptomicina} = \mathbf{1.0}$, $\text{neomicina} = \mathbf{0.1}$ $\implies$ la penicilina es **$200\times$ más potente** ($1/0.005 = 200$) que la estreptomicina.
  * **Gram negativas:** $\text{penicilina} = \mathbf{100.0}$, $\text{estreptomicina} = \mathbf{1.0}$, $\text{neomicina} = \mathbf{0.1}$ $\implies$ gana la neomicina (`0.1`) y la penicilina pasa de `0.005` a `100` (**$20\,000\times$ menos potente**).
* **Pregunta 2 (del biólogo) — Dos etiquetas que no calzaban y que anticiparon la microbiología por décadas (Diaps. 34–36):** Al graficar $\log_{10}(\text{penicilina})$ vs. $\log_{10}(\text{neomicina})$ con color por género:
  1. **`Diplococcus pneumoniae` (`pen=0.005, neo=10`):** no se llamaba estreptococo en 1951, pero cae justo junto a *S. hemolyticus* (`0.001, 10`) y *S. viridans* (`0.005, 40`). En **1974** la microbiología la reclasificó como **`Streptococcus pneumoniae`**.
  2. **`Streptococcus fecalis` (`pen=1.0, neo=0.1`):** sí se llamaba estreptococo, pero requiere entre **$200\times$ y $1\,000\times$ más penicilina** que los verdaderos estreptococos. En **1984** pasó al nuevo género **`Enterococcus faecalis`**.

### 0.6.3. Integridad Gráfica (Tufte, 1983), Efectividad Perceptual (Stevens, Cleveland-McGill, Mackinlay) y Uso del Color (Diapositivas 39–49 y 57–75)
* **Factor de Mentira (*Lie Factor*, Tufte, 1983, Diaps. 40–41):**
  $$\text{LieFactor} = \frac{\text{tamaño del efecto en el gráfico}}{\text{tamaño del efecto en los datos}}$$
  En *If Bush Tax Cuts Expire* (Fox Business, 2012: $35\% \to 39.6\%$ con el eje truncado en $34\%$), las alturas visuales son $1.0$ y $5.6$ (**$5.6\times$ más alta** $\implies$ efecto gráfico $+460\% = 4.6$), mientras el efecto real en los datos es $(39.6-35)/35 \approx 13.14\% = 0.1314$, dando $\text{LieFactor} = 4.6 / 0.131428 = \mathbf{35.0}$. En la auditoría de las 4 marcas $A=52\%..D=56\%$ con base en $50\%$ (Diap. 49), $D$ parece el triple de $A$ ($\text{LieFactor} = \mathbf{26.0}$).
* **Ley de Potencias de Stevens (1961, Diaps. 61–62) y Rankings de Cleveland-McGill (1984) y Mackinlay (1986, Diaps. 63–64):** $\text{percibido} \propto \text{intensidad}^\beta$:
  * **Longitud ($\beta \approx 1.0$):** un valor $10\times$ se percibe como $10^{1.0} = \mathbf{10.0\times}$ (fiel).
  * **Área ($\beta \approx 0.7$):** un valor $10\times$ se percibe como $10^{0.7} \approx \mathbf{5.0\times}$ (a la mitad).
  * **Brillo ($\beta \approx 0.33$):** un valor $10\times$ se percibe como $10^{0.33} \approx \mathbf{2.1\times}$ (a un quinto).
  * **Descarga eléctrica ($\beta = 3.5$):** el doble de corriente ($2\times$) se siente $2^{3.5} \approx \mathbf{11.3\times}$ más fuerte.
* **Uso del Color y Luminancia Relativa Rec. 709 (Diaps. 67–74):** La luminancia relativa $Y = 0.2126R + 0.7152G + 0.0722B$ en la paleta arcoíris (`jet`) sube y baja 3 veces creando bordes falsos (Rogowitz y Treinish, 1998), mientras que en **`viridis`** crece monótonamente de $\mathbf{0.02}$ a $\mathbf{0.78}$. Regla de Ware: máximo **5 a 8 colores** para categorías; daltonismo: ~**8% de los hombres** (**azul vs. naranjo** como pareja segura).

### 0.6.4. Visualización Engañosa: Taxonomía de Correll, Espejismos (*Mirages*, CHI 2020) y Ética Visual (Diapositivas 76–103)
* **Las 4 categorías de Michael Correll (Tableau Research, Diap. 77):**
  1. **Incorrectas:** codifican mal los datos (CNN Hong Kong en Brasil, Fox $2385^\circ$, *Daily Nexus* sin escala, circular de Fox News que suma $70\%+63\%+60\%=\mathbf{193\%}$ y doble eje manipulado `300->340` vs `88->101`).
  2. **Ilegibles:** ahogan al lector en detalle u ocultan el patrón (circular de 24 porciones y serie temporal de cesantía en Alabama graficada como burbujas).
  3. **Bullshit (Harry Frankfurt):** indiferencia por la verdad (*Decline of Artistic Standards* sin unidades ni fuente).
  4. **No convencionales:** invierten convenciones de lectura (*Gun deaths in Florida*, Reuters 2014: eje con el **0 arriba** donde el alza a **873 muertes** tras la ley *Stand Your Ground* de 2005 parece una caída, frente a *Iraq's Bloody Toll* de S. Scarr; mapa de ITS donde más oscuro = menos infecciones).
* **Espejismos de visualización (*Visualization Mirages*, McNutt, Kindlmann y Correll, CHI 2020, Diaps. 96–98):** Cuando en 2012 los países sin dato saltan de $\approx 72$ a $\approx 177$, sobre la misma tabla `SUM(Energy Usage)` muestra un **desplome** ($-39.3\%$) y `AVG(Energy Usage)` muestra un **auge brusco** ($+119.9\%$). Auditar `COUNT(NULL)` va antes de leer cualquier agregado.
* **Clasificación de los 4 casos de la Diapositiva 100:** (1) Mapa donde el verde intenso marca menos áreas verdes $\to$ **No convencional**; (2) Curva de felicidad sin unidades ni fuente $\to$ **Bullshit**; (3) Circular que suma $140\%$ $\to$ **Incorrecta**; (4) Red de $3\,000$ nodos y $40\,000$ aristas como resumen ejecutivo $\to$ **Ilegible**.

---

## 0.7. Cuarta Unidad CD04 (Cuarta en Orden de Cátedra): Introducción a Machine Learning (56 Diapositivas)

Siguiendo inmediatamente a `CD03` en la secuencia didáctica, el módulo [`introduccion_ml.py`](introduccion_ml.py) y los laboratorios `04.1` a `04.4` de [`web/index.html`](web/index.html) implementan y demuestran las 56 diapositivas de **CD04 — Introducción a Machine Learning** (basado en material de Stanford CS221 de Percy Liang et al.):

### 0.7.1. Panorama General de IA, Modelos de Reflejo y Tipos de Tareas de Predicción (Diapositivas 1–17)
* **El Mapa de 4 Niveles en Inteligencia Artificial (Diaps. 2–8):**
  1. **Reflejos (*Reflex-based models*):** Procesamiento instantáneo $x \to y$ de costo computacional fijo por inferencia, sin búsqueda explícita de trayectorias (regresión lineal, clasificadores lineales, redes neuronales feedforward). Es el nivel cubierto en el inicio del modelado predictivo.
  2. **Estados (*State-based models*):** Modelan secuencias de acciones en entornos dinámicos (búsqueda en grafos $A^*$, programación dinámica, MDPs y Aprendizaje por Refuerzo).
  3. **Variables (*Variable-based models*):** Razonamiento sobre asignaciones conjuntas y relaciones de incertidumbre (redes bayesianas, modelos gráficos probabilísticos, satisfacción de restricciones CSP).
  4. **Lógica (*Logic-based models*):** Deducción formal de primer orden, demostración de teoremas y conocimiento declarativo composicional.
* **Paradigma de Modelos de Reflejo (Diap. 8):** Entrada $x$ $\to$ Extracción de características $\phi(x) \in \mathbb{R}^d$ $\to$ Ponderación con parámetros $w \in \mathbb{R}^d$ $\to$ Salida predicha $f_w(x)$.
* **Taxonomía de Tareas de Predicción (Diaps. 9–17):**
  * **Clasificación Binaria:** $y \in \{-1, +1\}$ o $\{0, 1\}$ (diagnóstico médico, detección de spam).
  * **Clasificación Multiclase:** $y \in \{1, 2, \dots, K\}$ (reconocimiento de dígitos manuscritos, etiquetas de visión).
  * **Regresión:** $y \in \mathbb{R}$ (predicción de precios, concentración de compuestos, trayectorias continuas).
  * **Predicción Estructurada (*Structured Prediction*):** $y = (y_1, y_2, \dots, y_m)$ es un objeto combinatorio complejo (árbol sintáctico, traducción de texto, segmentación densa). Se descompone en decisiones locales de reflejo o búsquedas en espacio de estados.

### 0.7.2. Regresión Lineal, Pérdida Cuadrática y Descenso por Gradiente (Diapositivas 18–33)
* **Gauss, Ceres (1801) y la Hipótesis Lineal (Diaps. 18–22):** En 1801, Carl Friedrich Gauss recuperó la órbita del planeta enano Ceres a partir de apenas 41 días de observaciones celestes usando el principio de **Mínimos Cuadrados**. Se define la clase de hipótesis lineal:
  $$f_w(x) = w \cdot \phi(x) = \sum_{j=1}^d w_j \phi_j(x)$$
* **Pérdida Cuadrática y Pérdida de Entrenamiento (Diaps. 23–25):**
  $$\text{Loss}_{\text{squared}}(x, y, w) = \left(f_w(x) - y\right)^2 = \left(w \cdot \phi(x) - y\right)^2$$
  $$\text{TrainLoss}(w) = \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{(x,y) \in \mathcal{D}_{\text{train}}} \left(w \cdot \phi(x) - y\right)^2$$
* **Verificación Numérica Exacta del Ejemplo de Cátedra (Diaps. 24–27):**
  * Datos: $\mathcal{D}_{\text{train}} = \{(1, 1), (2, 3), (4, 3)\}$ con características $\phi(x) = [1, x]$ ($d=2$).
  * Para $w = [1, 0.57]$:
    * Predicción en $x=3$: $f_w(3) = 1 + 0.57(3) = \mathbf{2.71}$.
    * Errores individuales: $e_1 = (1 + 0.57(1)) - 1 = 0.57 \implies e_1^2 \approx 0.3249$; $e_2 = (1 + 0.57(2)) - 3 = -0.86 \implies e_2^2 \approx 0.7396$; $e_3 = (1 + 0.57(4)) - 3 = 0.28 \implies e_3^2 \approx 0.0784$.
    * Pérdida empírica: $\text{TrainLoss}([1, 0.57]) = \frac{0.3249 + 0.7396 + 0.0784}{3} \approx \mathbf{0.38}$.
* **Gradiente y Tabla de Descenso por Gradiente (Diaps. 28–33):**
  $$\nabla_w \text{Loss}_{\text{squared}}(x, y, w) = 2\left(w \cdot \phi(x) - y\right)\phi(x)$$
  $$\nabla_w \text{TrainLoss}(w) = \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{(x,y) \in \mathcal{D}_{\text{train}}} 2\left(w \cdot \phi(x) - y\right)\phi(x)$$
  Con inicio $w^{(0)} = [0, 0]$ y tasa de aprendizaje $\eta = 0.1$:
  * **$t=1$:** $f_{w^{(0)}} - y = [-1, -3, -3] \implies \nabla = \frac{2}{3} ([-1,-1] + [-3,-6] + [-3,-12]) = \frac{2}{3}[-7, -19] \approx [-4.67, -12.67] \implies w^{(1)} = [0, 0] - 0.1[-4.67, -12.67] = \mathbf{[0.47, 1.27]}$.
  * **$t=2$:** $\nabla \approx [-0.60, 4.31] \implies w^{(2)} = \mathbf{[0.53, 0.84]}$.
  * **$t=200$ (convergencia óptima):** $w^* = \mathbf{[1.00, 0.57]}$, con $\text{TrainLoss} = \mathbf{0.38}$.

### 0.7.3. Clasificación Lineal, Margen y Pérdida Hinge (Diapositivas 34–43)
* **Clasificador Lineal en $\mathbb{R}^2$ y Superficie de Decisión (Diaps. 34–36):**
  $$f_w(x) = \text{sign}(w \cdot \phi(x)) = \begin{cases} +1 & \text{si } w \cdot \phi(x) > 0 \\ -1 & \text{si } w \cdot \phi(x) < 0 \\ 0 & \text{indeciso} \end{cases}$$
  El límite de decisión es el hiperplano $\{x : w \cdot \phi(x) = 0\}$, ortogonal al vector de pesos $w$.
* **Puntuación (*Score*) y Margen (Diap. 37):**
  $$\text{score} = w \cdot \phi(x), \qquad \text{margen } m = (w \cdot \phi(x)) y = \text{score} \cdot y$$
  * Si $m > 0$: predicción correcta. Cuanto mayor sea $m$, mayor la confianza y holgura.
  * Si $m < 0$: predicción errónea.
* **Falla de la Pérdida Cero-Uno $\text{Loss}_{0-1}(m) = \mathbb{I}[m \le 0]$ (Diap. 38):**
  Su gradiente respecto a $w$ es idénticamente cero en casi todo $\mathbb{R}^d$ ($\nabla_w \text{Loss}_{0-1} = 0$ para $m \ne 0$) e indefinido en el salto discontinuo $m=0$. Por consiguiente, los métodos basados en gradiente no pueden optimizar la tasa de error directa.
* **Pérdida Hinge (Bisagra) y Subgradiente (Diaps. 39–43):**
  $$\text{Loss}_{\text{hinge}}(x, y, w) = \max\{0, 1 - m\} = \max\{0, 1 - (w \cdot \phi(x)) y\}$$
  $$\nabla_w \text{Loss}_{\text{hinge}}(x, y, w) = \begin{cases} -\phi(x) y & \text{si } m < 1 \\ 0 & \text{si } m > 1 \end{cases}$$
* **Verificación Numérica del Ejemplo de Cátedra (Diaps. 41–42):**
  * Datos: $\mathcal{D}_{\text{train}} = \{([0, 2], +1), ([-2, 0], +1), ([1, -1], -1)\}$ con pesos $w = [2, -1]$.
  * Para el punto $([1, -1], y = -1)$:
    * $\text{score} = w \cdot x = 2(1) + (-1)(-1) = 2 + 1 = \mathbf{3}$.
    * $\text{margen } m = \text{score} \cdot y = 3 \cdot (-1) = \mathbf{-3}$.
    * $\text{Loss}_{\text{hinge}} = \max\{0, 1 - (-3)\} = \mathbf{4}$.
    * Dado que $m = -3 < 1$, el gradiente es $-\phi(x)y = -[1, -1](-1) = \mathbf{[1, -1]}$.
  * Evaluando los 3 puntos con $w=[2,-1]$:
    * $x^{(1)}=[0,2], y=+1 \implies m = -2 \implies \text{hinge} = 3, \nabla = [0, -2]$.
    * $x^{(2)}=[-2,0], y=+1 \implies m = -4 \implies \text{hinge} = 5, \nabla = [2, 0]$.
    * $x^{(3)}=[1,-1], y=-1 \implies m = -3 \implies \text{hinge} = 4, \nabla = [1, -1]$.
    * Promedios sobre $\mathcal{D}_{\text{train}}$: $\text{TrainLoss}_{0-1} = \frac{3}{3} = \mathbf{1.0}$, $\text{TrainLoss}_{\text{hinge}} = \frac{3+5+4}{3} = \mathbf{4.0}$, y $\nabla_w = \frac{1}{3}[3, -3] = \mathbf{[1.0, -1.0]}$.
  * Y con el vector $w = [-1, 1]$ que separa el conjunto:
    * $x^{(1)}=[0,2], y=+1 \implies m = 2 \implies \text{hinge} = 0$.
    * $x^{(2)}=[-2,0], y=+1 \implies m = 2 \implies \text{hinge} = 0$.
    * $x^{(3)}=[1,-1], y=-1 \implies m = 2 \implies \text{hinge} = 0$.
    * $\text{TrainLoss}_{0-1} = \mathbf{0.0}$, $\text{TrainLoss}_{\text{hinge}} = \mathbf{0.0}$.

### 0.7.4. Regresión Logística, Máxima Verosimilitud y Extensión Multiclase (Diapositivas 44–56)
* **Función Logística Sigmoide (Diap. 45):**
  $$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad p(y=+1 \mid x) = \sigma(w \cdot \phi(x)) = \frac{1}{1 + e^{-w \cdot \phi(x)}}$$
  Propiedad de simetría fundamental:
  $$1 - \sigma(z) = \sigma(-z), \qquad p(y \mid x) = \sigma((w \cdot \phi(x)) y) = \sigma(m)$$
* **Deducción de la Pérdida Logística por Máxima Verosimilitud (MLE) (Diaps. 46–48):**
  Maximizar la verosimilitud condicional bajo ejemplos independientes equivale a minimizar la log-verosimilitud negativa:
  $$\text{Loss}_{\text{logistic}}(x, y, w) = -\log p(y \mid x) = -\log \sigma(m) = -\log \left(\frac{1}{1 + e^{-m}}\right) = \mathbf{\log\left(1 + e^{-m}\right)}$$
* **Gradiente de la Pérdida Logística (Diap. 49):**
  $$\nabla_w \text{Loss}_{\text{logistic}}(x, y, w) = \frac{-e^{-m}}{1 + e^{-m}} \nabla_w m = -\left(1 - \sigma(m)\right)\phi(x)y = \mathbf{-\sigma(-m)\phi(x)y}$$
  Interpretación física: $-\sigma(-m)$ es la probabilidad que el modelo asigna a la etiqueta equivocada; cuanto más seguro esté del error ($m \ll 0 \implies \sigma(-m) \to 1$), mayor es la corrección $-\phi(x)y$. Si clasifica con amplio margen correcto ($m \gg 0 \implies \sigma(-m) \to 0$), el gradiente se desvanece de manera suave.
* **Comportamiento en Datos Separables y la Necesidad de Regularización (Diap. 50):**
  En un conjunto linealmente separable, para que $\text{Loss}_{\text{logistic}} \to 0$ se requiere que $m = (w \cdot \phi(x))y \to +\infty$ para todo ejemplo, lo que obliga a que **$\|w\| \to \infty$**. Al simular descenso por gradiente ($\eta=0.5$):
  * A $t=100$: $w = [-2.29, 2.29]$, $\|w\| \approx 3.24$, $\text{Loss} \approx 0.010$.
  * A $t=1000$: $w = [-3.45, 3.45]$, $\|w\| \approx 4.88$, $\text{Loss} \approx 0.001$.
  Los pesos no se estabilizan en ningún valor finito a menos que se introduzca un término de regularización L2 ($\lambda \|w\|_2^2$), motivación directa para las unidades posteriores.
* **Umbral de Decisión como Perilla de Control (*Threshold Knob*, Diap. 52):**
  Decidir $\hat{y} = +1$ si $p(y=+1 \mid x) \ge \tau$:
  * En diagnóstico de cáncer: bajar $\tau$ (e.g. $0.1$) para maximizar *Recall* (evitar falsos negativos críticos).
  * En filtro de spam: subir $\tau$ (e.g. $0.9$) para maximizar *Precision* (evitar enviar correos legítimos al basurero).
* **Extensión Multiclase: Softmax y Riesgo Empírico (Diaps. 53–56):**
  Para $K$ clases con vectores de pesos $w_1, \dots, w_K$:
  $$p(y = k \mid x) = \frac{e^{w_k \cdot \phi(x)}}{\sum_{j=1}^K e^{w_j \cdot \phi(x)}}$$
  El objetivo del aprendizaje en todos los modelos de reflejo se resume en la **Minimización del Riesgo Empírico (ERM)**:
  $$\min_w \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{(x,y) \in \mathcal{D}_{\text{train}}} \text{Loss}(x, y, w)$$

---

## 0.8. Quinta Unidad CD05 (Quinta en Orden de Cátedra): Generalización y Validación (63 Diapositivas)

Continuando la progresión de los modelos de reflejos lineales y antes de adentrarnos en las redes neuronales profundas y la optimización avanzada, el módulo [`generalizacion_validacion.py`](generalizacion_validacion.py) y los laboratorios `05.1` a `05.4` de [`web/index.html`](web/index.html) desarrollan y demuestran las 63 diapositivas de **CD05 — Generalización y Validación** (basado en material de Stanford CS221 de Percy Liang et al., Harvard CS109 y CMU 11-785):

### 0.8.1. El Objetivo Real: Generalizar, Sobreajuste, Correlación Espuria y Descomposición de Liang (Diapositivas 1–16)
* **La Falacia de la Memorización Empírica (Diaps. 4–5):**
  Minimizar el riesgo de entrenamiento $\text{TrainLoss}(w) = 0.0$ es un objetivo instrumental engañoso. El algoritmo trivial *Memorize*:
  $$h(x) = \begin{cases} y_i & \text{si } x = x_i \text{ para algún } (x_i, y_i) \in \mathcal{D}_{\text{train}} \\ \text{default} & \text{en otro caso} \end{cases}$$
  alcanza $\text{TrainLoss} = 0.0$ exacto pero es incapaz de predecir en puntos no vistos. El objetivo real del aprendizaje automático es **generalizar** a la distribución no observada $\mathcal{D}$.
* **Correlación Espuria y el Falso Atajo Clínico (Oakden-Rayner et al., 2019, Diap. 6):**
  Un modelo de visión profunda entrenado para detectar neumotórax en radiografías de tórax alcanzó un aparente $95\%$ de precisión. La auditoría reveló que la red no detectaba el tejido colapsado del pulmón, sino la presencia del **tubo de drenaje torácico** (catéter naranja) ya instalado por el equipo médico de urgencias. En pacientes graves que aún no habían recibido el tubo, el clasificador falló catastróficamente.
* **Maldición de la Dimensionalidad e Imposibilidad de Cobertura Pura (Diaps. 7–8):**
  En un espacio de $d = 100$ variables binarias, existen $2^{100} \approx 1.2676 \times 10^{30}$ estados posibles. Incluso si dispusiéramos de un conjunto colosal de un petabyte de datos ($n = 10^{15}$ ejemplos), la fracción del espacio muestreada es:
  $$\frac{n}{2^d} = \frac{10^{15}}{1.2676 \times 10^{30}} \approx 7.89 \times 10^{-16}$$
  El $99.9999999999999\%$ del espacio queda completamente inexplorado. Por consiguiente, es **matemáticamente imposible generalizar** sin restringir la clase de hipótesis $\mathcal{F}$ (sesgo inductivo).
* **Descomposición Teórica del Error (Percy Liang / Stanford CS221, Diap. 12):**
  Sea $\mathcal{A}$ la clase universal de todos los predictores, $\mathcal{F}$ nuestra clase de hipótesis elegida (e.g. lineal), $f^* = \arg\min_{f \in \mathcal{A}} \text{Err}(f)$ el predictor óptimo de Bayes, $g = \arg\min_{f \in \mathcal{F}} \text{Err}(f)$ la mejor hipótesis de la clase, y $\hat{f} \in \mathcal{F}$ el predictor estimado por minimización del riesgo empírico sobre $\mathcal{D}_{\text{train}}$. El exceso de error se descompone de forma exacta como:
  $$\text{Err}(\hat{f}) - \text{Err}(f^*) = \underbrace{\left(\text{Err}(\hat{f}) - \text{Err}(g)\right)}_{\text{Error de Estimación (Varianza)}} + \underbrace{\left(\text{Err}(g) - \text{Err}(f^*)\right)}_{\text{Error de Aproximación (Sesgo)}}$$
  * Aumentar la riqueza de $\mathcal{F}$ reduce el error de aproximación pero dispara el de estimación.
  * Restringir $\mathcal{F}$ reduce el error de estimación pero incrementa el de aproximación.
* **Auditoría de los 3 Equipos de Fraude (Diaps. 14–15, baseline 2%):**
  * **Equipo A (Train 0.0%, Test 25.0%):** Sobreajuste severo / alto error de estimación. Solución: regularización $\lambda$, reducción de características o recopilar más datos $n$.
  * **Equipo B (Train 23.0%, Test 24.0%):** Subajuste severo / alto error de aproximación. Solución: enriquecer la hipótesis con no linealidades o nuevas features predictivas.
  * **Equipo C (Probó 50 modelos en Test y eligió el de 5.0%):** Conjunto de prueba contaminado (*Data Leakage* / *P-Hacking*). La métrica del $5\%$ carece de validez estadística fuera de muestra.

### 0.8.2. Controlar la Clase de Hipótesis: Dimensión, Cota de Cauchy-Schwarz, Regularización L2/L1 y Early Stopping (Diapositivas 17–27)
* **Estrategia 1 — Reducir Dimensión y Selección Secuencial Hacia Adelante (Diap. 18):**
  Para $J$ variables, la búsqueda exhaustiva sobre los subconjuntos requiere evaluar $2^J$ modelos (para $J=20 \implies 2^{20} = 1\,048\,576$, combinatoriamente intratable). La selección codiciosa (*Greedy Forward Selection*) incorpora en cada paso la variable que más reduce la pérdida, evaluando solo:
  $$\sum_{k=1}^J k = \frac{J(J+1)}{2} \implies \text{Para } J=20: \quad \frac{20 \times 21}{2} = \mathbf{210} \text{ modelos}$$
  lo que representa una aceleración de **$4\,993\times$** menos cómputo.
* **Estrategia 2 — Cota de Suavidad por Cauchy-Schwarz (Diap. 21):**
  La cota de variación del puntaje entre dos entradas $x$ y $x'$ satisface la desigualdad de Cauchy-Schwarz:
  $$|s(x) - s(x')| = |w \cdot (\phi(x) - \phi(x'))| \le \|w\|_2 \|\phi(x) - \phi(x')\|_2$$
  Por ende, acotar la norma euclidiana $\|w\|_2$ controla de manera directa la constante de Lipschitz del predictor, forzando funciones suaves e insensibles a fluctuaciones locales de ruido.
* **Regularización L2 (Ridge / Weight Decay) y el Paso a Mano Exacto de la Diapositiva 26 (Págs. 103–107):**
  La función objetivo regularizada es:
  $$\mathcal{L}_{\text{reg}}(w) = \text{TrainLoss}(w) + \frac{\lambda}{2}\|w\|_2^2 \implies \nabla_w \mathcal{L}_{\text{reg}}(w) = \nabla_w \text{TrainLoss}(w) + \lambda w$$
  Dando lugar a la regla de contracción de pesos (*weight decay*):
  $$w \leftarrow w - \eta(\nabla_w \text{TrainLoss}(w) + \lambda w) = (1 - \eta\lambda)w - \eta\nabla_w \text{TrainLoss}(w)$$
  **Verificación del Cálculo Numérico Paso a Paso (Diapositiva 26):**
  Con $w = [2.0, -2.0]$, $\nabla\text{TrainLoss}(w) = [1.0, 1.0]$, $\lambda = 0.5$ y $\eta = 0.1$:
  1. Penalización L2: $\lambda w = 0.5 \times [2, -2] = \mathbf{[1.0, -1.0]}$.
  2. Gradiente total regularizado: $\nabla_{\text{total}} = [1, 1] + [1, -1] = \mathbf{[2.0, 0.0]}$.
  3. Actualización de pesos:
     $$w_{\text{nuevo}} = [2.0, -2.0] - 0.1 \times [2.0, 0.0] = [2.0 - 0.2, -2.0 - 0.0] = \mathbf{[1.8, -2.0]}$$
     Coincidencia exacta con la diapositiva y con la contracción: $(1 - 0.05)[2, -2] - 0.1[1, 1] = [1.9 - 0.1, -1.9 - 0.1] = [1.8, -2.0]$.
* **Regularización L1 (LASSO) y Early Stopping (Diaps. 24 y 27):**
  * L1 penaliza $\|w\|_1 = \sum |w_j|$ con subgradiente $\text{sign}(w_j)$, cuyos contornos en diamante colisionan en los ejes coordenados forzando coeficientes irrelevantes exactamente a cero (**selección intrínseca de características**).
  * *Early Stopping*: Partiendo de $w^{(0)} = 0$, tras $T$ iteraciones con gradientes acotados $\|\nabla\| \le G$, se cumple $\|w^{(T)}\| \le \eta G T$. Detener el descenso antes actúa como un radio de bola implícito sobre la norma de pesos.

### 0.8.3. Hiperparámetros, Validación y K-Fold Cross-Validation: El Experimento Controlado Oficial (Diapositivas 28–52)
* **La Trinidad de Datos y la Regla Sagrada (Diaps. 28–33):**
  $\mathcal{D}_{\text{train}} \setminus \mathcal{D}_{\text{val}}$, $\mathcal{D}_{\text{val}}$ y $\mathcal{D}_{\text{test}}$. El conjunto de prueba solo se consulta una única vez al final del proyecto para certificar el rendimiento en producción.
* **El Experimento Numérico Controlado Oficial (Diapositivas 42–47 / Págs. 173–199):**
  Se genera un conjunto sintético con la verdad terreno conocida:
  $$y = x^3 - 3x + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, 2^2) \implies \sigma^2 = \mathbf{4.0} \quad (\text{piso de error irreducible})$$
  con $n = 30$ observaciones distribuidas uniformemente en $[-2.2, 2.2]$ ($22$ para entrenamiento y $8$ para validación simple). Se ajustan polinomios por mínimos cuadrados con grados $d \in \{1, \dots, 8\}$. El grado real del generador es $d = 3$.
* **Tabla Comparativa Oficial: Desenmascarando la Ilusión de la Validación Simple (Diaps. 44–45):**
  $$\begin{array}{c|c|c|l}
  \hline
  \text{Grado } d & \text{Validación Simple (8 pts)} & \text{5-Fold Cross-Validation } (K=5) & \text{Diagnóstico} \\
  \hline
  d = 1 & 26.7 & 23.5 & \text{Subajuste severo (línea recta rígida)} \\
  d = 2 & 23.4 & 29.8 & \text{Parábola sesgada} \\
  \mathbf{d = 3} & 3.1 & \mathbf{3.8} & \mathbf{\text{Mínimo óptimo real (CV elige la verdad } d=3)} \\
  d = 4 & 3.2 & 5.6 & \text{Inicio de varianza innecesaria} \\
  d = 5 & 2.2 & 4.5 & \text{Fluctuaciones de frontera} \\
  \mathbf{d = 6} & \mathbf{1.9} & 9.5 & \mathbf{\text{Falso mínimo engañoso (Val. Simple engañada)}} \\
  d = 7 & 2.0 & 5.7 & \text{Inestabilidad} \\
  d = 8 & 2.1 & \mathbf{80.0} & \mathbf{\text{Colapso catastrófico por sobreajuste extremo}} \\
  \hline
  \end{array}$$
  * *La ilusión:* En validación simple, $d=6$ parece milagroso con $\text{MSE} = 1.9$, ¡incluso inferior al piso irreducible de ruido $\sigma^2 = 4.0$! Esto delata que el modelo sobreajustó a los 8 puntos particulares de esa partición afortunada.
  * *La realidad:* 5-Fold Cross-Validation promedia sobre todos los datos y desenmascara la alta inestabilidad de $d=6$ ($\text{MSE} = 9.5$) y el colapso de $d=8$ ($\text{MSE} = 80.0$), seleccionando con honestidad el grado verdadero **$d = 3$** ($\text{MSE} = 3.8$).
* **¿Por qué 5-Fold Desenmascara a $d=6$? Desglose por Rondas (Diapositiva 46 / Pág. 190):**
  * Para $d=3$: Folds $= [2.4, 2.6, 7.6, 4.7, 1.5] \implies \text{Promedio} = \mathbf{3.8}$, $\text{std} = 2.18$ (comportamiento parejo y controlado).
  * Para $d=6$: Folds $= [1.9, 1.7, 8.2, \mathbf{32.3}, 3.3] \implies \text{Promedio} = \mathbf{9.5}$, $\text{std} = 11.66$.
  * ¡En el **Fold 4**, el polinomio de grado 6 explota violentamente a un error de $\mathbf{32.3}$! La validación simple tuvo la casualidad de no evaluar esa región difícil, pero K-Fold no perdona ningún punto.
* **Auditoría Final en 200,000 Datos Nuevos No Vistos (Diapositiva 47):**
  Al evaluar los dos modelos candidatos en una muestra independiente masiva de $200\,000$ puntos:
  * Modelo elegido por 5-Fold CV ($d=3$): $\text{MSE}_{\text{real}} = \mathbf{4.2}$ (a solo $+5\%$ del piso óptimo $\sigma^2 = 4.0$).
  * Modelo elegido por Validación Simple ($d=6$): $\text{MSE}_{\text{real}} = \mathbf{5.1}$ ($+27.5\%$ de error respecto al óptimo).
  * Conclusión: El modelo elegido por Cross-Validation rinde **$21.4\%$ mejor** en el mundo real.
* **Estabilidad y Variantes de Validación (Diaps. 48–52):**
  * Comparación de modelos con idéntico promedio: Modelo A ($[3.1, 2.7, 3.4, 2.9, 3.0] \implies \mu=3.02, \sigma=0.23$) es netamente superior a Modelo B ($[1.9, 4.8, 2.1, 4.5, 2.2] \implies \mu=3.10, \sigma=1.25$) por su baja dispersión y previsibilidad.
  * *Stratified K-Fold*: Preserva la proporción de clases en datasets desbalanceados (e.g. 98% no fraude / 2% fraude).
  * *TimeSeriesSplit (Purged Walk-Forward)*: Ventana expansiva temporal donde el futuro jamás se filtra para predecir el pasado.

### 0.8.4. Descomposición Sesgo-Varianza, Matriz de Control de Ingeniería y Curvas de Aprendizaje (Diapositivas 53–63)
* **Deducción de la Descomposición del Error Cuadrático Medio Esperado (Diaps. 54–57):**
  Para una entrada fija $x_0$, con $y = f^*(x_0) + \varepsilon$, donde $\mathbb{E}[\varepsilon]=0$ y $\text{Var}(\varepsilon)=\sigma^2$:
  $$\mathbb{E}_{\mathcal{D}, \varepsilon}\left[\left(y - \hat{f}(x_0)\right)^2\right] = \text{Bias}\left[\hat{f}(x_0)\right]^2 + \text{Var}\left[\hat{f}(x_0)\right] + \sigma^2$$
  donde:
  $$\text{Bias}\left[\hat{f}(x_0)\right] = \mathbb{E}_{\mathcal{D}}\left[\hat{f}(x_0)\right] - f^*(x_0)$$
  $$\text{Var}\left[\hat{f}(x_0)\right] = \mathbb{E}_{\mathcal{D}}\left[\left(\hat{f}(x_0) - \mathbb{E}_{\mathcal{D}}\left[\hat{f}(x_0)\right]\right)^2\right]$$
* **La Analogía de la Diana (*Bullseye Target*):**
  * **Alto Sesgo, Baja Varianza:** Disparos agrupados compactamente pero desplazados lejos del centro (modelo demasiado simple / subajuste).
  * **Bajo Sesgo, Alta Varianza:** Disparos cuyo centro de masa está en el blanco pero dispersos por toda la diana (modelo hipersensible a la muestra particular / sobreajuste).
  * **Alto Sesgo, Alta Varianza:** Disparos dispersos y descentrados (el peor escenario).
  * **Bajo Sesgo, Baja Varianza:** Disparos compactos y centrados en el punto medio (el óptimo del aprendizaje).
* **La Matriz de Control de Ingeniería (Diapositiva 58):**
  $$\begin{array}{l|c|c|l}
  \hline
  \text{Perilla de Ingeniería} & \text{Efecto en Sesgo} & \text{Efecto en Varianza} & \text{Riesgo Clave} \\
  \hline
  \text{Más features / mayor grado } d & \downarrow & \uparrow & \text{Sobreajuste si supera la señal} \\
  \text{Mayor regularización } \lambda \uparrow & \uparrow & \downarrow & \text{Subajuste si la contracción es excesiva} \\
  \text{Early Stopping } T \downarrow & \uparrow & \downarrow & \text{Subajuste si detiene antes de aprender} \\
  \mathbf{\text{Más datos de entrenamiento } n \uparrow} & \mathbf{=} & \mathbf{\downarrow} & \mathbf{\text{¡Ninguno! Es el único botón con costo estadístico cero}} \\
  \hline
  \end{array}$$
* **Diagnóstico Mediante Curvas de Aprendizaje (Diapositivas 59–60):**
  * **Caso Alta Varianza (Sobreajuste):** Pérdida de entrenamiento muy baja ($\text{Train} \approx 0.1$) y pérdida de validación alta ($\text{Val} \approx 12.0$). La brecha abierta indica que aumentar el tamaño muestral $n$ cerrará la brecha reduciendo la varianza.
  * **Caso Alto Sesgo (Subajuste):** Pérdida de entrenamiento alta ($\text{Train} \approx 11.8$) y pérdida de validación casi igual ($\text{Val} \approx 12.1$). Ambas curvas están estancadas en un nivel inaceptable. **Recopilar más datos no ayudará en absoluto**; se requiere obligatoriamente una hipótesis más expresiva o nuevas variables.

---

## 0.9. Sexta Unidad CD06 (Sexta en Orden de Cátedra): Clustering — Agrupamiento No Supervisado (57 Diapositivas)

### 0.9.1. K-Means: Compresión, Minimización Alternada de Lloyd y Codo (Diapositivas 1–21)
* **Doble naturaleza de K-Means (Diapositiva 4):** En la literatura coexisten dos conceptos bajo el término *K-Means*:
  1. **La función objetivo de pérdida cuadrática de compresión:**
     $$\text{Loss}_{\text{kmeans}}(z, \mu) = \frac{1}{n}\sum_{i=1}^n \|\phi(x_i) - \mu_{z_i}\|_2^2$$
     donde $z_i \in \{1, \dots, K\}$ asigna el punto $x_i$ a un cluster, y $\mu_k \in \mathbb{R}^d$ representa el centroide prototípico del cluster $k$.
  2. **El algoritmo de Lloyd (1982):** Un esquema iterativo de optimización alternada por bloques de coordenadas (*coordinate descent*):
     * **Paso 1 (Asignar $z$ fijando $\mu$):** $z_i \leftarrow \arg\min_{k \in \{1,\dots,K\}} \|\phi(x_i) - \mu_k\|_2^2$ (partición de Voronoi).
     * **Paso 2 (Recentrar $\mu$ fijando $z$):** $\mu_k \leftarrow \frac{1}{|C_k|}\sum_{i \in C_k}\phi(x_i)$ (media aritmética del cluster $C_k = \{i : z_i = k\}$).
* **Teorema de Convergencia Monótona en Pasos Finitos:**
  En cada paso, el algoritmo minimiza la pérdida respecto a un subconjunto de variables:
  $$\text{Loss}(z^{(t+1)}, \mu^{(t)}) \le \text{Loss}(z^{(t)}, \mu^{(t)}) \quad \text{y} \quad \text{Loss}(z^{(t+1)}, \mu^{(t+1)}) \le \text{Loss}(z^{(t+1)}, \mu^{(t)})$$
  Como el número de asignaciones posibles es finito ($K^n$), y la función es estrictamente decreciente ante cambios de asignación, el algoritmo **garantiza converger monótonamente en un número finito de pasos a un mínimo local**. Sin embargo, la pérdida no es convexa y **no garantiza el óptimo global**.
* **Traza Manual Exacta 1D (Diapositiva 12 / Págs. 36–45):**
  Puntos $\{0, 2, 10, 12\}$, $K=2$, inicializados en centroides $\mu_1 = 0, \mu_2 = 2$:
  * **Iteración 1:**
    - Distancias a $\mu_1=0$: $[0, 2, 10, 12]$. Distancias a $\mu_2=2$: $[2, 0, 8, 10]$.
    - Asignaciones: $z_1 = 1$, $z_2 = 2, z_3 = 2, z_4 = 2$ (empate en $2$ asignado a $2$).
    - Nuevos centroides: $\mu_1 = 0 / 1 = \mathbf{0}$, $\mu_2 = (2 + 10 + 12)/3 = \mathbf{8}$.
  * **Iteración 2:**
    - Distancias a $\mu_1=0$: $[0, 4, 100, 144]$. Distancias a $\mu_2=8$: $[64, 36, 4, 16]$.
    - Reasignación: $x_2=2$ prefiere $\mu_1=0$ (distancia 2 vs 6) $\implies z = [1, 1, 2, 2]$.
    - Nuevos centroides: $\mu_1 = (0 + 2)/2 = \mathbf{1}$, $\mu_2 = (10 + 12)/2 = \mathbf{11}$.
  * **Convergencia y Pérdida Final:**
    $$\text{Loss} = (0-1)^2 + (2-1)^2 + (10-11)^2 + (12-11)^2 = 1 + 1 + 1 + 1 = \mathbf{4.0}$$
* **Paso 2D a Mano Exacto (Diapositiva 15 / Págs. 58–63):**
  Puntos $(0,0), (2,0), (6,4), (8,4)$ con $\mu_1 = (0,0), \mu_2 = (2,0)$:
  * **Iteración 1:** Punto 1 a $\mu_1$; Puntos 2, 3 y 4 a $\mu_2$ $\implies \mu_1 = (0,0)$, $\mu_2 = \frac{(2,0)+(6,4)+(8,4)}{3} = \mathbf{(\frac{16}{3}, \frac{8}{3})} \approx (5.33, 2.67)$. Pérdida $\text{Loss} = 26.67$.
  * **Iteración 2:** Puntos 1 y 2 a $\mu_1$; Puntos 3 y 4 a $\mu_2$ $\implies \mu_1 = \mathbf{(1,0)}, \mu_2 = \mathbf{(7,4)}$.
    $$\text{Loss} = \|(0,0)-(1,0)\|^2 + \|(2,0)-(1,0)\|^2 + \|(6,4)-(7,4)\|^2 + \|(8,4)-(7,4)\|^2 = 1 + 1 + 1 + 1 = \mathbf{4.0}$$
    Convergencia total alcanzada en 2 pasos.
* **Simulación de 210 Puntos en 3 Grupos (Diapositiva 18):**
  Demostración de la vulnerabilidad ante inicializaciones desfavorables:
  * Inicialización aleatoria atrapada: 2 centroides en el cluster superior y 1 en el punto medio de los inferiores $\implies \text{Loss} \approx \mathbf{1\,450}$ (mínimo local severo con 2 clusters fundidos).
  * Inicialización K-Means++: separa los 3 centros en $(0, 3.5), (-3, -2), (3, -2) \implies \text{Loss} \approx \mathbf{328}$ (óptimo global).
* **Curva Oficial del Codo (Elbow Method, Diapositiva 21):**
  Para $K=1..8$, la pérdida decrece como $[1863.2, 955.1, \mathbf{107.5}, 90.1, 77.9, 66.1, 58.4, 50.7]$.
  La reducción marginal de $K=2 \to 3$ es $\Delta = 847.6$, mientras que de $K=3 \to 4$ cae bruscamente a $\Delta = 17.4$ ($48.7\times$ menor), estableciendo el codo formal en **$K=3$**.
* **Sensibilidad Extrema a Outliers:**
  Al incorporar el dato atípico $x=30$ a $\{0, 2, 10, 12\}$, con $K=2$ el punto $30$ secuestra el centroide $\mu_2=30.0$, forzando a $\mu_1=6.0$ a absorber los 4 datos reales, con pérdida $\text{Loss} = 104.0$ frente a $365.0$ si se respeta la partición natural, evidenciando la no robustez de la norma $L_2$ cuadrática.

### 0.9.2. Evaluación de Clusters: Índice de Rand (RI), Rand Ajustado (ARI) y Ruido (Diapositivas 22–30)
* **Matriz de Contingencia y Conteo de Pares $\binom{n}{2}$ (Diapositiva 24):**
  Dada una verdad terreno $Y$ y una partición generada $Z$, se clasifican los $\binom{n}{2} = \frac{n(n-1)}{2}$ pares de observaciones:
  * $a$: puntos en el **mismo** cluster en $Y$ y en el **mismo** en $Z$ (acuerdo positivo).
  * $b$: puntos en **distintos** clusters en $Y$ y en **distintos** en $Z$ (acuerdo negativo).
  * $c$: puntos en el **mismo** cluster en $Y$ pero en **distintos** en $Z$ (falso negativo).
  * $d$: puntos en **distintos** clusters en $Y$ pero en el **mismo** en $Z$ (falso positivo).
* **Rand Index Simple:**
  $$\text{RI} = \frac{a + b}{\binom{n}{2}} = \frac{a + b}{a + b + c + d}$$
* **Patología de Inflación de Base del Rand Index (Diapositivas 25–26):**
  En un conjunto de $n=6$ puntos con dos clases balanceadas ($Y = [1,1,1,2,2,2]$):
  * **Asignación Trivial 1 (Un solo supercluster $Z=[1,1,1,1,1,1]$):** $a=6, b=0, c=0, d=9 \implies \text{RI} = 6/15 = \mathbf{0.40}$.
  * **Asignación Trivial 2 (Cada punto en su propio cluster $Z=[1,2,3,4,5,6]$):** $a=0, b=9, c=6, d=0 \implies \text{RI} = 9/15 = \mathbf{0.60}$.
  ¡El índice en bruto otorga hasta 60% de exactitud a agrupamientos completamente inútiles debido a la sobreabundancia natural de pares separados!
* **Ajuste por Azar de Hubert y Arabie (1985, Diapositiva 27):**
  Bajo el modelo hipergeométrico de asignación aleatoria con marginales fijos:
  $$\text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]} = \frac{\sum_{ij} \binom{n_{ij}}{2} - \frac{\sum_i \binom{a_i}{2}\sum_j \binom{b_j}{2}}{\binom{n}{2}}}{\frac{1}{2}\left[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}\right] - \frac{\sum_i \binom{a_i}{2}\sum_j \binom{b_j}{2}}{\binom{n}{2}}}$$
  * En ambas patologías triviales, $\text{ARI} = \mathbf{0.0000}$ (acuerdo nulo descontado el azar).
  * En la partición con 1 error de la Diapositiva 27 ($Z=[1,1,2,2,2,2]$): $a=4, b=6, c=2, d=3 \implies \text{RI} = 10/15 \approx \mathbf{0.6667}$, corregido a $\text{ARI} = \mathbf{0.3243}$.

### 0.9.3. Agrupamiento Jerárquico: Dendrogramas, Criterios de Enlace y GMM (Diapositivas 31–37)
* **Algoritmo Aglomerativo Bottom-Up sobre $\{1, 2, 5, 9, 12\}$:**
  * **Single Linkage (Distancia mínima):** $D(A, B) = \min_{x \in A, y \in B} d(x, y)$.
    - Fusiones: $\{1, 2\}$ a $d=1.0$; $\{1,2\}$ con $5$ a $d=3.0$; $\{9, 12\}$ a $d=3.0$; $\{1,2,5\}$ con $\{9,12\}$ a $d=4.0$.
    - **Empate en $h=3.0$:** Como $d(2,5) = 3.0$ y $d(9,12) = 3.0$ ocurren simultáneamente, **es imposible aislar $K=3$ clusters**. Cortar a $h=3.5$ produce 2 clusters ($\{1,2,5\}$ y $\{9,12\}$); cortar a $h=2.9$ produce 4 clusters ($\{1,2\}, \{5\}, \{9\}, \{12\}$).
  * **Complete Linkage (Distancia máxima):** $D(A, B) = \max_{x \in A, y \in B} d(x, y)$.
    - Fusiones: $\{1,2\}$ a $d=1.0$; $\{9,12\}$ a $d=3.0$; $\{1,2\}$ con $5$ a $d=\max(4,3)=4.0$; raíz uniendo ambos a $d=12-1=11.0$. Permite aislar $K=3$ en $h \in [3.0, 4.0)$.
* **Modelos de Mezclas Gaussianas (GMM, Diapositivas 36–37):**
  Supera la asignación dura (*hard clustering*) de K-Means calculando responsabilidades probabilísticas continuas (*soft clustering*):
  $$\gamma_{ik} = P(z_i = k \mid x_i) = \frac{\pi_k \mathcal{N}(x_i; \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i; \mu_j, \Sigma_j)}$$
  En la zona fronteriza entre clusters, un punto $x$ recibe responsabilidades suaves (p. ej. $\gamma_{i1} = 0.70, \gamma_{i2} = 0.30$) optimizadas mediante el algoritmo Esperanza-Maximización (EM).

### 0.9.4. Basado en Densidad: DBSCAN, Dos Lunas, Densidades Heterogéneas y HDBSCAN (Diapositivas 38–57)
* **Fundamento de DBSCAN (Ester et al., 1996):**
  Definido por radio $\varepsilon > 0$ y umbral $\text{minPts} \in \mathbb{N}$:
  1. $p$ es **Punto Núcleo (Core):** si $|N_\varepsilon(p)| \ge \text{minPts}$, donde $N_\varepsilon(p) = \{q : \|p - q\| \le \varepsilon\}$.
  2. $q$ es **Directamente Alcanzable por Densidad** desde $p$: si $p$ es núcleo y $q \in N_\varepsilon(p)$.
  3. $q$ es **Alcanzable por Densidad** desde $p$: si existe una cadena $p_1, \dots, p_k$ con $p_1=p, p_k=q$ donde cada $p_{i+1}$ es directamente alcanzable desde $p_i$.
  4. $p$ y $q$ son **Conectados por Densidad:** si existe $o$ tal que ambos son alcanzables desde $o$.
  5. Punto de **Frontera (Border):** no es núcleo pero pertenece a la $\varepsilon$-vecindad de un núcleo.
  6. Punto de **Ruido (Noise):** no es núcleo ni frontera.
* **Ejemplo Numérico de 10 Puntos (Diapositivas 43–48):**
  Arco de 5 puntos, bloque de 4 puntos y 1 punto atípico:
  * Con $\varepsilon = 1.05, \text{minPts} = 3$: 7 núcleos, 2 fronteras y 1 ruido (2 clusters conectados).
  * Con $\varepsilon = 1.05, \text{minPts} = 4$: El bloque mantiene 4 núcleos. Pero los 5 puntos del arco tienen a lo más 3 vecinos, de modo que **ninguno es núcleo** $\implies$ el arco completo colapsa catastróficamente a 5 puntos de ruido.
* **Experimento de las Dos Lunas (Diapositiva 38):**
  En 120 puntos entrelazados no convexos:
  * K-Means ($K=2$): impone fronteras hiperplanares y corta ambas lunas a la mitad $\implies \mathbf{31\text{ a }33}$ errores de clasificación ($\text{ARI} \approx \mathbf{0.490}$).
  * DBSCAN ($\varepsilon=0.25, \text{minPts}=5$): conecta la densidad continua a lo largo de las curvas $\implies \mathbf{0}$ errores, $0$ puntos de ruido y $\mathbf{\text{ARI} = 1.0000}$.
* **Dilema de Densidades Heterogéneas y HDBSCAN (Diapositivas 50–57):**
  DBSCAN falla ante densidades dispares porque un único $\varepsilon$ global crea un compromiso imposible:
  - $\varepsilon$ pequeño ($0.25$): detecta los clusters compactos pero fragmenta el cluster difuso enviando 22 puntos a ruido.
  - $\varepsilon$ grande ($0.60$): rescata el cluster difuso pero funde los dos clusters compactos vecinos en uno solo.
  * **Solución HDBSCAN (Campello et al., 2013):** Utiliza la distancia de alcanzabilidad mutua:
    $$d_{\text{mreach}, k}(a, b) = \max\{d_{\text{core}, k}(a), d_{\text{core}, k}(b), d(a, b)\}$$
    y construye un árbol condensado variando la escala $\lambda = 1/\varepsilon$, seleccionando clusters por máxima persistencia (estabilidad integral $\sum (\lambda_{\text{end}} - \lambda_{\text{birth}})$) con un único parámetro intuitivo: `min_cluster_size`.

---

## 1. Introducción y Contexto Histórico (Unidad CD13 / Unidad I)

El estudio de las redes neuronales artificiales experimentó un hito fundacional en 1943 con el modelo de neurona formal de Warren McCulloch y Walter Pitts, y posteriormente en 1958 con el Perceptrón propuesto por Frank Rosenblatt. El perceptrón simple demostró ser capaz de aprender y clasificar patrones linealmente separables (como las funciones lógicas AND y OR).

Sin embargo, en 1969, Marvin Minsky y Seymour Papert publicaron su célebre monografía *Perceptrons: An Introduction to Computational Geometry*, donde expusieron con rigor matemático la limitación intrínseca de los clasificadores lineales de una sola capa: la imposibilidad de resolver problemas no linealmente separables, tomando como contraejemplo arquetípico la función lógica **XOR** (o disyunción exclusiva).

Este hallazgo precipitó el periodo conocido en la literatura de la Inteligencia Artificial como el *primer invierno de la IA*. No obstante, la solución conceptual ya radicaba en la composición jerárquica de funciones: la introducción de una **capa oculta** (*hidden layer*) cuyas salidas intermedias no aparecen directamente en los datos observados, pero permiten transformar las coordenadas del problema de forma no lineal. El presente trabajo valida en detalle dicha arquitectura.

---

## 2. Fundamento Lógico y Descomposición Booleana

La disyunción exclusiva ($X \oplus Y$) se define como verdadera si y solo si una de las dos proposiciones es verdadera, pero no ambas. 

Formalmente, puede descomponerse en una conjunción de dos términos:
1. Una disyunción inclusiva (compuerta **OR**): $X \lor Y$ (al menos uno de los dos es 1).
2. Una disyunción de las negaciones (equivalente a la negación de la conjunción o compuerta **NAND**): $\bar{X} \lor \bar{Y} \equiv \overline{X \land Y}$ (no ambos son 1 simultáneamente).

Por consiguiente:

$$X \oplus Y = (X \lor Y) \land (\bar{X} \lor \bar{Y})$$

Esta formulación proporciona la estructura topológica directa de la red neuronal:
* **Capa oculta:** Dos neuronas en paralelo que computan respectivamente $X \lor Y$ y $\bar{X} \lor \bar{Y}$.
* **Capa de salida:** Una neurona que computa la conjunción ($\land$) de las dos activaciones previas.

---

## 3. Modelo Matemático y Parametrización de la Red

### 3.1. Modelo Neuronal de Umbral (McCulloch-Pitts)
Cada neurona artificial de la red recibe un vector de entradas $\mathbf{x} = [x_1, x_2]^T$, ponderado por un vector de pesos sinápticos $\mathbf{w} = [w_1, w_2]^T$, y evalúa la suma ponderada neta:

$$z = \mathbf{w}^T \mathbf{x} = w_1 x_1 + w_2 x_2$$

La activación $a \in \{0, 1\}$ está gobernada por la función escalón unitario de Heaviside sujeta a un umbral de disparo $\theta$:

$$a = f(z - \theta) = \begin{cases} 1 & \text{si } z \ge \theta \\ 0 & \text{si } z < \theta \end{cases}$$

*(Nota matemática: Equivalentemente, definiendo el sesgo o bias $b = -\theta$, la condición de disparo es $z + b \ge 0$).*

---

### 3.2. Topología y Pesos de la Red

```
          Entradas                     Capa Oculta                    Capa de Salida
                     w11 = 1.0     +-----------------+
           X --------------------> |   Neurona 1     | w_out1 = 1.0
             \       w21 = -1.0    |  (Umbral: 1.0)  | -------------\
              \   /--------------> |   h1 = X ∨ Y    |               \
               \ /                 +-----------------+                \   +-----------------+
                X                                                      -> |  Neurona Salida |
               / \                 +-----------------+                /   |  (Umbral: 2.0)  | ---> y (X ⊕ Y)
              /   \--------------> |   Neurona 2     |               /    |  y = h1 ∧ h2    |
             /       w12 = 1.0     |  (Umbral: -1.0) | -------------/     +-----------------+
           Y --------------------> |   h2 = X̄ ∨ Ȳ    | w_out2 = 1.0
                     w22 = -1.0    +-----------------+
```

#### A. Neurona Oculta 1 ($h_1$: Compuerta OR)
* **Vector de pesos:** $\mathbf{w}_1 = [1.0, 1.0]^T$
* **Umbral de activación:** $\theta_1 = 1.0$
* **Suma ponderada:**
  $$z_1 = 1.0 \cdot X + 1.0 \cdot Y = X + Y$$
* **Función de disparo:**
  $$h_1 = \begin{cases} 1 & \text{si } X + Y \ge 1.0 \\ 0 & \text{si } X + Y < 1.0 \end{cases}$$
* **Evaluación de casos:**
  - $(0, 0) \implies z_1 = 0.0 < 1.0 \implies h_1 = 0$
  - $(0, 1) \implies z_1 = 1.0 \ge 1.0 \implies h_1 = 1$
  - $(1, 0) \implies z_1 = 1.0 \ge 1.0 \implies h_1 = 1$
  - $(1, 1) \implies z_1 = 2.0 \ge 1.0 \implies h_1 = 1$
* **Comportamiento:** Representa fielmente $X \lor Y$.

#### B. Neurona Oculta 2 ($h_2$: Compuerta NAND)
* **Vector de pesos:** $\mathbf{w}_2 = [-1.0, -1.0]^T$
* **Umbral de activación:** $\theta_2 = -1.0$
* **Suma ponderada:**
  $$z_2 = -1.0 \cdot X - 1.0 \cdot Y = -(X + Y)$$
* **Función de disparo:**
  $$h_2 = \begin{cases} 1 & \text{si } -(X + Y) \ge -1.0 \iff X + Y \le 1.0 \\ 0 & \text{si } -(X + Y) < -1.0 \iff X + Y > 1.0 \end{cases}$$
* **Evaluación de casos:**
  - $(0, 0) \implies z_2 = 0.0 \ge -1.0 \implies h_2 = 1$
  - $(0, 1) \implies z_2 = -1.0 \ge -1.0 \implies h_2 = 1$
  - $(1, 0) \implies z_2 = -1.0 \ge -1.0 \implies h_2 = 1$
  - $(1, 1) \implies z_2 = -2.0 < -1.0 \implies h_2 = 0$
* **Comportamiento:** Representa fielmente $\bar{X} \lor \bar{Y} \equiv \text{NAND}(X, Y)$.

#### C. Neurona de Salida ($y$: Compuerta AND sobre la capa oculta)
* **Entradas:** El vector de estados ocultos $\mathbf{h} = [h_1, h_2]^T$
* **Vector de pesos:** $\mathbf{w}_{\text{out}} = [1.0, 1.0]^T$
* **Umbral de activación:** $\theta_{\text{out}} = 2.0$
* **Suma ponderada:**
  $$z_{\text{out}} = 1.0 \cdot h_1 + 1.0 \cdot h_2 = h_1 + h_2$$
* **Función de disparo:**
  $$y = \begin{cases} 1 & \text{si } h_1 + h_2 \ge 2.0 \\ 0 & \text{si } h_1 + h_2 < 2.0 \end{cases}$$
* **Comportamiento:** Dado que $h_1, h_2 \in \{0, 1\}$, la condición $h_1 + h_2 \ge 2.0$ solo se satisface si $h_1 = 1$ **y** $h_2 = 1$. Por ende:
  $$y = h_1 \land h_2 = (X \lor Y) \land (\bar{X} \lor \bar{Y}) \equiv X \oplus Y$$

---

## 4. Análisis Geométrico y Transformación de Representación

### 4.1. El Espacio de Entrada Original $\mathbb{R}^2$
En el plano $(X, Y)$, los cuatro puntos del dominio de verdad corresponden a los vértices del cuadrado unitario:
- Clase $0$ ($X \oplus Y = 0$): $\{(0, 0), (1, 1)\}$
- Clase $1$ ($X \oplus Y = 1$): $\{(0, 1), (1, 0)\}$

Cualquier clasificador lineal único define un hiperplano (recta) de la forma $w_1 X + w_2 Y = \theta$. Es geométricamente evidente que no existe ninguna línea recta que pueda separar simultáneamente a los puntos $\{(0, 1), (1, 0)\}$ de los puntos $\{(0, 0), (1, 1)\}$, dado que los segmentos que unen las clases opuestas se intersecan en el punto $(0.5, 0.5)$.

### 4.2. La Transformación No Lineal de la Capa Oculta
La capa oculta define un mapeo $\Phi: \{0, 1\}^2 \to \{0, 1\}^2$ tal que $(X, Y) \mapsto (h_1, h_2)$:

| Entrada $(X, Y)$ | Espacio Oculto $(h_1, h_2)$ | Salida XOR $y$ |
|:----------------:|:---------------------------:|:--------------:|
|     $(0, 0)$     |          $(0, 1)$           |       0        |
|     $(0, 1)$     |          $(1, 1)$           |       1        |
|     $(1, 0)$     |          $(1, 1)$           |       1        |
|     $(1, 1)$     |          $(1, 0)$           |       0        |

**Propiedad fundamental:** En el espacio de características transformado $(h_1, h_2)$:
- Los dos patrones de clase 1 ($(0, 1)$ y $(1, 0)$) colapsan exactamente en el mismo punto: **$(1, 1)$**.
- Los patrones de clase 0 se ubican en **$(0, 1)$** y **$(1, 0)$**.
- El punto $(0, 0)$ en el espacio oculto no es alcanzable para ninguna entrada discreta.

En este nuevo espacio $(h_1, h_2)$, la clase 1 ($h_1=1, h_2=1$) es **linealmente separable** de la clase 0 mediante la recta de decisión:

$$1.0 \cdot h_1 + 1.0 \cdot h_2 = 1.5$$

La neurona de salida utiliza el umbral $\theta_{\text{out}} = 2.0$, lo cual separa limpiamente el punto $(1, 1)$ del resto.

### 4.3. Fronteras en el Espacio Original
En el espacio de entrada, las dos neuronas ocultas definen sendas semirrectas frontera:
- Neurona 1: $X + Y \ge 1$ (elimina el punto $(0, 0)$).
- Neurona 2: $X + Y \le 1$ (elimina el punto $(1, 1)$).

La conjunción de ambas condiciones define la franja cerrada:

$$1 \le X + Y \le 1 \iff X + Y = 1$$

cuya intersección con el dominio binario $\{0, 1\}^2$ contiene únicamente los puntos $\{(0, 1), (1, 0)\}$.

![Espacio de Decisión y Frontera](xor_espacio_decision.png)

---

## 5. Implementación Computacional y Validación

Se desarrolló el programa en Python [`perceptron_xor.py`](perceptron_xor.py), estructurado con orientación a objetos mediante clases desacopladas (`Perceptron` y `PerceptronMulticapaXOR`), libre de dependencias externas obligatorias para los cálculos, garantizando portabilidad y reproducibilidad.

### 5.1. Resultados Numéricos de la Verificación

La ejecución del modelo sobre la totalidad del espacio de estados arrojó la siguiente traza:

```text
================================================================================
 COMPROBACIÓN DE TEORÍA: EL PERCEPTRÓN MULTICAPA RESUELVE XOR
================================================================================
 X  |  Y  ||  z_h1   | h1 (OR)  ||  z_h2   | h2(NAND)  ||  z_out  |  X ⊕ Y  | Esperado | ¿Correcto?
--------------------------------------------------------------------------------
 0  |  0  ||   0.0   |    0     ||   0.0   |     1     ||   1.0   |    0    |    0     |  SÍ [OK]  
 0  |  1  ||   1.0   |    1     ||  -1.0   |     1     ||   2.0   |    1    |    1     |  SÍ [OK]  
 1  |  0  ||   1.0   |    1     ||  -1.0   |     1     ||   2.0   |    1    |    1     |  SÍ [OK]  
 1  |  1  ||   2.0   |    1     ||  -2.0   |     0     ||   1.0   |    0    |    0     |  SÍ [OK]  
================================================================================
```

### 5.2. Desglose de Activación Caso a Caso

1. **Caso $(X=0, Y=0)$:**
   - $z_1 = 0.0 < 1.0 \implies h_1 = 0$
   - $z_2 = 0.0 \ge -1.0 \implies h_2 = 1$
   - $z_{\text{out}} = 1(0) + 1(1) = 1.0 < 2.0 \implies \text{Salida} = 0$. *(XOR esperado: 0)* $\to$ **Correcto**.

2. **Caso $(X=0, Y=1)$:**
   - $z_1 = 1.0 \ge 1.0 \implies h_1 = 1$
   - $z_2 = -1.0 \ge -1.0 \implies h_2 = 1$
   - $z_{\text{out}} = 1(1) + 1(1) = 2.0 \ge 2.0 \implies \text{Salida} = 1$. *(XOR esperado: 1)* $\to$ **Correcto**.

3. **Caso $(X=1, Y=0)$:**
   - $z_1 = 1.0 \ge 1.0 \implies h_1 = 1$
   - $z_2 = -1.0 \ge -1.0 \implies h_2 = 1$
   - $z_{\text{out}} = 1(1) + 1(1) = 2.0 \ge 2.0 \implies \text{Salida} = 1$. *(XOR esperado: 1)* $\to$ **Correcto**.

4. **Caso $(X=1, Y=1)$:**
   - $z_1 = 2.0 \ge 1.0 \implies h_1 = 1$
   - $z_2 = -2.0 < -1.0 \implies h_2 = 0$
   - $z_{\text{out}} = 1(1) + 1(0) = 1.0 < 2.0 \implies \text{Salida} = 0$. *(XOR esperado: 0)* $\to$ **Correcto**.

---

## 6. Aproximación Universal con Redes Neuronales (Extensión Completa CMU 11-785)

En adición al caso fundamental de XOR, el módulo [`aproximacion_universal.py`](aproximacion_universal.py) y la suite interactiva [`web/index.html`](web/index.html) implementan los cuatro pilares de la teoría de **Aproximación Universal**:

### 6.1. El MLP como Función Booleana Universal y el Costo de Una Sola Capa
* **Construcción de Friedland para XOR con 2 neuronas:** Si se permiten conexiones directas desde las entradas hacia la neurona de salida (*skip connections*), basta **una única neurona oculta** $h = \mathbf{1}[X + Y \ge 2] = X \land Y$ y una neurona de salida que evalúa:
  $$y = \mathbf{1}[X + Y - 2(X \land Y) \ge 1]$$
  empleando en total solo **2 neuronas, 5 pesos y 2 umbrales**.
* **Síntesis por Forma Normal Disyuntiva (DNF) y Mapas de Karnaugh:** Cualquier función booleana $f:\{0,1\}^N \to \{0,1\}$ se puede expresar como una disyunción (`OR`) de conjunciones (`AND`), asignando una neurona oculta por cada fila activa de la tabla de verdad. La simplificación mediante Mapas de Karnaugh reduce directamente el ancho de la capa oculta (por ejemplo, de 7 a 3 neuronas ocultas para la función $Y = \bar{Y}\bar{Z} + \bar{W}X\bar{Y} + \bar{X}\bar{Z}$).
* **El peor caso (Tablero de Ajedrez / Paridad de $N$ variables):** Cuando dos entradas que difieren en un solo bit siempre producen salidas distintas ($O = X_1 \oplus X_2 \oplus \dots \oplus X_N$), ningún término del mapa de Karnaugh se puede agrupar:
  * **Con una sola capa oculta:** requiere **$2^{N-1}$ neuronas ocultas** ($2^{N-1} + 1$ neuronas en total), creciendo exponencialmente. Para $N=16$, requiere $32,769$ neuronas.
  * **Con una red profunda en árbol:** componiendo bloques XOR en un árbol binario de $2\log_2(N)$ capas, el costo cae a **$3(N-1)$ neuronas** (o $2(N-1)$ con bloques de Friedland), creciendo linealmente con $N$ ($45$ neuronas para $N=16$).

### 6.2. El MLP como Clasificador Universal en $\mathbb{R}^D$
* **Polígonos y el límite del cilindro circular:** Cada neurona oculta define un semiplano. Con $N$ neuronas ocultas rodeando una región central y una neurona de salida `AND` con umbral $N$, se captura cualquier polígono convexo de $N$ lados.
* **El piso asintótico $N/2$:** Cuando $N \to \infty$, el polígono tiende a un círculo de radio $r$. Adentro la suma vale $N$, mientras que afuera (a distancia $d > r$ del centro) la suma obedece:
  $$\sum_{i=1}^N y_i = N\left(\frac{1}{2} + \frac{1}{\pi}\arcsin\frac{r}{d}\right) \xrightarrow{d \to \infty} \frac{N}{2}$$
  Incorporando un **sesgo $b = -N/2$** en la salida, el piso exterior se anula ($\approx 0$) y el cilindro interior queda con altura $N/2$, permitiendo sumar múltiples cilindros circulares en una misma capa oculta para aproximar cualquier región de decisión con precisión arbitraria.

### 6.3. El MLP como Aproximador Universal de Funciones Continuas
* **Suma de pulsos cuadrados:** En 1D, dos neuronas de umbral en la capa oculta ($\mathbf{1}[x \ge T_1]$ con peso $+h_i$ y $\mathbf{1}[x \ge T_2]$ con peso $-h_i$) generan un pulso rectangular de altura $h_i$ en el intervalo $[T_1, T_2]$. Sumando $P$ pulsos (usando $2P$ neuronas ocultas y una salida lineal sumadora) se aproxima cualquier función continua $f(x)$.
* **Distinción conceptual clave:**
  * **Representar:** Calcular exactamente $f(x)$ (con unidades de umbral solo es posible si $f$ ya es escalonada).
  * **Aproximar:** Para cualquier tolerancia $\varepsilon > 0$, existe una red *finita* cuyo error es menor que $\varepsilon$.
  * **El límite:** Aunque $\varepsilon \to 0$ al afinar los pulsos, ninguna suma de pulsos deja de ser escalonada; la universalidad es de aproximación, no de igualdad exacta.

### 6.4. Suficiencia de la Arquitectura: Ancho, Profundidad y Activaciones
* **Cuello de botella por ancho insuficiente (Franjas vs. Celdas):** Una primera capa con pocas neuronas de umbral particiona el plano en un número reducido de celdas. Como la función escalón entrega únicamente `0` o `1`, todos los puntos que caen en la misma celda reciben exactamente el mismo vector binario; la posición fina dentro de la celda se pierde irreversiblemente y ninguna cantidad de capas posteriores puede recuperarla.
* **Activaciones graduadas ($\sigma(z)$, $\tanh(z)$, $\text{ReLU}(z)$):** Al variar continuamente con la distancia a la frontera $z = \mathbf{w}\cdot\mathbf{x} + b$, transmiten información de magnitud hacia las capas siguientes, compensando parcialmente la falta de ancho si se dispone de mayor profundidad (**Triángulo de compensaciones: Ancho $\leftrightarrow$ Profundidad $\leftrightarrow$ Activaciones**).

---

## 7. Unidad II: Aprendizaje de Redes Neuronales (CMU 11-785 & Stanford CS221)

El módulo [`aprendizaje_redes.py`](aprendizaje_redes.py) y los Laboratorios 2.1 a 2.4 de [`web/index.html`](web/index.html) implementan formal y computacionalmente la segunda parte del curso: cómo encontrar automáticamente el vector de parámetros $W$ (pesos y sesgos) a partir de datos.

### 7.1. Minimización del Riesgo Empírico vs. Error Real (Diapositivas 5–14)
Una red neuronal define una familia paramétrica de funciones $Y = f(X; W)$. Idealmente quisiéramos minimizar el error real sobre todo el dominio:
$$\text{TotalErr}(W) = \int \text{error}\big(f(X; W),\, g(X)\big)\,dX$$
pero como la función objetivo verdadera $g(X)$ es desconocida en casi todo el espacio, muestreamos un conjunto finito de pares de entrenamiento $\{(X_1, d_1), \dots, (X_N, d_N)\}$ y minimizamos el **Riesgo Empírico**:
$$\min_W \; \text{TrainLoss}(W) = \frac{1}{N}\sum_{i=1}^N \text{error}\big(f(X_i; W),\, d_i\big)$$

### 7.2. Regla de Aprendizaje del Perceptrón de Rosenblatt y Teorema de Convergencia (Diapositivas 15–22)
* **Absorción del umbral:** Añadiendo una coordenada constante $x_{N+1} = 1$ con peso $w_{N+1} = -T$, la desigualdad $\sum_{i=1}^N w_i x_i \ge T$ se reescribe como el hiperplano homogéneo $\mathbf{w}\cdot\mathbf{x} \ge 0$ en $\mathbb{R}^{N+1}$.
* **Algoritmo en línea de Rosenblatt (1958):** Con etiquetas $y_i \in \{-1, +1\}$ y $\mathbf{w}_0 = \mathbf{0}$, ante cada muestra mal clasificada ($y_i(\mathbf{w}\cdot\mathbf{x}_i) \le 0$) se actualiza:
  $$\mathbf{w} \leftarrow \mathbf{w} + y_i\mathbf{x}_i$$
* **Teorema de Convergencia:** Si el conjunto es linealmente separable con margen geométrico $\gamma = \min_i \frac{y_i(\mathbf{w}^*\cdot\mathbf{x}_i)}{\|\mathbf{w}^*\|} > 0$ y radio $R = \max_i \|\mathbf{x}_i\|$, el algoritmo converge en a lo más **$(R/\gamma)^2$ actualizaciones**.
* **Verificación exacta de la Diapositiva 22:** Evaluando los 8 puntos de la cátedra, `aprendizaje_redes.py` converge exactamente en **3 actualizaciones** al vector $\mathbf{w} = [3.25, 1.90]$ (con $R = 3.3971$, $\gamma = 0.4190$ y cota $(R/\gamma)^2 = 65.72 \ge 3$).

### 7.3. ¿Por qué no escala a Redes Multicapa? y ADALINE (Diapositivas 23–29)
* **El problema del doble pentágono:** En un MLP con unidades de umbral, incluso si todos los pesos excepto una sola neurona de primera capa estuvieran fijos en sus valores óptimos, las etiquetas globales del problema no son linealmente separables para esa neurona individual. Entrenar redes de umbral multicapa exige decidir qué código booleano intermedio asignar a cada muestra en cada capa oculta (un problema combinatorio exponencial).
* **ADALINE y la Regla Delta / LMS (Widrow y Hoff, 1960):** Mueve el aprendizaje *antes* del escalón minimizando $\frac{1}{2}(d - z)^2$ respecto a la pre-activación continua $z = \mathbf{w}\cdot\mathbf{x}$:
  $$w_i \leftarrow w_i + \eta\,(d - z)\,x_i$$

### 7.4. Activaciones Diferenciables y Señal de Aprendizaje (Diapositivas 30–39)
* **Falla del conteo de errores:** El número de errores de clasificación es una función escalonada plana a trozos: su derivada respecto a los pesos es **idénticamente cero** en todas partes salvo en los puntos donde cruza una muestra (donde es discontinua).
* **Suavizado mediante Sigmoide ($\sigma(z) = \frac{1}{1+e^{-z}}$):** Al reemplazar el escalón por una activación diferenciable y medir una divergencia continua, cualquier desplazamiento infinitesimal de los pesos modifica suavemente la pérdida total, haciendo que toda la red sea diferenciable de extremo a extremo por la regla de la cadena.

### 7.5. Planteamiento Formal: Representaciones de Salida y Funciones de Divergencia (Diapositivas 40–56)

| Tipo de Problema | Objetivo $d$ | Salida de la Red $Y$ | Divergencia $\text{div}(Y, d)$ | Gradiente $\partial\text{div}/\partial z$ |
| :--- | :--- | :--- | :--- | :--- |
| **Regresión** | Real / Vector $\mathbb{R}^K$ | Lineal ($Y = z$) | **L2:** $\frac{1}{2}\|Y - d\|^2$ | $Y - d$ |
| **Clasificación Binaria** | Escalar $\{0, 1\}$ | Sigmoide $\sigma(z) \in (0, 1)$ | **Entropía Cruzada Binaria:** $-d\log Y - (1-d)\log(1-Y)$ | $Y - d$ |
| **Clasificación Multiclase** | Vector *One-Hot* | **Softmax:** $y_i = \frac{e^{z_i}}{\sum_k e^{z_k}}$ | **Entropía Cruzada / KL:** $-\sum_i d_i\log y_i = -\log y_c$ | $y_i - d_i$ |

* **Cancelación de la saturación:** Con salida sigmoide, usar divergencia L2 produce un gradiente $(Y-d)\sigma'(z)$ que se anula cuando la neurona está muy equivocada ($\sigma'(z) \approx 0$). La **Entropía Cruzada** tiene derivada $-1/Y$ (para $d=1$) que cancela exactamente el factor $Y(1-Y)$ de $\sigma'(z)$, manteniendo un gradiente lineal y fuerte $Y - d$.
* **Equivalencia con la Divergencia de Kullback-Leibler:** Dado que $H(d, Y) = H(d) + \text{KL}(d \parallel Y)$ y la entropía de una distribución *one-hot* es $H(d) = 0$, minimizar la entropía cruzada $-\log y_c$ es matemáticamente equivalente a minimizar $\text{KL}(d \parallel Y)$.

---

## 8. Unidad III: Descenso por Gradiente y Convergencia (CMU 11-785 & Stanford CS221)

El módulo [`descenso_gradiente.py`](descenso_gradiente.py) y los Laboratorios 3.1 a 3.4 de [`web/index.html`](web/index.html) implementan formal y numéricamente las 47 diapositivas de la tercera unidad del curso: la optimización de funciones diferenciables no lineales mediante cálculo diferencial multivariable, la Matriz Hessiana y el algoritmo de Descenso por Gradiente.

### 8.1. Optimización de Funciones, Geometría del Gradiente y Curvas de Nivel (Diapositivas 1–16)
* **Puntos críticos en 1D:** Una derivada nula $df(x)/dx = 0$ caracteriza todos los puntos estacionarios del terreno: mínimos locales, mínimos globales, máximos y **puntos de inflexión** (puntos silla).
* **Diferencial multivariable y el Gradiente:** Para $y = f(x_1, x_2, \dots, x_n)$, la expansión de primer orden es:
  $$\Delta y \approx \sum_{i=1}^n \frac{\partial f}{\partial x_i}\Delta x_i = \nabla_{\mathbf{x}} f(\mathbf{x})\,\Delta\mathbf{x} = \|\nabla_{\mathbf{x}} f(\mathbf{x})\|\,\|\Delta\mathbf{x}\|\cos\theta$$
  * El cambio $\Delta y$ es máximo cuando $\cos\theta = +1$ ($\theta = 0^\circ$, dirección de $+\nabla f$) y mínimo (máximo descenso) cuando $\cos\theta = -1$ ($\theta = 180^\circ$, dirección de $-\nabla f$).
  * A lo largo de cualquier **curva de nivel** $f(\mathbf{x}) = c$, la función permanece constante ($\Delta y = 0$), por lo que todo vector tangente $\mathbf{t}$ cumple $\nabla_{\mathbf{x}} f(\mathbf{x})\cdot \mathbf{t} = 0$ ($\theta = 90^\circ$): **el gradiente es siempre ortogonal a las curvas de nivel**.
* **Las 3 afirmaciones de la Diapositiva 16:**
  1. *«Hay muchas direcciones en las que podemos movernos para que la función crezca»* $\to$ **VERDADERO** (todo el semiespacio donde $\nabla f \cdot \Delta\mathbf{x} > 0$, es decir, $|\theta| < 90^\circ$).
  2. *«La dirección del gradiente es la dirección de máximo crecimiento»* $\to$ **VERDADERO** ($\theta = 0$ maximiza $\cos\theta$).
  3. *«Si $\nabla_{\mathbf{x}} f(\mathbf{x}) = \mathbf{0}$, entonces $\mathbf{x}$ es un mínimo»* $\to$ **FALSO** (también puede ser un máximo o un punto silla).

### 8.2. Segunda Derivada, Matriz Hessiana y Ejemplo 3D Resuelto (Diapositivas 17–25)
* **Caso 1D (Diapositiva 20):** Para $f(x) = x^2 - 4x + 5$, la condición estacionaria $f'(x) = 2x - 4 = 0$ entrega $x^* = 2$. Dado que $f''(2) = 2 > 0$, la parábola es estrictamente convexa y alcanza su mínimo global en $f(2) = 1$.
* **Serie de Taylor de segundo orden y Matriz Hessiana:** Alrededor de un punto crítico $\mathbf{x}^*$ donde $\nabla_{\mathbf{x}} f(\mathbf{x}^*) = \mathbf{0}$:
  $$f(\mathbf{x}^* + \mathbf{w}) \approx f(\mathbf{x}^*) + \frac{1}{2}\mathbf{w}^T \nabla_{\mathbf{x}}^2 f(\mathbf{x}^*)\,\mathbf{w}$$
  * Si $\mathbf{w}^T \nabla_{\mathbf{x}}^2 f(\mathbf{x}^*)\,\mathbf{w} > 0$ para todo $\mathbf{w} \neq \mathbf{0}$ (**Hessiana definida positiva**, todos los autovalores $\lambda_i > 0$), $\mathbf{x}^*$ es un **mínimo local**.
  * Si $\mathbf{w}^T \nabla_{\mathbf{x}}^2 f(\mathbf{x}^*)\,\mathbf{w} < 0$ (**Hessiana definida negativa**, todos los $\lambda_i < 0$), $\mathbf{x}^*$ es un **máximo local**.
  * Si tiene autovalores de signos opuestos (**Hessiana indefinida**), $\mathbf{x}^*$ es un **punto silla**.
* **Ejemplo 3D Resuelto (Diapositivas 24–25):** Para
  $$f(x_1, x_2, x_3) = x_1^2 + x_1(1 - x_2) + x_2^2 - x_2 x_3 + x_3^2 + x_3$$
  el sistema lineal $\nabla_{\mathbf{x}} f = [2x_1 + 1 - x_2,\; -x_1 + 2x_2 - x_3,\; -x_2 + 2x_3 + 1]^T = \mathbf{0}$ tiene solución única $\mathbf{x}^* = (-1, -1, -1)$ con $f(-1,-1,-1) = -1$. La Hessiana constante:
  $$\nabla_{\mathbf{x}}^2 f = \begin{bmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 2 \end{bmatrix}$$
  tiene autovalores $\lambda_1 = 2 + \sqrt{2} \approx 3.4142 > 0$, $\lambda_2 = 2 > 0$ y $\lambda_3 = 2 - \sqrt{2} \approx 0.5858 > 0$, confirmando que es **definida positiva** y que $\mathbf{x}^* = (-1, -1, -1)$ es el **mínimo global**.

### 8.3. Descenso por Gradiente Iterativo y 200 Épocas de Regresión Lineal (Diapositivas 26–38)
Cuando $\nabla_{\mathbf{x}} f(\mathbf{x}) = \mathbf{0}$ no tiene solución analítica cerrada, actualizamos iterativamente en dirección opuesta al gradiente:
$$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} - \eta_k \nabla_{\mathbf{x}} f(\mathbf{x}^{(k)})^T$$
Evaluando sobre los 3 puntos de las Diapositivas 34–35 $\{(1, 1), (2, 3), (4, 3)\}$ con $\phi(x) = [1, x]$, pérdida $\text{TrainLoss}(\mathbf{w}) = \frac{1}{3}\sum_{i=1}^3 (\mathbf{w}\cdot\phi(x_i) - y_i)^2$, tasa de aprendizaje $\eta = 0.1$ y $\mathbf{w}^{(0)} = [0, 0]$:

| Época $t$ | $\nabla_{\mathbf{w}}\text{TrainLoss}$ (Diap. 35) | Pesos $\mathbf{w}^{(t)} = [w_0, w_1]$ | $\text{TrainLoss}(\mathbf{w}^{(t)})$ |
| :---: | :---: | :---: | :---: |
| **$t = 0$** | — | $[0.00, \; 0.00]$ | $6.3333$ |
| **$t = 1$** | $[-4.67, \; -12.67]$ | $[0.47, \; 1.27]$ | $1.9763$ |
| **$t = 2$** | $[2.18, \; 7.24]$ | $[0.25, \; 0.54]$ | $0.8407$ |
| **$t = 200$** | $[0.00, \; 0.00]$ | $[1.00, \; 0.57]$ | $0.4762$ |

### 8.4. Curvatura por Dirección y los 4 Regímenes del Tamaño de Paso (Diapositivas 39–47)
* **Paso óptimo cuadrático:** En 1D, minimizar $E(w)$ con curvatura $E''$ alcanza el mínimo en un solo paso con $\eta_{\text{opt}} = (E'')^{-1}$.
* **4 Regímenes sobre $E(w) = w^2$ ($E'' = 2 \implies \eta_{\text{opt}} = 0.5$, 20 iteraciones desde $w^{(0)} = 1.0$, Diapositivas 44–45):**
  1. $\eta = 0.1 < \eta_{\text{opt}} \implies w^{(20)} = (0.8)^{20} = \mathbf{0.01153}$ (convergencia monótona sin sobrepaso).
  2. $\eta = 0.5 = \eta_{\text{opt}} \implies w^{(1)} = w^{(20)} = \mathbf{0.0}$ (convergencia exacta en 1 paso).
  3. $\eta = 0.75 \in (\eta_{\text{opt}}, 2\eta_{\text{opt}}) \implies w^{(20)} = (-0.5)^{20} = \mathbf{9.5367 \times 10^{-7}}$ (oscila cambiando de signo y converge).
  4. $\eta = 1.1 > 2\eta_{\text{opt}} \implies w^{(20)} = (-1.2)^{20} = \mathbf{38.3376}$ (divergencia exponencial).
* **Curvatura anisotrópica en 2D (Diapositivas 41–43):** Para $E(\mathbf{w}) = \frac{1}{2}(w_1^2 + 3w_2^2)$, los pasos óptimos por eje son $\eta_{1,\text{opt}} = 1$ y $\eta_{2,\text{opt}} = 1/3$. Para evitar divergencia en la dirección de mayor curvatura ($w_2$), el tamaño de paso global debe cumplir:
  $$\eta < 2\min_i \eta_{i,\text{opt}} = \frac{2}{3}$$

---

## 9. Unidad IV: Algoritmo de Backpropagation (CMU 11-785 & Stanford CS221)

El módulo [`backpropagation.py`](backpropagation.py) y los Laboratorios 4.1 a 4.4 de [`web/index.html`](web/index.html) implementan de manera formal, numérica y visual las 60 diapositivas de la cuarta unidad del curso: el cálculo eficiente del gradiente en redes neuronales arbitrariamente profundas mediante grafos de computación y programación dinámica.

### 9.1. El Problema y las Dos Reglas de la Cadena (Diapositivas 5–13 y 18)
* **Descomposición del riesgo empírico:** Dado que $\text{TrainLoss}(W) = \frac{1}{T}\sum_{t=1}^T \text{div}(f(X_t; W), d_t)$, por linealidad de la derivada:
  $$\frac{\partial \text{TrainLoss}(W)}{\partial w_{ij}^{(k)}} = \frac{1}{T}\sum_{t=1}^T \frac{\partial \text{div}(Y_t, d_t)}{\partial w_{ij}^{(k)}}$$
  Basta saber calcular eficientemente las derivadas parciales de la divergencia de una sola instancia respecto a todos los pesos de la red.
* **La derivada como factor de amplificación local en la arista:** Si $y = f(x)$, una perturbación $\Delta x$ se amplifica según $\Delta y \approx \frac{dy}{dx}\Delta x$.
* **Regla de la cadena simple (composición en serie, Diaps. 10, 12 y 18):** Para $z \to y \to l$:
  $$\frac{dl}{dz} = \frac{dl}{dy}\frac{dy}{dz}$$
  En el ejemplo de la Diapositiva 18 ($b = a^2, c = b^2$ evaluado en $a = 2$): $b = 4, c = 16$ y $\frac{\partial c}{\partial a} = (2b)(2a) = 8 \times 4 = 32 = 4a^3$.
* **Regla de la cadena distribuida (múltiples caminos, Diaps. 11–12):** Cuando un nodo $x$ influye en la pérdida $l$ a través de $M$ caminos intermedios $z_1, \dots, z_M$, se multiplica a lo largo de cada camino y se suma sobre todos los caminos:
  $$\frac{dl}{dx} = \sum_{i=1}^M \frac{\partial l}{\partial z_i}\frac{dz_i}{dx}$$

### 9.2. Grafos de Computación y Ejemplos Paso a Paso (Diapositivas 14–23 y 40–41)
En cada nodo $i$ de un grafo acíclico dirigido se definen dos cantidades de igual dimensión (Diapositiva 22):
1. **Valor hacia adelante $f_i$ (en azul):** valor de la subexpresión con raíz en $i$.
2. **Gradiente hacia atrás $g_i = \frac{\partial \text{Loss}}{\partial f_i}$ (en rojo):** sensibilidad de la pérdida final respecto a $f_i$, partiendo con $g_{\text{raíz}} = 1$.

Para el grafo de regresión lineal con pérdida cuadrática $\text{Loss}(x, y, \mathbf{w}) = (\mathbf{w}\cdot\phi(x) - y)^2$:

| Ejemplo | Entradas $(\mathbf{w}, \phi(x), y)$ | Forward $(\text{puntaje}, \text{residuo}, \text{Loss})$ | Backward $(g_{\text{res}}, g_{\text{punt}}, \nabla_{\mathbf{w}}\text{Loss})$ |
| :--- | :---: | :---: | :---: |
| **Diapositiva 21** | $\mathbf{w}=[3,1], \phi(x)=[1,2], y=2$ | $5 \to 3 \to \mathbf{9}$ | $6 \to 6 \to \mathbf{[6, 12]}$ |
| **Diapositiva 23** | $\mathbf{w}=[1,2], \phi(x)=[2,1], y=3$ | $4 \to 1 \to \mathbf{1}$ | $2 \to 2 \to \mathbf{[4, 2]}$ |

### 9.3. Forward Pass y Backward Pass en una Red ReLU de 2 Capas (Diapositivas 24–41 y 57)
* **La Regla de Oro (Diapositiva 29):** Todos los valores intermedios $z_j^{(k)}$ e $y_j^{(k)}$ calculados durante el *forward pass* deben almacenarse en memoria porque el *backward pass* necesita evaluar $f_k'(z_j^{(k)})$ y multiplicar por $y_i^{(k-1)}$.
* **Simetría Forward vs. Backward con notación $\dot{u} \equiv \partial\text{div}/\partial u$ (Diapositiva 37):**
  * Combinación afín: hacia adelante $z_j^{(k)} = \sum_i w_{ij}^{(k)} y_i^{(k-1)}$ vs. hacia atrás $\dot{y}_i^{(k)} = \sum_j w_{ij}^{(k+1)}\dot{z}_j^{(k+1)}$.
  * Activación: hacia adelante $y_i^{(k)} = f_k(z_i^{(k)})$ vs. hacia atrás $\dot{z}_i^{(k)} = f_k'(z_i^{(k)})\dot{y}_i^{(k)}$.
  * Derivada de cada peso: $\frac{\partial\text{div}}{\partial w_{ij}^{(k)}} = y_i^{(k-1)}\dot{z}_j^{(k)}$.
* **Verificación exacta del ejemplo numérico en la Red ReLU de 2 Capas (Diapositivas 38–39 y 57):**
  Con $\mathbf{x} = [1, 2]$, objetivo $d = 3$, divergencia $\text{div} = (Y - d)^2$, $W_1 = \begin{bmatrix}1 & 1\\1 & -1\end{bmatrix}, \mathbf{b}_1 = [-1, 0], W_2 = [[2, 1]], \mathbf{b}_2 = [1]$:
  * **Forward pass:** $z_1^{(1)} = 2 \implies y_1^{(1)} = \text{ReLU}(2) = 2$ (activa); $z_2^{(1)} = -1 \implies y_2^{(1)} = \text{ReLU}(-1) = 0$ (inactiva); $z_1^{(2)} = 2(2)+1(0)+1 = 5 \implies Y = 5$; $\text{div} = (5-3)^2 = \mathbf{4}$.
  * **Backward pass:** $\dot{Y} = \dot{z}^{(2)} = 2(5-3) = \mathbf{4}$; $\nabla_{W_2}\text{div} = \mathbf{[[8, 0]]}, \nabla_{\mathbf{b}_2}\text{div} = \mathbf{[4]}$; bajada a capa 1: $\dot{y}^{(1)} = [8, 4]$, pero al cruzar la derivada de ReLU resulta $\dot{z}_1^{(1)} = 8\cdot 1 = \mathbf{8}$ y $\dot{z}_2^{(1)} = 4\cdot 0 = \mathbf{0}$ (**la neurona inactiva corta el gradiente**); finalmente $\nabla_{W_1}\text{div} = \begin{bmatrix}8 & 16\\0 & 0\end{bmatrix}$ y $\nabla_{\mathbf{b}_1}\text{div} = \mathbf{[8, 0]}$.

### 9.4. Vista Vectorial y las 4 Fórmulas con Jacobianos Transpuestos (Diapositivas 42–60)
Para una capa vectorial $\mathbf{z}_k = W_k\mathbf{y}_{k-1} + \mathbf{b}_k$, $\mathbf{y}_k = f_k(\mathbf{z}_k)$ con $W_k \in \mathbb{R}^{D_k \times D_{k-1}}$:
1. **Cruzar la activación:** $\nabla_{\mathbf{z}_k}\text{div} = J_{\mathbf{y}_k}(\mathbf{z}_k)^\top \nabla_{\mathbf{y}_k}\text{div}$ (diagonal si $f_k$ es componente a componente; matriz llena $J_{ij} = y_i(\mathbf{1}[i=j]-y_j)$ si $f_k$ es Softmax).
2. **Bajar a la capa anterior:** $\nabla_{\mathbf{y}_{k-1}}\text{div} = W_k^\top \nabla_{\mathbf{z}_k}\text{div}$.
3. **Gradiente de los pesos (producto externo):** $\nabla_{W_k}\text{div} = (\nabla_{\mathbf{z}_k}\text{div})\,\mathbf{y}_{k-1}^\top$.
4. **Gradiente del sesgo:** $\nabla_{\mathbf{b}_k}\text{div} = \nabla_{\mathbf{z}_k}\text{div}$.

---

## 10. Problemas de Convergencia en el Entrenamiento (Unidad V — 58 Diapositivas)

### 10.1. ¿Encuentra *Backprop* la Solución Correcta? Experimento de Brady, Raghavan y Slawny (1989) (Diapositivas 7–16)
* **La pérdida es un sustituto (*proxy*):** En clasificación interesa minimizar el error de clasificación $0\text{-}1$, pero por no ser diferenciable se minimiza una divergencia suave ($L_2$ o entropía cruzada). Minimizar el sustituto **no garantiza** minimizar el error de clasificación.
* **Caso de 3 puntos linealmente separables (Diaps. 8–9):** Con una neurona logística $y = \sigma(w_1 x_1 + w_2 x_2 + b)$ y margen $\varepsilon = 0.01$ ($u = \sigma^{-1}(1-\varepsilon) = \ln 99 \approx 4.59512$), los puntos $(1,0)\to +1$, $(0,1)\to +1$, $(-1,0)\to -1$ imponen el sistema $w_1+b=u, \; w_2+b=u, \; -w_1+b=-u$, cuya única solución es $w_1 = w_2 = u, \; b = 0$ (recta $x_1 + x_2 = 0$).
* **El cuarto punto «saboteador» (*spoiler*) lejano $(0, -t)\to +1$ (Diaps. 10–13):** Al agregar $(0, -t)$ con $t \gg 1$, el conjunto de 4 puntos sigue siendo linealmente separable (el Perceptrón encuentra una recta casi vertical y acierta $4/4$). Sin embargo, evaluando la solución de 3 puntos en $(0, -t)$, la preactivación es $z_4 = -ut \to -\infty$ y la regla de la cadena entrega:
  $$\frac{d\,\text{div}_4}{dw_2} = 2(1 - \varepsilon - \sigma(z_4))\,\sigma'(z_4)\,t$$
  Como $\sigma'(z_4) = \sigma(z_4)(1-\sigma(z_4)) \sim e^{-ut}$ decae **exponencialmente** mientras $t$ crece solo **linealmente**, resulta $\frac{d\,\text{div}_4}{dw_2} \to 0$ y $\frac{d\,\text{div}_4}{db} \to 0$ (por ejemplo, para $t=50$, $\text{div}_4 = 0.9801$ pero $\frac{d\,\text{div}_4}{dw_2} = 1.64\times 10^{-98}$). La solución de 3 puntos permanece como un **mínimo local ancho** que deja $(0,-t)$ mal clasificado ($3/4 = 75\%$).

### 10.2. Superficie de Pérdida, las 6 Trayectorias desde $(-8, 4)$ y Número de Condición $\kappa$ (Diapositivas 17–33)
* **Puntos de silla y paisaje en redes grandes (Diaps. 18–19):** Un punto de silla tiene $\nabla E = \mathbf{0}$ pero el Hessiano posee autovalores positivos y negativos (ej. $E = \frac{1}{2}(w_1^2 - w_2^2)$). En redes grandes, los puntos de silla son exponencialmente más abundantes que los mínimos locales (**Dauphin et al., 2014**) y la mayoría de los mínimos locales son cercanos al global (**Choromanska et al., 2015**), aunque en redes finitas con datos finitos pueden existir malos mínimos locales (**Swirszcz et al., 2016**).
* **Las 6 trayectorias sobre $E(w_1, w_2) = \frac{1}{2}(w_1^2 + 3w_2^2)$ desde $\mathbf{w}^{(0)} = (-8, 4)$ (Diapositiva 27):** Con $a_1 = 1 \implies \eta_{1,\text{opt}} = 1$ y $a_2 = 3 \implies \eta_{2,\text{opt}} = 1/3$, el umbral global es $\eta < 2\min_i\eta_{i,\text{opt}} = 2/3$:
  1. $\eta = 0.25$: converge monótonamente en ambos ejes ($\mathbf{w}^{(1)} = (-6, 1)$).
  2. $\eta = 1/3 = \eta_{2,\text{opt}}$: resuelve $w_2 = 0$ en un solo paso ($\mathbf{w}^{(1)} = (-5.33, 0)$) mientras $w_1$ baja lento.
  3. $\eta = 0.50 = \eta^*$: oscila en $w_2$ y converge con factor óptimo $\frac{\kappa-1}{\kappa+1} = 0.5$.
  4. $\eta = 2/3 = 2\eta_{2,\text{opt}}$: rebota sin fin en $w_2$ entre $+4$ y $-4$ mientras $w_1 \to 0$.
  5. $\eta = 0.75$: diverge oscilando en $w_2$.
  6. $\eta = 1.00 = \eta_{1,\text{opt}}$: anula $w_1 = 0$ en un paso, pero $w_2$ explota ($\mathbf{w}^{(1)} = (0, -8)$).
* **Número de condición y velocidades por familia (Diaps. 29–32):** Con $\kappa = \frac{a_{\max}}{a_{\min}}$, el mejor factor geométrico por paso es $\frac{\kappa-1}{\kappa+1}$. Las funciones **fuertemente convexas** convergen linealmente en $O(\log(1/\delta))$ pasos, mientras que las funciones **convexas solo Lipschitz-suaves** caen como $O(1/k)$ y requieren $O(1/\delta)$ pasos.

### 10.3. Métodos de Segundo Orden: Normalización $A^{1/2}$, Método de Newton y Punto de Silla (Diapositivas 34–44)
* **Escalamiento de coordenadas y regla de Newton (Diaps. 35–40):** El cambio de variable $\hat{\mathbf{w}} = A^{1/2}\mathbf{w}$ transforma las elipses en círculos ($\kappa = 1$). De vuelta en el espacio original, produce la **actualización de Newton**:
  $$\mathbf{w}^{(k+1)} = \mathbf{w}^{(k)} - \eta H_E(\mathbf{w}^{(k)})^{-1}\nabla_{\mathbf{w}}E(\mathbf{w}^{(k)})$$
  Con $\eta = 1$, alcanza el mínimo de cualquier cuadrática en **un solo paso**.
* **Los dos problemas de Newton (Diaps. 41–43):**
  1. **Tamaño prohibitivo:** para $N = 100\,000$ parámetros, $H_E$ tiene $N^2 = 10^{10}$ entradas e invertirla cuesta $O(N^3)$.
  2. **Atracción hacia puntos de silla:** en zonas de curvatura negativa ($E = \frac{1}{2}(w_1^2 - w_2^2)$), $H_E^{-1}$ invierte el signo en el eje negativo y salta directo al punto de silla $(0,0)$, mientras que **Levenberg-Marquardt** $(H_E + \lambda I)^{-1}$ con $\lambda > |\lambda_{\min}|$ restaura el descenso.

### 10.4. Tasas Decrecientes, RProp y QuickProp (Diapositivas 45–58)
* **Explorar primero, converger después (Diaps. 45–47):** Un $\eta_0$ inicial grande permite saltar barreras de mínimos locales mediocres y luego aterrizar con esquemas decrecientes ($\eta_0/k$, $\eta_0/k^2$, $\eta_0 e^{-\beta(k-1)}$ o por escalones).
* **RProp (*Resilient Propagation*, Diaps. 48–53):** Mantiene un paso adaptativo $\Delta_i$ por coordenada guiado solo por $\text{signo}(\partial E/\partial w_i)$: si el signo se mantiene, $\Delta_i \leftarrow \min(\alpha\Delta_i, \Delta_{\max})$ ($\alpha=1.2$); si el signo cambia, se pasó del mínimo, por lo que **deshace el paso** y reduce $\Delta_i \leftarrow \max(\beta\Delta_i, \Delta_{\min})$ ($\beta=0.5$).
* **QuickProp (Diaps. 54–56):** Aproxima Newton coordenada a coordenada estimando $E''(w^{(k)})$ por diferencia finita de derivadas sucesivas:
  $$\Delta w^{(k)} = \Delta w^{(k-1)}\cdot \frac{E'(w^{(k)})}{E'(w^{(k-1)}) - E'(w^{(k)})}$$

---

## 11. Descenso por Gradiente Estocástico, Mini-lotes y Optimizadores con Tendencia (Unidad VI — 41 Diapositivas)

### 11.1. El Costo del Lote, «SGD a Mano: Una Época» y el Barajado por Época (Diapositivas 5–14)
* **El costo computacional del lote (Diaps. 5–10):** La función objetivo empírica es el promedio sobre el conjunto de entrenamiento:
  $$\text{TrainLoss}(\mathbf{w}) = \frac{1}{|\mathcal{D}_{\text{train}}|}\sum_{(x,y)\in\mathcal{D}_{\text{train}}}\text{Loss}(x,y,\mathbf{w}), \qquad \nabla_{\mathbf{w}}\text{TrainLoss}(\mathbf{w}) = \frac{1}{|\mathcal{D}_{\text{train}}|}\sum_{(x,y)\in\mathcal{D}_{\text{train}}}\nabla_{\mathbf{w}}\text{Loss}(x,y,\mathbf{w})$$
  Con $N = |\mathcal{D}_{\text{train}}| = 1\,000\,000$, una actualización por lote evalúa $1\,000\,000$ de gradientes para mover $\mathbf{w}$ **una sola vez**. En cambio, tras $T$ épocas (*epochs*), el descenso por gradiente estocástico (**SGD**) evalúa los mismos $TN$ gradientes pero actualiza $\mathbf{w}$ **$TN$ veces** ($\mathbf{w} \leftarrow \mathbf{w} - \eta\nabla_{\mathbf{w}}\text{Loss}(x,y,\mathbf{w})$).
* **Verificación exacta de «SGD a mano: una época» (Diapositiva 14):** Para $\mathcal{D}_{\text{train}} = \{(1, 1), (2, 3), (4, 3)\}$, $\phi(x) = [1, x]$, $\eta = 0.1$, $\mathbf{w}^{(0)} = [0, 0]$, $\text{Loss}(x,y,\mathbf{w}) = (\mathbf{w}\cdot\phi(x) - y)^2$ y $\nabla_{\mathbf{w}}\text{Loss} = 2r\phi(x)$ con $r = \mathbf{w}\cdot\phi(x) - y$:

| Paso | Ejemplo $(x, y)$ | Predicción $\mathbf{w}\cdot\phi(x)$ | Residuo $r$ | $\nabla_{\mathbf{w}}\text{Loss} = 2r\phi(x)$ | $\mathbf{w}_{\text{nuevo}}$ (exacto / Diap. 14) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $(1, 1)$ | $0.00$ | $-1.00$ | $[-2.00, -2.00]$ | $[0.20, 0.20]$ |
| **2** | $(2, 3)$ | $0.60$ | $-2.40$ | $[-4.80, -9.60]$ | $[0.68, 1.16]$ |
| **3** | $(4, 3)$ | $5.32$ | $+2.32$ | $[4.64, 18.56]$ | $[0.216, -0.696] \approx \mathbf{[0.22, -0.70]}$ |

  En comparación, una época de lote realiza una sola actualización hacia $\mathbf{w}_{\text{lote}}^{(1)} = [0.47, 1.27]$, y el óptimo de mínimos cuadrados es $\mathbf{w}^* = [1.00, 0.57]$.
* **Orden de presentación y barajado (*shuffling*, Diaps. 12–13):** Si los datos están agrupados (p. ej., todo el Grupo A y luego todo el Grupo B) y se recorren siempre en el mismo orden, los pesos oscilan cíclicamente persiguiendo al último grupo visto. Permutar $\mathcal{D}_{\text{train}}$ al azar al inicio de **cada época** es parte esencial del algoritmo de SGD.

### 11.2. Tasa de Aprendizaje Decreciente, Condiciones de Robbins-Monro (1951) y Velocidad (Diapositivas 15–20)
* **Condiciones suficientes de convergencia (Robbins y Monro, 1951, Diap. 17):** Como cada ejemplo empuja en una dirección ruidosa distinta, un paso constante $\eta$ nunca deja de oscilar alrededor del mínimo. Para garantizar convergencia se exige:
  $$\sum_{k=1}^{\infty}\eta_k = \infty \quad (\text{alcanza cualquier región}), \qquad \sum_{k=1}^{\infty}\eta_k^2 < \infty \quad (\text{amortigua el ruido del último ejemplo})$$
  El esquema $\eta_k \propto 1/k$ cumple ambas ($\sum_{k=1}^\infty 1/k = \infty$ y $\sum_{k=1}^\infty 1/k^2 = \pi^2/6 \approx 1.6449 < \infty$) y es óptimo cuando la pérdida es fuertemente convexa.
* **Velocidad de convergencia y experimento en Python (Diaps. 18–20):** Para situarse a distancia $\varepsilon$ del óptimo en una pérdida fuertemente convexa, el lote requiere $O(\log(1/\varepsilon))$ iteraciones de costo $N$, mientras que SGD requiere $O(1/\varepsilon)$ iteraciones de costo $1$. En el experimento en Python (Diap. 19), tras recorrer apenas $1/10$ de la primera época, SGD ya aproxima $\mathbf{w}_{\text{real}}$ mientras el lote aún lleva $0$ actualizaciones.

### 11.3. ¿Por qué Funciona? Varianza del Estimador $\text{Var}(\text{Loss})/b$ y Mini-lotes (Diapositivas 21–29 y 39)
* **Estimador insesgado y escalamiento de varianza (Diaps. 21–27):** Tanto el lote completo como un único ejemplo $(x,y)$ o un mini-lote $B_j$ de tamaño $b$ son estimadores **insesgados** del riesgo real $R(\mathbf{w})$:
  $$\mathbb{E}\!\left[\frac{1}{b}\sum_{(x,y)\in B_j}\text{Loss}(x,y,\mathbf{w})\right] = R(\mathbf{w}), \qquad \text{Var}\!\left(\frac{1}{b}\sum_{(x,y)\in B_j}\text{Loss}(x,y,\mathbf{w})\right) = \frac{1}{b}\text{Var}(\text{Loss})$$
* **Las tres formas de dar un paso (Diapositiva 39):** El mini-lote ($b$ entre $32$ y $256$) reduce la varianza en un factor $1/b$, aprovecha el paralelismo vectorial de las GPUs (calcular $b$ gradientes cuesta casi lo mismo en tiempo que calcular $1$) y recupera como casos extremos el SGD puro ($b=1$) y el lote completo ($b=N$).

### 11.4. Optimizadores con Tendencia: Momentum, Nesterov, RMSprop y Adam (Diapositivas 30–38)
* **Momentum (Polyak, 1964, Diaps. 31–32) y Nesterov (1983, Diap. 33):**
  $$\text{Momentum: }\Delta\mathbf{w}^{(k)} = \beta\Delta\mathbf{w}^{(k-1)} - \eta\nabla_{\mathbf{w}}\text{Loss}(\mathbf{w}^{(k-1)}), \qquad \text{Nesterov: }\Delta\mathbf{w}^{(k)} = \beta\Delta\mathbf{w}^{(k-1)} - \eta\nabla_{\mathbf{w}}\text{Loss}\!\left(\mathbf{w}^{(k-1)} + \beta\Delta\mathbf{w}^{(k-1)}\right)$$
  donde Nesterov primero extiende el paso de inercia $\beta\Delta\mathbf{w}^{(k-1)}$ y evalúa el gradiente en el punto adelantado donde va a caer.
* **Tasa por dirección y RMSprop (Tieleman y Hinton, 2012, Diaps. 34–35):** En la tabla de 5 pasos de las Diapositivas 34–35 ($\Delta x = [1, 1, 2, 1, 1.5] \implies \sum|\Delta x|=6.5, \sum\Delta x=6.5, \text{RMS}_x = \sqrt{9.25/5} \approx \mathbf{1.36}$; $\Delta y = [+2.5, -3, +2.5, -2, +1.5] \implies \sum|\Delta y|=11.5, \sum\Delta y=1.5, \text{RMS}_y = \sqrt{27.75/5} \approx \mathbf{2.36}$), la dirección vertical se mueve casi el doble pero avanza 4 veces menos. RMSprop mantiene el promedio móvil cuadrático $E[\partial_w^2]^{(k)} = \gamma E[\partial_w^2]^{(k-1)} + (1-\gamma)(\partial_w^{(k)})^2$ y divide el paso por $\sqrt{E[\partial_w^2]^{(k)}+\epsilon}$.
* **Adam (*Adaptive Moment Estimation*, Kingma y Ba, 2015, Diaps. 36–38):** Combina el primer momento (tendencia media) y el segundo momento (agite cuadrático) con corrección de sesgo inicial ($\beta_1=0.9, \beta_2=0.999, \eta=0.001$):
  $$m^{(k)} = \beta_1 m^{(k-1)} + (1-\beta_1)\partial_w^{(k)}, \quad v^{(k)} = \beta_2 v^{(k-1)} + (1-\beta_2)(\partial_w^{(k)})^2, \quad \hat{m} = \frac{m^{(k)}}{1-\beta_1^k}, \quad \hat{v} = \frac{v^{(k)}}{1-\beta_2^k}, \quad w^{(k+1)} = w^{(k)} - \frac{\eta}{\sqrt{\hat{v}}+\epsilon}\hat{m}$$

---

## 12. Conclusiones

1. **Comprobación de la teoría (Unidad I):** Se demostró de manera analítica y computacional que el MLP de 3 neuronas y la variante compacta de 2 neuronas de Friedland resuelven de forma exacta el operador lógico XOR, y que un MLP de una sola capa oculta es un aproximador universal booleano, geométrico y continuo cuyo ancho puede crecer como $2^{N-1}$ si no se emplea profundidad ($3(N-1)$).
2. **Aprendizaje y diferenciabilidad (Unidad II):** Se comprobó empíricamente la convergencia del perceptrón de Rosenblatt en 3 pasos sobre el conjunto de la Diapositiva 22 ($\mathbf{w} = [3.25, 1.9]$), la regla Delta de ADALINE, la diferencia entre el paisaje plano del conteo de errores y la señal continua de la sigmoide, y la superioridad del par Sigmoide/Softmax + Entropía Cruzada (KL) frente a L2 en problemas de clasificación.
3. **Generalización, Regularización y Validación (Unidad CD05):** Se demostró de forma concluyente la falacia de la memorización empírica pura ($\text{TrainLoss}=0$ pero colapso en generalización), el sesgo por correlación espuria de Oakden-Rayner et al. (2019) con el tubo torácico de drenaje, la imposibilidad geométrica de generalizar en hipercubos $\{0, 1\}^{100}$ sin sesgo inductivo, la cota de suavidad lipschitziana de Cauchy-Schwarz ($|s(x)-s(x')| \le \|w\|\|\Delta\phi\|$), el paso a mano exacto de L2 ($w^{(1)} = [1.8, -2.0]$ de la Diapositiva 26), la reducción de $10^6$ a 210 modelos en Greedy Forward Selection, el desenmascaramiento del falso mínimo $d=6$ (1.9) frente al grado verdadero $d=3$ (3.8) mediante 5-Fold Cross-Validation en el experimento $y=x^3-3x$ (con explosión a 32.3 en el Fold 4), la auditoría final en $200\,000$ datos (MSE 4.2 vs 5.1), la descomposición Sesgo-Varianza en la diana y la propiedad de $n$ como único botón de ingeniería con costo estadístico cero.
4. **Optimización por Descenso de Gradiente (Unidad III):** Se verificaron analítica y numéricamente la ortogonalidad del gradiente respecto a las curvas de nivel, la clasificación del punto crítico 3D $\mathbf{x}^*=(-1,-1,-1)$ mediante su Matriz Hessiana definida positiva ($\lambda = \{3.414, 2, 0.586\} > 0$), la trayectoria de 200 épocas en Regresión Lineal hacia $\mathbf{w}^*=[1, 0.57]$, y los 4 regímenes de convergencia/divergencia gobernados por la relación entre $\eta$ y la curvatura $(E'')^{-1}$.
5. **Algoritmo de Backpropagation (Unidad IV):** Se verificaron las reglas de la cadena simple y distribuida, los grafos de computación de las Diapositivas 21 y 23 ($\nabla_{\mathbf{w}}\text{Loss} = [6, 12]$ y $[4, 2]$), el cálculo completo hacia adelante y hacia atrás en la red ReLU de 2 capas de las Diapositivas 38–39 y 57 ($\nabla_{W_1} = \begin{bmatrix}8 & 16\\0 & 0\end{bmatrix}$), y la equivalencia exacta con la formulación vectorial basada en Jacobianos transpuestos.
6. **Problemas de Convergencia y Métodos Adaptativos (Unidad V):** Se verificó numéricamente el fenómeno de los puntos «saboteadores» de Brady et al. (1989) donde la saturación exponencial de la sigmoide apaga el gradiente y atrapa a *Backprop* en un mínimo local ($75\%$) mientras el Perceptrón separa los 4 puntos ($100\%$); se reprodujeron las 6 trayectorias de la Diapositiva 27 gobernadas por el número de condición $\kappa$; se demostró la convergencia en 1 paso de Newton en cuencos cuadráticos junto con su atracción patológica hacia puntos de silla (corregida por Levenberg-Marquardt); y se compararon las tasas decrecientes, RProp y QuickProp.
7. **Descenso por Gradiente Estocástico, Mini-lotes y Optimizadores con Tendencia (Unidad VI):** Se reprodujo exactamente la tabla de «SGD a mano: una época» de la Diapositiva 14 ($[0.20, 0.20] \to [0.68, 1.16] \to [0.22, -0.70]$), la necesidad del barajado por época (*shuffling*), las condiciones de convergencia de Robbins y Monro (1951), la ley de reducción de varianza $\frac{1}{b}\text{Var}(\text{Loss})$ en mini-lotes, la tabla direccional de RMSprop ($\text{RMS}_x \approx 1.36, \text{RMS}_y \approx 2.36$) y la superioridad de Momentum, Nesterov, RMSprop y Adam en valles angostos y puntos de silla.
8. **Validación integral:** Todos los módulos evaluados en [`perceptron_xor.py`](perceptron_xor.py), [`introduccion_curso.py`](introduccion_curso.py), [`preprocesamiento_eda.py`](preprocesamiento_eda.py), [`visualizacion_datos.py`](visualizacion_datos.py), [`introduccion_ml.py`](introduccion_ml.py), [`generalizacion_validacion.py`](generalizacion_validacion.py), [`aproximacion_universal.py`](aproximacion_universal.py), [`aprendizaje_redes.py`](aprendizaje_redes.py), [`descenso_gradiente.py`](descenso_gradiente.py), [`backpropagation.py`](backpropagation.py), [`problemas_convergencia.py`](problemas_convergencia.py), [`sgd_lotes.py`](sgd_lotes.py) y la suite web [`web/index.html`](web/index.html) arrojaron concordancia del 100% con el material de cátedra (12 Unidades Académicas • 739 Diapositivas • 49 Laboratorios Interactivos).

---

## 13. Referencias Bibliográficas

1. **Cauchy, A. (1847).** *Méthode générale pour la résolution des systèmes d'équations simultanées.* Comptes Rendus Hebdomadaires des Séances de l'Académie des Sciences, 25, 536-538.
2. **McCulloch, W. S., & Pitts, W. (1943).** *A logical calculus of the ideas immanent in nervous activity.* The Bulletin of Mathematical Biophysics, 5(4), 115-133.
3. **Robbins, H., & Monro, S. (1951).** *A stochastic approximation method.* The Annals of Mathematical Statistics, 22(3), 400-407.
4. **Rosenblatt, F. (1958).** *The perceptron: A probabilistic model for information storage and organization in the brain.* Psychological Review, 65(6), 386-408.
5. **Widrow, B., & Hoff, M. E. (1960).** *Adaptive switching circuits.* IRE WESCON Convention Record, Part 4, 96-104.
6. **Polyak, B. T. (1964).** *Some methods of speeding up the convergence of iteration methods.* USSR Computational Mathematics and Mathematical Physics, 4(5), 1-17.
7. **Minsky, M., & Papert, S. (1969).** *Perceptrons: An Introduction to Computational Geometry.* MIT Press, Cambridge, MA.
8. **Nesterov, Y. (1983).** *A method for unconstrained convex minimization problem with the rate of convergence $O(1/k^2)$.* Doklady AN SSSR, 269, 543-547.
9. **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).** *Learning representations by back-propagating errors.* Nature, 323(6088), 533-536.
10. **Fahlman, S. E. (1988).** *An empirical study of learning speed in back-propagation networks (QuickProp).* CMU-CS-88-162.
11. **Brady, M. L., Raghavan, R., & Slawny, J. (1989).** *Back propagation fails to separate where perceptrons succeed.* IEEE Transactions on Circuits and Systems, 36(5), 665-674.
12. **Baldi, P., & Hornik, K. (1989).** *Neural networks and principal component analysis: Learning from examples without local minima.* Neural Networks, 2(1), 53-58.
13. **Cybenko, G. (1989).** *Approximation by superpositions of a sigmoidal function.* Mathematics of Control, Signals and Systems, 2(4), 303-314.
14. **Riedmiller, M., & Braun, H. (1993).** *A direct adaptive method for faster backpropagation learning: The RPROP algorithm.* IEEE International Conference on Neural Networks, 586-591.
15. **Tieleman, T., & Hinton, G. (2012).** *Lecture 6.5—rmsprop: Divide the gradient by a running average of its recent magnitude.* COURSERA: Neural Networks for Machine Learning.
16. **Dauphin, Y. N., Pascanu, R., Gulcehre, C., Cho, K., Ganguli, S., & Bengio, Y. (2014).** *Identifying and attacking the saddle point problem in high-dimensional non-convex optimization.* NIPS 2014.
17. **Kingma, D. P., & Ba, J. (2015).** *Adam: A method for stochastic optimization.* ICLR 2015.
18. **Choromanska, A., Henaff, M., Mathieu, M., Ben Arous, G., & LeCun, Y. (2015).** *The loss surfaces of multilayer networks.* AISTATS 2015.
19. **Swirszcz, G., Czarnecki, W. M., & Pascanu, R. (2016).** *Local minima in training of neural networks.* arXiv:1611.06310.
20. **Oakden-Rayner, L., Gale, W., Bonham, T. A., Lungren, M. P., Carneiro, G., Bradley, A. P., & Palmer, L. J. (2019).** *Validation and Algorithmic Audit of a Deep Learning System for the Detection of Pneumothorax in Chest Radiographs.* Scientific Reports, 10, 16140.
21. **Rand, W. M. (1971).** *Objective criteria for the evaluation of clustering methods.* Journal of the American Statistical Association, 66(336), 846-850.
22. **Lloyd, S. (1982).** *Least squares quantization in PCM.* IEEE Transactions on Information Theory, 28(2), 129-137.
23. **Hubert, L., & Arabie, P. (1985).** *Comparing partitions.* Journal of Classification, 2(1), 193-218.
24. **Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996).** *A density-based algorithm for discovering clusters in large spatial databases with noise.* KDD 1996, 226-231.
25. **Arthur, D., & Vassilvitskii, S. (2007).** *k-means++: The advantages of careful seeding.* SODA 2007, 1027-1035.
26. **Campello, R. J., Moulavi, D., & Zimek, A. (2013).** *Density-based clustering based on hierarchical density estimates.* PAKDD 2013, 160-172.
27. **Raj, B., et al. (CMU 11-785), Protopapas, P., et al. (Harvard CS109), Liang, P., et al. (Stanford CS221) & Kolter, Z. (CMU 15-388).** *Aproximación Universal, Aprendizaje de Redes Neuronales, Descenso por Gradiente, Backpropagation, Problemas de Convergencia, Descenso Estocástico, Introducción a Machine Learning & Generalización y Validación.* Ciencia de Datos — Doctorado UCN.
