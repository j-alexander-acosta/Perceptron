# Contexto 12: Integración de la Unidad CD01 (Primera en Orden) — Introducción al Curso (63 Diapositivas)
**Fecha de registro:** 2026-11-09

## 1. Instrucciones
* Incorporar la nueva presentación académica de 63 diapositivas (**«CD01-Introducción-al-Curso.pdf»**, Ciencia de Datos — UCN, basado en material de CMU 15-388/688 de Z. Kolter, Harvard CS109 de P. Protopapas et al., Stanford CS221 de P. Liang et al., y UC Berkeley DS100 para la sección de LLMs) **al inicio del proyecto (de primeras)**, respetando estrictamente el orden numérico del nombre de archivo (`CD01`, `CD02`, ..., `CD13`–`CD18`) para que las futuras unidades previas (`CD02` a `CD12`) puedan insertarse secuencialmente en su posición correspondiente.
* Desarrollar el módulo computacional interactivo y autoverificable en Python puro (`introduccion_curso.py`) que implemente y verifique numéricamente las 4 secciones de la unidad `CD01`:
  1. **Caso Real 1 — Ciencia de Datos en la Pandemia, Sesgo en la Tasa de Letalidad y Toque de Queda (Diaps. 5–11, 19–21):**
     * Los 5 pasos del proceso en la pandemia y la clasificación de las 4 preguntas clave (*¿Qué tan rápido se propaga?* $\to$ modelamiento; *¿Cuándo se saturarán los hospitales?* $\to$ predicción; *¿Funcionan las cuarentenas?* $\to$ inferencia causal; *¿Cómo distribuir vacunas con equidad?* $\to$ optimización) junto con el dilema ético de elegir la métrica de éxito (Flaxman et al., 2020, *Nature*).
     * Fuentes de datos y sus sesgos estructurales (testeo PCR/antígenos, movilidad de celulares Google/Apple, registros civiles de defunciones).
     * **Verificación exacta del sesgo en la tasa de letalidad (Diapositiva 9):** con $100$ muertes y $2\,000$ casos confirmados, la letalidad aparente (CFR) es $100/2\,000 = \mathbf{5.0\%}$, mientras que con $20\,000$ infecciones reales la letalidad real (IFR) es $100/20\,000 = \mathbf{0.5\%}$ (el denominador sesgado infla la letalidad **$10$ veces**, cambiando el orden de magnitud).
     * **Ejercicio del Toque de Queda Nocturno (Diapositiva 21):** inferencia causal, sesgo de cobertura de *smartphones* y variable de confusión (el miedo al virus reduce movilidad y contagios simultáneamente).
  2. **El Modelo Epidemiológico SIR a Mano, Número Reproductivo Básico $R_0$ y Cuarentenas (Diaps. 12–18):**
     * Ecuaciones en diferencias diarias para $S_t, I_t, R_t$ con población $N = S + I + R$: $\text{contagios}_t = \beta S_t I_t / N$, $\text{recuperaciones}_t = \gamma I_t$.
     * **Reproducción exacta de la tabla «La dinámica SIR, a mano» (Diapositiva 13):** con $N = 1\,000$, $\beta = 0.3$, $\gamma = 0.1$ ($1/\gamma = 10$ días), $S_0 = 990.0$, $I_0 = 10.0$, $R_0 = 0.0$:
       - **Día 0:** contagios $= 2.97$, recuperaciones $= 1.00$, $S = 990.0$, $I = 10.0$, $R = 0.0$.
       - **Día 1:** contagios $= 3.54$, recuperaciones $= 1.20$, $S = 987.0$, $I = 12.0$, $R = 1.0$.
       - **Día 2:** contagios $= 4.22$, recuperaciones $= 1.43$, $S = 983.5$, $I = 14.3$, $R = 2.2$.
       - **Día 28 (pico):** contagios $= 30.24$, recuperaciones $= 31.59$, $S = 319.0$, $I = \mathbf{315.9}$, $R = 365.0$, y cerca del **$95\%$** de la población infectada al final de la epidemia ($R_{160} \approx 947$, Diapositiva 14).
     * **Efecto de las cuarentenas sobre $R_0 = \beta/\gamma$ (Diapositiva 15):** bajar $\beta$ de $0.3$ ($R_0 = 3.0$) a $0.16$ ($R_0 = 1.6$) reduce el pico de infectados simultáneos de $\approx \mathbf{316}$ (día 28) a $\approx \mathbf{89}$ (día 61), «aplanando y retrasando la curva», junto con la lección del modelo IHME (Diapositivas 16–18).
  3. **¿Qué es Ciencia de Datos?, El Caso Netflix Prize (2006–2009) y la Paradoja de la Sobre-optimización (Diaps. 22–40):**
     * Espectro Máquina $\leftrightarrow$ Humano (Daniel Keim, *Visual Analytics*) y las 5 etapas del ciclo iterativo de Ciencia de Datos, incluyendo los dos casos donde el producto no es un modelo predictivo complejo: *Our World in Data* (visualización como producto final, Diap. 39) y *Mapear la pobreza desde el cielo* mediante el material del techo como variable sustituta o *proxy* (Abelson, Varshney y Sun, 2014, Diap. 40).
     * **Verificación exacta de la matriz del Netflix Prize (Diapositivas 27–28):** $480\,189 \text{ usuarios} \times 17\,770 \text{ películas} = \mathbf{8\,532\,958\,530} \approx \mathbf{8\,533\text{ millones}}$ de celdas, con $\mathbf{100\,480\,507}$ calificaciones observadas ($\approx \mathbf{1.18\% \approx 1.2\%}$ observado; **falta casi el $99\%$** de la tabla con datos faltantes no al azar, MNAR).
     * **Progresión de 3 años del RMSE (Diapositivas 30–31):** Cinematch ($2006$, $\text{RMSE} = 0.951$) $\to$ Año 1 ($+8.43\%$) $\to$ Año 2 ($+9.44\%$) $\to$ Año 3 ($+10.06\%$, $\text{RMSE} = 0.856$, *BellKor's Pragmatic Chaos* por 20 minutos de diferencia), y la **Paradoja de la Sobre-optimización** (Amatriain y Basilico, 2012; Kästner, CMU 17-645): por qué el ensamble ganador nunca llegó a producción.
  4. **Los Tres Relatos de la IA, el Mapa de 4 Niveles y la Política de Uso de LLMs (Diaps. 1–4, 41–63):**
     * El Test de Turing (1950) y las tres tradiciones históricas de la IA: **Relato simbólico** (Dartmouth 1956, traducción rusa e informe ALPAC 1966 $\to$ 1.er invierno; sistemas expertos $\to$ 2.º invierno en 1987), **Relato neuronal** (McCulloch-Pitts 1943, Hebb 1949, Rosenblatt 1958, Minsky y Papert 1969, Backpropagation 1986, AlexNet 2012 impulsado por datos y cómputo) y **Relato estadístico** (Gauss y Legendre 1801 con Ceres, Fisher 1936, Bellman 1957, Vapnik 1995).
     * **El Mapa de 4 Niveles de Modelos en IA (Diapositivas 3 y 50):** Reflejos (se recorre entero), Estados (búsqueda, MDP y refuerzo, sin juegos), Variables (modelos probabilísticos, sin CSP) y Lógica (fuera del curso).
     * **Política de Uso de LLMs (Diapositivas 58–61, UC Berkeley DS100):** desglose en 3 partes (Contexto, Planificación, Ejecución) y evaluación de los 4 casos prácticos.
* Actualizar `web/index.html` situando **CD01: Introducción al Curso (63 Diaps.)** **de primeras** en el selector de unidades (`.unit-switcher`) y como vista activa inicial al cargar la plataforma, incorporando 4 nuevos laboratorios interactivos (`panel-25` a `panel-28`) y 4 nuevas preguntas de autoevaluación (`quiz-item-25` a `quiz-item-28`), alcanzando **7 Unidades Académicas (386 Diapositivas) y 29 Laboratorios Interactivos** ordenados por código de cátedra (`CD01`, `CD13`–`CD18`).
* Integrar `introduccion_curso.py` en primera posición dentro de `perceptron_xor.py` (`--intro` / `-i`, opción `[0]` en el menú interactivo y primer bloque de `--auto`), `README.md` e `INFORME.md`.

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Integrar teórica, computacional y visualmente la unidad inicial **CD01 — Introducción al Curso (63 Diapositivas)** en la primera posición de toda la arquitectura del proyecto, dejando la estructura indexada por código `CDxx` para futuras unidades intermedias (`CD02`–`CD12`).
* **Fases/Hitos principales:**
  1. Creación del registro estructurado `contexto/context12.md`.
  2. Implementación del módulo `introduccion_curso.py` con verificaciones numéricas exactas de las Diapositivas 9, 13–15, 21, 27–34, 41–51 y 58–61.
  3. Actualización de `web/index.html` ubicando **CD01 — Introducción al Curso** como la primera unidad activa en la barra superior y añadiendo sus 4 laboratorios (`panel-25` a `panel-28`) y 4 reactivos de autoevaluación.
  4. Integración en primera posición dentro de `perceptron_xor.py`, actualización de `README.md` e `INFORME.md`, y verificación automatizada (`python3 perceptron_xor.py --auto`).
* **Entregables:** `contexto/context12.md`, `introduccion_curso.py`, `web/index.html` actualizado (7 unidades ordenadas `CD01`, `CD13`–`CD18`; 29 laboratorios), `perceptron_xor.py`, `README.md` e `INFORME.md`.

## 3. Referencias
* **Path del plan original:** `contexto/context12.md`
* **Material de cátedra:** Presentación *«CD01-Introducción-al-Curso.pdf»* (Ciencia de Datos — UCN, 63 diapositivas, basado en CMU 15-388/688 de Z. Kolter, Harvard CS109 de P. Protopapas et al., Stanford CS221 de P. Liang et al., y UC Berkeley DS100).
* **Artículos y fuentes fundamentales citados en la unidad CD01:**
  * Flaxman, S., Mishra, S., Gandy, A., et al. (2020). *Estimating the effects of non-pharmaceutical interventions on COVID-19 in Europe*. Nature, 584, 257–261.
  * Hasell, J., Mathieu, E., Beltekian, D., et al. (2020). *A cross-country database of COVID-19 testing*. Scientific Data, 7, 345.
  * Dong, E., Du, H., & Gardner, L. (2020). *An interactive web-based dashboard to track COVID-19 in real time*. The Lancet Infectious Diseases, 20(5), 533–534.
  * Amatriain, X., & Basilico, J. (2012). *Netflix Recommendations: Beyond the 5 stars (Part 1)*. Netflix TechBlog.
  * Abelson, B., Varshney, K. R., & Sun, J. (2014). *Targeting direct cash transfers to the extremely poor*. Proceedings of the 20th ACM SIGKDD (pp. 1563–1572).
  * Turing, A. M. (1950). *Computing Machinery and Intelligence*. Mind, 59(236), 433–460.
