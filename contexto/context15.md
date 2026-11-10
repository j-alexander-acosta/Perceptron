# Contexto 15 — Integración de CD04: Introducción a Machine Learning (56 Diapositivas)

## 1. Instrucción del Usuario y Orden Secuencial
- **Solicitud:** «Sigue con esta sección» (integración del documento adjunto **CD04 — Introducción a Machine Learning**, Ciencia de Datos — UCN, basado en Stanford CS221 de P. Liang et al., Harvard CS109 de P. Protopapas et al. y CMU 11-785 de B. Raj et al., 56 diapositivas / 232 páginas PDF con transiciones).
- **Posición en el orden de cátedra:** Conforme a la regla establecida por el usuario y a la conexión explícita entre unidades (Diapositiva 102 de `CD03`: *«Siguiente paso: machine learning: con los datos ya explorados y visualizados, empezamos a modelar»*), esta unidad corresponde a **CD04** y se ubica en **cuarta posición** dentro de la suite:
  $$\text{CD01} \to \text{CD02} \to \text{CD03} \to \mathbf{\text{CD04}} \to \text{CD13} \to \text{CD14} \to \text{CD15} \to \text{CD16} \to \text{CD17} \to \text{CD18}$$

---

## 2. Síntesis Completa de Contenidos de CD04 (Diapositivas 1–56)

1. **Panorama de Machine Learning y Tareas de Predicción (Diapositivas 1–10):**
   - **¿Dónde estamos en el mapa de la IA? (Diap. 3):** Modelos ordenados de bajo a alto nivel: **Reflejos** (aprendizaje automático) $\to$ **Estados** (búsqueda, MDP, juegos) $\to$ **Variables** (CSP, redes de Markov y bayesianas) $\to$ **Lógica** (proposicional, primer orden).
   - **¿Qué es Machine Learning? (Diap. 5):** Mueve la complejidad **del código a los datos** (en vez de programar la solución, se define una familia de modelos y se aprende desde ejemplos) y requiere el salto de fe de **generalizar** a entradas nunca vistas.
   - **Modelos basados en reflejos (Diap. 6):** $x \to \boxed{f\text{ predictor}} \to y$. Ejecutan una secuencia **fija** de cómputos sobre la entrada (sin vuelta atrás ni explorar alternativas); la inferencia es directa y rápida, y el trabajo duro está en aprender $f$.
   - **Los 3 tipos de tareas de predicción según la forma de la salida (Diaps. 7–10):**
     1. **Clasificación binaria (Diap. 7):** $x \to f \to y \in \{+1, -1\}$ (ej. detección de fraude en tarjeta, moderación de foros tóxicos, decaimiento del bosón de Higgs). **Extensión multiclase:** $y \in \{1, \dots, K\}$ (ej. dígitos manuscritos con $K = 10$).
     2. **Regresión (Diap. 8):** $x \to f \to y \in \mathbb{R}$ salida continua (ej. mapeo de pobreza desde imagen satelital, precio de vivienda, tiempo de llegada en logística).
     3. **Predicción estructurada (Diap. 9):** $x \to f \to y$ es un objeto complejo (traducción automática español $\to$ inglés, descripción de imágenes, segmentación de imágenes en conducción autónoma). **El secreto:** suele descomponerse en una **secuencia de clasificaciones multiclase** (predecir una palabra a la vez).
     4. **Cuatro casos y su tipo de tarea (Diap. 10):**
        - *Predecir si un correo es spam* $\to$ **clasificación binaria**.
        - *Estimar el precio de una vivienda en Antofagasta* $\to$ **regresión**.
        - *Transcribir un audio a texto* $\to$ **estructurada (secuencia de palabras)**.
        - *Reconocer qué dígito (0–9) hay en una imagen* $\to$ **multiclase ($K = 10$)**.

2. **Regresión Lineal y las Tres Decisiones de Diseño (Diapositivas 11–23):**
   - **1801: El descubrimiento de Ceres y el triunfo de Gauss (Diaps. 11–12):** Piazzi registra **19 observaciones** de Ceres (`01 ene: Asc. recta 50.91, Decl. 15.24`; `02 ene: 50.84, 15.30`; ...; `11 feb: 53.51, 18.43`) antes de perderse en el resplandor del Sol. En septiembre de 1801 Gauss realiza más de 100 horas de cálculo a mano con **regresión lineal por mínimos cuadrados** y el 7 de diciembre de 1801 Ceres reaparece a **medio grado** de lo predicho.
   - **El marco de las 3 decisiones de diseño (Diap. 13):** Con $\mathcal{D}_{\text{train}} = \{(1, 1), (2, 3), (4, 3)\}$ y predicción $f(3) = 2.71$:
     1. *¿Qué predictores son posibles?* $\to$ **Clase de hipótesis** $\mathcal{F} = \{f_{\mathbf{w}}(x) = \mathbf{w}\cdot\phi(x) : \mathbf{w}\in\mathbb{R}^2\}$ con $\mathbf{w}=[w_1, w_2]$ y extractor $\phi(x)=[1, x]$ (ej. con $\mathbf{w}=[1, 0.57]$, $f_{\mathbf{w}}(3) = [1, 0.57]\cdot[1, 3] = 1 + 1.71 = \mathbf{2.71}$, Diaps. 14–15).
     2. *¿Qué tan bueno es un predictor?* $\to$ **Función de pérdida** cuadrática sobre el **residuo** $(f_{\mathbf{w}}(x) - y)$: $\text{Loss}(x, y, \mathbf{w}) = (f_{\mathbf{w}}(x) - y)^2$. Con $\mathbf{w} = [1, 0.57]$ (Diaps. 16–17):
        - $\text{Loss}(1, 1) = (1.57 - 1)^2 = \mathbf{0.3249 \approx 0.32}$
        - $\text{Loss}(2, 3) = (2.14 - 3)^2 = \mathbf{0.7396 \approx 0.74}$
        - $\text{Loss}(4, 3) = (3.28 - 3)^2 = \mathbf{0.0784 \approx 0.08}$
        - **Pérdida de entrenamiento (riesgo empírico):** $\text{TrainLoss}([1, 0.57]) = \frac{0.32 + 0.74 + 0.08}{3} = \mathbf{0.38}$.
     3. *¿Cómo calculamos el mejor?* $\to$ **Algoritmo de optimización** (descenso por gradiente, Diaps. 18–22):
        $$\nabla_{\mathbf{w}}\text{TrainLoss}(\mathbf{w}) = \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{(x,y)\in\mathcal{D}_{\text{train}}} 2(\underbrace{\mathbf{w}\cdot\phi(x) - y}_{\text{residuo}})\phi(x), \qquad \mathbf{w} \leftarrow \mathbf{w} - \eta\nabla_{\mathbf{w}}\text{TrainLoss}(\mathbf{w})$$
        - **Tabla numérica exacta con $\eta = 0.1$ desde $\mathbf{w}^{(0)} = [0, 0]$ (Diap. 21):**
          - $t=0$: $\mathbf{w} = [0, 0]$
          - $t=1$: $\nabla\text{TrainLoss} = \mathbf{[-4.67, -12.67]} \implies \mathbf{w} = \mathbf{[0.47, 1.27]}$
          - $t=2$: $\nabla\text{TrainLoss} = \mathbf{[2.18, 7.24]} \implies \mathbf{w} = \mathbf{[0.25, 0.54]}$
          - $t=200$: $\nabla\text{TrainLoss} = \mathbf{[0, 0]} \implies \mathbf{w}^* = \mathbf{[1, 0.57]}$ ($\text{TrainLoss} = 0.38$).

3. **Clasificación Lineal: Puntaje, Margen, Pérdida 0-1 y Pérdida Hinge (Diapositivas 24–37):**
   - **Datos de entrenamiento en $\mathbb{R}^2$ (Diaps. 24–25):** $\mathcal{D}_{\text{train}} = \{([0, 2], +1), ([-2, 0], +1), ([1, -1], -1)\}$ con $\phi(x) = [x_1, x_2]$ y clasificador $f_{\mathbf{w}}(x) = \text{sign}(\mathbf{w}\cdot\phi(x))$.
     - Con $\mathbf{w} = [-0.6, 0.6]$ (Diap. 25): $f([0, 2]) = \text{sign}(1.2) = +1$, $f([-2, 0]) = \text{sign}(1.2) = +1$, $f([1, -1]) = \text{sign}(-1.2) = -1$. La frontera de decisión $\mathbf{w}\cdot\phi(x) = 0$ es perpendicular a $\mathbf{w}$, y **escalar $\mathbf{w}$ no cambia la frontera** (Diap. 26).
   - **Pérdida 0-1 con $\mathbf{w} = [0.5, 1]$ (Diap. 27):** $\text{Loss}_{0\text{-}1}(x, y, \mathbf{w}) = \mathbf{1}[f_{\mathbf{w}}(x) \neq y] = \mathbf{1}[(\mathbf{w}\cdot\phi(x))y \le 0]$:
     - $([0, 2], +1) \implies \text{sign}(2) = +1 \implies \text{pérdida} = 0$
     - $([-2, 0], +1) \implies \text{sign}(-1) = -1 \implies \text{pérdida} = 1$
     - $([1, -1], -1) \implies \text{sign}(-0.5) = -1 \implies \text{pérdida} = 0$
     - $\text{TrainLoss}_{0\text{-}1}([0.5, 1]) = \frac{0 + 1 + 0}{3} = \mathbf{0.33}$ (tasa de error).
   - **Puntaje y Margen (Diaps. 28–30):**
     - **Puntaje:** $z = \mathbf{w}\cdot\phi(x)$ (qué tan seguros estamos de predecir $+1$).
     - **Margen:** $m = (\mathbf{w}\cdot\phi(x))y$ (qué tan acertada es la predicción: $m > 0$ acierto, $m \le 0$ error, $m$ grande = acierto con holgura).
     - **Por qué falla el descenso por gradiente con $\text{Loss}_{0\text{-}1}$ (Diap. 30):** El problema real no es la discontinuidad en $m=0$, sino que **la pérdida es plana ($\nabla_{\mathbf{w}}\text{Loss}_{0\text{-}1} = \mathbf{0}$) en todo el resto**, dejando al descenso por gradiente atascado.
   - **La solución: Pérdida Hinge (bisagra, Diaps. 31–36):**
     $$\text{Loss}_{\text{hinge}}(x, y, \mathbf{w}) = \max\{1 - (\mathbf{w}\cdot\phi(x))y,\, 0\}, \qquad \nabla_{\mathbf{w}}\text{Loss}_{\text{hinge}}(x, y, \mathbf{w}) = \begin{cases} -\phi(x)y & \text{si margen } < 1 \\ \mathbf{0} & \text{si margen } > 1 \end{cases}$$
     - El $1$ exige un **margen de seguridad** (acertar con holgura $m \ge 1$) y es cota superior convexa de la pérdida $0\text{-}1$.
     - **Tabla exacta en nuestros 3 datos con $\mathbf{w} = [0.5, 1]$ (Diap. 35):**
       1. $([0, 2], +1)$: margen $= 2 \implies \text{Loss}_{\text{hinge}} = \max\{1-2, 0\} = \mathbf{0}$, $\nabla = \mathbf{[0, 0]}$.
       2. $([-2, 0], +1)$: margen $= -1 \implies \text{Loss}_{\text{hinge}} = \max\{1-(-1), 0\} = \mathbf{2}$, $\nabla = \mathbf{[2, 0]}$.
       3. $([1, -1], -1)$: margen $= +0.5 > 0$ (acierta, pero con margen $< 1$) $\implies \text{Loss}_{\text{hinge}} = \max\{1-0.5, 0\} = \mathbf{0.5}$, $\nabla = \mathbf{[1, -1]}$.
       - **Promedio:** $\text{TrainLoss}_{\text{hinge}} = \frac{0 + 2 + 0.5}{3} = \mathbf{0.83}$, $\nabla\text{TrainLoss}_{\text{hinge}} = \mathbf{[1, -0.33]}$.
     - **Ejemplo a mano de la Diapositiva 36:** Con $\mathbf{w} = [2, -1]$, $x = [1, 1], y = -1$: puntaje $= 2(1)+(-1)(1) = \mathbf{1} \implies \text{predice } +1$; margen $= 1\cdot(-1) = \mathbf{-1}$ (errada); $\text{Loss}_{\text{hinge}} = \max\{1-(-1), 0\} = \mathbf{2}$; gradiente $= -\phi(x)y = \mathbf{[1, 1]}$ $\implies \mathbf{w} \leftarrow \mathbf{w} - \eta[1, 1]$ baja el puntaje de $[1, 1]$.

4. **Regresión Logística, Máxima Verosimilitud, Crecimiento de $\|\mathbf{w}\|$, Umbral de Decisión, Softmax y Riesgo Empírico (Diapositivas 38–56):**
   - **¿Por qué no regresión lineal sobre $\{0, 1\}$? (Diap. 39):** Al ajustar la recta $y = 0.27x - 0.45$, en $x = 6.4$ «predice» probabilidad $\mathbf{1.29} > 1$ y en $x = 0.4$ predice $\mathbf{-0.35} < 0$.
   - **La función sigmoide y el modelo logístico (Diaps. 40–42):**
     $$\sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1), \quad \sigma(0) = \frac{1}{2}, \quad \sigma(-z) = 1 - \sigma(z), \quad p_{\mathbf{w}}(x) = P(y=+1 \mid x) = \sigma(\mathbf{w}\cdot\phi(x))$$
     - **La magnitud de $\mathbf{w}$ ahora importa (Diap. 42):** Con $\text{sign}$, escalar $\mathbf{w}$ no cambiaba nada; con la sigmoide, $\sigma(3z)$ es más empinada que $\sigma(z)$ (la frontera $p_{\mathbf{w}}(x) = 1/2 \iff \mathbf{w}\cdot\phi(x) = 0$ no se mueve, pero norma grande = modelo más confiado).
   - **Máxima verosimilitud = Pérdida logística (Diaps. 43–45):**
     $$P(y \mid x) = \sigma(\underbrace{(\mathbf{w}\cdot\phi(x))y}_{\text{margen } m}) \implies -\log P(y \mid x) = -\log\sigma(m) = \log\left(1 + e^{-m}\right) = \text{Loss}_{\text{logistic}}(x, y, \mathbf{w})$$
     - **Gradiente:** $\nabla_{\mathbf{w}}\text{Loss}_{\text{logistic}} = -\underbrace{\sigma(-m)}_{P(\text{etiqueta equivocada})}\phi(x)y$ (graduado por cuán equivocado se cree el modelo).
   - **Tabla exacta de la pérdida logística en nuestros 3 datos con $\mathbf{w} = [0.5, 1]$ (Diap. 46):**
     1. $([0, 2], +1)$: $z = 2$, $P(y\mid x) = \sigma(2) = \mathbf{0.881}$, pérdida $= \mathbf{0.127}$, gradiente $= \mathbf{[0, -0.238]}$.
     2. $([-2, 0], +1)$: $z = -1$, $P(y\mid x) = \sigma(-1) = \mathbf{0.269}$, pérdida $= \mathbf{1.313}$, gradiente $= \mathbf{[1.462, 0]}$.
     3. $([1, -1], -1)$: $z = -0.5$, $P(y\mid x) = \sigma(0.5) = \mathbf{0.622}$, pérdida $= \mathbf{0.474}$, gradiente $= \mathbf{[0.378, -0.378]}$.
     - **Promedio:** $\text{TrainLoss}_{\text{logistic}} = \mathbf{0.638}$, $\nabla\text{TrainLoss}_{\text{logistic}} = \mathbf{[0.613, -0.205]}$.
   - **Simulación en Python y «La logística nunca queda satisfecha» en datos separables (Diaps. 47–48):**
     - Con $\eta = 0.5$ desde $\mathbf{w} = [0, 0]$ sobre los 3 puntos:
       - Tras **100 iteraciones**: $\mathbf{w} = \mathbf{[-2.29, 2.29]}$, $\sigma(X\mathbf{w}) = [0.99, 0.99, 0.01]$, pérdida $= \mathbf{0.010}$.
       - Tras **1000 iteraciones**: $\mathbf{w} = \mathbf{[-3.45, 3.45]}$, pérdida $= \mathbf{0.001}$ y bajando (la norma crece sin freno $\to$ motivación directa para la **regularización**).
   - **El umbral de decisión como perilla ($\tau \in (0, 1)$, Diap. 49), Softmax multiclase (Diap. 50), el ejemplo $x=[1,1], y=-1, \mathbf{w}=[2,-1]$ ($\sigma(1)=0.731, P(y=-1\mid x)=0.269, \text{Loss}_{\text{logistic}} = -\ln(0.269) = 1.313$ vs. Hinge $= 2$, Diap. 51) y el principio del Riesgo Empírico (Diaps. 52–56).**

---

## 3. Plan de Acción
1. Crear `introduccion_ml.py` con verificaciones automáticas (`--auto`) y menú interactivo para las 4 secciones de **CD04**.
2. Actualizar `web/index.html` situando **CD04 • Introducción a Machine Learning (56 Diaps.)** en **cuarto lugar** en `.unit-switcher` (justo después de `CD03` y antes de `CD13`), agregando los 4 laboratorios interactivos `#panel-37` a `#panel-40` y las 4 preguntas de autoevaluación `quiz-item-37` a `quiz-item-40` (**10 Unidades Académicas • 619 Diapositivas → 41 Laboratorios**).
3. Integrar `introduccion_ml.py` en cuarta posición dentro de `perceptron_xor.py`, `README.md` e `INFORME.md`, y ejecutar `python3 perceptron_xor.py --auto`.
