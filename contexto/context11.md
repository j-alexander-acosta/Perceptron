# Contexto 11: Integración de la Unidad VI — Descenso por Gradiente Estocástico y Lotes (41 Diapositivas)
**Fecha de registro:** 2026-11-09

## 1. Instrucciones
* Incorporar la nueva presentación académica de 41 diapositivas (**«Descenso por Gradiente Estocástico y Lotes»**, Ciencia de Datos — UCN, basado en material de CMU 11-785 de B. Raj et al. y Stanford CS221 de P. Liang et al.) como la **Unidad Temática VI** del proyecto.
* Desarrollar el módulo computacional interactivo y autoverificable en Python puro (`sgd_lotes.py`) que implemente y verifique numéricamente las 5 secciones del material:
  1. **El costo del lote y actualizaciones incrementales: SGD, épocas y barajado (Diaps. 5–14):**
     * Costo por iteración del descenso por gradiente de lote ($N$ gradientes por cada actualización de $\mathbf{w}$) frente a SGD ($N$ actualizaciones por época con el mismo costo de $TN$ evaluaciones de gradiente).
     * Efecto del orden de presentación (oscilación cíclica al presentar primero el grupo A y luego el grupo B vs. convergencia mediante **barajado / *shuffling* en cada época**).
     * **Verificación exacta de la traza manual de SGD de la Diapositiva 14:** sobre $\{(1,1),(2,3),(4,3)\}$ con $\phi(x)=[1,x]$, $\eta=0.1$, $\mathbf{w}^{(0)}=[0,0]$ y $\text{Loss}=(\mathbf{w}\cdot\phi(x)-y)^2$:
       - Paso 1 $(1,1)$: predicción $=0$, residuo $r=-1$, $\nabla_{\mathbf{w}}\text{Loss}=[-2,-2]$, $\mathbf{w}=[0.2, 0.2]$.
       - Paso 2 $(2,3)$: predicción $=0.6$, residuo $r=-2.4$, $\nabla_{\mathbf{w}}\text{Loss}=[-4.8,-9.6]$, $\mathbf{w}=[0.68, 1.16]$.
       - Paso 3 $(4,3)$: predicción $=5.32$, residuo $r=+2.32$, $\nabla_{\mathbf{w}}\text{Loss}=[4.64,18.56]$, $\mathbf{w}=[0.216,-0.696]\approx[0.22,-0.70]$ (comparado con el paso único de lote $\mathbf{w}=[0.47, 1.27]$ y el óptimo $\mathbf{w}^*=[1, 0.57]$).
  2. **Tasa de aprendizaje decreciente, condiciones de Robbins y Monro (1951) y velocidad de convergencia (Diaps. 15–20):**
     * El riesgo de perseguir al último ejemplo con paso fijo grande frente a tasas decrecientes $\eta_k = 1/\sqrt{k}$ y $\eta_k \propto 1/k$.
     * Verificación de las **condiciones suficientes de Robbins y Monro (1951)**: $\sum_k \eta_k = \infty$ y $\sum_k \eta_k^2 < \infty$.
     * Comparación de complejidad para quedar a distancia $\varepsilon$ del óptimo en pérdidas fuertemente convexas: Lote $O(\log(1/\varepsilon))$ iteraciones ($\times N$ gradientes) vs. SGD $O(1/\varepsilon)$ iteraciones ($\times 1$ gradiente).
     * Reproducción del experimento en Python de la **Diapositiva 19** (regresión en $\mathbb{R}^{10}$ donde SGD se acerca a `w_real` viendo solo $1/10$ de los datos antes de que el lote complete una sola actualización).
  3. **¿Por qué funciona? La varianza del estimador y los mini-lotes (Diaps. 21–29):**
     * Insesgadez de $\text{TrainLoss}(\mathbf{w})$, del paso de SGD y del mini-lote de tamaño $b$: $\mathbb{E}[\hat{L}_b(\mathbf{w})] = \mathbb{E}[\text{Loss}(x,y,\mathbf{w})]$.
     * Ley de reducción de varianza: $\text{Var}(\text{Lote}) = \frac{1}{N}\text{Var}(\text{Loss})$, $\text{Var}(\text{Mini-lote}) = \frac{1}{b}\text{Var}(\text{Loss})$, $\text{Var}(\text{SGD}) = \text{Var}(\text{Loss})$.
     * Continuo entre $b=1$ (SGD) y $b=N$ (lote), vectorización en GPU y desmontaje de la trampa conceptual de la Diapositiva 29 ($b=1$ **es** SGD).
  4. **Optimizadores con tendencia: Momentum, Nesterov, RMSprop y Adam (Diaps. 30–38):**
     * **Momentum** ($\Delta\mathbf{w}^{(k)} = \beta\Delta\mathbf{w}^{(k-1)} - \eta\nabla_{\mathbf{w}}\text{Loss}(\mathbf{w}^{(k-1)})$) y **Gradiente Acelerado de Nesterov** ($\Delta\mathbf{w}^{(k)} = \beta\Delta\mathbf{w}^{(k-1)} - \eta\nabla_{\mathbf{w}}\text{Loss}(\mathbf{w}^{(k-1)} + \beta\Delta\mathbf{w}^{(k-1)})$).
     * **Verificación exacta de la tabla de la Diapositiva 34 y del cálculo RMS de la Diapositiva 35:** pasos $\Delta x = [1, 1, 2, 1, 1.5]$ (recorrido total $6.5$, avance neto $6.5$, $\text{RMS}_x = \sqrt{1.85} \approx 1.36$) y $\Delta y = [+2.5, -3, +2.5, -2, +1.5]$ (recorrido total $11.5$, avance neto $1.5$, $\text{RMS}_y = \sqrt{5.55} \approx 2.36$).
     * **RMSprop** y **Adam** (primer momento $m^{(k)}$, segundo momento $v^{(k)}$ y correcciones de sesgo $\hat{m} = \frac{m^{(k)}}{1-\beta_1^k}$, $\hat{v} = \frac{v^{(k)}}{1-\beta_2^k}$).
     * Simulación comparativa de trayectorias (**SGD vs. Momentum vs. Nesterov vs. RMSprop vs. Adam**) en un **valle largo y angosto** y en un **punto de silla** (Diapositiva 38).
* Expandir la plataforma interactiva `web/index.html` para incorporar la **Unidad VI: Descenso por Gradiente Estocástico y Lotes** con 4 nuevos laboratorios interactivos (`panel-21` a `panel-24`, alcanzando **6 Unidades y 25 Laboratorios Interactivos**) y ampliar el banco de autoevaluación a 24 preguntas.
* Integrar el nuevo módulo en `perceptron_xor.py` (`--sgd` / `-s`, menú interactivo `[9]` y suite `--auto`), `README.md` e `INFORME.md`.

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Integrar teórica, computacional y visualmente la Unidad VI sobre Descenso por Gradiente Estocástico, Mini-lotes y Optimizadores con Tendencia (Momentum, Nesterov, RMSprop y Adam).
* **Fases/Hitos principales:**
  1. Creación del registro estructurado `contexto/context11.md`.
  2. Implementación del módulo `sgd_lotes.py` con verificaciones numéricas exactas de las Diapositivas 10, 12–14, 17–19, 21–29 y 30–39.
  3. Ampliación de la suite web interactiva `web/index.html` con la Unidad VI (`panel-21` a `panel-24`) y las preguntas 21 a 24 del cuestionario.
  4. Integración en `perceptron_xor.py`, actualización de `README.md` e `INFORME.md`, y ejecución de pruebas automatizadas (`python3 perceptron_xor.py --auto`).
* **Entregables:** `contexto/context11.md`, `sgd_lotes.py`, `web/index.html` actualizado (6 unidades, 25 laboratorios), `perceptron_xor.py`, `README.md` e `INFORME.md`.

## 3. Referencias
* **Path del plan original:** `contexto/context11.md`
* **Material de cátedra:** Presentación *«Descenso por Gradiente Estocástico y Lotes»* (Ciencia de Datos — UCN, 41 diapositivas, basado en CMU 11-785 de B. Raj et al. y Stanford CS221 de P. Liang et al.).
* **Artículos fundamentales:**
  * Robbins, H., & Monro, S. (1951). *A stochastic approximation method*. The Annals of Mathematical Statistics, 22(3), 400–407.
  * Polyak, B. T. (1964). *Some methods of speeding up the convergence of iteration methods*. USSR Computational Mathematics and Mathematical Physics, 4(5), 1–17.
  * Nesterov, Y. (1983). *A method for solving the convex programming problem with convergence rate $O(1/k^2)$*. Doklady AN SSSR, 269, 543–547.
  * Tieleman, T., & Hinton, G. (2012). *Lecture 6.5—rmsprop: Divide the gradient by a running average of its recent magnitude*. COURSERA: Neural Networks for Machine Learning.
  * Kingma, D. P., & Ba, J. (2015). *Adam: A method for stochastic optimization*. ICLR 2015.
