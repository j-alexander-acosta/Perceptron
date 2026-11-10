# Contexto 17 — Integración de la Unidad CD05: Generalización y Validación (63 Diapositivas)
**Fecha de registro:** 2026-11-10

## 1. Instrucciones y Directrices
* **Archivo fuente cargado:** `/Users/alexanderacosta/.gemini/antigravity/brain/c8f17cb7-96ad-4898-8b55-e7d88fe53835/.user_uploaded/media_1794308515273_66855210.pdf` (`CD05-Generalización-y-Validación.pdf`).
* **Ubicación en el plan de estudios:** Quinta unidad en orden estricto de cátedra:
  $$\text{CD01} \to \text{CD02} \to \text{CD03} \to \text{CD04} \to \mathbf{\text{CD05 (Generalización y Validación)}} \to \text{CD13} \to \dots \to \text{CD18}$$
* **Compatibilidad técnica:** Mantener implementación en Python puro (librería estándar, sin dependencias externas obligatorias) para la CLI, suite web interactiva en `web/index.html` con renderizado Canvas nativo, e integración en el verificador central `perceptron_xor.py`.
* **Totales del proyecto:** 11 Unidades Académicas • 682 Diapositivas • 45 Laboratorios Interactivos.

---

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Implementar, simular, verificar y documentar exhaustivamente todas las fórmulas, teoremas, cálculos manuales y experimentos controlados de las 63 diapositivas de la Unidad CD05 (Generalización y Validación).
* **Fases e hitos principales:**
  1. **Análisis de contenidos de CD05:** 4 Módulos temáticos extraídos de las 63 diapositivas (276 páginas de transición).
  2. **Módulo computacional Python (`generalizacion_validacion.py`):** Implementación de las 4 demostraciones y verificación automática `--auto`.
  3. **Integración en orquestador (`perceptron_xor.py`):** Inclusión en `UNIDADES_CATEDRA`, `menu_fase1()`, banderas CLI (`--cd05`, `--gen`, `-g5`) y navegación interactiva.
  4. **Suite Web Interactiva (`web/index.html`):** Incorporación de botón en `.unit-switcher`, pestañas `#unit05Tabs`, paneles `#panel-41` a `#panel-44`, simuladores interactivos (Maldición de la dimensionalidad, Cauchy-Schwarz, Paso L2 de Diap. 26, Experimento $y=x^3-3x$ con 5-Fold vs Simple y Diana Bullseye) y Quizzes 41 a 44.
  5. **Documentación formal:** Sección 0.8 en `INFORME.md` con demostraciones matemáticas completas, actualización de `README.md` y registro en `contexto/context17.md`.

---

## 3. Síntesis Teórica y Numérica Implementada

### 3.1. Módulo 1: El Objetivo Real: Generalizar, Sobreajuste y Correlación Espuria (Diaps. 1–16)
* **Algoritmo Memorize:** $h(x) = y_i$ si $x = x_i \in \mathcal{D}_{\text{train}}$, de lo contrario $\text{default}$. Obtiene $\text{TrainLoss} = 0.0$ pero colapsa a $\text{TestLoss} = 41.0$ fuera de muestra.
* **Correlación espuria (Oakden-Rayner et al., 2019):** Detección de neumotórax en radiografías a través del falso atajo del tubo torácico de drenaje ya insertado en el hospital.
* **Maldición de la dimensionalidad:** En $\{0, 1\}^{100}$ hay $2^{100} \approx 1.27 \times 10^{30}$ estados. $10^{15}$ muestras (1 petabyte) cubren apenas una fracción de $7.89 \times 10^{-16}$. Es matemáticamente imposible generalizar sin sesgo inductivo sobre $\mathcal{F}$.
* **Descomposición del error de Liang (Stanford CS221):**
  $$\text{Err}(\hat{f}) - \text{Err}(f^*) = \underbrace{(\text{Err}(\hat{f}) - \text{Err}(g))}_{\text{Error de Estimación (Varianza)}} + \underbrace{(\text{Err}(g) - \text{Err}(f^*))}_{\text{Error de Aproximación (Sesgo)}}$$
* **Auditoría de fraude:** Equipo A (Train 0%, Test 25% $\implies$ sobreajuste / alta estimación); Equipo B (Train 23%, Test 24% $\implies$ subajuste / alta aproximación); Equipo C (50 pruebas en Test $\implies$ test contaminado / data leakage).

### 3.2. Módulo 2: Controlar Hipótesis: Dimensión, Cauchy-Schwarz, Regularización L2/L1 y Early Stopping (Diaps. 17–27)
* **Greedy Forward Selection:** Evalúa $\sum_{k=1}^J k = J(J+1)/2$ modelos frente a $2^J$ de la búsqueda exhaustiva. Para $J=20$: evalúa solo $210$ modelos frente a $1\,048\,576$ (aceleración de $4\,993\times$).
* **Cota de Suavidad por Cauchy-Schwarz:**
  $$|s(x) - s(x')| = |w \cdot (\phi(x) - \phi(x'))| \le \|w\|_2 \|\phi(x) - \phi(x')\|_2$$
  Acotar $\|w\|$ restringe la constante de Lipschitz del predictor y previene oscilaciones bruscas.
* **Paso a mano exacto de la Diapositiva 26 (Págs. 103–107):**
  Con $w = [2.0, -2.0]$, $\nabla\text{TrainLoss} = [1.0, 1.0]$, $\lambda = 0.5$ y $\eta = 0.1$:
  $$\lambda w = 0.5[2, -2] = [1.0, -1.0] \implies \nabla_{\text{total}} = [1, 1] + [1, -1] = [2.0, 0.0]$$
  $$w_{\text{nuevo}} = [2.0, -2.0] - 0.1[2.0, 0.0] = \mathbf{[1.8, -2.0]}$$
* **L1 vs L2 vs Early Stopping:** L1 ($\|w\|_1$) fuerza dispersión exacta ($w_j=0$); Early Stopping impone cota de radio $\|w^{(T)}\| \le \eta G T$ partiendo de $w^{(0)}=0$.

### 3.3. Módulo 3: Hiperparámetros, Experimento Controlado y Validación Cruzada K-Fold (Diaps. 28–52)
* **Experimento controlado:** Ground truth $y = x^3 - 3x + \varepsilon$, con $\varepsilon \sim \mathcal{N}(0, 4.0)$ ($n=30$ puntos). Grado verdadero $d=3$.
* **Tabla comparativa oficial (Diaps. 44–45):**
  * Validación simple (8 pts): $d=1: 26.7$, $d=2: 23.4$, $d=3: 3.1$, $d=4: 3.2$, $d=5: 2.2$, $\mathbf{d=6: 1.9}$ (falso mínimo), $d=7: 2.0$, $d=8: 2.1$.
  * 5-Fold CV ($K=5$): $d=1: 23.5$, $d=2: 29.8$, $\mathbf{d=3: 3.8}$ (mínimo real óptimo), $d=4: 5.6$, $d=5: 4.5$, $d=6: 9.5$, $d=7: 5.7$, $d=8: 80.0$.
* **Desglose de rondas (Diap. 46 / Pág. 190):**
  * $d=3$: $[2.4, 2.6, 7.6, 4.7, 1.5] \implies \text{media } \mathbf{3.8}$ (estable).
  * $d=6$: $[1.9, 1.7, 8.2, \mathbf{32.3}, 3.3] \implies \text{media } \mathbf{9.5}$ (la ronda 4 explota a $32.3$).
* **Auditoría final en 200,000 datos (Diap. 47):** $d=3$ (CV) obtiene MSE real $\mathbf{4.2}$ (+5% del óptimo 4.0) vs $\mathbf{5.1}$ (+27.5%) de $d=6$ ($d=3$ rinde $21.4\%$ mejor en producción).
* **Estabilidad:** Modelo A ($\mu=3.02, \sigma=0.23$) vs Modelo B ($\mu=3.10, \sigma=1.25$).

### 3.4. Módulo 4: Descomposición Sesgo-Varianza, Diana y Curvas de Aprendizaje (Diaps. 53–63)
* **Descomposición del MSE esperado:**
  $$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2$$
* **La Diana (Bullseye):** Alto/Bajo Sesgo vs Alto/Bajo Varianza.
* **Matriz de control de 4 perillas (Diap. 58):**
  * Complejidad $d \uparrow \implies \text{Sesgo} \downarrow, \text{Varianza} \uparrow$.
  * Regularización $\lambda \uparrow \implies \text{Sesgo} \uparrow, \text{Varianza} \downarrow$.
  * Early Stopping $T \downarrow \implies \text{Sesgo} \uparrow, \text{Varianza} \downarrow$.
  * **Tamaño muestral $n \uparrow \implies \text{Sesgo} =, \mathbf{\text{Varianza} \downarrow}$ (¡El único botón con costo estadístico cero!).**
* **Diagnóstico:** Alta Varianza (Train << Val, brecha abierta) vs Alto Sesgo (Train $\approx$ Val estancados en alto error; más datos no ayuda).

---

## 4. Estado de Archivos y Verificación
* `generalizacion_validacion.py`: Módulo CLI en Python puro con 100% de verificaciones pasando (`python3 generalizacion_validacion.py --auto`).
* `perceptron_xor.py`: Unidad CD05 cableada con menú jerárquico, barra de navegación interactiva y suite automática (`python3 perceptron_xor.py --auto`).
* `web/index.html`: Selector, tabs `#unit05Tabs`, paneles `#panel-41` a `#panel-44`, simuladores interactivos Canvas y Quizzes 41 a 44 validados sintácticamente con Node.js.
* `README.md` e `INFORME.md`: Actualizados con la Sección 0.8, fórmulas LaTeX y referencias completas.
