# Contexto 6: Implementación Educativa de Aproximación Universal con Redes Neuronales
**Fecha de registro:** 2026-10-26

## 1. Instrucciones
* Implementar la temática completa del documento PDF adjunto (*"Aproximación Universal con Redes Neuronales — Ciencia de Datos UCN, basado en CMU 11-785 de B. Raj et al."*) de manera altamente educativa, interactiva y demostrativa.
* Cubrir los 4 pilares fundamentales de la clase:
  1. **El MLP como función booleana universal:** Compuertas de umbral universales ($N$ entradas), construcción automática de cualquier tabla de verdad mediante Forma Normal Disyuntiva (DNF), reducción con Mapas de Karnaugh ($4\times 4$), el peor caso del tablero de ajedrez (paridad / XOR de $N$ variables) y la explosión exponencial de una sola capa oculta ($2^{N-1}$) frente al crecimiento lineal de redes profundas en cascada o árbol ($3(N-1)$ neuronas en $2\log_2 N$ capas). Incluye también el diseño compacto de XOR con 2 neuronas (Friedland).
  2. **El MLP como clasificador universal:** Composición de fronteras de decisión en $\mathbb{R}^2$ usando semiplanos (polígonos convexos como el cuadrado, pentágono, hexágono y límite circular con $N$ lados), el fenómeno de la "torta de escalones" y el cilindro con piso $N/2$, ajuste de sesgo ($-N/2$) para combinar múltiples regiones, unión de polígonos en 3 capas y el problema del tablero de rombos ($16$ y $64$ líneas).
  3. **El MLP como aproximador universal de funciones continuas:** Construcción de un pulso cuadrado con 2 unidades de umbral en la capa oculta ($T_1, T_2$) y suma lineal en la salida, aproximación de funciones reales $f(x)$ mediante suma de pulsos escalados ($h_i$) y distinción rigurosa entre *representar*, *aproximar* ($\varepsilon > 0$) y *el límite*.
  4. **Suficiencia de la arquitectura (Profundidad, Ancho y Activaciones):** Cuello de botella por ancho insuficiente (franjas vs. celdas, pérdida irreversible de información con activación de umbral), el rol de las activaciones graduadas (sigmoide, tanh, ReLU) que transmiten cuánta distancia hay respecto a la frontera, y el triángulo de compensaciones (Ancho $\leftrightarrow$ Profundidad $\leftrightarrow$ Activaciones).
* Integrar tanto una **plataforma web educativa interactiva** (`web/index.html` y `web/aproximacion_universal.html`) como un **módulo interactivo y verificador en Python** (`aproximacion_universal.py`) conectado al flujo del proyecto (`perceptron_xor.py`, `README.md`, `INFORME.md`).

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Transformar la presentación teórica de 61 diapositivas sobre *Aproximación Universal con Redes Neuronales* en un laboratorio educativo interactivo (Python + Web + Documentación Académica).
* **Fases/Hitos principales:**
  1. Registro del contexto secuencial `contexto/context6.md`.
  2. Creación del módulo educativo en Python `aproximacion_universal.py` con demostraciones ejecutables, constructor automático DNF, comparador exponencial vs. profundo (XOR $N$-ario), clasificador geométrico de polígonos/círculos y aproximador de funciones continuas por pulsos, además de generación de gráficas educativas.
  3. Expansión de la aplicación web interactiva (`web/index.html`) en una **Suite Educativa Interactiva con 5 Laboratorios Visuales** que cubren todos los temas del PDF en tiempo real.
  4. Actualización de `perceptron_xor.py`, `README.md` e `INFORME.md` para integrar los nuevos laboratorios y fundamentos teóricos.
* **Entregables:**
  - `contexto/context6.md`
  - `aproximacion_universal.py`
  - `web/index.html` (Suite Educativa Interactiva completa)
  - `aproximacion_universal_resumen.png` (Gráfico resumen multipanel)
  - `README.md` e `INFORME.md` actualizados

## 3. Referencias
* **Path del plan original:** `contexto/context6.md`
