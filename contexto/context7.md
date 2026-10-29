# Contexto 7: Integración de "Aprendizaje de Redes Neuronales" y Reorganización Modular por Unidades
**Fecha de registro:** 2026-10-29

## 1. Instrucciones
* Agregar la nueva sección correspondiente al PDF **"Aprendizaje de Redes Neuronales"** (*Ciencia de Datos — UCN, basado en material de CMU 11-785 de B. Raj et al. y Stanford CS221 de P. Liang et al.*, 56 diapositivas).
* Distribuir y reorganizar todos los temas del proyecto para que se vean mucho más ordenados, limpios y estructurados pedagógicamente (divididos en **Unidades Temáticas** y **Sub-secciones / Laboratorios** tanto en la interfaz web como en los scripts de Python y documentación).
* Cubrir los 5 ejes de la nueva presentación (*Aprendizaje de Redes Neuronales*):
  1. **¿Qué significa aprender una red? (Diaps. 5–14):** La red como función paramétrica $Y = f(X; W)$, ajuste manual vs. minimización del error real $\text{TotalErr}(W) = \int \text{error}(f(X;W), g(X))\,dX$, minimización del riesgo empírico $\text{TrainLoss}(W) = \frac{1}{N}\sum_i \text{error}(f(X_i;W), d_i)$, generalización y los dos malentendidos sobre aproximación universal (Diap. 14).
  2. **La regla de aprendizaje del perceptrón (Diaps. 15–22):** Truco para absorber el umbral ($x_{N+1}=1, w_{N+1}=-T$), algoritmo en línea de Rosenblatt ($\mathbf{w} \leftarrow \mathbf{w} + y\mathbf{x}$ cuando $y(\mathbf{w}\cdot\mathbf{x}) \le 0$), teorema de convergencia en a lo más $(R/\gamma)^2$ actualizaciones, no convergencia en datos no separables y reproducción exacta del ejemplo en Python de la Diapositiva 22 (`w = [3.25, 1.9]` en 3 actualizaciones).
  3. **¿Por qué no escala a redes multicapa? (Diaps. 23–29):** El problema del doble pentágono (no separabilidad lineal de una neurona individual incluso con el resto de la red fija, problema combinatorio de etiquetas intermedias), ADALINE (Widrow & Hoff, 1960) con la regla Delta / LMS $w_i \leftarrow w_i + \eta(d - z)x_i$ y MADALINE.
  4. **La salida: activaciones diferenciables (Diaps. 30–39):** Por qué contar errores da derivada nula en casi todas partes (función escalonada plana a trozos), comparación entre señal de umbral vs. suma de distancias continuas con sigmoide (3 afirmaciones de la Diap. 34), regresión logística por ventana deslizante y diferenciabilidad extremo a extremo mediante la regla de la cadena.
  5. **El planteamiento formal: red, representaciones y pérdidas (Diaps. 40–56):** Neurona estándar diferenciable $y = \sigma(z)$, tabla resumen de representaciones de salida y funciones de divergencia (Diap. 54):
     - **Regresión:** Salida lineal + Divergencia **L2** $\frac{1}{2}\|Y - d\|^2$ ($\nabla_Y = Y - d$).
     - **Clasificación binaria:** Salida sigmoide + **Entropía cruzada binaria** $-d\log Y - (1-d)\log(1-Y)$.
     - **Clasificación multiclase:** Salida **Softmax** $y_i = e^{z_i}/\sum_k e^{z_k}$ + **Entropía cruzada / Divergencia KL** $-\sum_i d_i \log y_i = -\log y_c$.

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Incorporar el módulo completo de *Aprendizaje de Redes Neuronales* y reestructurar toda la suite educativa en **2 Unidades Temáticas** claramente organizadas (Unidad I: *Representación y Aproximación Universal* y Unidad II: *Aprendizaje de Redes Neuronales*).
* **Fases/Hitos principales:**
  1. Registro del plan en `contexto/context7.md`.
  2. Desarrollo del módulo en Python `aprendizaje_redes.py` con demostraciones ejecutables de los 5 temas (Rosenblatt en Diap. 22, cota $(R/\gamma)^2$, ADALINE LMS, paisaje de error escalón vs. sigmoide, Softmax y divergencias L2 / Entropía Cruzada / KL con gradientes).
  3. Rediseño organizacional de `web/index.html` con selector superior de **Unidades Académicas**, índice estructurado de **9 Laboratorios Interactivos** (5 en Unidad I + 4 en Unidad II) y autoevaluaciones interactivas validadas.
  4. Integración en `perceptron_xor.py`, `README.md` e `INFORME.md`, y verificación automática completa.
* **Entregables:**
  - `contexto/context7.md`
  - `aprendizaje_redes.py`
  - `web/index.html` (reorganizado en 2 Unidades y 9 Laboratorios)
  - `perceptron_xor.py`, `README.md` e `INFORME.md` actualizados

## 3. Referencias
* **Path del plan original:** `contexto/context7.md`
