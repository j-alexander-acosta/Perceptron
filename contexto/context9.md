# Contexto de Trabajo 9

**Fecha y Hora:** 2026-11-02

## Instrucciones del Usuario
Sube la siguiente unidad adjuntando el documento PDF `CD16-Algoritmo-de-Backpropagation.pdf` ("Algoritmo de Backpropagation", Ciencia de Datos — UCN, basado en el material de CMU 11-785 de B. Raj et al. y Stanford CS221 de P. Liang et al., 60 diapositivas / 253 páginas), para integrarlo como **Unidad IV: Algoritmo de Backpropagation** tanto en los módulos Python interactivos/automatizados como en la plataforma educativa web interactiva (`web/index.html`) y la documentación técnica.

## Plan de Trabajo
1. **Registro de Contexto (`almacenamiento`):**
   - Documentar la solicitud y el plan de trabajo estructurado en `contexto/context9.md`.

2. **Análisis Integral del Material (`CD16-Algoritmo-de-Backpropagation.pdf` — 60 Diapositivas):**
   - **Sección 1 (Diaps. 5–13): El Problema y las Dos Reglas de la Cadena.**
     - Descomposición de la derivada del riesgo empírico: $\frac{\partial \text{TrainLoss}}{\partial w_{ij}^{(k)}} = \frac{1}{T}\sum_{t=1}^T \frac{\partial \text{div}(Y_t, d_t)}{\partial w_{ij}^{(k)}}$.
     - La derivada como factor de amplificación local sobre aristas de un diagrama de influencia ($\Delta y \approx \frac{dy}{dx}\Delta x$).
     - **Cadena simple (composición, Diap. 10 y 12):** $\frac{dl}{dz} = \frac{dl}{dy}\frac{dy}{dz}$ (*multiplicar por el camino*).
     - **Cadena distribuida (varios caminos, Diap. 11 y 12):** $\frac{dl}{dx} = \sum_{i=1}^M \frac{\partial l}{\partial z_i}\frac{dz_i}{dx}$ (*multiplicar por cada camino y sumar sobre los caminos*).
   - **Sección 2 (Diaps. 14–23): Grafos de Computación.**
     - 6 bloques básicos de construcción (suma, resta, producto, cuadrado, máximo, sigmoide $\sigma'(a)=\sigma(a)(1-\sigma(a))$).
     - Composición en cadena (Diap. 18): $b=a^2, c=b^2 \implies \frac{\partial c}{\partial a}=(2b)(2a)=4a^3$; en $a=2$: $b=4, c=16, \frac{\partial c}{\partial a}=32$.
     - Pérdida Hinge como grafo (Diap. 19): $\text{Loss}(x,y,\mathbf{w})=\max\{1-\mathbf{w}\cdot\phi(x)y, 0\} \implies \nabla_{\mathbf{w}}\text{Loss} = -\mathbf{1}[\text{margen}<1]\phi(x)y$.
     - Red de 2 capas como grafo (Diap. 20): $\text{Loss}=(\mathbf{w}\cdot\sigma(V\phi(x))-y)^2$, compartiendo el factor común $2(\text{residuo})$.
     - Backpropagation paso a paso sobre $\text{Loss}(x,y,\mathbf{w})=(\mathbf{w}\cdot\phi(x)-y)^2$:
       - **Ejemplo 1 (Diap. 21):** $\mathbf{w}=[3,1], \phi(x)=[1,2], y=2 \implies \text{puntaje}=5, \text{residuo}=3, \text{pérdida}=9, \nabla_{\mathbf{w}}\text{Loss}=[6,12]$.
       - **Ejemplo 2 (Diap. 23):** $\mathbf{w}=[1,2], \phi(x)=[2,1], y=3 \implies \text{puntaje}=4, \text{residuo}=1, \text{pérdida}=1, \nabla_{\mathbf{w}}\text{Loss}=[4,2]$.
     - Definición nodal (Diap. 22): $f_i$ (valor hacia adelante) y $g_i = \partial\text{Loss}/\partial f_i$ (gradiente hacia atrás de igual dimensión).
   - **Sección 3 (Diaps. 24–41): Forward Pass y Backward Pass en una Red por Capas.**
     - Convención del sesgo como peso ($y_0^{(k)}=1, w_{0j}^{(k)}=b_j^{(k)}$) y **Regla de Oro** (almacenar todos los $z_j^{(k)}$ e $y_j^{(k)}$ durante el *forward pass*).
     - Simetría Forward vs. Backward con notación punto $\dot{u} = \partial\text{div}/\partial u$ (Diap. 37):
       - $\dot{y}_i^{(k)} = \sum_j w_{ij}^{(k+1)}\dot{z}_j^{(k+1)}$
       - $\dot{z}_i^{(k)} = f_k'(z_i^{(k)})\dot{y}_i^{(k)}$
       - $\frac{\partial\text{div}}{\partial w_{ij}^{(k)}} = y_i^{(k-1)}\dot{z}_j^{(k)}$
     - **Ejemplo numérico completo en una red ReLU de 2 capas (Diaps. 38–39 y 57):**
       - $\mathbf{x}=[1,2], d=3, W_1=\begin{bmatrix}1&1\\1&-1\end{bmatrix}, \mathbf{b}_1=[-1,0], W_2=[[2,1]], \mathbf{b}_2=[1]$.
       - Forward: $z_1^{(1)}=2 \to y_1^{(1)}=2$, $z_2^{(1)}=-1 \to y_2^{(1)}=0$, $z_1^{(2)}=5 \to Y=5$, $\text{div}=(5-3)^2=4$.
       - Backward: $\dot{Y}=\dot{z}^{(2)}=4$, $\nabla_{W_2}=[[8,0]], \nabla_{\mathbf{b}_2}=[4]$; en $h_1$: $\dot{y}_1=8, \dot{z}_1=8$; en $h_2$: $\dot{y}_2=4, \dot{z}_2=0$ (neurona inactiva corta el gradiente); $\nabla_{W_1}=\begin{bmatrix}8&16\\0&0\end{bmatrix}, \nabla_{\mathbf{b}_1}=[8,0]$.
       - Verificación por perturbación finita ($\Delta w_{12}^{(1)} \implies \Delta\text{div}/\Delta w \approx 16$).
     - Subgradientes en ReLU ($z=0$) y función $\max(z_1, z_2, z_3)$ como enrutador del gradiente hacia el ganador (Diaps. 40–41).
   - **Sección 4 (Diaps. 42–60): Ciclo de Entrenamiento Completo y Vista Vectorial.**
     - Qué es y qué NO es Backpropagation (Diap. 44).
     - Derivadas vectoriales y Jacobianos transpuestos: $\nabla_{\mathbf{z}}s = J_{\mathbf{y}}(\mathbf{z})^\top\nabla_{\mathbf{y}}s$ (Diaps. 50–52).
     - Jacobiano diagonal (activación componente a componente) vs. Jacobiano lleno (Softmax, Diap. 51).
     - Las 4 fórmulas del Backward Pass Vectorial y su traducción directa a código NumPy (Diaps. 54–58):
       1. $\nabla_{\mathbf{z}_k}\text{div} = J_{\mathbf{y}_k}(\mathbf{z}_k)^\top \nabla_{\mathbf{y}_k}\text{div}$
       2. $\nabla_{\mathbf{y}_{k-1}}\text{div} = W_k^\top \nabla_{\mathbf{z}_k}\text{div}$
       3. $\nabla_{W_k}\text{div} = \nabla_{\mathbf{z}_k}\text{div}\,\mathbf{y}_{k-1}^\top$
       4. $\nabla_{\mathbf{b}_k}\text{div} = \nabla_{\mathbf{z}_k}\text{div}$

3. **Nuevo Módulo Computacional Python (`backpropagation.py`) e Integración en `perceptron_xor.py`:**
   - Implementar un script en Python puro (sin dependencias externas, ejecutable en modo interactivo y `--auto`) que verifique todas las demostraciones y ejemplos numéricos de las 60 diapositivas e integrarlo al menú y flujo `--auto` de `perceptron_xor.py`.

4. **Expansión de la Plataforma Web Interactiva (`web/index.html` — Unidad IV, Módulos 4.1 a 4.4):**
   - Ampliar el selector superior de unidades temáticas a **4 Unidades Temáticas** y **17 Laboratorios Interactivos**:
     - **Unidad I:** El Perceptrón y Fundamentos de Redes Neuronales (Módulos 1.1 a 1.5).
     - **Unidad II:** Aproximación Universal por Redes Neuronales (Módulos 2.1 a 2.4).
     - **Unidad III:** Optimización y Descenso por Gradiente (Módulos 3.1 a 3.4).
     - **Unidad IV:** Algoritmo de Backpropagation (Módulos 4.1 a 4.4):
       - **Módulo 4.1 (`panel-13`):** Regla de la Cadena Simple vs. Distribuida (Diagramas de Influencia Interactivos, Diaps. 8–13) + Composición en cadena $c=(a^2)^2$ (Diap. 18).
       - **Módulo 4.2 (`panel-14`):** Grafos de Computación Interactivos: Simulador paso a paso de los ejemplos de las Diapositivas 21 y 23 ($\text{Loss}=(\mathbf{w}\cdot\phi(x)-y)^2$), Pérdida Hinge (Diap. 19) y Enrutador `max(z₁, z₂, z₃)` (Diap. 41).
       - **Módulo 4.3 (`panel-15`):** Red ReLU de 2 Capas Paso a Paso (Diapositivas 38–39 y 57): visualización del grafo neuronal completo con valores azules de *forward pass* ($z, y$) y valores rojos de *backward pass* ($\dot{y}, \dot{z}, \nabla_{W_1}, \nabla_{\mathbf{b}_1}, \nabla_{W_2}, \nabla_{\mathbf{b}_2}$), destacando la neurona inactiva que corta el gradiente y verificación por diferencias finitas.
       - **Módulo 4.4 (`panel-16`):** Vista Vectorial, Jacobianos (Diagonal vs. Softmax lleno, Diaps. 49–58), tabla de traducción Fórmula $\leftrightarrow$ Código NumPy (Diap. 58) y **Autoevaluación Interactiva IV** (preguntas 13 a 16).

5. **Documentación y Verificación:**
   - Actualizar `README.md` e `INFORME.md` con la nueva Unidad IV.
   - Ejecutar `python3 backpropagation.py --auto`, `python3 perceptron_xor.py --auto` y verificar la sintaxis JS y coincidencia total de IDs del DOM en `web/index.html`.
