# Suite Educativa de Redes Neuronales: Aproximación Universal, Aprendizaje y Descenso por Gradiente

Este repositorio implementa de manera interactiva, visual y educativa los fundamentos teóricos y prácticos de las tres unidades del curso (*Ciencia de Datos — Doctorado UCN, basado en CMU 11-785 de B. Raj et al. y Stanford CS221 de P. Liang et al.*):

* **Unidad I — Representación y Aproximación Universal (61 Diapositivas):** Laboratorios 1.1 a 1.5 ([`aproximacion_universal.py`](aproximacion_universal.py)).
* **Unidad II — Aprendizaje de Redes Neuronales (56 Diapositivas):** Laboratorios 2.1 a 2.4 ([`aprendizaje_redes.py`](aprendizaje_redes.py)).
* **Unidad III — Descenso por Gradiente y Convergencia (47 Diapositivas):** Laboratorios 3.1 a 3.4 ([`descenso_gradiente.py`](descenso_gradiente.py)).

> 📄 **Informe Académico Completo:** Consulta el desarrollo matemático y las demostraciones en [`INFORME.md`](INFORME.md).

---

## 🏛️ Organización Temática por Unidades

### 🟦 UNIDAD I: Representación y Aproximación Universal con Redes Neuronales

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

## 🚀 Cómo Ejecutar los Laboratorios Educativos

### 1. 🌐 Suite Web Educativa Interactiva (3 Unidades • 13 Laboratorios en Navegador)
```bash
open web/index.html
# O desde Python:
python3 perceptron_xor.py --web
```

### 2. ⚡ Laboratorios en Consola por Unidad (Python Puro)
```bash
# Ejecutar TODAS las verificaciones de las Unidades I, II y III automáticamente:
python3 perceptron_xor.py --auto

# Laboratorio interactivo de la Unidad I (Aproximación Universal):
python3 aproximacion_universal.py

# Laboratorio interactivo de la Unidad II (Aprendizaje de Redes Neuronales):
python3 aprendizaje_redes.py

# Laboratorio interactivo de la Unidad III (Descenso por Gradiente):
python3 descenso_gradiente.py

# Menú principal integrado:
python3 perceptron_xor.py
```

---

## 📁 Estructura del Proyecto
```
├── README.md                    # Guía general organizada por Unidades I, II y III
├── INFORME.md                   # Informe académico detallado de las tres unidades
├── perceptron_xor.py            # Lanzador central y verificador del MLP XOR
├── aproximacion_universal.py    # Unidad I: DNF, Karnaugh, Paridad N-aria, Polígonos y Pulsos
├── aprendizaje_redes.py         # Unidad II: Rosenblatt (Diap. 22), ADALINE, Sigmoide, Softmax y KL
├── descenso_gradiente.py        # Unidad III: Gradiente, Hessiana 3D (Diap. 24), 200 Épocas y Step Size
├── gui_xor.py                   # Interfaz gráfica de escritorio nativa (Tkinter)
├── web/
│   └── index.html               # Suite Web Interactiva (3 Unidades Temáticas • 13 Laboratorios)
└── contexto/                    # Historial secuencial de planes de trabajo (context1..context8)
```
