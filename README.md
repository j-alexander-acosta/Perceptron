# Aproximación Universal con Redes Neuronales y Perceptrón Multicapa (XOR)

Este repositorio implementa de manera interactiva, visual y educativa los fundamentos teóricos de la clase **«Aproximación Universal con Redes Neuronales»** (*Ciencia de Datos — Doctorado UCN, basado en el curso CMU 11-785 de B. Raj et al.*).

> 📄 **Informe Académico Completo:** Consulta el desarrollo matemático y las demostraciones en [`INFORME.md`](INFORME.md).

---

## 🎓 Los 4 Ejes Educativos Implementados

### 1. El MLP como Función Booleana Universal (Diapositivas 11–29)
* **Compuertas lógicas y universales de $N$ entradas:** Una sola unidad de umbral calcula «al menos $K$ de $N$» asignando pesos $\pm 1$ y umbral $L - N + K$.
* **Dos arquitecturas para XOR:**
  1. **MLP Clásico (3 neuronas):** Capa oculta con `OR` ($X+Y \ge 1$) y `NAND` ($-X-Y \ge -1$), combinadas por una salida `AND` ($h_1 + h_2 \ge 2$).
  2. **Construcción de Friedland (Solo 2 neuronas):** Una única neurona oculta calcula $X \land Y$ (umbral 2) y la neurona de salida recibe conexiones directas de $X$ e $Y$ evaluando $X + Y - 2(X \land Y) \ge 1$.
* **Síntesis vía Forma Normal Disyuntiva (DNF) y Mapas de Karnaugh:** Cualquier tabla de verdad se convierte en un MLP de **una sola capa oculta** (una neurona `AND` por minitérmino con salida `1` y una neurona `OR` de salida). Reducir el mapa de Karnaugh equivale a reducir el ancho de la capa oculta (por ejemplo, de 7 a 3 neuronas en la diapositiva 19).
* **El costo exponencial de una sola capa oculta vs. redes profundas:** El peor caso booleano es el «tablero de ajedrez» (la función de paridad $X_1 \oplus X_2 \oplus \dots \oplus X_N$).
  * Con **una sola capa oculta**: requiere **$2^{N-1}$ neuronas ocultas** ($2^{N-1} + 1$ en total). Para $N=16$, exige $32,769$ neuronas.
  * Con una **red profunda en árbol** de bloques XOR: requiere solo **$3(N-1)$ neuronas** (o $2(N-1)$ con bloques de 2 neuronas) organizadas en $2\log_2(N)$ capas. Para $N=16$, apenas $45$ neuronas.

### 2. El MLP como Clasificador Universal en $\mathbb{R}^2$ (Diapositivas 30–48)
* **De polígonos al cilindro circular:** Un polígono convexo de $N$ lados se construye con $N$ neuronas ocultas (una por semiplano) y una salida `AND` con umbral $N$.
* **El piso asintótico $N/2$ y el rol del sesgo:** Al aumentar $N \to \infty$, el polígono tiende a un círculo de radio $r$. Adentro la suma vale $N$, y afuera (a distancia $d > r$) cae según:
  $$\sum_{i=1}^N y_i = N\left(\frac{1}{2} + \frac{1}{\pi}\arcsin\frac{r}{d}\right) \xrightarrow{d \to \infty} \frac{N}{2}$$
  Agregando un **sesgo $-N/2$** en la salida, el piso exterior baja a $\approx 0$, permitiendo sumar múltiples círculos en una sola capa oculta para aproximar cualquier región arbitraria.

### 3. El MLP como Aproximador Universal de Funciones Continuas (Diapositivas 49–54)
* **Construcción con pulsos cuadrados:** Dos neuronas de umbral en la capa oculta ($\mathbf{1}[x \ge T_1]$ con peso $+h_i$ y $\mathbf{1}[x \ge T_2]$ con peso $-h_i$) generan un pulso cuadrado de altura $h_i$ en $[T_1, T_2]$. Sumando múltiples pulsos se aproxima cualquier función continua $f(x)$ con error $\varepsilon > 0$ arbitrariamente pequeño.
* **Tres verbos que no dicen lo mismo:** *Representar* (igualdad exacta), *Aproximar* (error $< \varepsilon$ con red finita) y *el límite* ($\varepsilon \to 0$).

### 4. Suficiencia de la Arquitectura: Ancho, Profundidad y Activaciones (Diapositivas 55–61)
* **Franjas y celdas:** Una capa con pocas neuronas de umbral colapsa regiones enteras del espacio al mismo código binario, perdiendo información fina que ninguna capa posterior puede recuperar.
* **Activaciones graduadas ($\sigma(z)$, $\tanh(z)$, $\text{ReLU}(z)$):** A diferencia del escalón binario, transmiten *cuánto más allá* de la frontera se encuentra la entrada.
* **El triángulo de compensaciones:** **Ancho $\leftrightarrow$ Profundidad $\leftrightarrow$ Activaciones** se compensan parcialmente entre sí, pero ninguno reemplaza por completo a los otros.

---

## 🚀 Cómo Ejecutar los Laboratorios Educativos

### 1. 🌐 Suite Web Educativa Interactiva (5 Laboratorios en Tiempo Real)
Incluye simulador de compuertas y XOR (3 vs. 2 neuronas), Mapa de Karnaugh $4\times 4$ interactivo, comparador exponencial vs. profundo, compositor geométrico 2D de polígonos/círculos, aproximador de funciones continuas por pulsos y autoevaluación interactiva:
```bash
open web/index.html
# O desde Python:
python3 perceptron_xor.py --web
```

### 2. 🎓 Laboratorio Educativo en Consola (`aproximacion_universal.py`)
Ejecuta todas las demostraciones matemáticas y tablas comparativas de las 61 diapositivas:
```bash
# Modo interactivo por módulos:
python3 aproximacion_universal.py

# Ejecutar todas las demostraciones de una vez:
python3 aproximacion_universal.py --auto
```

### 3. 🖥️ Aplicación de Escritorio Nativa XOR (`gui_xor.py`) y Script Principal (`perceptron_xor.py`)
```bash
# Menú integrado completo:
python3 perceptron_xor.py

# Interfaz nativa con Tkinter:
python3 gui_xor.py
```

---

## 📁 Estructura del Proyecto
```
├── README.md                    # Guía general y síntesis de los 4 pilares educativos
├── INFORME.md                   # Informe académico detallado de toda la teoría y pruebas
├── aproximacion_universal.py    # Implementación en Python de los 4 módulos de Aproximación Universal
├── perceptron_xor.py            # Verificador del MLP XOR y lanzador central
├── gui_xor.py                   # Interfaz gráfica de escritorio nativa (Tkinter)
├── web/
│   └── index.html               # Suite Educativa Web con 5 laboratorios interactivos
├── xor_espacio_decision.png     # Gráfico 2D de separación lineal para XOR
└── contexto/                    # Historial secuencial de planes de trabajo (context1..context6)
```
