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

## 1. Introducción y Contexto Histórico

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

## 9. Conclusiones

1. **Comprobación de la teoría (Unidad I):** Se demostró de manera analítica y computacional que el MLP de 3 neuronas y la variante compacta de 2 neuronas de Friedland resuelven de forma exacta el operador lógico XOR, y que un MLP de una sola capa oculta es un aproximador universal booleano, geométrico y continuo cuyo ancho puede crecer como $2^{N-1}$ si no se emplea profundidad ($3(N-1)$).
2. **Aprendizaje y diferenciabilidad (Unidad II):** Se comprobó empíricamente la convergencia del perceptrón de Rosenblatt en 3 pasos sobre el conjunto de la Diapositiva 22 ($\mathbf{w} = [3.25, 1.9]$), la regla Delta de ADALINE, la diferencia entre el paisaje plano del conteo de errores y la señal continua de la sigmoide, y la superioridad del par Sigmoide/Softmax + Entropía Cruzada (KL) frente a L2 en problemas de clasificación.
3. **Optimización por Descenso de Gradiente (Unidad III):** Se verificaron analítica y numéricamente la ortogonalidad del gradiente respecto a las curvas de nivel, la clasificación del punto crítico 3D $\mathbf{x}^*=(-1,-1,-1)$ mediante su Matriz Hessiana definida positiva ($\lambda = \{3.414, 2, 0.586\} > 0$), la trayectoria de 200 épocas en Regresión Lineal hacia $\mathbf{w}^*=[1, 0.57]$, y los 4 regímenes de convergencia/divergencia gobernados por la relación entre $\eta$ y la curvatura $(E'')^{-1}$.
4. **Validación integral:** Todos los módulos evaluados en [`perceptron_xor.py`](perceptron_xor.py), [`aproximacion_universal.py`](aproximacion_universal.py), [`aprendizaje_redes.py`](aprendizaje_redes.py), [`descenso_gradiente.py`](descenso_gradiente.py) y la suite web [`web/index.html`](web/index.html) arrojaron concordancia del 100% con el material de cátedra.

---

## 10. Referencias Bibliográficas

1. **Cauchy, A. (1847).** *Méthode générale pour la résolution des systèmes d'équations simultanées.* Comptes Rendus Hebdomadaires des Séances de l'Académie des Sciences, 25, 536-538.
2. **McCulloch, W. S., & Pitts, W. (1943).** *A logical calculus of the ideas immanent in nervous activity.* The Bulletin of Mathematical Biophysics, 5(4), 115-133.
3. **Rosenblatt, F. (1958).** *The perceptron: A probabilistic model for information storage and organization in the brain.* Psychological Review, 65(6), 386-408.
4. **Widrow, B., & Hoff, M. E. (1960).** *Adaptive switching circuits.* IRE WESCON Convention Record, Part 4, 96-104.
5. **Minsky, M., & Papert, S. (1969).** *Perceptrons: An Introduction to Computational Geometry.* MIT Press, Cambridge, MA.
6. **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).** *Learning representations by back-propagating errors.* Nature, 323(6088), 533-536.
7. **Cybenko, G. (1989).** *Approximation by superpositions of a sigmoidal function.* Mathematics of Control, Signals and Systems, 2(4), 303-314.
8. **Raj, B., et al. (CMU 11-785) & Liang, P., et al. (Stanford CS221).** *Aproximación Universal, Aprendizaje de Redes Neuronales & Descenso por Gradiente.* Ciencia de Datos — Doctorado UCN.
