# Perceptrón Multicapa para Resolver la Función XOR

Este repositorio implementa y demuestra la teoría según la cual **un perceptrón multicapa (MLP) es capaz de resolver el problema del operador lógico XOR ($\oplus$)**, superando la limitación histórica de separabilidad lineal del perceptrón simple (Minsky & Papert, 1969).

> 📄 **Informe Académico Completo:** Consulta el documento formal en [`INFORME.md`](INFORME.md).

---

## 1. Fundamento Teórico

El operador lógico **XOR** se define mediante la siguiente equivalencia booleana:

$$X \oplus Y = (X \lor Y) \land (\bar{X} \lor \bar{Y})$$

> *"Disparar cuando al menos uno es 1 y no ambos. Componer capas sí resuelve XOR."*

### Función de Activación
Se utiliza la función escalón unitario (Heaviside) con umbral $\theta$:

$$f(z - \theta) = \begin{cases} 1 & \text{si } z \ge \theta \\ 0 & \text{si } z < \theta \end{cases} \quad \text{donde } z = \sum_{i} w_i x_i$$

---

## 2. Arquitectura de la Red

```
       X (Entrada)               Capa Oculta                       Salida
          \        1        +-------------------+
           \ -------------> |   Neurona 1       | 1
            \        -1     |   (Umbral: 1)     | ----\
             \    /-------> |   h1 = X ∨ Y      |      \
              \  /          +-------------------+       \   +-------------------+
               \/                                        -> |   Neurona Salida  |
               /\                                        -> |   (Umbral: 2)     | ---> X ⊕ Y
              /  \          +-------------------+       /   |   X ⊕ Y = h1 ∧ h2 |
             /    \-------> |   Neurona 2       |      /    +-------------------+
            /        1      |   (Umbral: -1)    | ----/ 1
           / -------------> |   h2 = X̄ ∨ Ȳ      |
          /                 +-------------------+
       Y (Entrada)
```

### Especificación de Neuronas:
1. **Neurona Oculta 1 ($h_1$ - Disyunción / OR):**
   - Pesos: $w_{1,X} = 1.0, w_{1,Y} = 1.0$
   - Umbral: $\theta_1 = 1.0$
   - Ecuación: $z_1 = 1\cdot X + 1\cdot Y$. Dispara ($h_1=1$) si $X + Y \ge 1$.
2. **Neurona Oculta 2 ($h_2$ - Negación de Conjunción / NAND):**
   - Pesos: $w_{2,X} = -1.0, w_{2,Y} = -1.0$
   - Umbral: $\theta_2 = -1.0$
   - Ecuación: $z_2 = -X - Y$. Dispara ($h_2=1$) si $-X - Y \ge -1 \iff X + Y \le 1$.
3. **Neurona de Salida (Conjunción / AND de la capa oculta):**
   - Pesos: $w_{\text{out},1} = 1.0, w_{\text{out},2} = 1.0$
   - Umbral: $\theta_{\text{out}} = 2.0$
   - Ecuación: $z_{\text{out}} = 1\cdot h_1 + 1\cdot h_2$. Dispara ($\text{Salida}=1$) si $h_1 + h_2 \ge 2$.

---

## 3. Tabla de Verdad y Comprobación

| $X$ | $Y$ | $z_1 = X+Y$ | $h_1$ ($z_1 \ge 1$) | $z_2 = -X-Y$ | $h_2$ ($z_2 \ge -1$) | $z_{\text{out}} = h_1+h_2$ | Salida ($z_{\text{out}} \ge 2$) | Esperado ($X \oplus Y$) | ¿Correcto? |
|:---:|:---:|:-----------:|:-------------------:|:------------:|:--------------------:|:--------------------------:|:-------------------------------:|:-----------------------:|:----------:|
|  0  |  0  |     0.0     |        **0**        |      0.0     |        **1**         |            1.0             |              **0**              |          **0**          |   ✅ SÍ    |
|  0  |  1  |     1.0     |        **1**        |     -1.0     |        **1**         |            2.0             |              **1**              |          **1**          |   ✅ SÍ    |
|  1  |  0  |     1.0     |        **1**        |     -1.0     |        **1**         |            2.0             |              **1**              |          **1**          |   ✅ SÍ    |
|  1  |  1  |     2.0     |        **1**        |     -2.0     |        **0**         |            1.0             |              **0**              |          **0**          |   ✅ SÍ    |

---

## 4. Frontera de Decisión Geométrica

![Espacio de Decisión](xor_espacio_decision.png)

---

## 5. Instrucciones de Uso

El script no requiere librerías externas para los cálculos (emplea la biblioteca estándar de Python). Para la generación del gráfico visual se utiliza opcionalmente `matplotlib`.

### Modo Interactivo (Menú):
```bash
python3 perceptron_xor.py
```
Permite seleccionar entre:
* **[1] Ingreso manual de datos:** Evalúa pares $(X, Y)$ con desglose aritmético paso a paso.
* **[2] Comprobación completa:** Muestra la tabla de verdad y trazado de todos los casos.
* **[3] Generar gráfico de decisión:** Exporta `xor_espacio_decision.png`.

### Paso Directo de Parámetros por Consola:
```bash
python3 perceptron_xor.py 1 0
python3 perceptron_xor.py 1 1
```

### Ejecución Automática Completa:
```bash
python3 perceptron_xor.py --auto
```

---

## Estructura del Repositorio
```
├── README.md                 # Resumen y guía de uso rápido
├── INFORME.md                # Informe técnico-científico académico formal
├── .gitignore                # Archivos ignorados por Git
├── perceptron_xor.py         # Implementación del perceptrón y modos de uso
├── xor_espacio_decision.png  # Gráfico del espacio de decisión 2D
└── contexto/                 # Historial estructurado de requerimientos
    ├── context1.md           # Verificación teórica inicial
    ├── context2.md           # Habilitación del modo manual interactivo
    ├── context3.md           # Sincronización con GitHub
    └── context4.md           # Redacción del informe académico formal
```
