# Contexto 1: Comprobación del Perceptrón Multicapa para la Función XOR
**Fecha de registro:** 2026-10-26

## 1. Instrucciones
* Implementar un Perceptrón Multicapa (MLP) respetando exactamente la arquitectura, pesos y umbrales especificados en la teoría de la diapositiva:
  - **Entradas:** $X, Y \in \{0, 1\}$
  - **Neurona oculta 1 ($h_1$ - Disyunción / OR):**
    - Pesos: $w_{1,X} = 1$, $w_{1,Y} = 1$
    - Umbral: $\theta_1 = 1$
    - Activación: $h_1 = 1$ si $1\cdot X + 1\cdot Y \ge 1$, de lo contrario $0$. Representa $X \lor Y$.
  - **Neurona oculta 2 ($h_2$ - NAND / $\bar{X} \lor \bar{Y}$):**
    - Pesos: $w_{2,X} = -1$, $w_{2,Y} = -1$
    - Umbral: $\theta_2 = -1$
    - Activación: $h_2 = 1$ si $-1\cdot X - 1\cdot Y \ge -1$, de lo contrario $0$. Representa $\bar{X} \lor \bar{Y}$.
  - **Neurona de salida ($y$ - Conjunción / AND de la capa oculta):**
    - Pesos: $w_{\text{out},1} = 1$, $w_{\text{out},2} = 1$
    - Umbral: $\theta_{\text{out}} = 2$
    - Activación: $y = 1$ si $1\cdot h_1 + 1\cdot h_2 \ge 2$, de lo contrario $0$. Representa $(X \lor Y) \land (\bar{X} \lor \bar{Y}) = X \oplus Y$.
* Validar el comportamiento para los cuatro pares de entrada: $(0,0), (0,1), (1,0), (1,1)$.
* Desarrollar el código en Python sin dependencias obligatorias (compatible con biblioteca estándar de Python y extensible con visualizaciones si se desea).
* Incluir explicación teórica y geométrica del funcionamiento de los hiperplanos separadores.

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Demostrar y verificar formal y computacionalmente que la red neuronal multicapa propuesta resuelve el problema no linealmente separable del operador XOR.
* **Fases/Hitos principales:**
  1. Formalización matemática de cada neurona (pesos, umbrales y funciones de activación tipo escalón).
  2. Implementación en Python (`perceptron_xor.py`) con cálculo paso a paso e impresión de tabla de verdad.
  3. Verificación exhaustiva de resultados frente a la tabla esperada de la compuerta XOR.
* **Entregables:**
  - Código fuente Python (`perceptron_xor.py`) ejecutable y documentado.
  - Tabla de verdad y traza de ejecución de todas las capas.
  - Informe y explicación analítica.

## 3. Referencias
* **Path del plan original:** `contexto/context1.md`
