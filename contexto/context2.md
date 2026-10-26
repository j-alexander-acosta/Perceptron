# Contexto 2: Modo Interactivo e Ingreso Manual de Datos para el Perceptrón XOR
**Fecha de registro:** 2026-10-26

## 1. Instrucciones
* Modificar el script `perceptron_xor.py` para permitir la entrada manual e interactiva de los valores de entrada $X$ e $Y$.
* Validar que los valores ingresados sean binarios ($0$ o $1$), con manejo amigable de errores de formato.
* Desglosar y mostrar el cálculo interno paso a paso para las entradas dadas:
  - Suma ponderada y estado de activación de la neurona oculta 1 ($X \lor Y$).
  - Suma ponderada y estado de activación de la neurona oculta 2 ($\bar{X} \lor \bar{Y}$).
  - Suma ponderada y estado de activación de la neurona de salida ($X \oplus Y$).
  - Comparación con el valor esperado del operador XOR.
* Ofrecer un menú interactivo que permita tanto el ingreso manual repetido como la ejecución de la tabla completa y la generación del gráfico.
* Añadir soporte para argumentos de línea de comandos (ej. `python3 perceptron_xor.py 1 0` o `python3 perceptron_xor.py --auto`).

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Habilitar el ingreso manual de datos para evaluar interactivamente cualquier par $(X, Y)$ en el perceptrón multicapa.
* **Fases/Hitos principales:**
  1. Implementar la función de entrada manual con validación y bucle interactivo en `perceptron_xor.py`.
  2. Incorporar soporte para argumentos por consola (`sys.argv`).
  3. Validar el funcionamiento del flujo interactivo y las entradas.
* **Entregables:**
  - Script `perceptron_xor.py` actualizado.
  - Documento `contexto/context2.md`.

## 3. Referencias
* **Path del plan original:** `contexto/context2.md`
