#!/usr/bin/env python3
"""
Comprobación del Perceptrón Multicapa para la compuerta XOR
Basado en la arquitectura, pesos y umbrales de la diapositiva:
"El perceptrón multicapa resuelve XOR"

Estructura de la red:
- Capa de Entrada:
    Variables booleanas X, Y in {0, 1}

- Capa Oculta (2 neuronas):
    1) Neurona Superior (OR):
       Pesos: w_X = 1, w_Y = 1
       Umbral: theta = 1
       Operación: X v Y (disparo si 1*X + 1*Y >= 1)
    2) Neurona Inferior (NAND):
       Pesos: w_X = -1, w_Y = -1
       Umbral: theta = -1
       Operación: ~X v ~Y  <=> ~(X ^ Y) (disparo si -1*X + -1*Y >= -1)

- Capa de Salida (1 neurona):
    Neurona de Salida (AND de capa oculta):
       Pesos: w_h1 = 1, w_h2 = 1
       Umbral: theta = 2
       Operación: (X v Y) ^ (~X v ~Y) <=> X (+) Y
       (disparo si 1*h1 + 1*h2 >= 2)
"""

import sys
from typing import Tuple, Dict, Any, List


class Perceptron:
    """
    Modelo de neurona artificial (Perceptrón clásico de McCulloch-Pitts / Rosenblatt)
    con función de activación de escalón unitario (Heaviside):
        f(z) = 1 si z >= umbral, de lo contrario 0
    donde z = sum(w_i * x_i)
    """
    def __init__(self, weights: List[float], threshold: float, name: str = ""):
        self.weights = weights
        self.threshold = threshold
        self.name = name

    def activate(self, inputs: List[int]) -> Tuple[int, float]:
        """
        Calcula la suma ponderada z y evalúa si alcanza el umbral.
        Retorna (activación: 0 ó 1, suma ponderada z).
        """
        if len(inputs) != len(self.weights):
            raise ValueError(f"Dimensión de entrada {len(inputs)} no coincide con dimensión de pesos {len(self.weights)}")
        
        z = sum(w * x for w, x in zip(self.weights, inputs))
        output = 1 if z >= self.threshold else 0
        return output, z


class PerceptronMulticapaXOR:
    """
    Red Neuronal Feedforward de 2 capas para resolver la función XOR.
    Implementa exactamente la arquitectura de la diapositiva.
    """
    def __init__(self):
        # Capa Oculta
        self.h1_or = Perceptron(weights=[1.0, 1.0], threshold=1.0, name="h1 (X ∨ Y)")
        self.h2_nand = Perceptron(weights=[-1.0, -1.0], threshold=-1.0, name="h2 (X̄ ∨ Ȳ)")
        
        # Capa de Salida
        self.out_xor = Perceptron(weights=[1.0, 1.0], threshold=2.0, name="Salida (X ⊕ Y)")

    def forward(self, x: int, y: int) -> Dict[str, Any]:
        """
        Propagación hacia adelante paso a paso.
        """
        inputs = [x, y]
        
        # 1. Capa oculta
        h1_val, z_h1 = self.h1_or.activate(inputs)
        h2_val, z_h2 = self.h2_nand.activate(inputs)
        
        # 2. Capa de salida
        hidden_outputs = [h1_val, h2_val]
        out_val, z_out = self.out_xor.activate(hidden_outputs)
        
        expected = x ^ y
        
        return {
            "inputs": (x, y),
            "z_h1": z_h1,
            "h1": h1_val,
            "z_h2": z_h2,
            "h2": h2_val,
            "z_out": z_out,
            "out": out_val,
            "expected_xor": expected,
            "correct": out_val == expected
        }


def print_step_by_step(res: Dict[str, Any]):
    """
    Imprime el desglose analítico y aritmético completo de la red para una entrada dada.
    """
    x, y = res["inputs"]
    h1 = res["h1"]
    h2 = res["h2"]
    z1 = res["z_h1"]
    z2 = res["z_h2"]
    z_out = res["z_out"]
    out = res["out"]
    exp = res["expected_xor"]
    
    print("\n" + "=" * 70)
    print(f" RESULTADO DETALLADO PARA ENTRADA: X = {x}, Y = {y}")
    print("=" * 70)
    print("1. CAPA OCULTA:")
    print(f"   • Neurona 1 (Disyunción - OR):")
    print(f"       Pesos: w = [1.0, 1.0], Umbral: theta = 1.0")
    print(f"       Suma ponderada z1 = (1.0)*{x} + (1.0)*{y} = {z1:.1f}")
    print(f"       Disparo: ¿z1 >= 1.0? -> {'SÍ' if h1 == 1 else 'NO'} => h1 (X ∨ Y) = {h1}")
    print()
    print(f"   • Neurona 2 (NAND - X̄ ∨ Ȳ):")
    print(f"       Pesos: w = [-1.0, -1.0], Umbral: theta = -1.0")
    print(f"       Suma ponderada z2 = (-1.0)*{x} + (-1.0)*{y} = {z2:.1f}")
    print(f"       Disparo: ¿z2 >= -1.0? -> {'SÍ' if h2 == 1 else 'NO'} => h2 (X̄ ∨ Ȳ) = {h2}")
    print("-" * 70)
    print("2. CAPA DE SALIDA:")
    print(f"   • Neurona de Salida (Conjunción - AND de capa oculta):")
    print(f"       Pesos: w = [1.0, 1.0], Umbral: theta = 2.0")
    print(f"       Suma ponderada z_out = (1.0)*{h1} + (1.0)*{h2} = {z_out:.1f}")
    print(f"       Disparo: ¿z_out >= 2.0? -> {'SÍ' if out == 1 else 'NO'} => Salida = {out}")
    print("-" * 70)
    print("3. CONCLUSIÓN Y VERIFICACIÓN:")
    print(f"   • Predicción de la Red: {out}")
    print(f"   • Valor Teórico XOR:   {exp}")
    print(f"   • Estado:               {'✅ CORRECTO (Coincide con XOR)' if res['correct'] else '❌ ERROR'}")
    print("=" * 70 + "\n")


def run_verification(show_details: bool = True) -> bool:
    """
    Ejecuta la comprobación completa para todas las combinaciones booleanas posibles.
    """
    mlp = PerceptronMulticapaXOR()
    cases = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    print("\n" + "=" * 80)
    print(" COMPROBACIÓN DE TEORÍA: EL PERCEPTRÓN MULTICAPA RESUELVE XOR")
    print("=" * 80)
    print(f"{'X':^3} | {'Y':^3} || {'z_h1':^7} | {'h1 (OR)':^8} || {'z_h2':^7} | {'h2(NAND)':^9} || {'z_out':^7} | {'X ⊕ Y':^7} | {'Esperado':^8} | {'¿Correcto?':^10}")
    print("-" * 80)
    
    all_passed = True
    results = []
    
    for x, y in cases:
        res = mlp.forward(x, y)
        results.append(res)
        if not res["correct"]:
            all_passed = False
            
        print(f"{x:^3} | {y:^3} || "
              f"{res['z_h1']:>5.1f}   | {res['h1']:^8} || "
              f"{res['z_h2']:>5.1f}   | {res['h2']:^9} || "
              f"{res['z_out']:>5.1f}   | {res['out']:^7} | "
              f"{res['expected_xor']:^8} | "
              f"{'SÍ [OK]' if res['correct'] else 'NO [ERROR]':^10}")
        
    print("=" * 80)
    
    if show_details:
        print("\nANÁLISIS PASO A PASO POR CASO:")
        for res in results:
            x, y = res["inputs"]
            print(f"▶ Caso (X={x}, Y={y}): "
                  f"h1(OR)={res['h1']}, h2(NAND)={res['h2']} -> Salida={res['out']} "
                  f"[XOR esperado: {res['expected_xor']}] -> {'OK' if res['correct'] else 'FALLÓ'}")
    
    if all_passed:
        print("\n✅ Conclusión: La teoría de la diapositiva ha sido comprobada con éxito al 100%.")
    else:
        print("\n❌ Conclusión: Hubo discrepancias con la tabla esperada.")
        
    return all_passed


def request_binary_input(prompt: str) -> int:
    """
    Solicita un valor binario (0 o 1) al usuario con validación.
    Permite escribir 'q' o 'salir' para cancelar.
    """
    while True:
        raw = input(prompt).strip().lower()
        if raw in ('q', 'salir', 'exit'):
            raise KeyboardInterrupt
        if raw in ('0', '1'):
            return int(raw)
        print("  ⚠️  Entrada inválida. Por favor ingrese únicamente 0 o 1 (o 'q' para salir).")


def run_manual_mode(mlp: PerceptronMulticapaXOR):
    """
    Modo interactivo que permite ingresar valores manualmente.
    """
    print("\n" + "=" * 60)
    print(" INGRESO MANUAL DE DATOS - PERCEPTRÓN XOR")
    print(" Ingrese valores binarios (0 o 1) para las entradas X e Y.")
    print(" (Puede escribir 'q' o presionar Ctrl+C en cualquier momento para salir)")
    print("=" * 60)
    
    while True:
        try:
            print()
            x = request_binary_input("• Ingrese valor de X (0 o 1): ")
            y = request_binary_input("• Ingrese valor de Y (0 o 1): ")
            
            res = mlp.forward(x, y)
            print_step_by_step(res)
            
            seguir = input("¿Desea ingresar otro par de datos? [S/n]: ").strip().lower()
            if seguir in ('n', 'no', 'q'):
                print("Volviendo al menú principal...")
                break
        except (KeyboardInterrupt, EOFError):
            print("\nOperación cancelada. Volviendo al menú principal...")
            break


def generate_plot(output_filename: str = "xor_espacio_decision.png"):
    """
    Genera un gráfico visual del espacio 2D mostrando los puntos de entrada,
    sus etiquetas XOR y las dos fronteras lineales definidas por la capa oculta.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n[Nota] Matplotlib o NumPy no están instalados. Se omite la generación del gráfico.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Puntos XOR
    points = [(0, 0), (0, 1), (1, 0), (1, 1)]
    labels = [0, 1, 1, 0]
    
    for (x, y), label in zip(points, labels):
        color = "#2b5c8f" if label == 1 else "#d9534f"
        marker = "o" if label == 1 else "s"
        ax.scatter(x, y, s=250, c=color, marker=marker, edgecolors="black", linewidths=1.5, zorder=5)
        ax.annotate(f"({x},{y})\nSalida={label}", (x, y), textcoords="offset points", 
                    xytext=(0, 14 if y == 0 else -25), ha='center', fontsize=10, fontweight='bold')
    
    x_vals = np.linspace(-0.5, 1.5, 200)
    y_line = 1 - x_vals
    
    ax.plot(x_vals, y_line, color="#2e7d32", linestyle="--", linewidth=2.5, 
            label="Frontera oculta: $X + Y = 1$\n($h_1: X+Y \\geq 1$, $h_2: X+Y \\leq 1$)")
    
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel("Entrada X", fontsize=12, fontweight='bold')
    ax.set_ylabel("Entrada Y", fontsize=12, fontweight='bold')
    ax.set_title("Espacio de Entrada y Separación de XOR con Capa Oculta", fontsize=13, fontweight='bold')
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close()
    print(f"[Info] Gráfico de decisión guardado en: {output_filename}")


def launch_gui():
    """
    Inicia la aplicación gráfica nativa de escritorio con Tkinter.
    """
    try:
        from gui_xor import launch
        print("\nIniciando entorno gráfico de escritorio (Tkinter)...")
        launch()
    except Exception as e:
        print(f"⚠️ Error al iniciar la interfaz gráfica: {e}")
        print("Puede ejecutarla directamente con: python3 gui_xor.py")


def launch_web():
    """
    Abre la aplicación gráfica web interactiva en el navegador predeterminado.
    """
    import os
    import webbrowser
    html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "web", "index.html"))
    print(f"\nAbriendo entorno gráfico web en: file://{html_path}")
    webbrowser.open(f"file://{html_path}")


def launch_universal_approx():
    """
    Ejecuta el laboratorio interactivo de la Unidad I: Aproximación Universal con Redes Neuronales.
    """
    try:
        from aproximacion_universal import interactive_menu
        interactive_menu()
    except Exception as e:
        print(f"⚠️ Error al cargar el módulo de Aproximación Universal: {e}")


def launch_learning_module():
    """
    Ejecuta el laboratorio interactivo de la Unidad II: Aprendizaje de Redes Neuronales.
    """
    try:
        from aprendizaje_redes import interactive_learning_menu
        interactive_learning_menu()
    except Exception as e:
        print(f"⚠️ Error al cargar el módulo de Aprendizaje de Redes Neuronales: {e}")


def main():
    mlp = PerceptronMulticapaXOR()
    
    # Argumentos por línea de comandos
    if len(sys.argv) >= 3:
        try:
            x = int(sys.argv[1])
            y = int(sys.argv[2])
            if x not in (0, 1) or y not in (0, 1):
                print("Error: Los valores de X e Y deben ser 0 o 1.")
                sys.exit(1)
            res = mlp.forward(x, y)
            print_step_by_step(res)
            return
        except ValueError:
            print("Error: Argumentos no válidos. Ejemplo de uso: python3 perceptron_xor.py 1 0")
            sys.exit(1)
    elif len(sys.argv) == 2:
        arg = sys.argv[1].lower()
        if arg in ('--auto', '-a', '--all'):
            run_verification()
            generate_plot()
            from aproximacion_universal import run_all_demos
            run_all_demos()
            from aprendizaje_redes import run_all_learning_demos
            run_all_learning_demos()
            return
        elif arg in ('--gui', '-g'):
            launch_gui()
            return
        elif arg in ('--web', '-w'):
            launch_web()
            return
        elif arg in ('--universal', '-u'):
            launch_universal_approx()
            return
        elif arg in ('--learning', '-l', '--aprendizaje'):
            launch_learning_module()
            return

    # Menú Interactivo en Consola Organizado por Unidades
    while True:
        print("\n" + "=" * 74)
        print(" SUITE EDUCATIVA DE REDES NEURONALES — DOCTORADO UCN")
        print(" Unidad I: Aproximación Universal  |  Unidad II: Aprendizaje de Redes")
        print("=" * 74)
        print(" Seleccione una opción:")
        print("   --- UNIDAD I: REPRESENTACIÓN Y APROXIMACIÓN UNIVERSAL ---")
        print("   [1] Ingresar datos manuales para el Perceptrón Multicapa XOR (X e Y)")
        print("   [2] Ver comprobación completa de la tabla de verdad XOR")
        print("   [3] Generar gráfico estático de frontera XOR (.png)")
        print("   [4] 🎓 Laboratorio Consola Unidad I: Aproximación Universal (DNF, Polígonos, Pulsos)")
        print("   --- UNIDAD II: APRENDIZAJE DE REDES NEURONALES ---")
        print("   [5] ⚡ Laboratorio Consola Unidad II: Aprendizaje (Rosenblatt, ADALINE, Softmax, KL)")
        print("   --- INTERFACES GRÁFICAS INTERACTIVAS ---")
        print("   [6] 🌐 Abrir Suite Web Interactiva Completa (2 Unidades • 9 Laboratorios)")
        print("   [7] 🖥️  Abrir Interfaz Gráfica de Escritorio XOR (Tkinter)")
        print("   [8] Salir")
        print("-" * 74)
        
        try:
            opc = input("Ingrese opción [1-8]: ").strip()
            if opc == '1':
                run_manual_mode(mlp)
            elif opc == '2':
                run_verification()
            elif opc == '3':
                generate_plot()
            elif opc == '4':
                launch_universal_approx()
            elif opc == '5':
                launch_learning_module()
            elif opc == '6':
                launch_web()
            elif opc == '7':
                launch_gui()
            elif opc in ('8', 'q', 'salir', 'exit'):
                print("Saliendo. ¡Hasta luego!")
                break
            else:
                print("⚠️ Opción no válida. Por favor elija un número del 1 al 8.")
        except (KeyboardInterrupt, EOFError):
            print("\nSaliendo. ¡Hasta luego!")
            break


if __name__ == "__main__":
    main()
