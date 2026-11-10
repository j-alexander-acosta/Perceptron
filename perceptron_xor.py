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


def launch_gradient_module():
    """
    Ejecuta el laboratorio interactivo de la Unidad III: Descenso por Gradiente.
    """
    try:
        from descenso_gradiente import interactive_gradient_menu
        interactive_gradient_menu()
    except Exception as e:
        print(f"⚠️ Error al cargar el módulo de Descenso por Gradiente: {e}")


def launch_backprop_module():
    """
    Ejecuta el laboratorio interactivo de la Unidad IV: Algoritmo de Backpropagation.
    """
    try:
        from backpropagation import interactive_backprop_menu
        interactive_backprop_menu()
    except Exception as e:
        print(f"⚠️ Error al cargar el módulo de Backpropagation: {e}")


def launch_convergence_module():
    """
    Ejecuta el laboratorio interactivo de la Unidad V: Problemas de Convergencia.
    """
    try:
        from problemas_convergencia import interactive_menu
        interactive_menu()
    except Exception as e:
        print(f"⚠️ Error al cargar el módulo de Problemas de Convergencia: {e}")


def launch_intro_module():
    """
    Ejecuta el laboratorio interactivo de la Unidad CD01 (primera en orden): Introducción al Curso.
    """
    try:
        from introduccion_curso import menu_interactivo
        menu_interactivo()
    except Exception as e:
        print(f"⚠️ Error al cargar el módulo CD01 de Introducción al Curso: {e}")


def launch_eda_module():
    """
    Ejecuta el laboratorio interactivo de la Unidad CD02 (segunda en orden): Preprocesamiento y EDA.
    """
    try:
        from preprocesamiento_eda import menu_interactivo
        menu_interactivo()
    except Exception as e:
        print(f"⚠️ Error al cargar el módulo CD02 de Preprocesamiento y EDA: {e}")


def launch_viz_module():
    """
    Ejecuta el laboratorio interactivo de la Unidad CD03 (tercera en orden): Visualización Efectiva y Engañosa.
    """
    try:
        from visualizacion_datos import menu_interactivo
        menu_interactivo()
    except Exception as e:
        print(f"⚠️ Error al cargar el módulo CD03 de Visualización de Datos: {e}")


def launch_ml_module():
    """
    Ejecuta el laboratorio interactivo de la Unidad CD04 (cuarta en orden): Introducción a Machine Learning.
    """
    try:
        from introduccion_ml import menu_interactivo
        menu_interactivo()
    except Exception as e:
        print(f"⚠️ Error al cargar el módulo CD04 de Introducción a Machine Learning: {e}")


def launch_gen_module():
    """
    Ejecuta el laboratorio interactivo de la Unidad CD05 (quinta en orden): Generalización y Validación.
    """
    try:
        from generalizacion_validacion import menu_interactivo
        menu_interactivo()
    except Exception as e:
        print(f"⚠️ Error al cargar el módulo CD05 de Generalización y Validación: {e}")


def launch_clustering_module():
    """
    Ejecuta el laboratorio interactivo de la Unidad CD06 (sexta en orden): Clustering (Agrupamiento no supervisado).
    """
    try:
        from clustering import menu_clustering
        menu_clustering()
    except Exception as e:
        print(f"⚠️ Error al cargar el módulo CD06 de Clustering: {e}")


def launch_dim_reduction_module():
    """
    Ejecuta el laboratorio interactivo de la Unidad CD07 (séptima en orden): Reducción de Dimensionalidad.
    """
    try:
        from reduccion_dimensionalidad import menu_reduccion_dim
        menu_reduccion_dim()
    except Exception as e:
        print(f"⚠️ Error al cargar el módulo CD07 de Reducción de Dimensionalidad: {e}")


def launch_sgd_module():
    """
    Ejecuta el laboratorio interactivo de la Unidad VI (CD18): Descenso por Gradiente Estocástico y Lotes.
    """
    try:
        from sgd_lotes import interactive_sgd_menu
        interactive_sgd_menu()
    except Exception as e:
        print(f"⚠️ Error al cargar el módulo de SGD y Lotes: {e}")


# =============================================================================
# ESTRUCTURA DE UNIDADES DE CÁTEDRA Y NAVEGACIÓN SECUENCIAL
# =============================================================================

UNIDADES_CATEDRA = [
    {
        "id": "CD01",
        "fase": 1,
        "num": 1,
        "nombre": "CD01: Introducción al Curso de Ciencia de Datos",
        "desc": "Pandemia, SIR (Diap. 13), Netflix Prize, 3 Relatos IA y LLMs (63 Diaps.)",
        "launcher": launch_intro_module,
    },
    {
        "id": "CD02",
        "fase": 1,
        "num": 2,
        "nombre": "CD02: Preprocesamiento y Análisis Exploratorio de Datos",
        "desc": "Gallup vs Yelp, CRISP-DM, Pandas epl_mini (23 jug.) y Anscombe (74 Diaps.)",
        "launcher": launch_eda_module,
    },
    {
        "id": "CD03",
        "fase": 1,
        "num": 3,
        "nombre": "CD03: Visualización de Datos Efectiva y Engañosa",
        "desc": "Datasaurus, Burtin 1951 (16 bact.), Tufte Lie Factor, Stevens, Correll y Mirages (103 Diaps.)",
        "launcher": launch_viz_module,
    },
    {
        "id": "CD04",
        "fase": 1,
        "num": 4,
        "nombre": "CD04: Introducción a Machine Learning",
        "desc": "Mapa IA, Tareas, Ceres, Regresión Lineal, Hinge, Logística y Riesgo Empírico (56 Diaps.)",
        "launcher": launch_ml_module,
    },
    {
        "id": "CD05",
        "fase": 1,
        "num": 5,
        "nombre": "CD05: Generalización y Validación",
        "desc": "Sobreajuste, L1/L2 Regularización, Cauchy-Schwarz, K-Fold CV vs Simple, Sesgo-Varianza (63 Diaps.)",
        "launcher": launch_gen_module,
    },
    {
        "id": "CD06",
        "fase": 1,
        "num": 6,
        "nombre": "CD06: Clustering (Agrupamiento no Supervisado)",
        "desc": "K-Means (Lloyd 1D/2D), Rand Index & ARI, Jerárquico/Dendrogramas, DBSCAN y HDBSCAN (57 Diaps.)",
        "launcher": launch_clustering_module,
    },
    {
        "id": "CD07",
        "fase": 1,
        "num": 7,
        "nombre": "CD07: Reducción de Dimensionalidad (PCA, Fisher LDA, t-SNE, UMAP)",
        "desc": "Maldición Dimensional, PCA a Mano (6 pts), Fisher LDA (8 pts), t-SNE y UMAP (73 Diaps.)",
        "launcher": launch_dim_reduction_module,
    },
    {
        "id": "CD13",
        "fase": 2,
        "num": 8,
        "nombre": "CD13 (Unidad I): Aproximación Universal con Redes Neuronales",
        "desc": "DNF, Karnaugh, Paridad N-aria, Polígonos y Pulsos (61 Diaps.)",
        "launcher": launch_universal_approx,
    },
    {
        "id": "CD14",
        "fase": 2,
        "num": 9,
        "nombre": "CD14 (Unidad II): Aprendizaje de Redes Neuronales",
        "desc": "Rosenblatt (Diap. 22), ADALINE, Sigmoide, Softmax y Entropía Cruzada KL (56 Diaps.)",
        "launcher": launch_learning_module,
    },
    {
        "id": "CD15",
        "fase": 2,
        "num": 10,
        "nombre": "CD15 (Unidad III): Descenso por Gradiente y Convergencia",
        "desc": "Ortogonalidad, Hessiana 3D, 200 Épocas y Relación Paso/Curvatura (47 Diaps.)",
        "launcher": launch_gradient_module,
    },
    {
        "id": "CD16",
        "fase": 2,
        "num": 11,
        "nombre": "CD16 (Unidad IV): Algoritmo de Backpropagation",
        "desc": "Regla de la Cadena, Grafos de Cómputo (Diaps. 21/23), ReLU 2 Capas y Jacobianos (60 Diaps.)",
        "launcher": launch_backprop_module,
    },
    {
        "id": "CD17",
        "fase": 2,
        "num": 12,
        "nombre": "CD17 (Unidad V): Problemas de Convergencia en el Entrenamiento",
        "desc": "Brady 1989, 6 Trayectorias, Condición κ, Newton, RProp y QuickProp (58 Diaps.)",
        "launcher": launch_convergence_module,
    },
    {
        "id": "CD18",
        "fase": 2,
        "num": 13,
        "nombre": "CD18 (Unidad VI): Descenso por Gradiente Estocástico y Lotes",
        "desc": "SGD a mano (Diap. 14), Robbins-Monro, Varianza 1/b, Momentum, RMSprop y Adam (41 Diaps.)",
        "launcher": launch_sgd_module,
    },
]


def ejecutar_unidad_con_navegacion(unit_idx: int):
    """
    Ejecuta el laboratorio de una unidad y ofrece una barra de navegación interactiva
    (Siguiente / Anterior / Repetir / Volver) para avanzar secuencialmente.
    """
    total = len(UNIDADES_CATEDRA)
    idx = unit_idx
    
    while 0 <= idx < total:
        u = UNIDADES_CATEDRA[idx]
        print("\n" + "═" * 84)
        print(f" ▶ INICIANDO: {u['nombre']}")
        print(f"   {u['desc']}")
        print("═" * 84)
        
        # Ejecutar módulo
        try:
            u["launcher"]()
        except Exception as e:
            print(f"⚠️ Error al ejecutar {u['id']}: {e}")
        
        # Barra de Navegación Secuencial
        has_prev = idx > 0
        has_next = idx < total - 1
        prev_name = UNIDADES_CATEDRA[idx - 1]["id"] if has_prev else ""
        next_name = UNIDADES_CATEDRA[idx + 1]["id"] if has_next else ""
        
        print("\n" + "─" * 84)
        print(f" 📍 Unidad actual: [{u['id']}] ({idx + 1} de {total})")
        nav_opts = []
        if has_next:
            nav_opts.append(f"[S] Siguiente unidad ({next_name})")
        if has_prev:
            nav_opts.append(f"[A] Anterior unidad ({prev_name})")
        nav_opts.append(f"[R] Repetir ({u['id']})")
        nav_opts.append("[V] Volver al menú")
        print("   " + "  |  ".join(nav_opts))
        print("─" * 84)
        
        try:
            choice = input("Seleccione acción de navegación: ").strip().lower()
            if choice in ('s', 'sig', 'siguiente') and has_next:
                idx += 1
            elif choice in ('a', 'ant', 'anterior') and has_prev:
                idx -= 1
            elif choice in ('r', 'repetir'):
                continue
            elif choice in ('v', 'volver', 'm', 'menu', 'q', 'salir', ''):
                break
            else:
                print("Opción no reconocida. Volviendo al menú.")
                break
        except (KeyboardInterrupt, EOFError):
            print("\nNavegación cancelada.")
            break


def menu_fase1():
    """
    Submenú enfocado y limpio para la Fase 1: Fundamentos y Análisis de Datos (CD01 a CD07).
    """
    while True:
        print("\n" + "═" * 84)
        print(" 📊 FASE 1: FUNDAMENTOS Y ANÁLISIS DE DATOS (CD01 a CD07 • 7 Unidades)")
        print("═" * 84)
        print(" [1] CD01: Introducción al Curso (Pandemia, SIR, Netflix Prize, 3 Relatos IA)")
        print(" [2] CD02: Preprocesamiento y EDA (Gallup/Yelp, CRISP-DM, epl_mini, Anscombe)")
        print(" [3] CD03: Visualización Efectiva y Engañosa (Datasaurus, Burtin, Tufte, Correll)")
        print(" [4] CD04: Introducción a Machine Learning (Ceres, Hinge, Logística, Softmax)")
        print(" [5] CD05: Generalización y Validación (Sobreajuste, L1/L2, K-Fold CV, Sesgo-Varianza)")
        print(" [6] CD06: Clustering (K-Means Lloyd 1D/2D, Rand/ARI, Jerárquico, DBSCAN/HDBSCAN)")
        print(" [7] CD07: Reducción de Dimensionalidad (Maldición Dim, PCA 6 pts, Fisher LDA, t-SNE/UMAP)")
        print(" [T] Ejecutar todas las demostraciones automáticas de Fase 1")
        print(" [V] Volver al Menú Principal")
        print("─" * 84)
        
        try:
            opc = input("Fase 1 [1-7, T, V]: ").strip().lower()
            if opc == '1':
                ejecutar_unidad_con_navegacion(0)
            elif opc == '2':
                ejecutar_unidad_con_navegacion(1)
            elif opc == '3':
                ejecutar_unidad_con_navegacion(2)
            elif opc == '4':
                ejecutar_unidad_con_navegacion(3)
            elif opc == '5':
                ejecutar_unidad_con_navegacion(4)
            elif opc == '6':
                ejecutar_unidad_con_navegacion(5)
            elif opc == '7':
                ejecutar_unidad_con_navegacion(6)
            elif opc in ('t', 'todos', 'all'):
                from introduccion_curso import ejecutar_suite_completa
                ejecutar_suite_completa()
                from preprocesamiento_eda import ejecutar_todas_las_demos
                ejecutar_todas_las_demos()
                from visualizacion_datos import ejecutar_todo_cd03
                ejecutar_todo_cd03()
                from introduccion_ml import ejecutar_todas_las_demos as ejecutar_todo_cd04
                ejecutar_todo_cd04()
                from generalizacion_validacion import ejecutar_todas_las_demos as ejecutar_todo_cd05
                ejecutar_todo_cd05()
                from clustering import ejecutar_suite_automatica as ejecutar_todo_cd06
                ejecutar_todo_cd06()
                from reduccion_dimensionalidad import ejecutar_suite_automatica as ejecutar_todo_cd07
                ejecutar_todo_cd07()
            elif opc in ('v', 'volver', 'q', 'salir', '0', ''):
                break
            else:
                print("⚠️ Opción no válida en Fase 1.")
        except (KeyboardInterrupt, EOFError):
            break


def menu_fase2():
    """
    Submenú enfocado y limpio para la Fase 2: Redes Neuronales y Optimización (CD13 a CD18).
    """
    while True:
        print("\n" + "═" * 84)
        print(" 🧠 FASE 2: REDES NEURONALES Y OPTIMIZACIÓN (CD13 a CD18 • 6 Unidades)")
        print("═" * 84)
        print(" [1] CD13 (Unidad I)  : Aproximación Universal (DNF, Karnaugh, Polígonos, Pulsos)")
        print(" [2] CD14 (Unidad II) : Aprendizaje de Redes (Rosenblatt, ADALINE, Softmax, KL)")
        print(" [3] CD15 (Unidad III): Descenso por Gradiente (Hessiana 3D, 200 Épocas, Paso)")
        print(" [4] CD16 (Unidad IV) : Backpropagation (Regla Cadena, Grafos, ReLU 2 Capas)")
        print(" [5] CD17 (Unidad V)  : Problemas de Convergencia (Brady 1989, Newton, RProp)")
        print(" [6] CD18 (Unidad VI) : SGD y Lotes (SGD a Mano, Robbins-Monro, Adam, RMSprop)")
        print(" [T] Ejecutar todas las demostraciones automáticas de Fase 2")
        print(" [V] Volver al Menú Principal")
        print("─" * 84)
        
        try:
            opc = input("Fase 2 [1-6, T, V]: ").strip().lower()
            if opc == '1':
                ejecutar_unidad_con_navegacion(7)
            elif opc == '2':
                ejecutar_unidad_con_navegacion(8)
            elif opc == '3':
                ejecutar_unidad_con_navegacion(9)
            elif opc == '4':
                ejecutar_unidad_con_navegacion(10)
            elif opc == '5':
                ejecutar_unidad_con_navegacion(11)
            elif opc == '6':
                ejecutar_unidad_con_navegacion(12)
            elif opc in ('t', 'todos', 'all'):
                from aproximacion_universal import run_all_demos
                run_all_demos()
                from aprendizaje_redes import run_all_learning_demos
                run_all_learning_demos()
                from descenso_gradiente import run_all_gradient_demos
                run_all_gradient_demos()
                from backpropagation import run_all_backprop_demos
                run_all_backprop_demos()
                from problemas_convergencia import run_all_convergence_demos
                run_all_convergence_demos()
                from sgd_lotes import run_all_sgd_demos
                run_all_sgd_demos()
            elif opc in ('v', 'volver', 'q', 'salir', '0', ''):
                break
            else:
                print("⚠️ Opción no válida en Fase 2.")
        except (KeyboardInterrupt, EOFError):
            break


def menu_xor_lab(mlp):
    """
    Submenú dedicado al experimento clásico del Perceptrón Multicapa XOR.
    """
    while True:
        print("\n" + "═" * 84)
        print(" ⚡ LABORATORIO CLÁSICO XOR — PERCEPTRÓN MULTICAPA (MLP 2 CAPAS)")
        print("═" * 84)
        print(" [1] Ingresar datos manuales (X e Y ∈ {0, 1}) y ver forward pass paso a paso")
        print(" [2] Ver comprobación completa de los 4 estados discretos {0, 1}²")
        print(" [3] Generar y guardar gráfico estático de frontera de decisión XOR (.png)")
        print(" [V] Volver al Menú Principal")
        print("─" * 84)
        
        try:
            opc = input("Laboratorio XOR [1-3, V]: ").strip().lower()
            if opc == '1':
                run_manual_mode(mlp)
            elif opc == '2':
                run_verification()
            elif opc == '3':
                generate_plot()
            elif opc in ('v', 'volver', 'q', 'salir', '0', ''):
                break
            else:
                print("⚠️ Opción no válida en Laboratorio XOR.")
        except (KeyboardInterrupt, EOFError):
            break


def menu_interfaces():
    """
    Submenú para el lanzamiento de interfaces gráficas (Web y Desktop).
    """
    while True:
        print("\n" + "═" * 84)
        print(" 🌐 INTERFACES GRÁFICAS INTERACTIVAS")
        print("═" * 84)
        print(" [1] 🌐 Suite Web Interactiva (10 Unidades • 41 Laboratorios en Navegador)")
        print(" [2] 🖥️  Interfaz Gráfica de Escritorio XOR (Tkinter)")
        print(" [V] Volver al Menú Principal")
        print("─" * 84)
        
        try:
            opc = input("Interfaces [1-2, V]: ").strip().lower()
            if opc == '1':
                launch_web()
            elif opc == '2':
                launch_gui()
            elif opc in ('v', 'volver', 'q', 'salir', '0', ''):
                break
            else:
                print("⚠️ Opción no válida en Interfaces Gráficas.")
        except (KeyboardInterrupt, EOFError):
            break


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
            from introduccion_curso import ejecutar_suite_completa
            ejecutar_suite_completa()
            from preprocesamiento_eda import ejecutar_todas_las_demos
            ejecutar_todas_las_demos()
            from visualizacion_datos import ejecutar_todo_cd03
            ejecutar_todo_cd03()
            from introduccion_ml import ejecutar_todas_las_demos as ejecutar_todo_cd04
            ejecutar_todo_cd04()
            from generalizacion_validacion import ejecutar_todas_las_demos as ejecutar_todo_cd05
            ejecutar_todo_cd05()
            from clustering import ejecutar_suite_automatica as ejecutar_todo_cd06
            ejecutar_todo_cd06()
            from reduccion_dimensionalidad import ejecutar_suite_automatica as ejecutar_todo_cd07
            ejecutar_todo_cd07()
            run_verification()
            generate_plot()
            from aproximacion_universal import run_all_demos
            run_all_demos()
            from aprendizaje_redes import run_all_learning_demos
            run_all_learning_demos()
            from descenso_gradiente import run_all_gradient_demos
            run_all_gradient_demos()
            from backpropagation import run_all_backprop_demos
            run_all_backprop_demos()
            from problemas_convergencia import run_all_convergence_demos
            run_all_convergence_demos()
            from sgd_lotes import run_all_sgd_demos
            run_all_sgd_demos()
            return
        elif arg in ('--intro', '-i', '--cd01'):
            ejecutar_unidad_con_navegacion(0)
            return
        elif arg in ('--eda', '-e', '--cd02'):
            ejecutar_unidad_con_navegacion(1)
            return
        elif arg in ('--viz', '-v', '--cd03'):
            ejecutar_unidad_con_navegacion(2)
            return
        elif arg in ('--ml', '-m', '--cd04'):
            ejecutar_unidad_con_navegacion(3)
            return
        elif arg in ('--gen', '-g5', '--cd05'):
            ejecutar_unidad_con_navegacion(4)
            return
        elif arg in ('--cluster', '-c6', '--cd06', '--kmeans', '--dbscan'):
            ejecutar_unidad_con_navegacion(5)
            return
        elif arg in ('--dim', '-c7', '--cd07', '--pca', '--lda', '--tsne', '--umap'):
            ejecutar_unidad_con_navegacion(6)
            return
        elif arg in ('--gui', '-g'):
            launch_gui()
            return
        elif arg in ('--web', '-w'):
            launch_web()
            return
        elif arg in ('--universal', '-u', '--cd13'):
            ejecutar_unidad_con_navegacion(7)
            return
        elif arg in ('--learning', '-l', '--aprendizaje', '--cd14'):
            ejecutar_unidad_con_navegacion(8)
            return
        elif arg in ('--gradient', '-d', '--gradiente', '--cd15'):
            ejecutar_unidad_con_navegacion(9)
            return
        elif arg in ('--backprop', '-b', '--backpropagation', '--cd16'):
            ejecutar_unidad_con_navegacion(10)
            return
        elif arg in ('--convergence', '-c', '--convergencia', '--cd17'):
            ejecutar_unidad_con_navegacion(11)
            return
        elif arg in ('--sgd', '-s', '--lotes', '--cd18'):
            ejecutar_unidad_con_navegacion(12)
            return

    # Menú Principal Organizado, Desaturado y con Navegación Jerárquica
    while True:
        print("\n" + "═" * 84)
        print("  CIENCIA DE DATOS Y REDES NEURONALES — SUITE ACADÉMICA (DOCTORADO UCN)")
        print("  13 Unidades de Cátedra (812 Diapositivas) • 53 Laboratorios Interactivos")
        print("═" * 84)
        print(" [1] 📊 FASE 1: Fundamentos y Análisis de Datos (CD01 a CD07)")
        print("     • CD01: Intro  • CD02: EDA  • CD03: Viz  • CD04: ML  • CD05: Gen.  • CD06: Cluster  • CD07: Reducción Dim")
        print("")
        print(" [2] 🧠 FASE 2: Redes Neuronales y Optimización (CD13 a CD18)")
        print("     • CD13: Aprox. Univ.  • CD14: Aprendizaje  • CD15: Gradiente  • CD16: Backprop")
        print("     • CD17: Convergencia  • CD18: SGD y Lotes")
        print("")
        print(" [3] ⚡ LABORATORIO CLÁSICO XOR: Perceptrón Multicapa (MLP 2 Capas)")
        print("     • Modo Manual (X, Y)  • Tabla de Verdad  • Gráfica de Frontera (.png)")
        print("")
        print(" [4] 🌐 INTERFACES GRÁFICAS: Suite Web (Navegador) & GUI de Escritorio (Tkinter)")
        print("")
        print(" [5] 🚀 VERIFICACIÓN COMPLETA AUTOMATIZADA (--auto, 13 Unidades)")
        print("")
        print(" [0] Salir del Sistema")
        print("─" * 84)
        
        try:
            opc = input("Seleccione opción [1-5, 0]: ").strip().lower()
            if opc == '1':
                menu_fase1()
            elif opc == '2':
                menu_fase2()
            elif opc == '3':
                menu_xor_lab(mlp)
            elif opc == '4':
                menu_interfaces()
            elif opc == '5':
                from introduccion_curso import ejecutar_suite_completa
                ejecutar_suite_completa()
                from preprocesamiento_eda import ejecutar_todas_las_demos
                ejecutar_todas_las_demos()
                from visualizacion_datos import ejecutar_todo_cd03
                ejecutar_todo_cd03()
                from introduccion_ml import ejecutar_todas_las_demos as ejecutar_todo_cd04
                ejecutar_todo_cd04()
                from generalizacion_validacion import ejecutar_todas_las_demos as ejecutar_todo_cd05
                ejecutar_todo_cd05()
                from clustering import ejecutar_suite_automatica as ejecutar_todo_cd06
                ejecutar_todo_cd06()
                from reduccion_dimensionalidad import ejecutar_suite_automatica as ejecutar_todo_cd07
                ejecutar_todo_cd07()
                run_verification()
                generate_plot()
                from aproximacion_universal import run_all_demos
                run_all_demos()
                from aprendizaje_redes import run_all_learning_demos
                run_all_learning_demos()
                from descenso_gradiente import run_all_gradient_demos
                run_all_gradient_demos()
                from backpropagation import run_all_backprop_demos
                run_all_backprop_demos()
                from problemas_convergencia import run_all_convergence_demos
                run_all_convergence_demos()
                from sgd_lotes import run_all_sgd_demos
                run_all_sgd_demos()
            elif opc in ('0', 'q', 'salir', 'exit'):
                print("Saliendo de la Suite Académica. ¡Hasta luego!")
                break
            # Atajos directos por compatibilidad
            elif opc in ('cd01', 'intro'):
                ejecutar_unidad_con_navegacion(0)
            elif opc in ('cd02', 'eda'):
                ejecutar_unidad_con_navegacion(1)
            elif opc in ('cd03', 'viz'):
                ejecutar_unidad_con_navegacion(2)
            elif opc in ('cd04', 'ml'):
                ejecutar_unidad_con_navegacion(3)
            elif opc in ('cd05', 'gen', 'val'):
                ejecutar_unidad_con_navegacion(4)
            elif opc in ('cd06', 'cluster', 'kmeans', 'dbscan'):
                ejecutar_unidad_con_navegacion(5)
            elif opc in ('cd07', 'pca', 'lda', 'tsne', 'umap', 'dim'):
                ejecutar_unidad_con_navegacion(6)
            elif opc in ('cd13', 'universal'):
                ejecutar_unidad_con_navegacion(7)
            elif opc in ('cd14', 'learning'):
                ejecutar_unidad_con_navegacion(8)
            elif opc in ('cd15', 'gradient'):
                ejecutar_unidad_con_navegacion(9)
            elif opc in ('cd16', 'backprop'):
                ejecutar_unidad_con_navegacion(10)
            elif opc in ('cd17', 'convergence'):
                ejecutar_unidad_con_navegacion(11)
            elif opc in ('cd18', 'sgd'):
                ejecutar_unidad_con_navegacion(12)
            elif opc == 'web':
                launch_web()
            elif opc == 'gui':
                launch_gui()
            else:
                print("⚠️ Opción no válida. Por favor elija un número del 0 al 5.")
        except (KeyboardInterrupt, EOFError):
            print("\nSaliendo de la Suite Académica. ¡Hasta luego!")
            break


if __name__ == "__main__":
    main()
