#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
CD12: REDES NEURONALES Y CONEXIONISMO (60 DIAPOSITIVAS)
Suite Educativa de Ciencia de Datos — Doctorado en Informática, UCN
Basado en Stanford CS221 (P. Liang et al.) y CMU 11-785 (B. Raj et al.)
===============================================================================

Este módulo implementa los fundamentos conceptuales, históricos y matemáticos
del surgimiento de las Redes Neuronales y el Paradigma Conexionista:
  12.1 Asociacionismo, Alexander Bain (1873), Escala Biológica vs Estimación,
       Máquinas de Turing Tipo A/B y los 8 Ingredientes de un Sistema PDP.
  12.2 La Neurona Lógica de McCulloch-Pitts (1943), Regla de Hebb (1949),
       Diagnóstico de Inestabilidad sin Cota y Estabilización con Regla de Oja (1982).
  12.3 El Perceptrón de Rosenblatt (1958), Aprendizaje de OR, Demostración
       Formal de Inconsistencia de XOR y Solución con MLP (3 Neuronas).
  12.4 Clasificador y Regresor Universal: Dos Triángulos Disjuntos (9 Neuronas),
       Pulsos de Riemann, Teorema de Tamaño vs Activación (cos(2x) con 1 neurona),
       y la Línea de Tiempo del Conexionismo (1943 a 2016).
"""

import math
import sys
import time


# =============================================================================
# MÓDULO 12.1: ASOCIACIONISMO, ALEXANDER BAIN Y MÁQUINAS CONEXIONISTAS
# =============================================================================

def info_leyes_asociacionismo():
    """
    Retorna la síntesis histórica del Asociacionismo (400 a.C. a 1900)
    y las 4 Leyes Fundamentales de Aristóteles.
    """
    return {
        "origen": "Aristóteles (~360 a.C.), Locke, Hume, los Mill, Bain, Pavlov",
        "postulado": "Pares de ideas quedan asociados según la experiencia pasada del organismo.",
        "leyes": [
            ("Contigüidad", "Cosas que ocurren juntas en el espacio o tiempo quedan enlazadas (ej. relámpago y trueno)."),
            ("Frecuencia", "Mientras más veces se enlazan dos cosas, más fuerte es la asociación. (Semilla de Hebb)."),
            ("Similitud", "Si dos cosas se parecen, percibir una evoca la otra."),
            ("Contraste", "Percibir un estímulo puede evocar de inmediato su opuesto.")
        ]
    }


def comparativa_escala_bain():
    """
    Compara las estimaciones de Alexander Bain (1873, Mind and Body)
    con la neurobiología moderna cuantitativa.
    """
    bain_1873 = {
        "neuronas": 1_000_000,           # 1 millón
        "conexiones": 5_000_000_000      # 5 mil millones (5e9)
    }
    cerebro_real = {
        "neuronas": 86_000_000_000,      # ~86 mil millones (8.6e10)
        "conexiones": 100_000_000_000_000 # ~100 billones (1e14)
    }
    factor_conexiones = cerebro_real["conexiones"] / bain_1873["conexiones"]
    return {
        "bain_1873": bain_1873,
        "cerebro_real": cerebro_real,
        "factor_subestimacion": factor_conexiones, # 20,000x
        "conclusion": "La idea de Bain era conceptualmente correcta; su error fue de escala."
    }


def ingredientes_sistema_pdp():
    """
    Retorna los 8 ingredientes de un sistema de Procesamiento Distribuido
    en Paralelo (PDP, Rumelhart, Hinton & McClelland, 1986).
    """
    return [
        "1. Conjunto de unidades de procesamiento (neuronas/nodos)",
        "2. Estado de activación de cada unidad",
        "3. Función de salida para cada unidad",
        "4. Patrón de conectividad entre unidades (matriz de pesos W)",
        "5. Regla de propagación de las activaciones",
        "6. Regla de activación (función no lineal)",
        "7. Regla de aprendizaje que ajusta las conexiones",
        "8. Entorno en el que el sistema opera (datos/estímulos externos)"
    ]


def demo_conexionismo_bain():
    print("\n" + "═" * 78)
    print(" 12.1 ASOCIACIONISMO, CONEXIONISMO Y LA ESCALA BIOLÓGICA DE BAIN")
    print("═" * 78)
    print("  Tesis Central de Alexander Bain (1873, 'Mind and Body'):")
    print("  «La información NO está en las unidades: está en las conexiones entre ellas.»\n")

    comp = comparativa_escala_bain()
    bain = comp["bain_1873"]
    real = comp["cerebro_real"]

    print("  ┌─────────────────────────┬──────────────────────┬──────────────────────┐")
    print("  │ Magnitud Biológica      │ Estimación Bain 1873 │ Neurociencia Actual  │")
    print("  ├─────────────────────────┼──────────────────────┼──────────────────────┤")
    print(f"  │ Neuronas totales        │ {bain['neuronas']:>20,d} │ {real['neuronas']:>20,d} │")
    print(f"  │ Conexiones (sinapsis)   │ {bain['conexiones']:>20,d} │ {real['conexiones']:>20,d} │")
    print("  └─────────────────────────┴──────────────────────┴──────────────────────┘")
    print(f"  • Factor de escala: El cerebro real tiene {comp['factor_subestimacion']:,.0f}x MÁS conexiones")
    print("    que la cifra que llevó a Alexander Bain a retractarse al final de su vida.\n")

    print("  Máquinas de Turing (1948, 'Intelligent Machinery'):")
    print("  • Tipo A: Redes aleatorias no entrenables de compuertas NAND ('desorganizadas').")
    print("  • Tipo B: Redes donde cada conexión posee un modificador entrenable/adaptativo.")
    print("    Turing separó la arquitectura fija de la regla de aprendizaje de pesos.\n")

    print("  Los 8 Ingredientes de un Sistema PDP (Rumelhart, Hinton & McClelland, 1986):")
    for ing in ingredientes_sistema_pdp():
        print(f"    {ing}")


# =============================================================================
# MÓDULO 12.2: NEURONA DE McCULLOCH-PITTS (1943) Y REGLA DE HEBB (1949)
# =============================================================================

class NeuronaMcCullochPitts:
    """
    Implementación del modelo matemático original de McCulloch & Pitts (1943):
    - Entradas excitatorias con pesos w_i y umbral theta.
    - Sinapsis inhibitoria absoluta: si se activa, bloquea el disparo de forma absoluta.
    """
    def __init__(self, pesos, umbral):
        self.pesos = list(pesos)
        self.umbral = float(umbral)

    def evaluar(self, entradas, inhibitoria=False):
        if inhibitoria:
            return 0  # Inhibición absoluta
        suma = sum(w * x for w, x in zip(self.pesos, entradas))
        return 1 if suma >= self.umbral else 0


def simular_regla_hebb(n_presentaciones=1000, eta=0.1):
    """
    Simulación de la regla hebbiana pura: w_{xy} <- w_{xy} + eta * x * y
    Con dos unidades continuamente coactivas (x=1, y=1) y w_0 = 0.
    Demuestra la inestabilidad matemática intrínseca (crecimiento sin cota).
    """
    pesos_hebb = []
    w = 0.0
    hitos = [10, 100, 1000]
    registro_hitos = {}

    for t in range(1, n_presentaciones + 1):
        x, y = 1.0, 1.0
        w += eta * x * y
        if t in hitos:
            registro_hitos[t] = w
        if t <= 10 or t % 100 == 0:
            pesos_hebb.append((t, w))

    return registro_hitos, w


def simular_regla_oja(n_presentaciones=1000, eta=0.05):
    """
    Simulación de la Regla de Oja (1982): Delta w = eta * y * (x - y * w)
    con normalización y olvido intrínseco. Estabiliza en ||w|| -> 1.0.
    """
    w = 0.0
    registro_oja = {}
    hitos = [10, 50, 100, 500, 1000]
    for t in range(1, n_presentaciones + 1):
        x = 1.0
        y = w * x  # Salida lineal
        if t == 1 and w == 0.0:
            y = 1.0  # Disparo inicial inducido
        w += eta * y * (x - y * w)
        if t in hitos:
            registro_oja[t] = w
    return registro_oja, w


def demo_mcculloch_pitts_hebb():
    print("\n" + "═" * 78)
    print(" 12.2 NEURONA McCULLOCH-PITTS (1943) Y REGLA DE APRENDIZAJE DE HEBB (1949)")
    print("═" * 78)

    # 1. McCulloch-Pitts compuertas
    and_gate = NeuronaMcCullochPitts([1, 1], umbral=2)
    or_gate = NeuronaMcCullochPitts([1, 1], umbral=1)
    not_gate = NeuronaMcCullochPitts([-1], umbral=0)

    print("  1. Compuertas Lógicas de McCulloch & Pitts (Sinapsis Excitatorias):")
    print("     • AND (w=[1, 1], θ=2):")
    for x in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        print(f"       ({x[0]}, {x[1]}) -> Disparo: {and_gate.evaluar(x)}")
    print("     • OR (w=[1, 1], θ=1):")
    for x in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        print(f"       ({x[0]}, {x[1]}) -> Disparo: {or_gate.evaluar(x)}")
    print("     • NOT (w=[-1], θ=0):")
    for x in [0, 1]:
        print(f"       ({x}) -> Disparo: {not_gate.evaluar([x])}")

    # 2. Hebb y su inestabilidad
    hitos_hebb, final_hebb = simular_regla_hebb(1000, eta=0.1)
    print("\n  2. Regla de Hebb (1949): w_xy <- w_xy + η * x * y")
    print("     «Neuronas que disparan juntas, se conectan».")
    print("     Simulación con x=1, y=1, η=0.1, w_0=0 (Diapositiva 30):")
    print("     ┌──────────────────┬────────────────────┐")
    print("     │ Presentaciones   │ Peso resultante w  │")
    print("     ├──────────────────┼────────────────────┤")
    for t, val in hitos_hebb.items():
        print(f"     │ {t:>16d} │ {val:>18.2f} │")
    print("     └──────────────────┴────────────────────┘")
    print("     ⚠️  Diagnóstico: Inestable por falta de competencia y cota superior.")
    print("         Los pesos crecen indefinidamente sin freno.")

    # 3. Solución Oja
    hitos_oja, final_oja = simular_regla_oja(1000, eta=0.05)
    print(f"\n  3. Estabilización con Regla de Oja (1982): w converge asintóticamente a {final_oja:.4f} (cercano a 1.0).")


# =============================================================================
# MÓDULO 12.3: PERCEPTRÓN DE ROSENBLATT (1958), LÍMITE DE XOR Y MLP
# =============================================================================

class PerceptronRosenblatt:
    """
    Perceptrón binario de Rosenblatt con regla de convergencia:
      w <- w + eta * (d - y) * x
    """
    def __init__(self, n_features, eta=1.0):
        self.w = [0.0] * (n_features + 1)  # w_0 es el sesgo b = -T
        self.eta = eta

    def predict(self, x):
        # x con 1-extendido [1, x1, x2]
        suma = self.w[0] + sum(w_i * x_i for w_i, x_i in zip(self.w[1:], x))
        return 1 if suma >= 0.0 else 0

    def fit(self, X, d, max_epocas=20):
        historial = []
        for epoca in range(max_epocas):
            errores = 0
            for xi, di in zip(X, d):
                yi = self.predict(xi)
                error = di - yi
                if error != 0:
                    self.w[0] += self.eta * error * 1.0
                    for j in range(len(xi)):
                        self.w[j+1] += self.eta * error * xi[j]
                    errores += 1
            historial.append((epoca + 1, list(self.w), errores))
            if errores == 0:
                break
        return historial


def demostracion_imposibilidad_xor():
    """
    Demostración analítica de la contradicción de XOR para 1 neurona (Diaps. 37-38):
    1) x=(0,0) -> y=0: w1*0 + w2*0 - T < 0  => T > 0
    2) x=(1,0) -> y=1: w1*1 + w2*0 - T >= 0 => w1 >= T
    3) x=(0,1) -> y=1: w1*0 + w2*1 - T >= 0 => w2 >= T
    4) x=(1,1) -> y=0: w1*1 + w2*1 - T < 0  => w1 + w2 < T
    Sumando (2) y (3): w1 + w2 >= 2T. Como T > 0, 2T > T => w1 + w2 > T (Contradicción con 4).
    """
    pasos = [
        ("Paso 1", "(0, 0) -> 0", "w1*0 + w2*0 - T < 0", "T > 0"),
        ("Paso 2", "(1, 0) -> 1", "w1*1 + w2*0 - T >= 0", "w1 >= T"),
        ("Paso 3", "(0, 1) -> 1", "w1*0 + w2*1 - T >= 0", "w2 >= T"),
        ("Paso 4", "(1, 1) -> 0", "w1*1 + w2*1 - T < 0", "w1 + w2 < T"),
        ("Suma (2)+(3)", "Adición", "w1 + w2 >= 2T", "Como T > 0, 2T > T => w1 + w2 > T"),
        ("Contradicción", "Fallo Lógico", "w1 + w2 >= 2T > T", "CONTRADICE w1 + w2 < T")
    ]
    return pasos


class MlpXorClasico:
    """
    Perceptrón Multicapa para XOR de 3 neuronas (Diapositiva 39):
    Capa oculta:
      h1 = X or Y = step(1*X + 1*Y - 1)
      h2 = not(X and Y) = step(-1*X - 1*Y - (-1))
    Capa de salida:
      y = h1 and h2 = step(1*h1 + 1*h2 - 2)
    """
    def forward(self, x1, x2):
        # Neurona 1: X or Y (w=[1, 1], T=1)
        h1 = 1 if (1.0 * x1 + 1.0 * x2 - 1.0) >= 0.0 else 0
        # Neurona 2: not(X and Y) = not X or not Y (w=[-1, -1], T=-1)
        h2 = 1 if (-1.0 * x1 + -1.0 * x2 - (-1.0)) >= 0.0 else 0
        # Salida: h1 and h2 (w=[1, 1], T=2)
        y = 1 if (1.0 * h1 + 1.0 * h2 - 2.0) >= 0.0 else 0
        return h1, h2, y


def demo_rosenblatt_xor_mlp():
    print("\n" + "═" * 78)
    print(" 12.3 EL PERCEPTRÓN DE ROSENBLATT (1958), EL LÍMITE DE XOR Y EL MLP")
    print("═" * 78)

    # 1. Entrenamiento OR con Perceptrón
    X_or = [[0, 0], [0, 1], [1, 0], [1, 1]]
    d_or = [0, 1, 1, 1]
    p = PerceptronRosenblatt(n_features=2, eta=1.0)
    hist = p.fit(X_or, d_or, max_epocas=10)

    print("  1. Aprendizaje de la Compuerta OR con la Regla de Rosenblatt (Diap. 34):")
    print(f"     • Convergió en {len(hist)} épocas con pesos finales: b={p.w[0]:.1f}, w1={p.w[1]:.1f}, w2={p.w[2]:.1f}")
    print("       Condición de disparo: 1.0*x1 + 1.0*x2 >= 1.0 (Exactamente la compuerta OR).\n")

    # 2. Imposibilidad de XOR
    print("  2. Demostración Formal de la Imposibilidad de XOR con 1 Neurona (Minsky & Papert 1969):")
    for paso, caso, ineq, ded in demostracion_imposibilidad_xor():
        print(f"     [{paso}] {caso:<14} : {ineq:<24} => {ded}")
    print("     => Ningún hiperplano lineal en R^2 puede separar {(0,1), (1,0)} de {(0,0), (1,1)}.\n")

    # 3. Solución con MLP
    mlp = MlpXorClasico()
    print("  3. Solución con Perceptrón Multicapa (MLP de 2 Capas Ocultas, Diap. 39):")
    print("     X ⊕ Y = (X ∨ Y) ∧ (not X ∨ not Y)")
    print("     ┌───────┬───────┬────────────────┬────────────────┬────────────────┐")
    print("     │   X   │   Y   │  h1 (X ∨ Y)    │  h2 (¬X ∨ ¬Y)  │  Salida XOR    │")
    print("     ├───────┼───────┼────────────────┼────────────────┼────────────────┤")
    for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        h1, h2, y = mlp.forward(x1, x2)
        print(f"     │   {x1}   │   {x2}   │       {h1}        │       {h2}        │       {y}        │")
    print("     └───────┴───────┴────────────────┴────────────────┴────────────────┘")
    print("     ✔ ¡El perceptrón multicapa resuelve XOR con 100% de exactitud!")


# =============================================================================
# MÓDULO 12.4: CLASIFICADOR Y REGRESOR UNIVERSAL, Y LÍNEA DE TIEMPO
# =============================================================================

class ClasificadorDosTriangulos:
    """
    Clasificador de región no conexa / no convexa: Dos Triángulos Disjuntos (Diap. 51).
    Cada triángulo: 3 semiplanos lineales + 1 AND = 4 neuronas.
    Dos triángulos: 8 neuronas en capas 1 y 2.
    Capa 3: 1 neurona OR que activa si cualquiera de los dos triángulos dispara.
    Total: 9 neuronas (demuestra la necesidad de profundidad para unir convexos).
    """
    def __init__(self):
        # Triángulo 1 (T1) en semiplano inferior izquierdo:
        # Vértices aprox: (0, 0), (1, 0), (0.5, 1)
        # Rectas: x2 >= 0, x2 <= 2*x1, x2 <= 2*(1 - x1)
        # Triángulo 2 (T2) en semiplano superior derecho:
        # Vértices aprox: (2, 2), (3, 2), (2.5, 3)
        pass

    def evaluar_punto(self, x1, x2):
        # T1: 3 semiplanos
        l1_1 = 1 if x2 >= 0.0 else 0
        l1_2 = 1 if (2.0 * x1 - x2) >= 0.0 else 0
        l1_3 = 1 if (2.0 * (1.0 - x1) - x2) >= 0.0 else 0
        and_t1 = 1 if (l1_1 + l1_2 + l1_3) == 3 else 0

        # T2: 3 semiplanos
        l2_1 = 1 if (x2 - 2.0) >= 0.0 else 0
        l2_2 = 1 if (2.0 * (x1 - 2.0) - (x2 - 2.0)) >= 0.0 else 0
        l2_3 = 1 if (2.0 * (3.0 - x1) - (x2 - 2.0)) >= 0.0 else 0
        and_t2 = 1 if (l2_1 + l2_2 + l2_3) == 3 else 0

        # Capa OR superior: dispara si and_t1 == 1 o and_t2 == 1
        salida_or = 1 if (and_t1 + and_t2) >= 1 else 0
        return {
            "T1_semiplanos": (l1_1, l1_2, l1_3),
            "T1_disparo": and_t1,
            "T2_semiplanos": (l2_1, l2_2, l2_3),
            "T2_disparo": and_t2,
            "salida_red": salida_or
        }


def aproximador_pulso_cuadrado(x, T1, T2):
    """
    Síntesis de pulso rectangular con 2 neuronas escalón (Diapositiva 49):
      p(x) = step(x - T1) - step(x - T2) con peso +1 y -1 en salida lineal.
    """
    s1 = 1.0 if x >= T1 else 0.0
    s2 = 1.0 if x >= T2 else 0.0
    return s1 - s2


def demo_activacion_sinusoidal():
    """
    Demuestra la tesis de la Diapositiva 52:
    Si la activación es sin(wx + b), para representar y = cos(2x)
    se necesita exactamente 1 neurona porque sin(2x + pi/2) == cos(2x).
    """
    puntos = [0.0, 0.5, 1.0, 1.5, 2.0, math.pi]
    max_diff = 0.0
    for x in puntos:
        f_real = math.cos(2.0 * x)
        f_neurona = math.sin(2.0 * x + (math.pi / 2.0))
        diff = abs(f_real - f_neurona)
        if diff > max_diff:
            max_diff = diff
    return max_diff


def cronologia_conexionismo():
    """
    Línea de tiempo con los hitos clave de las redes neuronales y conexionismo (Diaps. 53, 56-57).
    """
    return [
        (1943, "McCulloch & Pitts", "Primer modelo matemático de neurona artificial (lógica proposicional)."),
        (1949, "Donald Hebb", "Primera regla de aprendizaje biológico asociativo ('wire together')."),
        (1958, "Frank Rosenblatt", "El Perceptrón y teorema de convergencia lineal (Cornell Lab)."),
        (1960, "Widrow & Hoff", "ADALINE y la regla delta / LMS (mínimos cuadrados antes del umbral)."),
        (1969, "Minsky & Papert", "Libro 'Perceptrons': límite de XOR y comienzo del primer invierno de la IA."),
        (1980, "Kunihiko Fukushima", "Neocognitrón: arquitectura jerárquica precursora de las CNNs."),
        (1986, "Rumelhart, Hinton & Williams", "Popularización de Backpropagation y Procesamiento Distribuido en Paralelo (PDP)."),
        (1989, "Yann LeCun et al.", "LeNet: Redes convolucionales para lectura de códigos postales en USPS."),
        (2006, "Geoffrey Hinton et al.", "Nacimiento del Deep Learning (preentrenamiento no supervisado capa por capa)."),
        (2009, "Dahl, Mohamed, Hinton", "Redes profundas superan ampliamente a modelos HMM en reconocimiento de voz."),
        (2012, "AlexNet (Krizhevsky et al.)", "Victoria aplastante en ImageNet; revolución del Deep Learning acelerado por GPU."),
        (2016, "AlphaGo (DeepMind)", "AlphaGo vence al campeón mundial de Go combinando Deep RL y MCTS.")
    ]


def demo_universalidad_hitos():
    print("\n" + "═" * 78)
    print(" 12.4 CLASIFICADOR Y REGRESOR UNIVERSAL, Y LÍNEA DE TIEMPO DEL CONEXIONISMO")
    print("═" * 78)

    # 1. Dos triángulos disjuntos (9 neuronas)
    clasif = ClasificadorDosTriangulos()
    print("  1. Clasificador Universal de Regiones No Convexas (Dos Triángulos Disjuntos, Diap. 51):")
    print("     Arquitectura: 6 semiplanos (Capa 1) -> 2 AND (Capa 2) -> 1 OR (Capa 3) = 9 Neuronas.")
    test_points = [
        (0.5, 0.5, "Centro Triángulo 1"),
        (2.5, 2.5, "Centro Triángulo 2"),
        (1.5, 1.5, "Espacio intermedio disjunto"),
        (-0.5, 0.5, "Fuera de la región"),
    ]
    for x1, x2, desc in test_points:
        res = clasif.evaluar_punto(x1, x2)
        print(f"     • Punto ({x1:>4.1f}, {x2:>4.1f}) [{desc:<26}] => Salida: {res['salida_red']} (T1={res['T1_disparo']}, T2={res['T2_disparo']})")

    # 2. Pulsos de Riemann
    print("\n  2. Regresor Universal mediante Pulsos de Riemann (Diap. 49-50):")
    print("     p(x) = step(x - T1) - step(x - T2) (2 neuronas escalón + salida lineal)")
    for x in [0.0, 0.5, 1.0, 1.5, 2.0]:
        val = aproximador_pulso_cuadrado(x, T1=0.5, T2=1.5)
        print(f"     • x = {x:>3.1f} en pulso [0.5, 1.5] -> Salida p(x) = {val:.1f}")

    # 3. Teorema de Tamaño vs Activación
    err_sin = demo_activacion_sinusoidal()
    print("\n  3. Teorema de Tamaño vs Activación (Diapositiva 52):")
    print("     ¿Cuántas neuronas se necesitan para representar y = cos(2x) exactamente?")
    print(f"     Con activación sinusoidal sin(wx + b): ¡Exactamente 1 neurona! Error máx = {err_sin:.2e}")
    print("     sin(2x + π/2) ≡ cos(2x). La capacidad de representación compacta depende de la activación.\n")

    # 4. Cronología
    print("  4. Línea de Tiempo: Del Estancamiento a la Revolución de la IA:")
    for año, autor, hito in cronologia_conexionismo():
        print(f"     [{año}] {autor:<28}: {hito}")


# =============================================================================
# SUITE DE VERIFICACIÓN AUTOMÁTICA (--auto)
# =============================================================================

def run_all_tests():
    """
    Ejecuta aserciones formales sobre todas las propiedades matemáticas de CD12.
    """
    print("\n" + "=" * 78)
    print(" EJECUTANDO SUITE DE TESTS AUTOMÁTICOS: CD12 REDES Y CONEXIONISMO")
    print("=" * 78)

    # Test 12.1: Escala Bain
    comp = comparativa_escala_bain()
    assert comp["factor_subestimacion"] == 20000.0, f"Error factor: {comp['factor_subestimacion']}"
    assert len(ingredientes_sistema_pdp()) == 8
    print("  [PASS] 12.1 Asociacionismo y Escala Bain: Factor 20,000x y 8 ingredientes PDP.")

    # Test 12.2: McCulloch-Pitts & Hebb
    and_g = NeuronaMcCullochPitts([1, 1], umbral=2)
    assert and_g.evaluar((1, 1)) == 1 and and_g.evaluar((1, 0)) == 0
    or_g = NeuronaMcCullochPitts([1, 1], umbral=1)
    assert or_g.evaluar((0, 0)) == 0 and or_g.evaluar((0, 1)) == 1
    # Inhibición absoluta
    assert and_g.evaluar((1, 1), inhibitoria=True) == 0

    # Hebb Divergencia
    hitos_hebb, _ = simular_regla_hebb(1000, eta=0.1)
    assert abs(hitos_hebb[10] - 1.0) < 1e-6
    assert abs(hitos_hebb[100] - 10.0) < 1e-6
    assert abs(hitos_hebb[1000] - 100.0) < 1e-6

    # Oja Convergencia
    _, final_oja = simular_regla_oja(1000, eta=0.05)
    assert abs(final_oja - 1.0) < 0.05
    print("  [PASS] 12.2 McCulloch-Pitts & Hebb: Lógica, Divergencia Hebb (100.0) y Estabilidad Oja.")

    # Test 12.3: Perceptrón OR y MLP XOR
    X_or = [[0, 0], [0, 1], [1, 0], [1, 1]]
    d_or = [0, 1, 1, 1]
    p = PerceptronRosenblatt(n_features=2, eta=1.0)
    p.fit(X_or, d_or, max_epocas=20)
    for xi, di in zip(X_or, d_or):
        assert p.predict(xi) == di

    mlp = MlpXorClasico()
    assert mlp.forward(0, 0)[2] == 0
    assert mlp.forward(0, 1)[2] == 1
    assert mlp.forward(1, 0)[2] == 1
    assert mlp.forward(1, 1)[2] == 0
    print("  [PASS] 12.3 Perceptrón OR convergente y Solución MLP XOR (100% exactitud).")

    # Test 12.4: Triángulos Disjuntos y sin(2x+pi/2)
    clasif = ClasificadorDosTriangulos()
    assert clasif.evaluar_punto(0.5, 0.5)["salida_red"] == 1
    assert clasif.evaluar_punto(2.5, 2.5)["salida_red"] == 1
    assert clasif.evaluar_punto(1.5, 1.5)["salida_red"] == 0
    assert clasif.evaluar_punto(-1.0, -1.0)["salida_red"] == 0

    assert aproximador_pulso_cuadrado(1.0, 0.5, 1.5) == 1.0
    assert aproximador_pulso_cuadrado(0.2, 0.5, 1.5) == 0.0

    err_sin = demo_activacion_sinusoidal()
    assert err_sin < 1e-12
    print("  [PASS] 12.4 Dos Triángulos Disjuntos (9 neuronas), Pulsos Riemann y cos(2x) exacto.")

    print("\nTODOS LOS TESTS DE CD12 PASARON EXITOSAMENTE (CÓDIGO 0).")
    print("=" * 78)


# =============================================================================
# MENÚ INTERACTIVO
# =============================================================================

def menu_redes_conexionismo():
    while True:
        print("\n" + "═" * 78)
        print(" 🧠 CD12: REDES NEURONALES Y CONEXIONISMO (60 DIAPOSITIVAS)")
        print("    Stanford CS221 & CMU 11-785 • Universidad Católica del Norte")
        print("═" * 78)
        print("  [1] Módulo 12.1: Asociacionismo, Alexander Bain, Turing y PDP (Diaps. 1–21)")
        print("  [2] Módulo 12.2: Neurona McCulloch-Pitts y Regla de Hebb vs Oja (Diaps. 22–30)")
        print("  [3] Módulo 12.3: Perceptrón de Rosenblatt, Límite de XOR y MLP (Diaps. 31–41)")
        print("  [4] Módulo 12.4: Clasificador/Regresor Universal y Línea de Tiempo (Diaps. 42–60)")
        print("  [5] Ejecutar suite de pruebas unitarias automáticas (--auto)")
        print("  [0] Volver al menú principal / Salir")
        print("─" * 78)
        try:
            opc = input("Seleccione una opción [0-5]: ").strip()
            if opc == '1':
                demo_conexionismo_bain()
            elif opc == '2':
                demo_mcculloch_pitts_hebb()
            elif opc == '3':
                demo_rosenblatt_xor_mlp()
            elif opc == '4':
                demo_universalidad_hitos()
            elif opc == '5':
                run_all_tests()
            elif opc in ('0', 'q', 'exit', 'volver'):
                break
            else:
                print("⚠️ Opción no válida. Intente de nuevo.")
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('--auto', '-a', '--test'):
        run_all_tests()
    else:
        menu_redes_conexionismo()
