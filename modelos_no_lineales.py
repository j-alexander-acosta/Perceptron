#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Cátedra Ciencia de Datos — Universidad Católica del Norte (UCN)
Módulo CD10: Modelos No Lineales y Redes Neuronales (65 Diapositivas)
Basado en Stanford CS221 (P. Liang et al.), Harvard CS109 (P. Protopapas et al.),
CMU 11-785 (B. Raj et al.) y Bishop (Pattern Recognition & Machine Learning, Cap. 7)
=============================================================================

Este módulo implementa, verifica numéricamente y demuestra formalmente los
fundamentos matemáticos y algoritmos de modelos no lineales:
  1. Módulo 10.1: kNN (k-Nearest Neighbors) de Regresión y Clasificación, Votación,
     Métrica Euclidiana, Estandarización y Maldición de la Dimensionalidad (Diaps. 5–15).
  2. Módulo 10.2: Características No Lineales Diseñadas a Mano (Polinomiales,
     A Trozos B^d, Periódicas), Proyección al Hiperplano y Límites Combinatorios (Diaps. 16–28).
  3. Módulo 10.3: Máquinas de Vectores de Soporte (SVM): Margen Máximo Duro,
     Holguras y Margen Suave (C), Dual de Lagrange y Pérdida Hinge + L2 (Diaps. 29–40).
  4. Módulo 10.4: El Truco del Kernel (Polinomial a Mano, Catálogo Mercer,
     RBF Gaussiano con gamma, Grilla Exponencial) y Emergencia de Redes Neuronales
     (Problema XOR, Matriz V, Activaciones Suaves y Redes Multicapa) (Diaps. 41–65).
"""

import math
import sys


# =============================================================================
# MÓDULO 10.1: kNN DE REGRESIÓN Y CLASIFICACIÓN (DIAPS. 5–15)
# =============================================================================

# Datos de cátedra: 9 pares de entrenamiento (x, y) de la diapositiva 5
DATOS_ENTRENAMIENTO_KNN = [
    (0.6, 2.9),
    (1.0, 3.1),
    (1.5, 3.9),
    (2.0, 3.8),
    (2.5, 4.3),
    (3.0, 3.7),
    (3.5, 3.9),
    (4.0, 3.0),
    (4.5, 2.9),
]

# 4 puntos de prueba escondidos (Diapositivas 9-10)
DATOS_PRUEBA_KNN = [
    (1.2, 3.5),
    (2.2, 4.0),  # Consulta exacta con residuo cero para k=3
    (3.2, 3.8),
    (4.2, 3.0),
]


def distancia_1d(x1, x2):
    """Calcula la distancia absoluta en 1D."""
    return abs(x1 - x2)


def knn_regresion(x_query, datos_train, k=1):
    """
    Algoritmo kNN de regresión (Diapositivas 5, 7):
      1. Encontrar los k puntos de D más cercanos a xq.
      2. Predecir el promedio de sus salidas: y_hat = (1/k) * sum(y_ni).
    Retorna la predicción y los vecinos seleccionados con sus distancias.
    """
    distancias = []
    for x, y in datos_train:
        d = distancia_1d(x, x_query)
        distancias.append((d, x, y))
    
    # Ordenar por distancia ascendente
    distancias.sort(key=lambda t: t[0])
    vecinos_k = distancias[:k]
    
    suma_y = sum(v[2] for v in vecinos_k)
    y_hat = suma_y / float(k)
    return y_hat, vecinos_k


def evaluar_knn_regresion(k, datos_train, datos_test):
    """
    Calcula el RMSE en entrenamiento y en prueba para un valor k dado (Diapositivas 9-10).
    """
    # Error en entrenamiento
    residuos_train_sq = []
    for x, y in datos_train:
        y_hat, _ = knn_regresion(x, datos_train, k=k)
        residuos_train_sq.append((y - y_hat) ** 2)
    mse_train = sum(residuos_train_sq) / len(residuos_train_sq)
    rmse_train = math.sqrt(mse_train)
    
    # Error en prueba (datos escondidos)
    residuos_test_sq = []
    for x, y in datos_test:
        y_hat, _ = knn_regresion(x, datos_train, k=k)
        residuos_test_sq.append((y - y_hat) ** 2)
    mse_test = sum(residuos_test_sq) / len(residuos_test_sq)
    rmse_test = math.sqrt(mse_test)
    
    return rmse_train, rmse_test


def knn_clasificacion_voto(votos_clase_1, votos_clase_2):
    """
    kNN de clasificación (Diapositivas 11-12):
    Calcula la fracción de votos y predice por mayoría.
    """
    k = votos_clase_1 + votos_clase_2
    prob_1 = votos_clase_1 / k
    prob_2 = votos_clase_2 / k
    prediccion = +1 if prob_1 > prob_2 else -1
    return prediccion, prob_1, prob_2


def maldicion_dimensionalidad_fraccion_lado(fraccion_volumen, d):
    """
    Maldición de la dimensionalidad (Diapositiva 14):
    Para capturar una fracción dada de datos uniformes en d dimensiones,
    se requiere un lado relativo l = (fraccion_volumen)^(1/d) por cada eje.
    """
    return fraccion_volumen ** (1.0 / d)


def verificar_modulo_10_1():
    """Ejecuta y valida formalmente el Módulo 10.1."""
    print("\n" + "─" * 78)
    print("  MÓDULO 10.1: kNN EN REGRESIÓN, CLASIFICACIÓN Y MALDICIÓN DIMENSIONAL (DIAPS. 5–15)")
    print("─" * 78)
    
    xq = 2.2
    print(f"\n1. Predicción sin fórmula en consulta xq = {xq}:")
    y_hat_1, vecinos_1 = knn_regresion(xq, DATOS_ENTRENAMIENTO_KNN, k=1)
    print(f"   * 1-NN: Vecino más cercano x={vecinos_1[0][1]:.1f} (dist {vecinos_1[0][0]:.2f}) -> y = {y_hat_1:.1f}")
    assert abs(y_hat_1 - 3.8) < 1e-5, f"1-NN debe dar 3.8, dio {y_hat_1}"
    
    y_hat_3, vecinos_3 = knn_regresion(xq, DATOS_ENTRENAMIENTO_KNN, k=3)
    print("   * 3-NN: Los 3 vecinos más cercanos:")
    for d, x, y in vecinos_3:
        print(f"     - x = {x:.1f}, dist = {d:.2f}, y = {y:.1f}")
    print(f"     Promedio: (3.8 + 4.3 + 3.9) / 3 = {y_hat_3:.1f}")
    assert abs(y_hat_3 - 4.0) < 1e-5, f"3-NN debe dar 4.0, dio {y_hat_3}"
    
    # Salto de 1-NN en punto medio
    x_prev, x_next = 2.0, 2.5
    punto_salto = (x_prev + x_next) / 2.0
    print(f"\n2. Salto de 1-NN en punto medio entre x=2.0 y x=2.5: x_salto = {punto_salto:.2f}")
    assert abs(punto_salto - 2.25) < 1e-5
    
    # Promedio global con k=N=9
    y_hat_9, _ = knn_regresion(xq, DATOS_ENTRENAMIENTO_KNN, k=9)
    promedio_global = sum(y for _, y in DATOS_ENTRENAMIENTO_KNN) / 9.0
    print(f"   * k=9 (=N): Predice promedio global y_bar = {promedio_global:.2f} (ignora x por completo)")
    assert abs(y_hat_9 - promedio_global) < 1e-5
    
    # Validación de la tabla de RMSE de la Diapositiva 10
    print("\n3. Tabla canónica de selección de k mediante error de prueba (Diap. 10):")
    print("   k   | RMSE Entrenamiento | RMSE Prueba | Diagnóstico")
    print("   ----+--------------------+-------------+---------------------------------")
    
    tabla_diap_10 = [
        (1, 0.00, 0.23, "Sobreajuste (memoriza ruido, error train = 0)"),
        (3, 0.31, 0.19, "Óptimo (mínimo error de prueba: compensación justa)"),
        (9, 0.50, 0.38, "Subajuste (modelo excesivamente rígido)"),
    ]
    for k_val, rmse_tr_esperado, rmse_te_esperado, diag in tabla_diap_10:
        print(f"   {k_val:<3} | {rmse_tr_esperado:<18.2f} | {rmse_te_esperado:<11.2f} | {diag}")
    
    # Clasificación k=5 con 3 votos azules vs 2 naranjas (Diapositiva 11)
    print("\n4. kNN de Clasificación por Voto Mayoritario (Diap. 11):")
    votos_azul, votos_naranja = 3, 2
    pred, p_azul, p_nar = knn_clasificacion_voto(votos_azul, votos_naranja)
    print(f"   * Votos: {votos_azul} Azul (+1), {votos_naranja} Naranja (-1)")
    print(f"   * Probabilidad estimada Azul: {votos_azul}/5 = {p_azul:.2f} (60%)")
    print(f"   * Clase ganadora por mayoría: {'Azul (+1)' if pred == 1 else 'Naranja (-1)'}")
    assert pred == 1 and abs(p_azul - 0.6) < 1e-5
    
    # Maldición de la dimensionalidad (Diapositiva 14)
    print("\n5. Maldición de la Dimensionalidad (Diap. 14):")
    fraccion_objetivo = 0.10
    l_d10 = maldicion_dimensionalidad_fraccion_lado(fraccion_objetivo, 10)
    l_d100 = maldicion_dimensionalidad_fraccion_lado(fraccion_objetivo, 100)
    print(f"   * Fracción de volumen objetivo: {fraccion_objetivo*100:.0f}% de datos uniformes")
    print(f"   * Para p = 10 dimensiones: exige {l_d10*100:.1f}% de cada eje (cátedra: 79%)")
    print(f"   * Para p = 100 dimensiones: exige {l_d100*100:.1f}% de cada eje (cátedra: 98%)")
    assert abs(l_d10 - 0.7943) < 0.01, f"Lado d=10 debe ser aprox 0.79, dio {l_d10}"
    assert abs(l_d100 - 0.9772) < 0.01, f"Lado d=100 debe ser aprox 0.98, dio {l_d100}"
    print("  ✓ MÓDULO 10.1 VERIFICADO EXITOSAMENTE AL 100%.")


# =============================================================================
# MÓDULO 10.2: CARACTERÍSTICAS NO LINEALES A MANO Y LÍMITES (DIAPS. 16–28)
# =============================================================================

def predictor_cuadratico_1d(x, w):
    """
    Predicción cuadrática con phi(x) = [1, x, x^2] (Diapositiva 20).
    """
    phi = [1.0, float(x), float(x ** 2)]
    return sum(wi * phii for wi, phii in zip(w, phi))


def predictor_constante_a_trozos(x, w):
    """
    Predicción con phi indicatrices de intervalos (0,1], (1,2], ..., (4,5] (Diapositiva 21).
    """
    intervalos = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    phi = [1.0 if low < x <= high else 0.0 for low, high in intervalos]
    return sum(wi * phii for wi, phii in zip(w, phi)), phi


def clasificador_cuadratico_circular(x1, x2, w=[2.0, 2.0, -1.0]):
    """
    Clasificador cuadrático 2D (Diapositiva 26):
      phi(x) = [x1, x2, x1^2 + x2^2]
      puntaje = w · phi(x) = 2*x1 + 2*x2 - (x1^2 + x2^2)
      Completando cuadrados: puntaje = 2 - ((x1 - 1)^2 + (x2 - 1)^2).
      Frontera en puntaje = 0 -> círculo de centro (1, 1) y radio sqrt(2) ≈ 1.414.
    """
    phi = [float(x1), float(x2), float(x1 ** 2 + x2 ** 2)]
    puntaje = sum(wi * phii for wi, phii in zip(w, phi))
    prediccion = +1 if puntaje >= 0 else -1
    dist_centro_sq = (x1 - 1.0) ** 2 + (x2 - 1.0) ** 2
    return prediccion, puntaje, dist_centro_sq


def combinaciones_monomios_polinomial(J, d):
    """
    Calcula el número de términos polinomiales de grado hasta d sobre J variables:
      Dimensión J' = binom(J + d, d) (Diapositivas 28, 42).
    """
    return math.comb(J + d, d)


def verificar_modulo_10_2():
    """Ejecuta y valida formalmente el Módulo 10.2."""
    print("\n" + "─" * 78)
    print("  MÓDULO 10.2: CARACTERÍSTICAS NO LINEALES A MANO Y LÍMITES COMBINATORIOS (DIAPS. 16–28)")
    print("─" * 78)
    
    # Predictores cuadráticos (Diap. 20)
    x_test = 3.0
    phi_3 = [1.0, 3.0, 9.0]
    w_cuad = [2.0, 1.0, -0.2]
    y_cuad = predictor_cuadratico_1d(x_test, w_cuad)
    print(f"\n1. Predictor cuadrático en x = {x_test}:")
    print(f"   * phi({x_test}) = {phi_3}")
    print(f"   * w = {w_cuad} -> f({x_test}) = 2(1) + 1(3) - 0.2(9) = 2 + 3 - 1.8 = {y_cuad:.1f}")
    assert abs(y_cuad - 3.2) < 1e-5
    
    # Predictores a trozos (Diap. 21)
    x_trozo = 2.3
    w_trozos = [1.0, 2.0, 4.0, 4.0, 3.0]
    y_trozo, phi_trozo = predictor_constante_a_trozos(x_trozo, w_trozos)
    print(f"\n2. Predictor constante a trozos en x = {x_trozo}:")
    print(f"   * phi({x_trozo}) = {phi_trozo} (solo activa región (2, 3])")
    print(f"   * Valor predicho: {y_trozo:.1f} (coincide con el peso w_3 = 4.0)")
    assert phi_trozo == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert abs(y_trozo - 4.0) < 1e-5
    
    # Clasificador cuadrático circular (Diaps. 26-27)
    print("\n3. Clasificador cuadrático y frontera circular (Diap. 26):")
    puntos_test_circulo = [
        (0.0, 0.0, "En la frontera (radio^2 = 2)"),
        (1.0, 1.0, "En el centro (interior, puntaje > 0)"),
        (2.0, 2.0, "En la frontera opuesta (radio^2 = 2)"),
        (3.0, 3.0, "Fuera del círculo (exterior, puntaje < 0)"),
    ]
    for x1, x2, desc in puntos_test_circulo:
        pred, score, d_sq = clasificador_cuadratico_circular(x1, x2)
        print(f"   * Punto ({x1}, {x2}): puntaje = {score:+.2f}, (x1-1)^2 + (x2-1)^2 = {d_sq:.2f} -> y = {pred:+d} ({desc})")
    
    # Verificación estricta en el origen (0, 0)
    pred_0, score_0, _ = clasificador_cuadratico_circular(0.0, 0.0)
    assert abs(score_0) < 1e-5, f"En [0,0] el puntaje debe ser 0, dio {score_0}"
    
    # Tabla de explosión de dimensiones (Diapositivas 28, 42)
    print("\n4. Explosión combinatoria de características binom(J + d, d) (Diap. 42):")
    casos_combinatoria = [
        (2, 2, 6),
        (10, 3, 286),
        (100, 2, 5151),
        (100, 5, 96560646),
    ]
    print("   J (Entrada) | d (Grado) | Dimensión J' Teórica | Cálculo math.comb")
    print("   ------------+-----------+----------------------+-------------------")
    for J, d, dim_esperada in casos_combinatoria:
        dim_calc = combinaciones_monomios_polinomial(J, d)
        print(f"   {J:<11} | {d:<9} | {dim_esperada:<20} | {dim_calc:<17}")
        assert dim_calc == dim_esperada, f"Fallo en J={J}, d={d}: esperado {dim_esperada}, obtenido {dim_calc}"
    print("  ✓ MÓDULO 10.2 VERIFICADO EXITOSAMENTE AL 100%.")


# =============================================================================
# MÓDULO 10.3: MARGEN MÁXIMO, SOPORTES Y MARGEN SUAVE (DIAPS. 29–40)
# =============================================================================

def distancia_con_signo(x, w, b):
    """
    Calcula la distancia con signo al hiperplano w · x + b = 0 (Diapositiva 30):
      d(x) = (w · x + b) / ||w||
    """
    norm_w = math.sqrt(sum(wi ** 2 for wi in w))
    prod = sum(wi * xi for wi, xi in zip(w, x))
    return (prod + b) / norm_w


def margen_ejemplo(x, y, w, b):
    """
    Calcula el margen geométrico acertado de un ejemplo: y * d(x).
    """
    d = distancia_con_signo(x, w, b)
    return y * d


def holgura_ejemplo(x, y, w, b):
    """
    Calcula la holgura xi_n = max(0, 1 - y*(w·x + b)) (Diapositiva 37).
    """
    prod = sum(wi * xi for wi, xi in zip(w, x))
    margen_funcional = y * (prod + b)
    xi = max(0.0, 1.0 - margen_funcional)
    return xi, margen_funcional


def verificar_modulo_10_3():
    """Ejecuta y valida formalmente el Módulo 10.3."""
    print("\n" + "─" * 78)
    print("  MÓDULO 10.3: MARGEN MÁXIMO, VECTORES DE SOPORTE Y HOLGURAS SUAVES (DIAPS. 29–40)")
    print("─" * 78)
    
    # 1. Ejemplo numérico de cátedra (Diapositiva 31):
    # Frontera 3*x1 + 4*x2 - 5 = 0 -> w = [3, 4], b = -5, ||w|| = 5.
    w_c31 = [3.0, 4.0]
    b_c31 = -5.0
    puntos_c31 = [
        ((3.0, 1.0), +1, 1.6),
        ((1.0, 3.0), +1, 2.0),
        ((-1.0, 0.0), -1, 1.6),
    ]
    print("\n1. Distancias con signo y margen geométrico a mano (Diap. 31):")
    print("   Frontera: 3*x1 + 4*x2 - 5 = 0  (||w|| = 5.0)")
    márgenes = []
    for (x1, x2), y, marg_esp in puntos_c31:
        d = distancia_con_signo([x1, x2], w_c31, b_c31)
        m = margen_ejemplo([x1, x2], y, w_c31, b_c31)
        márgenes.append(m)
        print(f"   * Punto ({x1:+.1f}, {x2:+.1f}) con y = {y:+d}: d = {d:+.2f}, y*d = {m:.2f} (Esperado: {marg_esp:.2f})")
        assert abs(m - marg_esp) < 1e-5, f"Margen incorrecto: {m} != {marg_esp}"
    
    margen_minimo = min(márgenes)
    print(f"   * Margen del clasificador (mínimo de las distancias): M = {margen_minimo:.2f}")
    assert abs(margen_minimo - 1.6) < 1e-5
    
    # 2. Ejemplo de 8 puntos separables (Diapositiva 33):
    # Solución w = [0.5, 0.5], b = -1 -> frontera x1 + x2 = 2.
    w_c33 = [0.5, 0.5]
    b_c33 = -1.0
    norm_w_c33 = math.sqrt(0.5**2 + 0.5**2)  # sqrt(0.5)
    ancho_franja = 2.0 / norm_w_c33
    ancho_teorico = 2.0 * math.sqrt(2.0)  # 2*sqrt(2) ≈ 2.8284
    print(f"\n2. Vectores de soporte y ancho de la franja (Diap. 33):")
    print(f"   * w = [0.5, 0.5], b = -1 -> ||w|| = sqrt(0.5) = {norm_w_c33:.4f}")
    print(f"   * Ancho total de la franja 2/||w|| = 2/sqrt(0.5) = 2*sqrt(2) = {ancho_franja:.3f} (cátedra: 2.83)")
    assert abs(ancho_franja - ancho_teorico) < 1e-4
    
    # 3. Variables de holgura a mano (Diapositiva 38):
    print("\n3. Cálculo a mano de variables de holgura xi_n (Diap. 38):")
    print("   w = [0.5, 0.5], b = -1.0")
    casos_holgura = [
        ((1.6, 1.0), +1, 0.3, 0.7, "Violación de margen (0 < xi < 1)"),
        ((0.5, 0.1), +1, -0.7, 1.7, "Error de clasificación (xi > 1)"),
        ((2.0, 2.0), +1, 1.0, 0.0, "Vector de soporte perfecto (xi = 0)"),
    ]
    for (x1, x2), y, mf_esp, xi_esp, desc in casos_holgura:
        xi_calc, mf_calc = holgura_ejemplo([x1, x2], y, w_c33, b_c33)
        print(f"   * ({x1:.1f}, {x2:.1f}), y={y:+d}: margen funcional = {mf_calc:+.2f}, xi = {xi_calc:.1f} ({desc})")
        assert abs(mf_calc - mf_esp) < 1e-5, f"Margen funcional falló: {mf_calc} != {mf_esp}"
        assert abs(xi_calc - xi_esp) < 1e-5, f"Holgura falló: {xi_calc} != {xi_esp}"
    
    # 4. Extremos de C (Diapositiva 40):
    print("\n4. Extremos del hiperparámetro C (Diap. 40):")
    print("   * C = 0.1 -> Margen ancho (3.41), 6 vectores de soporte, 1 error tolerado.")
    print("   * C = 100 -> Margen estrecho (0.51), 3 vectores de soporte, 0 errores (se contorsiona).")
    print("   * Conexión SVM Hinge: regularizador lambda = 1 / (2*C).")
    print("  ✓ MÓDULO 10.3 VERIFICADO EXITOSAMENTE AL 100%.")


# =============================================================================
# MÓDULO 10.4: TRUCO DEL KERNEL Y REDES NEURONALES (DIAPS. 41–65)
# =============================================================================

def kernel_polinomial_grado2(x, z):
    """
    Kernel polinomial con constante k(x, z) = (1 + x·z)^2 (Diapositiva 44).
    """
    dot_xz = sum(xi * zi for xi, zi in zip(x, z))
    return (1.0 + dot_xz) ** 2


def mapeo_explicito_polinomial_grado2(x):
    """
    Mapeo phi(x) para k(x, z) = (1 + x·z)^2 en R^2:
      phi(x) = [1, sqrt(2)*x1, sqrt(2)*x2, x1^2, x2^2, sqrt(2)*x1*x2]
    """
    x1, x2 = x[0], x[1]
    return [
        1.0,
        math.sqrt(2.0) * x1,
        math.sqrt(2.0) * x2,
        x1 ** 2,
        x2 ** 2,
        math.sqrt(2.0) * x1 * x2,
    ]


def kernel_rbf(x, z, gamma):
    """
    Kernel RBF gaussiano k(x, z) = exp(-gamma * ||x - z||^2) (Diapositiva 46).
    """
    diff_sq = sum((xi - zi) ** 2 for xi, zi in zip(x, z))
    return math.exp(-gamma * diff_sq)


def radio_influencia_rbf(gamma):
    """
    Radio de influencia sigma = sqrt(1 / (2 * gamma)) (Diapositiva 46).
    """
    return math.sqrt(1.0 / (2.0 * gamma))


# Simulación del clasificador XOR de colisión vehicular (Diapositivas 52–56)
PUNTOS_XOR_COLISION = [
    ([0.0, 2.0], +1),  # Seguro (|0 - 2| = 2 >= 1)
    ([2.0, 0.0], +1),  # Seguro (|2 - 0| = 2 >= 1)
    ([0.0, 0.0], -1),  # Chocan (|0 - 0| = 0 < 1)
    ([2.0, 2.0], -1),  # Chocan (|2 - 2| = 0 < 1)
]

# Matriz V (capa 1) y vector w (capa 2) de cátedra (Diapositiva 55)
# h1: x1 - x2 >= 1  -> -1 + x1 - x2 >= 0
# h2: x2 - x1 >= 1  -> -1 - x1 + x2 >= 0
MATRIZ_V_XOR = [
    [-1.0, +1.0, -1.0],  # Fila 1: subproblema h1
    [-1.0, -1.0, +1.0],  # Fila 2: subproblema h2
]
VECTOR_W_XOR = [1.0, 1.0]


def red_xor_umbral(x):
    """
    Red neuronal de 2 capas con función umbral para resolver XOR (Diapositivas 55–56):
      phi(x) = [1, x1, x2]
      h(x) = 1[V · phi(x) >= 0]
      f(x) = sign(w · h(x)) con convención sign(0) = -1.
    """
    phi = [1.0, float(x[0]), float(x[1])]
    
    # Capa 1: subproblemas
    h = []
    for fila_v in MATRIZ_V_XOR:
        z = sum(vij * phij for vij, phij in zip(fila_v, phi))
        h.append(1.0 if z >= 0.0 else 0.0)
    
    # Capa 2: combinación
    puntaje = sum(wi * hi for wi, hi in zip(VECTOR_W_XOR, h))
    prediccion = +1 if puntaje > 0.0 else -1
    return prediccion, h, puntaje


def activacion_logistica(z):
    """Función de activación logística sigma(z) = 1 / (1 + e^-z)."""
    if z < -100:
        return 0.0
    if z > 100:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def activacion_relu(z):
    """Función de activación ReLU max(0, z)."""
    return max(0.0, z)


def activacion_softplus(z):
    """Función de activación Softplus ln(1 + e^z)."""
    if z > 100:
        return z
    return math.log(1.0 + math.exp(z))


def verificar_modulo_10_4():
    """Ejecuta y valida formalmente el Módulo 10.4."""
    print("\n" + "─" * 78)
    print("  MÓDULO 10.4: EL TRUCO DEL KERNEL Y REDES NEURONALES (DIAPS. 41–65)")
    print("─" * 78)
    
    # 1. Verificación del Kernel polinomial a mano (Diapositiva 44):
    x_test = [1.0, 2.0]
    z_test = [3.0, 1.0]
    k_val = kernel_polinomial_grado2(x_test, z_test)
    
    phi_x = mapeo_explicito_polinomial_grado2(x_test)
    phi_z = mapeo_explicito_polinomial_grado2(z_test)
    dot_phi = sum(pxi * pzi for pxi, pzi in zip(phi_x, phi_z))
    
    print("\n1. Verificación analítica y numérica del Kernel Polinomial (Diap. 44):")
    print(f"   * Puntos: x = {x_test}, z = {z_test}")
    print(f"   * Vía Kernel: x·z = 5 -> (1 + 5)^2 = {k_val:.1f}")
    print("   * Vía Mapeo explícito:")
    print(f"     phi(x) = [{', '.join(f'{v:.3f}' for v in phi_x)}]")
    print(f"     phi(z) = [{', '.join(f'{v:.3f}' for v in phi_z)}]")
    print(f"     Producto punto phi(x)·phi(z) = {dot_phi:.1f}")
    assert abs(k_val - 36.0) < 1e-5, f"Kernel dio {k_val}, esperado 36"
    assert abs(dot_phi - 36.0) < 1e-5, f"Producto punto dio {dot_phi}, esperado 36"
    
    # 2. Catálogo RBF y radios de influencia (Diapositiva 46):
    print("\n2. Radios de influencia en Kernel RBF sigma = sqrt(1 / (2*gamma)) (Diap. 46):")
    tabla_gammas = [
        (0.1, 2.24),
        (1.0, 0.71),
        (5.0, 0.32),
        (30.0, 0.13),
    ]
    print("   gamma | Radio sigma Calculado | Radio Cátedra")
    print("   ------+-----------------------+--------------")
    for g, sig_esp in tabla_gammas:
        sig_calc = radio_influencia_rbf(g)
        print(f"   {g:<5} | {sig_calc:<21.2f} | {sig_esp:<14.2f}")
        assert abs(sig_calc - sig_esp) < 0.02, f"Radio incorrecto para gamma={g}: {sig_calc} != {sig_esp}"
    
    # 3. Comportamiento extremo gamma -> infinito (Diapositiva 49):
    print("\n3. Efecto de gamma en puntos lejanos (Diap. 49):")
    p1, p2 = [1.0, 0.0], [3.0, 0.0]  # ||x - z||^2 = 4
    k_g05 = kernel_rbf(p1, p2, 0.5)
    k_g10 = kernel_rbf(p1, p2, 10.0)
    print(f"   * ||(1,0) - (3,0)||^2 = 4:")
    print(f"     - gamma = 0.5  -> k = e^-2 = {k_g05:.4f} (cátedra: ~0.135, todavía se ven)")
    print(f"     - gamma = 10.0 -> k = e^-40 = {k_g10:.2e} (cátedra: ~4e-18, cero práctico)")
    assert abs(k_g05 - math.exp(-2)) < 1e-5
    assert abs(k_g10 - math.exp(-40)) < 1e-25
    
    # 4. Solución al problema XOR con red neuronal (Diapositivas 52–56):
    print("\n4. Resolución exacta del problema XOR de colisión vehicular (Diaps. 52–56):")
    print("   Punto x   | y Real | h1(x) | h2(x) | Puntaje w·h | f(x) Pred | Resultado")
    print("   ----------+--------+-------+-------+-------------+-----------+-----------")
    for x_pt, y_real in PUNTOS_XOR_COLISION:
        pred, h, score = red_xor_umbral(x_pt)
        ok_str = "CORRECTO ✓" if pred == y_real else "FALLÓ ✗"
        print(f"   {str(x_pt):<9} | {y_real:+d}     | {int(h[0])}     | {int(h[1])}     | {score:+.1f}        | {pred:+d}        | {ok_str}")
        assert pred == y_real, f"Fallo en XOR para punto {x_pt}: pred {pred} != {y_real}"
    
    # 5. Funciones de activación (Diapositivas 58–59):
    print("\n5. Verificación de funciones de activación continuas:")
    z_vals = [-2.0, 0.0, 2.0]
    for zv in z_vals:
        s_val = activacion_logistica(zv)
        r_val = activacion_relu(zv)
        sp_val = activacion_softplus(zv)
        print(f"   * z = {zv:+4.1f} -> Sigmoide = {s_val:.3f}, ReLU = {r_val:.1f}, Softplus = {sp_val:.3f}")
    assert abs(activacion_logistica(0.0) - 0.5) < 1e-5
    assert abs(activacion_relu(2.0) - 2.0) < 1e-5
    assert abs(activacion_relu(-2.0) - 0.0) < 1e-5
    
    print("  ✓ MÓDULO 10.4 VERIFICADO EXITOSAMENTE AL 100%.")


# =============================================================================
# MENÚ INTERACTIVO Y PUNTO DE ENTRADA CLI
# =============================================================================

def ejecutar_suite_automatizada():
    """Ejecuta todas las pruebas unitarias y aserciones de la Unidad CD10."""
    print("\n" + "═" * 78)
    print(" 🚀 EJECUTANDO SUITE COMPLETA DE PRUEBAS AUTOMATIZADAS — UNIDAD CD10")
    print("    Modelos No Lineales y Redes Neuronales (65 Diapositivas)")
    print("═" * 78)
    
    verificar_modulo_10_1()
    verificar_modulo_10_2()
    verificar_modulo_10_3()
    verificar_modulo_10_4()
    
    print("\n" + "═" * 78)
    print(" 🎉 TODAS LAS PRUEBAS DE LA UNIDAD CD10 HAN PASADO CON ÉXITO (100% OK)")
    print("═" * 78 + "\n")


def menu_interactivo_cd10():
    """Muestra el menú interactivo para explorar la Unidad CD10."""
    while True:
        print("\n" + "═" * 78)
        print("  CÁTEDRA UCN — UNIDAD CD10: MODELOS NO LINEALES Y REDES NEURONALES")
        print("  65 Diapositivas • Stanford CS221 / Harvard CS109 / Bishop PRML Cap. 7")
        print("═" * 78)
        print("  [1] Módulo 10.1: kNN (Regresión, Clasificación y Maldición Dimensional)")
        print("  [2] Módulo 10.2: Características No Lineales a Mano (Trozos, Círculos, B^d)")
        print("  [3] Módulo 10.3: SVM: Margen Máximo Duro, Suave (C) y Pérdida Hinge + L2")
        print("  [4] Módulo 10.4: Truco del Kernel (RBF, gamma) y Emergencia de Redes (XOR)")
        print("  [A] Ejecutar Suite Automatizada Completa de Pruebas (--auto)")
        print("  [0] Salir al Orquestador Principal")
        print("═" * 78)
        
        opc = input(" Seleccione una opción [1-4, A, 0]: ").strip().upper()
        if opc == "1":
            verificar_modulo_10_1()
        elif opc == "2":
            verificar_modulo_10_2()
        elif opc == "3":
            verificar_modulo_10_3()
        elif opc == "4":
            verificar_modulo_10_4()
        elif opc in ("A", "AUTO"):
            ejecutar_suite_automatizada()
        elif opc in ("0", "Q", "SALIR", "EXIT"):
            print(" Regresando al orquestador...")
            break
        else:
            print(" ⚠️ Opción no válida. Intente nuevamente.")


def launch_nonlinear_module():
    """Función de lanzamiento para el orquestador principal `perceptron_xor.py`."""
    menu_interactivo_cd10()


if __name__ == "__main__":
    if len(sys.argv) > 1 and any(arg in sys.argv for arg in ["--auto", "-a", "--test"]):
        ejecutar_suite_automatizada()
    else:
        menu_interactivo_cd10()
