#!/usr/bin/env python3
"""
Módulo de Verificación Numérica y Conceptual: CD08 — Modelos de Árboles de Clasificación y Regresión (CART)
==========================================================================================================
Ciencia de Datos — Doctorado UCN
Basado en material de:
  - Harvard CS109 (P. Protopapas, K. Rader et al.)
  - L. Breiman, J. Friedman, R. Olshen, C. Stone, "Classification and Regression Trees" (CART, 1984)
  - J. R. Quinlan, "Induction of Decision Trees" (Machine Learning, 1986 - ID3)
  - L. Hyafil & R. L. Rivest, "Constructing Optimal Binary Decision Trees is NP-complete" (1976)
  - T. Hastie, R. Tibshirani & J. Friedman, "The Elements of Statistical Learning" (Cap. 9)

Módulos temáticos incluidos:
  1. De la Frontera Lineal al Árbol:
     - Límites de separabilidad lineal (XOR, círculos concéntricos, costos de polinomios de grado superior).
     - Formulación de ML: hipótesis de árbol h_theta(x), pérdida Bernoulli / entropía cruzada.
     - Explosión combinatoria 2^(2^n) y justificación de optimización greedy (voraz).
     - Partición ortogonal del espacio y efecto de codificación de atributos categóricos.
  2. Criterios de División e Inducción ID3:
     - Comparación analítica de impureza: Error de clasificación vs Índice de Gini vs Entropía de Shannon.
     - Demostración de ceguera del error de clasificación vs concavidad de Gini/Entropía (casos 19 pts y 12 pts).
     - Algoritmo ID3 paso a paso a mano con la tabla de 10 días ("¿Saldrá a correr?"):
       H(S) = 1.0, IG(Clima)=0.60, IG(Viento)=0.125, IG(Temp)=0.049 -> Clima en la raíz.
       Recursión en Soleado: IG(Viento)=1.0 -> Árbol con 100% pureza y selección natural de variables.
  3. Control de Complejidad, Sesgo-Varianza y Poda por Costo-Complejidad:
     - Profundidad como perilla de sesgo-varianza, riesgo de sobreajuste y detención temprana (early stopping).
     - Miopía del criterio de ganancia mínima (min_impurity_decrease) frente a patrones tipo paridad/XOR.
     - Poda por costo-complejidad (CART): C(T) = Error(T) + alpha * |T|.
     - Algoritmo de eslabón más débil (Weakest Link Pruning) a mano sobre el árbol de 100 puntos (Diaps. 51-54):
       Cálculo exacto de g(t), secuencia T0 -> T1 -> T2, umbrales alpha en 0.04 y 0.32, y verificación con alpha=0.10.
  4. Árboles de Regresión (CART Continuo):
     - Criterio de división por varianza / error cuadrático medio ponderado (MSE).
     - Predicción por promedio de hoja.
     - Traza a mano de los 6 puntos continuos (x < 3.5 con MSE=0.17 ganador frente a x < 2.5 y x < 4.5).
     - Progresión de profundidades 1 (MSE=0.39), 2 (0.06), 3 (0.04) a 10 (0.00 memorización).
     - Balance analítico: fortalezas, debilidades y fundamento para ensambles (Bagging, Random Forest, Boosting).
"""

import math
import sys
import random

# =============================================================================
# UTILIDADES MATEMÁTICAS EN PYTHON ESTÁNDAR
# =============================================================================

def entropia_binaria(p):
    """Calcula la entropía de Shannon en bits para dos clases con proporción p y 1-p."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))

def entropia_distribucion(conteos):
    """
    Calcula la entropía de Shannon H(S) = -sum p_k * log2(p_k) en bits.
    conteos: lista o tupla de frecuencias por clase.
    """
    total = sum(conteos)
    if total == 0:
        return 0.0
    h = 0.0
    for c in conteos:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h

def gini_distribucion(conteos):
    """
    Calcula el índice de impureza de Gini: Gini(S) = 1 - sum p_k^2.
    """
    total = sum(conteos)
    if total == 0:
        return 0.0
    sum_sq = sum((c / total) ** 2 for c in conteos)
    return 1.0 - sum_sq

def error_clasificacion_distribucion(conteos):
    """
    Calcula la tasa de error de clasificación al predecir la clase mayoritaria:
    Error(S) = 1 - max p_k.
    """
    total = sum(conteos)
    if total == 0:
        return 0.0
    p_max = max(conteos) / total
    return 1.0 - p_max

def mse_region(valores):
    """Calcula el Error Cuadrático Medio de un conjunto continuo respecto a su promedio."""
    n = len(valores)
    if n == 0:
        return 0.0, 0.0
    promedio = sum(valores) / n
    var = sum((y - promedio) ** 2 for y in valores) / n
    return promedio, var

# =============================================================================
# MÓDULO 08.1: LÍMITES LINEALES, HIPÓTESIS, EXPLOSIÓN 2^(2^n) Y GEOMETRÍA
# =============================================================================

def verificar_limites_lineales_y_explosion():
    print("\n" + "═" * 78)
    print("  MÓDULO 08.1: DE FRONTERAS LINEALES A ÁRBOLES DE DECISIÓN (DIAPS. 1–22)")
    print("═" * 78)
    
    # 1. Separación satelital lineal amigable (Diap. 6)
    # Ecuación: -0.8 * x1 + x2 = 0  => Latitud = 0.8 * Longitud
    def satelite_lineal(lon, lat):
        return -0.8 * lon + lat
    
    # Puntos de vegetación (arriba de la recta: lat > 0.8 * lon)
    pts_veg = [(1.0, 1.5), (2.0, 2.5), (3.0, 3.2), (4.0, 4.0)]
    # Puntos de suelo (debajo de la recta: lat < 0.8 * lon)
    pts_suelo = [(2.0, 1.0), (3.0, 2.0), (4.0, 2.8), (5.0, 3.5)]
    
    veg_ok = all(satelite_lineal(lon, lat) > 0 for lon, lat in pts_veg)
    suelo_ok = all(satelite_lineal(lon, lat) < 0 for lon, lat in pts_suelo)
    print(f"  [1] Problema satelital lineal (Latitud = 0.8 * Longitud):")
    print(f"      • Puntos Vegetación clasificados correctamente (> 0): {veg_ok}")
    print(f"      • Puntos Suelo clasificados correctamente (< 0):      {suelo_ok}")
    
    # 2. Explosión combinatoria de funciones booleanas y árboles (Diaps. 14-15)
    # Con n entradas binarias: 2^n combinaciones de entrada, 2^(2^n) funciones
    tabla_explosion = [
        (1, 2, 4),
        (2, 4, 16),
        (3, 8, 256),
        (7, 128, 2**128)
    ]
    print(f"\n  [2] Explosión combinatoria del espacio de hipótesis 2^(2^n) (Diap. 15):")
    for n, filas, funcs in tabla_explosion:
        if n < 7:
            print(f"      • n = {n}: 2^{n} = {filas} filas de verdad -> 2^{filas} = {funcs} árboles distinguibles")
        else:
            print(f"      • n = {n}: 2^{n} = {filas} filas de verdad -> 2^{filas} ≈ {funcs:.4e} árboles (~3.4 x 10^38)")
    
    # 3. Partición ortogonal del espacio (Diap. 17)
    # Reglas: x1 < 2.6 -> R1; no -> x2 < 1.8 -> R2 (sí), R3 (no)
    def clasificar_espacio_ortogonal(x1, x2):
        if x1 < 2.6:
            return "R1"
        else:
            if x2 < 1.8:
                return "R2"
            else:
                return "R3"
                
    test_pts = [
        ((1.5, 2.0), "R1"),
        ((3.0, 1.2), "R2"),
        ((3.0, 2.5), "R3")
    ]
    particion_ok = all(clasificar_espacio_ortogonal(p[0], p[1]) == r for p, r in test_pts)
    print(f"\n  [3] Partición ortogonal del espacio R1, R2, R3 (Diap. 17):")
    for p, r in test_pts:
        res = clasificar_espacio_ortogonal(p[0], p[1])
        print(f"      • Punto {p} evaluado en árbol -> Región {res} (Esperado: {r})")

    # 4. Dilema de codificación categórica arbitraria (Diap. 21)
    # Color {Rojo, Azul, Amarillo}.
    # Codificación A: R=0, A=1, Am=2. Cortes permitidos sobre valores contiguos:
    #   umbral 0.5: {R} vs {A, Am}
    #   umbral 1.5: {R, A} vs {Am}
    #   Partición {A} vs {Am, R} es inalcanzable con un único corte en orden A.
    cortes_A = [
        set(["Rojo"]),                # corte < 0.5 -> {Rojo}
        set(["Rojo", "Azul"])         # corte < 1.5 -> {Rojo, Azul}
    ]
    particion_prohibida_en_A = set(["Azul"])
    codificacion_limita = particion_prohibida_en_A not in cortes_A
    print(f"\n  [4] Efecto de orden numérico forzado en categóricas (Diap. 21):")
    print(f"      • Codificación A (R=0, A=1, Am=2): partición {{Azul}} vs {{Rojo, Amarillo}} inalcanzable: {codificacion_limita}")
    
    ok = veg_ok and suelo_ok and particion_ok and codificacion_limita
    print(f"\n  => Módulo 08.1 Verificación: {'✓ APROBADA (100%)' if ok else '✗ REPROBADA'}")
    return ok

# =============================================================================
# MÓDULO 08.2: CRITERIOS DE IMPUREZA Y TRAZA A MANO ID3 "¿SALGO A CORRER?"
# =============================================================================

def verificar_criterios_impureza_y_traza_id3():
    print("\n" + "═" * 78)
    print("  MÓDULO 08.2: CRITERIOS DE IMPUREZA Y CONSTRUCCIÓN COMPLETA ID3 (DIAPS. 23–43)")
    print("═" * 78)
    
    # 1. Ejemplo Falla del Error vs Gini vs Entropía (Diaps. 25, 27, 30)
    # Región R: 14 A, 5 B (N=19)
    # R1: 6 A, 0 B (N1=6)
    # R2: 8 A, 5 B (N2=13)
    N = 19
    N1 = 6
    N2 = 13
    
    err_antes = error_clasificacion_distribucion([14, 5])
    err_r1 = error_clasificacion_distribucion([6, 0])
    err_r2 = error_clasificacion_distribucion([8, 5])
    err_despues = (N1 / N) * err_r1 + (N2 / N) * err_r2
    ganancia_err = err_antes - err_despues
    
    gini_antes = gini_distribucion([14, 5])
    gini_r1 = gini_distribucion([6, 0])
    gini_r2 = gini_distribucion([8, 5])
    gini_despues = (N1 / N) * gini_r1 + (N2 / N) * gini_r2
    ganancia_gini = gini_antes - gini_despues
    
    ent_antes = entropia_distribucion([14, 5])
    ent_r1 = entropia_distribucion([6, 0])
    ent_r2 = entropia_distribucion([8, 5])
    ent_despues = (N1 / N) * ent_r1 + (N2 / N) * ent_r2
    ganancia_ent = ent_antes - ent_despues
    
    print(f"  [1] Comparación analítica de impureza en división (14 A, 5 B) -> (6 A, 0 B) y (8 A, 5 B):")
    print(f"      • Error Clasificación: Antes = {err_antes:.3f}, Después = {err_despues:.3f}, Ganancia = {ganancia_err:.3f} (¡Ceguera total!)")
    print(f"      • Índice de Gini:      Antes = {gini_antes:.3f}, Después = {gini_despues:.3f}, Ganancia = {ganancia_gini:.3f} (Registra la mejora)")
    print(f"      • Entropía (Shannon):  Antes = {ent_antes:.3f}, Después = {ent_despues:.3f}, Ganancia = {ganancia_ent:.3f} (Máxima sensibilidad)")
    
    test1_ok = (abs(ganancia_err) < 1e-6) and (abs(ganancia_gini - 0.064) < 0.002) and (abs(ganancia_ent - 0.174) < 0.002)
    
    # 2. Segundo caso de ceguera del error (Diap. 31)
    # Región: 9 A, 3 B -> R1: 6 A, 0 B; R2: 3 A, 3 B
    err2_antes = error_clasificacion_distribucion([9, 3])
    err2_despues = (6 / 12) * error_clasificacion_distribucion([6, 0]) + (6 / 12) * error_clasificacion_distribucion([3, 3])
    gini2_antes = gini_distribucion([9, 3])
    gini2_despues = (6 / 12) * gini_distribucion([6, 0]) + (6 / 12) * gini_distribucion([3, 3])
    test2_ok = (abs(err2_antes - err2_despues) < 1e-6) and (abs(gini2_antes - gini2_despues - 0.125) < 1e-6)
    print(f"\n  [2] Segundo caso (9 A, 3 B) -> (6 A, 0 B) y (3 A, 3 B) (Diap. 31):")
    print(f"      • Error: Antes = {err2_antes:.3f}, Después = {err2_despues:.3f} (Ganancia = 0.000)")
    print(f"      • Gini:  Antes = {gini2_antes:.3f}, Después = {gini2_despues:.3f} (Ganancia = {gini2_antes - gini2_despues:.3f})")

    # 3. Tabla canónica oficial CS109: "¿Salgo a correr hoy?" (Diaps. 34–43)
    dataset_correr = [
        {"id": 1, "clima": "Soleado",  "temp": "Alta",  "viento": "Suave",  "corre": "Sí"},
        {"id": 2, "clima": "Soleado",  "temp": "Alta",  "viento": "Fuerte", "corre": "No"},
        {"id": 3, "clima": "Nublado",  "temp": "Alta",  "viento": "Suave",  "corre": "Sí"},
        {"id": 4, "clima": "Nublado",  "temp": "Media", "viento": "Fuerte", "corre": "Sí"},
        {"id": 5, "clima": "Nublado",  "temp": "Baja",  "viento": "Suave",  "corre": "Sí"},
        {"id": 6, "clima": "Lluvioso", "temp": "Media", "viento": "Suave",  "corre": "No"},
        {"id": 7, "clima": "Lluvioso", "temp": "Baja",  "viento": "Fuerte", "corre": "No"},
        {"id": 8, "clima": "Soleado",  "temp": "Media", "viento": "Suave",  "corre": "Sí"},
        {"id": 9, "clima": "Lluvioso", "temp": "Alta",  "viento": "Suave",  "corre": "No"},
        {"id": 10,"clima": "Soleado",  "temp": "Baja",  "viento": "Fuerte", "corre": "No"}
    ]
    
    # Entropía base H(S)
    si_base = sum(1 for d in dataset_correr if d["corre"] == "Sí")
    no_base = sum(1 for d in dataset_correr if d["corre"] == "No")
    h_s = entropia_distribucion([si_base, no_base])
    print(f"\n  [3] Traza manual ID3 paso a paso (Diaps. 34–43):")
    print(f"      • Entropía base H(S) con {si_base} Sí y {no_base} No: {h_s:.4f} bit (Máxima incertidumbre)")
    
    def calcular_ig(datos, atributo):
        h_padre = entropia_distribucion([
            sum(1 for d in datos if d["corre"] == "Sí"),
            sum(1 for d in datos if d["corre"] == "No")
        ])
        valores = set(d[atributo] for d in datos)
        h_cond = 0.0
        n_total = len(datos)
        desglose = {}
        for v in sorted(valores):
            sub = [d for d in datos if d[atributo] == v]
            n_sub = len(sub)
            si_sub = sum(1 for d in sub if d["corre"] == "Sí")
            no_sub = sum(1 for d in sub if d["corre"] == "No")
            h_v = entropia_distribucion([si_sub, no_sub])
            peso = n_sub / n_total
            h_cond += peso * h_v
            desglose[v] = (n_sub, si_sub, no_sub, h_v, peso)
        ig = h_padre - h_cond
        return ig, h_cond, desglose

    ig_clima, hc_clima, d_clima = calcular_ig(dataset_correr, "clima")
    ig_viento, hc_viento, d_viento = calcular_ig(dataset_correr, "viento")
    ig_temp, hc_temp, d_temp = calcular_ig(dataset_correr, "temp")
    
    print(f"      • Ganancias de información en la Raíz (Paso 2):")
    print(f"        - IG(Clima)       = 1.0 - {hc_clima:.3f} = {ig_clima:.3f}  [GANA LA RAÍZ]")
    print(f"        - IG(Viento)      = 1.0 - {hc_viento:.3f} = {ig_viento:.3f}")
    print(f"        - IG(Temperatura) = 1.0 - {hc_temp:.3f} = {ig_temp:.3f}")
    
    test_ig_ok = (abs(ig_clima - 0.60) < 0.01) and (abs(ig_viento - 0.125) < 0.01) and (abs(ig_temp - 0.049) < 0.01)
    
    # Recursión en rama Soleado (filas 1, 2, 8, 10)
    sub_soleado = [d for d in dataset_correr if d["clima"] == "Soleado"]
    ig_viento_sol, hc_v_sol, _ = calcular_ig(sub_soleado, "viento")
    ig_temp_sol, hc_t_sol, _ = calcular_ig(sub_soleado, "temp")
    print(f"      • Recursión en la rama 'Soleado' (4 puntos: 2 Sí, 2 No) (Paso 4):")
    print(f"        - IG(Viento | Soleado)      = 1.0 - {hc_v_sol:.2f} = {ig_viento_sol:.2f}  [GANA LA RAMA SOLEADO]")
    print(f"        - IG(Temperatura | Soleado) = 1.0 - {hc_t_sol:.2f} = {ig_temp_sol:.2f}")
    
    test_rec_ok = (abs(ig_viento_sol - 1.0) < 1e-6) and (abs(ig_temp_sol - 0.5) < 1e-6)
    
    # Función del árbol aprendido para predecir
    def predecir_arbol_correr(clima, temp, viento):
        if clima == "Nublado":
            return "Sí"
        elif clima == "Lluvioso":
            return "No"
        elif clima == "Soleado":
            if viento == "Suave":
                return "Sí"
            else:
                return "No"
        return "Indeciso"

    # Verificación de error de entrenamiento (10/10)
    ent_preds = [predecir_arbol_correr(d["clima"], d["temp"], d["viento"]) == d["corre"] for d in dataset_correr]
    acc_train = sum(ent_preds) / len(ent_preds)
    print(f"      • Precisión sobre los 10 datos de entrenamiento: {acc_train * 100:.1f}% (Error 0%)")
    
    # Evaluación sobre días nuevos (Diap. 43)
    dias_nuevos = [
        ("Soleado", "Baja", "Suave", "Sí"),
        ("Nublado", "Media", "Suave", "Sí"),
        ("Lluvioso", "Alta", "Fuerte", "No")
    ]
    test_dias_ok = all(predecir_arbol_correr(c, t, v) == esper for c, t, v, esper in dias_nuevos)
    print(f"      • Predicciones sobre días nuevos (Diap. 43):")
    for c, t, v, esper in dias_nuevos:
        p = predecir_arbol_correr(c, t, v)
        print(f"        - Día ({c}, {t}, {v}) -> {p} (Esperado: {esper})")

    ok = test1_ok and test2_ok and test_ig_ok and test_rec_ok and (acc_train == 1.0) and test_dias_ok
    print(f"\n  => Módulo 08.2 Verificación: {'✓ APROBADA (100%)' if ok else '✗ REPROBADA'}")
    return ok

# =============================================================================
# MÓDULO 08.3: CONTROL DE COMPLEJIDAD Y PODA POR COSTO-COMPLEJIDAD (CART)
# =============================================================================

def verificar_poda_costo_complejidad():
    print("\n" + "═" * 78)
    print("  MÓDULO 08.3: CONTROL DE COMPLEJIDAD Y PODA WEAKEST-LINK CART (DIAPS. 44–55)")
    print("═" * 78)
    
    # Problema de las diapositivas 51-54:
    # 100 puntos totales (50 Clase A, 50 Clase B).
    # Árbol T0 completo:
    #   Raíz x1 < 5:
    #     sí -> L1: 36 A, 4 B (predice A, 4 errores)
    #     no -> x2 < 3:
    #             sí -> L2: 12 A, 8 B (predice A, 8 errores)
    #             no -> L3: 2 A, 38 B (predice B, 2 errores)
    # Hojas totales |T0| = 3. Errores totales = 4 + 8 + 2 = 14 -> Error(T0) = 0.14.
    
    err_t0 = (4 + 8 + 2) / 100.0
    hojas_t0 = 3
    print(f"  [1] Árbol completo T0 (Diap. 51):")
    print(f"      • Hojas |T0| = {hojas_t0}")
    print(f"      • Errores por hoja: L1=4, L2=8, L3=2 -> Total = 14")
    print(f"      • Error(T0) = {err_t0:.2f}")
    
    # Candidatos a poda (Diap. 52):
    # Candidato 1: Colapsar nodo x2 < 3 a una hoja (L2 + L3 = 14 A, 46 B -> predice B, 14 errores)
    #   Nuevo error: (4 + 14) / 100 = 0.18
    #   Hojas eliminadas: 3 - 2 = 1
    #   g(x2 < 3) = (0.18 - 0.14) / (3 - 2) = 0.04
    err_cand1 = (4 + 14) / 100.0
    hojas_cand1 = 2
    g_cand1 = (err_cand1 - err_t0) / (hojas_t0 - hojas_cand1)
    
    # Candidato 2: Colapsar la raíz x1 < 5 a una sola hoja (50 A, 50 B -> 50 errores)
    #   Nuevo error: 50 / 100 = 0.50
    #   Hojas eliminadas: 3 - 1 = 2
    #   g(raíz) = (0.50 - 0.14) / (3 - 1) = 0.36 / 2 = 0.18
    err_cand2 = 50.0 / 100.0
    hojas_cand2 = 1
    g_cand2 = (err_cand2 - err_t0) / (hojas_t0 - hojas_cand2)
    
    print(f"\n  [2] Evaluación de subárboles candidatos (Diap. 52):")
    print(f"      • Colapsar nodo x2 < 3: Error = {err_cand1:.2f}, g = {g_cand1:.4f}  [ESLABÓN MÁS DÉBIL]")
    print(f"      • Colapsar raíz x1 < 5:  Error = {err_cand2:.2f}, g = {g_cand2:.4f}")
    
    # T1 resulta de podar x2 < 3:
    #   T1 tiene 2 hojas (L1 y la nueva hoja B con 14 errores).
    #   Error(T1) = 0.18, |T1| = 2.
    # Desde T1, solo queda podar la raíz:
    #   g(T1 -> T2) = (0.50 - 0.18) / (2 - 1) = 0.32
    g_t1_a_t2 = (0.50 - err_cand1) / (2 - 1)
    print(f"\n  [3] Secuencia completa anidada de subárboles (Diap. 53):")
    print(f"      • T0: |T| = 3, Error = {err_t0:.2f}")
    print(f"      • T1: |T| = 2, Error = {err_cand1:.2f} (transición en alpha = {g_cand1:.2f})")
    print(f"      • T2: |T| = 1, Error = 0.50 (transición en alpha = {g_t1_a_t2:.2f})")
    
    # Verificación analítica de la perilla alpha (Diap. 54)
    # C(T) = Error(T) + alpha * |T|
    #   C(T0) = 0.14 + 3 * alpha
    #   C(T1) = 0.18 + 2 * alpha
    #   C(T2) = 0.50 + 1 * alpha
    def costo_complejidad(err, hojas, alpha):
        return err + alpha * hojas

    # Prueba con alpha = 0.10:
    alpha_test = 0.10
    c_t0 = costo_complejidad(0.14, 3, alpha_test)
    c_t1 = costo_complejidad(0.18, 2, alpha_test)
    c_t2 = costo_complejidad(0.50, 1, alpha_test)
    
    print(f"\n  [4] Verificación de costos con alpha = {alpha_test} (Diap. 54):")
    print(f"      • C(T0) = 0.14 + 3*(0.10) = {c_t0:.2f}")
    print(f"      • C(T1) = 0.18 + 2*(0.10) = {c_t1:.2f}  [ÓPTIMO]")
    print(f"      • C(T2) = 0.50 + 1*(0.10) = {c_t2:.2f}")
    
    t1_gana = (c_t1 < c_t0) and (c_t1 < c_t2)
    test_g_ok = (abs(g_cand1 - 0.04) < 1e-6) and (abs(g_t1_a_t2 - 0.32) < 1e-6)
    
    ok = (err_t0 == 0.14) and (err_cand1 == 0.18) and test_g_ok and t1_gana
    print(f"\n  => Módulo 08.3 Verificación: {'✓ APROBADA (100%)' if ok else '✗ REPROBADA'}")
    return ok

# =============================================================================
# MÓDULO 08.4: ÁRBOLES DE REGRESIÓN CART Y TRAZA MANUAL A MANO 6 PUNTOS
# =============================================================================

def verificar_arboles_regresion_y_traza_6_puntos():
    print("\n" + "═" * 78)
    print("  MÓDULO 08.4: ÁRBOLES DE REGRESIÓN (CART CONTINUO) Y TRAZA A MANO (DIAPS. 56–68)")
    print("═" * 78)
    
    # Datos de la Diapositiva 59:
    # x = [1, 2, 3, 4, 5, 6]
    # y = [2, 3, 2.5, 8, 9, 8.5]
    puntos = [(1, 2.0), (2, 3.0), (3, 2.5), (4, 8.0), (5, 9.0), (6, 8.5)]
    N = len(puntos)
    
    def evaluar_corte_regresion(umbral):
        r1 = [p[1] for p in puntos if p[0] < umbral]
        r2 = [p[1] for p in puntos if p[0] >= umbral]
        n1 = len(r1)
        n2 = len(r2)
        mean1, var1 = mse_region(r1)
        mean2, var2 = mse_region(r2)
        mse_pond = (n1 / N) * var1 + (n2 / N) * var2
        return {
            "umbral": umbral,
            "r1": r1, "mean1": mean1, "var1": var1,
            "r2": r2, "mean2": mean2, "var2": var2,
            "mse_pond": mse_pond
        }
        
    candidatos = [2.5, 3.5, 4.5]
    resultados = [evaluar_corte_regresion(u) for u in candidatos]
    
    print(f"  [1] Evaluación a mano de umbrales candidatos en 6 puntos (Diap. 59):")
    print(f"      Puntos: x={ [p[0] for p in puntos] }, y={ [p[1] for p in puntos] }")
    for res in resultados:
        print(f"      • Corte x < {res['umbral']}:")
        print(f"        - R1 (n={len(res['r1'])}): media = {res['mean1']:.2f}, MSE = {res['var1']:.2f}")
        print(f"        - R2 (n={len(res['r2'])}): media = {res['mean2']:.2f}, MSE = {res['var2']:.2f}")
        print(f"        - MSE Ponderado = {res['mse_pond']:.2f}")
        
    mejor = min(resultados, key=lambda r: r["mse_pond"])
    print(f"      => Ganador: Umbral x < {mejor['umbral']} con MSE Ponderado = {mejor['mse_pond']:.2f}")
    
    # Verificaciones numéricas de diapositiva 59:
    # x < 2.5: MSE pond ≈ 4.67
    # x < 3.5: MSE pond ≈ 0.17
    # x < 4.5: MSE pond ≈ 3.89
    res_25 = next(r for r in resultados if r["umbral"] == 2.5)
    res_35 = next(r for r in resultados if r["umbral"] == 3.5)
    res_45 = next(r for r in resultados if r["umbral"] == 4.5)
    
    test_mse_ok = (
        abs(res_25["mse_pond"] - 4.67) < 0.05 and
        abs(res_35["mse_pond"] - 0.17) < 0.05 and
        abs(res_45["mse_pond"] - 3.89) < 0.05 and
        mejor["umbral"] == 3.5
    )
    
    # Predicciones del modelo ganador:
    # x = 2 -> cae en x < 3.5 -> predice mean1 = 2.5
    # x = 5 -> cae en x >= 3.5 -> predice mean2 = 8.5
    pred_x2 = res_35["mean1"]
    pred_x5 = res_35["mean2"]
    print(f"\n  [2] Predicciones del árbol de regresión (Diap. 59):")
    print(f"      • x = 2 -> ŷ = {pred_x2:.1f} (Esperado: 2.5)")
    print(f"      • x = 5 -> ŷ = {pred_x5:.1f} (Esperado: 8.5)")
    test_pred_ok = (pred_x2 == 2.5) and (pred_x5 == 8.5)
    
    # Comparativa de profundidades (Diaps. 60–62):
    # Prof 1: MSE 0.39 (Sesgo alto)
    # Prof 2: MSE 0.06
    # Prof 3: MSE 0.04 (Equilibrio)
    # Prof 10: MSE 0.00 (Memorización / Varianza extrema)
    print(f"\n  [3] Escala de profundidades y sobreajuste (Diaps. 60–62):")
    print(f"      • Profundidad 1 (2 hojas): MSE ≈ 0.39 (Función escalonada rústica, subajuste)")
    print(f"      • Profundidad 2 (4 hojas): MSE ≈ 0.06 (Sigue la curva general)")
    print(f"      • Profundidad 3 (8 hojas): MSE ≈ 0.04 (Buen balance sesgo-varianza)")
    print(f"      • Profundidad 10 (24 hojas para 24 pts): MSE = 0.00 (Memorización de ruido)")

    ok = test_mse_ok and test_pred_ok
    print(f"\n  => Módulo 08.4 Verificación: {'✓ APROBADA (100%)' if ok else '✗ REPROBADA'}")
    return ok

# =============================================================================
# EJECUCIÓN INTEGRADA Y MENÚ CLI
# =============================================================================

def ejecutar_suite_cd08():
    print("\n" + "═" * 78)
    print("  EJECUTANDO SUITE COMPLETA CD08 — ÁRBOLES DE CLASIFICACIÓN Y REGRESIÓN")
    print("═" * 78)
    ok1 = verificar_limites_lineales_y_explosion()
    ok2 = verificar_criterios_impureza_y_traza_id3()
    ok3 = verificar_poda_costo_complejidad()
    ok4 = verificar_arboles_regresion_y_traza_6_puntos()
    
    print("\n" + "═" * 78)
    if ok1 and ok2 and ok3 and ok4:
        print("  🎉 ¡TODAS LAS VERIFICACIONES DE CD08 (ÁRBOLES CART/ID3) APROBADAS! (100%)")
    else:
        print("  ❌ HUBO ERRORES EN LA VERIFICACIÓN DE CD08.")
    print("═" * 78 + "\n")
    return ok1 and ok2 and ok3 and ok4

def ejecutar_suite_automatica():
    """Alias para orquestador unificado."""
    return ejecutar_suite_cd08()

def menu_arboles_decision():
    """Menú interactivo para CD08."""
    while True:
        print("\n" + "═" * 78)
        print("  🌲 MÓDULO CD08: ÁRBOLES DE DECISIÓN (CART / ID3) — DOCTORADO UCN")
        print("═" * 78)
        print(" [1] Módulo 08.1: Límites Lineales, Hipótesis de Árbol y Espacio 2^(2^n)")
        print(" [2] Módulo 08.2: Impureza (Error, Gini, Entropía) y Traza ID3 ('¿Corre?')")
        print(" [3] Módulo 08.3: Poda por Costo-Complejidad (CART Weakest Link 100 pts)")
        print(" [4] Módulo 08.4: Árboles de Regresión Continuos (Criterio MSE y 6 Puntos)")
        print(" [5] Ejecutar Suite Completa Automática (Todos los Módulos)")
        print(" [0] Volver al Menú Principal")
        print("─" * 78)
        try:
            opc = input("Selecciona una opción [0-5]: ").strip().lower()
            if opc == '1':
                verificar_limites_lineales_y_explosion()
            elif opc == '2':
                verificar_criterios_impureza_y_traza_id3()
            elif opc == '3':
                verificar_poda_costo_complejidad()
            elif opc == '4':
                verificar_arboles_regresion_y_traza_6_puntos()
            elif opc in ('5', 't', 'all'):
                ejecutar_suite_cd08()
            elif opc in ('0', 'q', 'v', 'salir'):
                break
            else:
                print("⚠️ Opción no válida.")
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('--auto', '-a', '--test'):
        success = ejecutar_suite_cd08()
        sys.exit(0 if success else 1)
    else:
        menu_arboles_decision()
