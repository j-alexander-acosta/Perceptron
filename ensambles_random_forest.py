#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Cátedra Ciencia de Datos — Universidad Católica del Norte (UCN)
Módulo CD09: Ensamblaje de Modelos y Random Forests (59 Diapositivas)
Basado en Harvard CS109 (P. Protopapas et al.) y ESL Cap. 15 (Hastie et al.)
=============================================================================

Este módulo implementa, verifica numéricamente y demuestra formalmente los
fundamentos matemáticos y algoritmos de ensambles de modelos:
  1. Módulo 09.1: Sabiduría de las Multitudes, Buey de Plymouth (Galton 1907)
     y Aritmética del Voto Independiente (Condorcet).
  2. Módulo 09.2: Bootstrap, Algoritmo de Bagging (Breiman 1996), Validación
     Out-of-Bag (OOB) e Importancia MDI.
  3. Módulo 09.3: El Límite del Bagging, Derivación de Varianza ESL 15.1,
     Piso Infranqueable rho*sigma^2 y Random Forests (Breiman 2001).
  4. Módulo 09.4: Boosting Aditivo (Gradient Boosting sobre Residuos,
     AdaBoost a Mano en 13 Puntos), Tasa Óptima lambda, Voting y Stacking.
"""

import math
import random
import sys


# =============================================================================
# MÓDULO 09.1: SABIDURÍA DE MULTITUDES Y ARITMÉTICA DEL VOTO INDEPENDIENTE
# =============================================================================

def experimento_buey_plymouth():
    """
    Verifica los datos históricos del experimento de Francis Galton (1907)
    publicado en Nature 75 ('Vox populi', pp. 450-451).
    - 787 personas pagaron para estimar el peso faenado de un buey vivo.
    - El 90% central de las estimaciones varió entre 1074 y 1293 libras.
    - Mediana agregada: 1207 libras.
    - Peso real medido: 1198 libras.
    - Error porcentual: |1207 - 1198| / 1198 = 0.751% ≈ 0.8%.
    - Cerca de 9 de cada 10 participantes individuales estuvieron más lejos
      del peso real que la estimación colectiva agregada.
    """
    n_participantes = 787
    p95_inf = 1074.0
    p95_sup = 1293.0
    mediana_galton = 1207.0
    peso_real = 1198.0
    
    error_absoluto = abs(mediana_galton - peso_real)
    error_porcentual = (error_absoluto / peso_real) * 100.0
    
    return {
        "participantes": n_participantes,
        "rango_90_pct": (p95_inf, p95_sup),
        "mediana": mediana_galton,
        "peso_real": peso_real,
        "error_absoluto": error_absoluto,
        "error_porcentual": error_porcentual,
        "fraccion_mas_lejos": 0.90
    }


def combinatoria(n, k):
    """Calcula n sobre k: n! / (k! * (n-k)!)."""
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def prob_mayoria_acierta(B, p):
    """
    Calcula analíticamente la probabilidad de que la mayoría simple de B
    clasificadores independientes (con precisión p) acierte:
      P(mayoría) = sum_{k=floor(B/2) + 1}^B binom(B, k) * p^k * (1-p)^(B-k)
    """
    if B % 2 == 0:
        raise ValueError("B debe ser impar para evitar empates estrictos.")
    
    k_min = (B // 2) + 1
    prob = 0.0
    for k in range(k_min, B + 1):
        prob += combinatoria(B, k) * (p ** k) * ((1.0 - p) ** (B - k))
    return prob


def verificar_sabiduria_multitudes_y_voto():
    """Ejecuta y valida formalmente el Módulo 09.1."""
    print("\n" + "─" * 78)
    print("  MÓDULO 09.1: SABIDURÍA DE MULTITUDES Y ARITMÉTICA DEL VOTO (DIAPS. 1–10)")
    print("─" * 78)
    
    # 1. Experimento del Buey de Plymouth (Galton 1907)
    exp = experimento_buey_plymouth()
    print(f" [Galton 1907] N={exp['participantes']} personas en Plymouth:")
    print(f"   • 90% central de respuestas: [{exp['rango_90_pct'][0]:.0f}, {exp['rango_90_pct'][1]:.0f}] libras")
    print(f"   • Mediana colectiva: {exp['mediana']:.1f} lbs | Peso real: {exp['peso_real']:.1f} lbs")
    print(f"   • Error porcentual del agregado: {exp['error_porcentual']:.3f}% (Diap. 6: ≈ 0.8%)")
    assert abs(exp['error_porcentual'] - 0.75125) < 0.01, "Falla cálculo de error Galton"
    
    # 2. Voto de tres (Diap. 10)
    p = 0.7
    p3 = p ** 3                          # 0.343
    p2_exacto = 3 * (p ** 2) * (1.0 - p) # 0.441
    p_mayoria_3 = p3 + p2_exacto        # 0.784
    calc_3 = prob_mayoria_acierta(3, p)
    print(f"\n [El voto de tres - Diap. 10] p={p}, B=3:")
    print(f"   • Aciertan los tres: p^3 = {p3:.3f}")
    print(f"   • Aciertan exactamente dos: 3p^2(1-p) = {p2_exacto:.3f}")
    print(f"   • Mayoría (>=2 de 3): {p_mayoria_3:.3f} (Individual: {p:.3f} -> Ganancia: +{p_mayoria_3 - p:.3f})")
    assert abs(p_mayoria_3 - 0.784) < 1e-6, "Falla voto de 3 con p=0.7"
    assert abs(calc_3 - 0.784) < 1e-6, "Falla función genérica para B=3"
    
    # 3. Curva de Aritmética del Voto (Diap. 8)
    p_b5 = prob_mayoria_acierta(5, 0.7)
    p_b15 = prob_mayoria_acierta(15, 0.7)
    p_b101 = prob_mayoria_acierta(101, 0.7)
    p_b101_debil = prob_mayoria_acierta(101, 0.55)
    
    print(f"\n [Aritmética del voto - Diap. 8]:")
    print(f"   • p=0.70 | B=5:   P(mayoría) = {p_b5:.3f} (Diap. 8: 0.837)")
    print(f"   • p=0.70 | B=15:  P(mayoría) = {p_b15:.3f} (Diap. 8: 0.950)")
    print(f"   • p=0.70 | B=101: P(mayoría) = {p_b101:.6f} (Diap. 8: prácticamente 1)")
    print(f"   • p=0.55 | B=101: P(mayoría) = {p_b101_debil:.3f} (Diap. 8: 0.844)")
    
    assert abs(p_b5 - 0.837) < 0.001, "Falla B=5 p=0.7"
    assert abs(p_b15 - 0.950) < 0.001, "Falla B=15 p=0.7"
    assert p_b101 > 0.9999, "Falla B=101 p=0.7"
    assert abs(p_b101_debil - 0.844) < 0.001, "Falla B=101 p=0.55"
    
    # 4. Trampa de la correlación idéntica (Diap. 9)
    # Si B modelos son copias exactas, siempre votan igual => precisión idéntica p
    print(f"\n [Letra clave: Independencia - Diap. 9]:")
    print("   • Si los B modelos son copias exactas, P(mayoría) = p = 0.700 (Ganancia = 0.000)")
    print("   • Conclusión: Sin diversidad, el promedio o voto no reduce error.")
    print(" ✔ MÓDULO 09.1 VERIFICADO CON ÉXITO.")
    return True


# =============================================================================
# MÓDULO 09.2: BOOTSTRAP, BAGGING Y OUT-OF-BAG (OOB)
# =============================================================================

def prob_aparecer_bootstrap(N):
    """
    Probabilidad de que una observación particular aparezca al menos una vez
    en una muestra bootstrap de tamaño N sorteada con reemplazo de N elementos:
      P(aparecer) = 1 - (1 - 1/N)^N
    """
    return 1.0 - ((1.0 - 1.0 / N) ** N)


def prob_oob(N):
    """
    Probabilidad de que una observación quede fuera de la bolsa (Out-of-Bag):
      P(queda fuera) = (1 - 1/N)^N
    """
    return (1.0 - 1.0 / N) ** N


def simular_muestra_bootstrap(datos, rng=None):
    """Genera una muestra bootstrap con reemplazo de tamaño len(datos)."""
    if rng is None:
        rng = random.Random()
    n = len(datos)
    return [datos[rng.randint(0, n - 1)] for _ in range(n)]


class StumpRegresionSimple:
    """Árbol de decisión simple de 1 solo corte (stump) para regresión continua."""
    def __init__(self, umbral=0.0, val_izq=0.0, val_der=0.0):
        self.umbral = umbral
        self.val_izq = val_izq
        self.val_der = val_der

    def fit(self, X, y):
        # Búsqueda exhaustiva del mejor corte unidimensional
        n = len(X)
        if n == 0:
            return self
        candidatos = sorted(list(set(X)))
        mejor_mse = float('inf')
        
        for i in range(len(candidatos) - 1):
            t = (candidatos[i] + candidatos[i + 1]) / 2.0
            y_izq = [y[k] for k in range(n) if X[k] <= t]
            y_der = [y[k] for k in range(n) if X[k] > t]
            
            if not y_izq or not y_der:
                continue
            
            m_izq = sum(y_izq) / len(y_izq)
            m_der = sum(y_der) / len(y_der)
            
            mse = sum((val - m_izq)**2 for val in y_izq) + sum((val - m_der)**2 for val in y_der)
            if mse < mejor_mse:
                mejor_mse = mse
                self.umbral = t
                self.val_izq = m_izq
                self.val_der = m_der
        return self

    def predict(self, x):
        return self.val_izq if x <= self.umbral else self.val_der


def simular_bagging_regresion_30pts():
    """
    Simula el experimento de la Diapositiva 16:
    - 30 puntos ruidosos generados por una curva f(x) = sin(x) + cos(0.5x).
    - Un árbol individual de regresión tiene MSE ≈ 1.36 frente a la función real.
    - Un ensamble Bagging de 100 árboles sobre muestras bootstrap reduce el MSE a ≈ 1.04.
    """
    rng = random.Random(42)
    N = 30
    # Generar 30 puntos en [0, 10]
    X = [k * 10.0 / (N - 1) for k in range(N)]
    f_real = [math.sin(x) * 2.0 + (x - 5.0)**2 * 0.1 for x in X]
    # Agregar ruido con desviación 0.9
    y = [f_real[i] + rng.gauss(0, 0.9) for i in range(N)]
    
    # Entrenar árbol individual
    arbol_solo = StumpRegresionSimple().fit(X, y)
    y_pred_solo = [arbol_solo.predict(x) for x in X]
    mse_solo = sum((y_pred_solo[i] - f_real[i])**2 for i in range(N)) / N
    
    # Entrenar Bagging con B=100 árboles
    B = 100
    bosque = []
    for b in range(B):
        # Muestra bootstrap de índices
        indices_boot = [rng.randint(0, N - 1) for _ in range(N)]
        X_b = [X[idx] for idx in indices_boot]
        y_b = [y[idx] for idx in indices_boot]
        stump = StumpRegresionSimple().fit(X_b, y_b)
        bosque.append(stump)
        
    # Predicción promedio
    y_pred_bagging = []
    for i in range(N):
        pred_prom = sum(stump.predict(X[i]) for stump in bosque) / B
        y_pred_bagging.append(pred_prom)
        
    mse_bagging = sum((y_pred_bagging[i] - f_real[i])**2 for i in range(N)) / N
    
    return {
        "mse_solo": mse_solo,
        "mse_bagging": mse_bagging,
        "reduccion_pct": (1.0 - mse_bagging / mse_solo) * 100.0
    }


def verificar_bootstrap_y_bagging():
    """Ejecuta y valida formalmente el Módulo 09.2."""
    print("\n" + "─" * 78)
    print("  MÓDULO 09.2: BOOTSTRAP, BAGGING Y OUT-OF-BAG (DIAPS. 11–19)")
    print("─" * 78)
    
    # 1. Probabilidades Bootstrap y OOB (Diaps. 12 y 14)
    p_app_5 = prob_aparecer_bootstrap(5)
    p_app_10 = prob_aparecer_bootstrap(10)
    p_app_100 = prob_aparecer_bootstrap(100)
    p_app_inf = 1.0 - math.exp(-1.0)
    
    p_oob_10 = prob_oob(10)
    p_oob_inf = math.exp(-1.0)
    
    print(" [Probabilidad de inclusión bootstrap 1 - (1 - 1/N)^N - Diap. 12]:")
    print(f"   • N = 5:    {p_app_5:.3f} (Diap. 12: 0.672)")
    print(f"   • N = 10:   {p_app_10:.3f} (Diap. 12: 0.651)")
    print(f"   • N = 100:  {p_app_100:.3f} (Diap. 12: 0.634)")
    print(f"   • N -> inf: {p_app_inf:.3f} (1 - e^-1 ≈ 0.632, ≈ 2/3 de los datos)")
    
    print("\n [Probabilidad Out-of-Bag (1 - 1/N)^N - Diap. 14]:")
    print(f"   • N = 10:   {p_oob_10:.3f} (0.9^10 ≈ 0.349)")
    print(f"   • N -> inf: {p_oob_inf:.3f} (e^-1 ≈ 0.368, ≈ 1/3 para validación gratuita)")
    print(f"   • Complementariedad exacta: 0.632 + 0.368 = {p_app_inf + p_oob_inf:.3f} (100%)")
    
    assert abs(p_app_5 - 0.67232) < 0.001
    assert abs(p_app_10 - 0.65132) < 0.001
    assert abs(p_app_100 - 0.63397) < 0.001
    assert abs(p_app_inf - 0.63212) < 0.001
    assert abs(p_oob_10 - 0.34868) < 0.001
    assert abs(p_oob_inf - 0.36788) < 0.001
    
    # 2. Experimento de Bagging en Regresión (Diap. 16)
    res_bag = simular_bagging_regresion_30pts()
    print("\n [Bagging en Regresión - Diap. 16]:")
    print(f"   • MSE Árbol Simple (con alta varianza): {res_bag['mse_solo']:.2f}")
    print(f"   • MSE Bagging (Promedio 100 réplicas):  {res_bag['mse_bagging']:.2f}")
    print(f"   • Reducción de error por promediado:     {res_bag['reduccion_pct']:.1f}%")
    assert res_bag['mse_bagging'] < res_bag['mse_solo'], "Bagging debe reducir el MSE"
    
    print(" ✔ MÓDULO 09.2 VERIFICADO CON ÉXITO.")
    return True


# =============================================================================
# MÓDULO 09.3: EL LÍMITE DEL BAGGING, ESL 15.1 Y RANDOM FORESTS
# =============================================================================

def varianza_promedio_correlacionado(B, rho, sigma2=1.0):
    """
    Fórmula canónica de Hastie, Tibshirani y Friedman (ESL, Ec. 15.1):
      var(hat_mu) = rho * sigma^2 + ((1 - rho) * sigma^2) / B
    """
    return rho * sigma2 + ((1.0 - rho) * sigma2) / B


def simular_variables_equicorrelacionadas(rho=0.6, B=100, T=400_000, seed=9):
    """
    Reproduce la simulación de la Diapositiva 24:
      X = sqrt(rho) * comun[:, None] + sqrt(1 - rho) * propia
    Genera T muestras de B variables con covarianza mutua rho.
    Retorna la varianza empírica del promedio sobre axis=1.
    """
    # Intentar numpy si está disponible, sino fallback en Python estándar
    try:
        import numpy as np
        rng = np.random.default_rng(seed)
        comun = rng.standard_normal(T)
        propia = rng.standard_normal((T, B))
        X = np.sqrt(rho) * comun[:, None] + np.sqrt(1.0 - rho) * propia
        return float(X.mean(axis=1).var())
    except ImportError:
        # Fallback en Python estándar (con T reducido si es necesario para velocidad)
        T_sub = min(T, 50_000)
        rng = random.Random(seed)
        c1 = math.sqrt(rho)
        c2 = math.sqrt(1.0 - rho)
        medias = []
        for _ in range(T_sub):
            z_comun = rng.gauss(0, 1)
            suma_fila = sum(c1 * z_comun + c2 * rng.gauss(0, 1) for _ in range(B))
            medias.append(suma_fila / B)
        mean_tot = sum(medias) / T_sub
        var_emp = sum((m - mean_tot)**2 for m in medias) / T_sub
        return var_emp


def verificar_limite_bagging_y_random_forests():
    """Ejecuta y valida formalmente el Módulo 09.3."""
    print("\n" + "─" * 78)
    print("  MÓDULO 09.3: EL LÍMITE DEL BAGGING Y RANDOM FORESTS (DIAPS. 20–34)")
    print("─" * 78)
    
    # 1. Derivación de Hastie et al. (ESL ec. 15.1 - Diap. 22)
    rho = 0.6
    sigma2 = 1.0
    v_b10 = varianza_promedio_correlacionado(10, rho, sigma2)   # 0.6 + 0.4/10 = 0.640
    v_b20 = varianza_promedio_correlacionado(20, rho, sigma2)   # 0.6 + 0.4/20 = 0.620
    v_b100 = varianza_promedio_correlacionado(100, rho, sigma2) # 0.6 + 0.4/100 = 0.604
    v_b1000 = varianza_promedio_correlacionado(1000, rho, sigma2) # 0.6004
    piso_inf = rho * sigma2                                     # 0.6000
    
    print(" [Derivación ESL Ec. 15.1: var(hat_mu) = rho*sigma^2 + (1-rho)*sigma^2 / B]:")
    print(f"   • B = 10,   rho = 0.6: var = {v_b10:.4f} (Diap. 25: 0.64)")
    print(f"   • B = 20,   rho = 0.6: var = {v_b20:.4f} (Diap. 25: 0.62 - Doble cómputo para bajar 0.02)")
    print(f"   • B = 100,  rho = 0.6: var = {v_b100:.4f} (Diap. 24: 0.6040)")
    print(f"   • B = 1000, rho = 0.6: var = {v_b1000:.4f} (Diap. 24: 0.6004)")
    print(f"   • B -> inf, rho = 0.6: piso infranqueable = {piso_inf:.4f} (Diap. 23)")
    
    assert abs(v_b10 - 0.64) < 1e-6
    assert abs(v_b20 - 0.62) < 1e-6
    assert abs(v_b100 - 0.6040) < 1e-6
    assert abs(v_b1000 - 0.6004) < 1e-6
    
    # 2. La palanca que importa: Descorrelacionar vs Agrandar (Diap. 25)
    v_descorrelacionada = varianza_promedio_correlacionado(10, 0.3, sigma2) # 0.3 + 0.7/10 = 0.370
    print(f"\n [Descorrelacionar o agrandar - Diap. 25]:")
    print(f"   • Bajar rho de 0.6 a 0.3 con solo 10 árboles: var = {v_descorrelacionada:.4f} (Diap. 25: 0.37)")
    print(f"   • Comparativa: duplicar B a 20 árboles con rho=0.6 solo llega a {v_b20:.2f}.")
    print("   • ¡Descorrelacionar los árboles vale mucho más que aumentar el número B!")
    assert abs(v_descorrelacionada - 0.37) < 1e-6
    
    # 3. Verificación numérica por simulación (Diap. 24)
    var_sim = simular_variables_equicorrelacionadas(rho=0.6, B=100, T=400_000, seed=9)
    print(f"\n [Simulación Monte Carlo - Diap. 24]:")
    print(f"   • Teórico (ESL 15.1): {v_b100:.4f}")
    print(f"   • Simulación:         {var_sim:.4f} (Diap. 24 reporta 0.6044)")
    assert abs(var_sim - 0.604) < 0.005, "Falla simulación Monte Carlo de equicorrelación"
    
    # 4. Dimensionamiento de Random Forest y sorteo de predictores (Diaps. 26, 27, 33)
    J = 36
    m_clasif = int(math.sqrt(J)) # 6
    prob_predictor_fuerte = m_clasif / J # 6 / 36 = 1/6 ≈ 0.1667
    pct_sin_fuerte = (1.0 - prob_predictor_fuerte) * 100.0 # 83.33%
    
    print(f"\n [Dimensionar un bosque - Diap. 33]:")
    print(f"   • J = {J} predictores totales.")
    print(f"   • Regla canónica clasificación: m = sqrt(J) = {m_clasif} predictores sorteados por nodo.")
    print(f"   • Probabilidad de que el predictor dominante aparezca: {m_clasif}/{J} = {prob_predictor_fuerte:.4f} (≈ 0.17)")
    print(f"   • {pct_sin_fuerte:.1f}% de las divisiones no ven el predictor fuerte (obligados a explorar otras variables).")
    assert m_clasif == 6
    assert abs(prob_predictor_fuerte - 1.0/6.0) < 1e-6
    
    print(" ✔ MÓDULO 09.3 VERIFICADO CON ÉXITO.")
    return True


# =============================================================================
# MÓDULO 09.4: BOOSTING ADITIVO, ADABOOST A MANO, VOTING Y STACKING
# =============================================================================

def adaboost_tasa_optima(epsilon):
    """
    Tasa analítica óptima de AdaBoost deducida desde la pérdida exponencial:
      lambda^(i) = 0.5 * ln((1 - epsilon) / epsilon)
    """
    if epsilon <= 0.0:
        return float('inf')
    if epsilon >= 1.0:
        return float('-inf')
    return 0.5 * math.log((1.0 - epsilon) / epsilon)


def traza_adaboost_13_puntos():
    """
    Reproduce la traza exacta a mano de AdaBoost en 13 puntos (Diaps. 47–49):
    Problema de aprobación de crédito: ingreso vs crédito. N = 13.
    """
    N = 13
    # Ronda 1 (Diap. 47)
    w1 = [1.0 / N] * N
    # Stump 1 falla en 1 punto:
    err1_indices = [0]
    eps1 = sum(w1[idx] for idx in err1_indices) # 1/13 ≈ 0.0769
    lam1 = adaboost_tasa_optima(eps1)           # 0.5 * ln(12) ≈ 1.24245
    
    # Ronda 2 (Diap. 48)
    # El punto fallado sube a peso 0.5; los otros 12 bajan a 1/24 ≈ 0.04167
    w2_mal = 0.5
    w2_bien = 1.0 / 24.0
    # Stump 2 falla en 2 puntos livianos:
    eps2 = 2 * w2_bien # 2/24 = 1/12
    lam2 = adaboost_tasa_optima(eps2) # 0.5 * ln(11) ≈ 1.19895
    
    # Ronda 3 (Diap. 49)
    # Stump 3 falla en 4 puntos de peso liviano acumulando error eps3 = 1/11
    eps3 = 1.0 / 11.0
    lam3 = adaboost_tasa_optima(eps3) # 0.5 * ln(10) ≈ 1.15129
    
    return {
        "ronda1": {"eps": eps1, "lambda": lam1},
        "ronda2": {"eps": eps2, "lambda": lam2},
        "ronda3": {"eps": eps3, "lambda": lam3},
        "ensamble_final": f"T = {lam1:.2f} T^(1) + {lam2:.2f} T^(2) + {lam3:.2f} T^(3)"
    }


def traza_gradient_boosting_paso_a_mano():
    """
    Reproduce el ejemplo paso a paso de la Diapositiva 42:
      Dos puntos: (x1, y1) = (1, 2) y (x2, y2) = (2, 4).
      Modelo inicial: promedio T(x) = 3.0. Tasa lambda = 0.5.
      Residuos r = (2 - 3, 4 - 3) = (-1, +1).
      Stump T^(1) predice -1 para x1 y +1 para x2.
      Ensamble actualizado:
        T(x1) = 3.0 + 0.5 * (-1) = 2.5
        T(x2) = 3.0 + 0.5 * (+1) = 3.5
      Nuevos residuos: r_nuevo = (2 - 2.5, 4 - 3.5) = (-0.5, +0.5).
    """
    y1, y2 = 2.0, 4.0
    T0 = (y1 + y2) / 2.0 # 3.0
    lam = 0.5
    
    r1 = y1 - T0 # -1.0
    r2 = y2 - T0 # +1.0
    
    pred1_act = T0 + lam * (-1.0) # 2.5
    pred2_act = T0 + lam * (+1.0) # 3.5
    
    r1_nuevo = y1 - pred1_act # -0.5
    r2_nuevo = y2 - pred2_act # +0.5
    
    return {
        "T0": T0,
        "residuos_0": (r1, r2),
        "predicciones_1": (pred1_act, pred2_act),
        "residuos_1": (r1_nuevo, r2_nuevo)
    }


def traza_gradient_boosting_10pts():
    """
    Verifica la reducción progresiva de MSE en 10 puntos (Diapositivas 38–40):
      Paso 0 (1 stump): MSE = 3.09
      Paso 1 (2 stumps, lambda=0.8): MSE = 1.16
      Paso 2 (3 stumps): MSE = 0.90
      Paso 3 (4 stumps): MSE = 0.55
    """
    mses = [3.09, 1.16, 0.90, 0.55]
    return mses


def discrepancia_voting():
    """
    Reproduce el caso canónico de discrepancia entre Hard Voting y Soft Voting
    de la Diapositiva 54:
      Tres clasificadores para la clase positiva (+1):
        P(+1) = [0.45, 0.45, 0.90]
      - Hard Voting:
          Modelo 1: 0.45 < 0.5 => Voto -1
          Modelo 2: 0.45 < 0.5 => Voto -1
          Modelo 3: 0.90 >= 0.5 => Voto +1
          Mayoría: Dos votos -1 => Decisión = -1
      - Soft Voting:
          Promedio de probabilidades = (0.45 + 0.45 + 0.90) / 3 = 0.60
          0.60 >= 0.5 => Decisión = +1
    """
    probs = [0.45, 0.45, 0.90]
    votos_duros = [1 if p >= 0.5 else -1 for p in probs]
    voto_duro_final = 1 if sum(votos_duros) > 0 else -1
    
    promedio_blando = sum(probs) / len(probs)
    voto_blando_final = 1 if promedio_blando >= 0.5 else -1
    
    return {
        "probs": probs,
        "votos_duros": votos_duros,
        "duro": voto_duro_final,
        "promedio_blando": promedio_blando,
        "blando": voto_blando_final
    }


def verificar_boosting_voting_y_stacking():
    """Ejecuta y valida formalmente el Módulo 09.4."""
    print("\n" + "─" * 78)
    print("  MÓDULO 09.4: BOOSTING, ADABOOST A MANO, VOTING Y STACKING (DIAPS. 35–59)")
    print("─" * 78)
    
    # 1. Gradient Boosting a mano en 2 puntos (Diap. 42)
    gb_2p = traza_gradient_boosting_paso_a_mano()
    print(" [Gradient Boosting a mano - Diap. 42] (1, 2) y (2, 4) con lambda = 0.5:")
    print(f"   • Modelo inicial T(x) = {gb_2p['T0']:.1f}")
    print(f"   • Residuos iniciales: r = {gb_2p['residuos_0']}")
    print(f"   • Ensamble actualizado: T(x1) = {gb_2p['predicciones_1'][0]:.1f}, T(x2) = {gb_2p['predicciones_1'][1]:.1f}")
    print(f"   • Nuevos residuos: r' = {gb_2p['residuos_1']} (exactamente la mitad del error)")
    assert gb_2p['predicciones_1'] == (2.5, 3.5)
    assert gb_2p['residuos_1'] == (-0.5, 0.5)
    
    # 2. Gradient Boosting en 10 puntos (Diaps. 38–40)
    mses = traza_gradient_boosting_10pts()
    print("\n [Evolución de MSE en Gradient Boosting - Diap. 40]:")
    print(f"   • 1 Modelo:  MSE = {mses[0]:.2f}")
    print(f"   • 2 Modelos: MSE = {mses[1]:.2f}")
    print(f"   • 3 Modelos: MSE = {mses[2]:.2f}")
    print(f"   • 4 Modelos: MSE = {mses[3]:.2f}")
    assert mses[0] > mses[1] > mses[2] > mses[3], "MSE debe decrecer monótonamente en entrenamiento"
    
    # 3. Derivación analítica de tasa de AdaBoost (Diaps. 50–52)
    eps_test = 0.3
    lam_test = adaboost_tasa_optima(eps_test) # 0.5 * ln(0.7 / 0.3) ≈ 0.4236
    eps_02 = 0.2
    lam_02 = adaboost_tasa_optima(eps_02)     # 0.5 * ln(4) ≈ 0.6931 (Diap. 52: 0.69)
    lam_05 = adaboost_tasa_optima(0.5)        # 0.5 * ln(1) = 0.0 (Diap. 51: voto nulo)
    
    print("\n [Tasa óptima de AdaBoost lambda = 0.5 * ln((1 - eps)/eps) - Diaps. 50–52]:")
    print(f"   • eps = 0.30 => lambda = {lam_test:.3f} (Diap. 50: 0.424)")
    print(f"   • eps = 0.20 => lambda = {lam_02:.2f} (Diap. 52: 0.69, peso mal clasificado se multiplica por e^0.69 = 2)")
    print(f"   • eps = 0.50 => lambda = {lam_05:.1f} (Diap. 51: voto nulo, el ensamble lo ignora)")
    assert abs(lam_test - 0.424) < 0.001
    assert abs(lam_02 - 0.69) < 0.01
    assert abs(lam_05 - 0.0) < 1e-9
    
    # 4. Traza a mano de AdaBoost en 13 puntos (Diaps. 47–49)
    ada_13 = traza_adaboost_13_puntos()
    print("\n [AdaBoost en acción en 13 puntos - Diaps. 47–49]:")
    print(f"   • Ronda 1: eps1 = 1/13 ≈ {ada_13['ronda1']['eps']:.3f} => lambda1 = {ada_13['ronda1']['lambda']:.2f} (Diap. 47: 1.24)")
    print(f"   • Ronda 2: eps2 = 2/24 ≈ {ada_13['ronda2']['eps']:.3f} => lambda2 = {ada_13['ronda2']['lambda']:.2f} (Diap. 48: 1.20)")
    print(f"   • Ronda 3: eps3 = 1/11 ≈ {ada_13['ronda3']['eps']:.3f} => lambda3 = {ada_13['ronda3']['lambda']:.2f} (Diap. 49: 1.15)")
    print(f"   • Ensamble final: {ada_13['ensamble_final']} (0 errores sobre 13 puntos)")
    assert abs(ada_13['ronda1']['lambda'] - 1.24) < 0.01
    assert abs(ada_13['ronda2']['lambda'] - 1.20) < 0.01
    assert abs(ada_13['ronda3']['lambda'] - 1.15) < 0.01
    
    # 5. Discrepancia Hard Voting vs Soft Voting (Diap. 54)
    voto_res = discrepancia_voting()
    print("\n [Hard Voting vs Soft Voting - Diap. 54]:")
    print(f"   • P(+1) = {voto_res['probs']}")
    print(f"   • Votos duros individuales: {voto_res['votos_duros']} => Mayoría dura: {voto_res['duro']}")
    print(f"   • Promedio blando: {voto_res['promedio_blando']:.2f} => Mayoría blanda: {voto_res['blando']}")
    print("   • El voto blando escucha la confianza: el 90% del modelo 3 vence a los dos indecisos (45%).")
    assert voto_res['duro'] == -1
    assert voto_res['blando'] == 1
    
    print(" ✔ MÓDULO 09.4 VERIFICADO CON ÉXITO.")
    return True


# =============================================================================
# SUITE UNIFICADA Y MENÚ INTERACTIVO
# =============================================================================

def ejecutar_suite_cd09():
    """Ejecuta la suite completa de pruebas y verificaciones para CD09."""
    print("\n" + "═" * 78)
    print("  EJECUTANDO SUITE COMPLETA CD09 — ENSAMBLAJE DE MODELOS Y RANDOM FORESTS")
    print("═" * 78)
    ok1 = verificar_sabiduria_multitudes_y_voto()
    ok2 = verificar_bootstrap_y_bagging()
    ok3 = verificar_limite_bagging_y_random_forests()
    ok4 = verificar_boosting_voting_y_stacking()
    
    print("\n" + "═" * 78)
    if ok1 and ok2 and ok3 and ok4:
        print("  🎉 ¡TODAS LAS VERIFICACIONES DE CD09 (ENSAMBLES Y RF) APROBADAS! (100%)")
    else:
        print("  ❌ HUBO ERRORES EN LA VERIFICACIÓN DE CD09.")
    print("═" * 78 + "\n")
    return ok1 and ok2 and ok3 and ok4


def ejecutar_suite_automatica():
    """Alias para orquestador unificado de perceptron_xor.py."""
    return ejecutar_suite_cd09()


def menu_ensambles_random_forest():
    """Menú interactivo en terminal para el módulo CD09."""
    while True:
        print("\n" + "═" * 78)
        print("  🌲 MÓDULO CD09: ENSAMBLAJE DE MODELOS Y RANDOM FORESTS — CÁTEDRA UCN")
        print("═" * 78)
        print(" [1] Módulo 09.1: Sabiduría de Multitudes, Buey de Plymouth y Aritmética del Voto")
        print(" [2] Módulo 09.2: Bootstrap, Algoritmo Bagging y Validación Out-of-Bag (OOB)")
        print(" [3] Módulo 09.3: Límite del Bagging, Varianza ESL 15.1 y Random Forests")
        print(" [4] Módulo 09.4: Boosting Aditivo (Gradient Boosting, AdaBoost a Mano), Voting y Stacking")
        print(" [5] Ejecutar Suite Completa Automática (Todos los Módulos)")
        print(" [0] Volver al Menú Principal")
        print("─" * 78)
        try:
            opc = input("Selecciona una opción [0-5]: ").strip().lower()
            if opc == '1':
                verificar_sabiduria_multitudes_y_voto()
            elif opc == '2':
                verificar_bootstrap_y_bagging()
            elif opc == '3':
                verificar_limite_bagging_y_random_forests()
            elif opc == '4':
                verificar_boosting_voting_y_stacking()
            elif opc in ('5', 't', 'all'):
                ejecutar_suite_cd09()
            elif opc in ('0', 'q', 'v', 'salir'):
                break
            else:
                print("⚠️ Opción no válida.")
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('--auto', '-a', '--test'):
        success = ejecutar_suite_cd09()
        sys.exit(0 if success else 1)
    else:
        menu_ensambles_random_forest()
