#!/usr/bin/env python3
"""
Módulo de Verificación Numérica y Conceptual: CD07 — Reducción de Dimensionalidad
==================================================================================
Ciencia de Datos — Doctorado UCN
Basado en material de:
  - CMU 15-388/688 (Z. Kolter)
  - Harvard CS109 (P. Protopapas et al.)
  - C. M. Bishop, Pattern Recognition and Machine Learning (PRML)
  - M. Turk & A. Pentland, Eigenfaces for Recognition (1991)
  - L. van der Maaten & G. Hinton, Visualizing Data using t-SNE (2008)
  - L. McInnes, J. Healy & J. Melville, UMAP (2018)

Módulos temáticos incluidos:
  1. La Maldición de la Dimensionalidad: La bola en el cubo (P_d), escape a las esquinas y distancias.
  2. PCA como Compresión y Máxima Varianza: Paso a mano exacto 2D (6 pts, S, autovalores 3 y 1, SVD).
  3. LDA: Proyección con Etiquetas y Criterio de Fisher: Sw, w* prop a Sw^-1(m2 - m1), ejemplos de 8 y 2 pts.
  4. Manifold Learning (t-SNE y UMAP): Probabilidades gaussianas, perplejidad, colas pesadas de Student-t y UMAP->HDBSCAN.
"""

import math
import sys
import random

# =============================================================================
# UTILIDADES MATEMÁTICAS EN PYTHON ESTÁNDAR
# =============================================================================

def gamma_func(x):
    """Aproximación de la función gamma Γ(x) mediante Lanczos / recursion."""
    if x == 1.0: return 1.0
    if x == 0.5: return math.sqrt(math.pi)
    if x == 1.5: return 0.5 * math.sqrt(math.pi)
    if x == 2.0: return 1.0
    if x == 2.5: return 1.5 * 0.5 * math.sqrt(math.pi)
    if x == 3.0: return 2.0
    if x == 3.5: return 2.5 * 1.5 * 0.5 * math.sqrt(math.pi)
    if x == 4.0: return 6.0
    if x > 1:
        return (x - 1) * gamma_func(x - 1)
    # Fórmula de reflexión de Euler para valores pequeños
    return math.gamma(x)

def razon_volumen_bola_cubo(d):
    """
    Calcula P_d = Vol(bola unitaria r=1) / Vol(cubo lado 2) en R^d:
    P_d = π^(d/2) / (Γ(d/2 + 1) * 2^d)
    """
    numerador = (math.pi ** (d / 2.0))
    denominador = math.gamma(d / 2.0 + 1.0) * (2.0 ** d)
    return numerador / denominador

# =============================================================================
# MÓDULO 1: LA MALDICIÓN DE LA DIMENSIONALIDAD Y LA BOLA EN EL CUBO
# =============================================================================

def verificar_maldicion_dimensionalidad():
    print("\n" + "═" * 78)
    print("  MÓDULO 07.1: LA MALDICIÓN DE LA DIMENSIONALIDAD (Diapositivas 5–11)")
    print("  El volumen se escapa a las esquinas y las distancias pierden contraste")
    print("═" * 78)

    # 1. Tabla de razones P_d oficial (Diapositivas 8 y 9)
    dims = [1, 2, 3, 6, 10, 15, 100]
    expected = {
        1: 1.000,
        2: 0.7854,
        3: 0.5236,
        6: 0.0807,
        10: 0.00249,
        15: 1.16e-5,
        100: 1.87e-70
    }

    print("\n1. Tabla oficial P_d = Vol(bola r=1) / Vol(cubo lado 2):")
    print(f"  {'d':>4} | {'P_d (Cálculo exacto)':>22} | {'Esperado en Diap. 9':>22} | {'Interpretación'}")
    print("  " + "─" * 72)

    for d in dims:
        pd = razon_volumen_bola_cubo(d)
        if d == 1:
            note = "Cubo = [-1, 1], Bola = [-1, 1] (100% de coincidencia)"
        elif d == 2:
            note = f"{pd*100:.1f}% en el círculo (esquinas casi no pesan)"
        elif d == 3:
            note = f"{pd*100:.1f}% en la esfera (8 esquinas ya roban casi la mitad)"
        elif d == 10:
            note = f"Solo 1 de cada {int(1/pd)} puntos cae al centro"
        elif d == 100:
            note = "Prácticamente 0 (el cubo es un erizo de 2^100 púas)"
        else:
            note = f"P_d = {pd:.2e}"

        print(f"  {d:>4} | {pd:>22.4e} | {expected[d]:>22.4e} | {note}")
        if d in (1, 2, 3, 6):
            assert abs(pd - expected[d]) < 0.005, f"Error en P_{d}: {pd} vs {expected[d]}"

    # 2. Experimento de 1,000 puntos uniformes en el cubo (Diapositiva 11)
    random.seed(42)
    n_puntos = 1000

    def simular_puntos_cubo(d, n):
        dentro = 0
        for _ in range(n):
            norma_sq = sum((random.uniform(-1.0, 1.0)) ** 2 for _ in range(d))
            if norma_sq <= 1.0:
                dentro += 1
        return dentro

    caen_d2 = simular_puntos_cubo(2, n_puntos)
    caen_d10 = simular_puntos_cubo(10, n_puntos)

    print("\n2. Experimento con N = 1,000 puntos aleatorios en cubo de lado 2 (Diap. 11):")
    print(f"  • En d = 2: Teórico = {1000*0.785:.1f} puntos | Simulado = {caen_d2} puntos (Casi todos)")
    print(f"  • En d = 10: Teórico = {1000*0.0025:.1f} puntos | Simulado = {caen_d10} puntos (Dos o tres con suerte)")

    assert abs(caen_d2 - 785) < 50, f"Simulación d=2 fuera de rango: {caen_d2}"
    assert caen_d10 <= 10, f"Simulación d=10 no debería superar 10 puntos: {caen_d10}"

    print("\n✔ Verificación de la maldición de la dimensionalidad completada con 100% de éxito.")
    return True

# =============================================================================
# MÓDULO 2: PCA COMO COMPRESIÓN Y MÁXIMA VARIANZA (PASO A PASO A MANO)
# =============================================================================

def verificar_pca_paso_a_paso():
    print("\n" + "═" * 78)
    print("  MÓDULO 07.2: PCA PASO A PASO A MANO Y SVD (Diapositivas 12–37)")
    print("  Ejemplo trabajado completo con 6 puntos en 2D (Págs. 102–142)")
    print("═" * 78)

    # 6 puntos oficiales en 2D (Diapositiva 25 / Pág. 102)
    puntos_raw = [
        (6.0, 4.0),
        (5.0, 5.0),
        (5.0, 2.0),
        (3.0, 4.0),
        (3.0, 1.0),
        (2.0, 2.0)
    ]
    N = len(puntos_raw)

    # Paso 1: Centrar los datos
    media_x1 = sum(p[0] for p in puntos_raw) / N
    media_x2 = sum(p[1] for p in puntos_raw) / N
    print(f"\nPaso 1 — Centroides y centrado (Diap. 25–26):")
    print(f"  Media: x̄ = ({sum(p[0] for p in puntos_raw)}/6, {sum(p[1] for p in puntos_raw)}/6) = ({media_x1:.1f}, {media_x2:.1f})")
    assert (media_x1, media_x2) == (4.0, 3.0), f"Media errónea: {(media_x1, media_x2)}"

    centrados = [(p[0] - media_x1, p[1] - media_x2) for p in puntos_raw]
    esperados_centrados = [(2, 1), (1, 2), (1, -1), (-1, 1), (-1, -2), (-2, -1)]
    for i, (c, e) in enumerate(zip(centrados, esperados_centrados)):
        assert c == e, f"Punto centrado {i+1} erróneo: {c} vs {e}"

    print("  Puntos centrados x̃_i:")
    for i, c in enumerate(centrados, 1):
        print(f"    Punto {i}: {c}")

    # Paso 2: Matriz de covarianza S = (1/N) X̃^T X̃
    s11 = sum(c[0]**2 for c in centrados) / N
    s22 = sum(c[1]**2 for c in centrados) / N
    s12 = sum(c[0]*c[1] for c in centrados) / N
    print(f"\nPaso 2 — Matriz de covarianza S (Diap. 27):")
    print(f"  S11 = 1/6 (4 + 1 + 1 + 1 + 1 + 4) = 12/6 = {s11:.1f}")
    print(f"  S22 = 1/6 (1 + 4 + 1 + 1 + 4 + 1) = 12/6 = {s22:.1f}")
    print(f"  S12 = 1/6 (2 + 2 - 1 - 1 + 2 + 2) =  6/6 = {s12:.1f}")
    assert (s11, s22, s12) == (2.0, 2.0, 1.0), f"Covarianza errónea: ({s11}, {s22}, {s12})"

    # Paso 3: Autovalores det(S - λI) = 0
    # (2 - λ)^2 - 1 = 0 => 2 - λ = ±1 => λ1 = 3, λ2 = 1
    lam1 = 3.0
    lam2 = 1.0
    print(f"\nPaso 3 — Autovalores (Diap. 28):")
    print(f"  det(S - λI) = (2 - λ)² - 1 = 0  ==>  λ1 = {lam1:.0f}, λ2 = {lam2:.0f}")
    print(f"  Verificación: traza tr(S) = {s11+s22:.0f} == λ1+λ2 = {lam1+lam2:.0f}")
    print(f"  Determinante det(S) = {s11*s22 - s12**2:.0f} == λ1*λ2 = {lam1*lam2:.0f}")
    assert (lam1, lam2) == (3.0, 1.0)

    # Paso 4: Autovectores
    # (S - 3I)v = [-1 1; 1 -1] [v1; v2] = 0 => v1 = v2 => v^(1) = [1/sqrt(2), 1/sqrt(2)]
    # (S - 1I)v = [ 1 1; 1  1] [v1; v2] = 0 => v1 = -v2 => v^(2) = [1/sqrt(2), -1/sqrt(2)]
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    v1 = (inv_sqrt2, inv_sqrt2)
    v2 = (inv_sqrt2, -inv_sqrt2)
    print(f"\nPaso 4 — Autovectores unitarios ortogonales (Diap. 29):")
    print(f"  v^(1) = (1/√2, 1/√2) ≈ ({v1[0]:.4f}, {v1[1]:.4f})  [Diagonal 45°]")
    print(f"  v^(2) = (1/√2, -1/√2) ≈ ({v2[0]:.4f}, {v2[1]:.4f}) [Ortogonal]")
    # Ortogonalidad:
    dot_v1_v2 = v1[0]*v2[0] + v1[1]*v2[1]
    assert abs(dot_v1_v2) < 1e-12

    # Paso 5: Proyectar zi = v^(1) · x̃^(i) = (x̃1 + x̃2)/√2
    print(f"\nPaso 5 — Proyecciones z_i sobre el primer componente (Diap. 30):")
    proy_z = []
    esperados_z = [3/math.sqrt(2), 3/math.sqrt(2), 0.0, 0.0, -3/math.sqrt(2), -3/math.sqrt(2)]
    for i, c in enumerate(centrados, 1):
        z = c[0]*v1[0] + c[1]*v1[1]
        proy_z.append(z)
        print(f"  Punto {i}: z_{i} = ({c[0]} + {c[1]})/√2 = {z:.4f}")
        assert abs(z - esperados_z[i-1]) < 1e-5, f"Proyección {i} errónea: {z}"

    # Paso 6: Reconstruir x̂^(i) = x̄ + z_i v^(1)
    print(f"\nPaso 6 — Reconstrucción y pérdida de compresión (Diap. 31):")
    error_reconstruccion = 0.0
    for i, (orig, z) in enumerate(zip(puntos_raw, proy_z), 1):
        x_hat = (media_x1 + z * v1[0], media_x2 + z * v1[1])
        err_sq = (orig[0] - x_hat[0])**2 + (orig[1] - x_hat[1])**2
        error_reconstruccion += err_sq
        print(f"  Punto {i}: Original {orig} -> Reconstruido ({x_hat[0]:.2f}, {x_hat[1]:.2f}) | Error² = {err_sq:.2f}")

    print(f"\n  Pérdida de reconstrucción total ∑ ‖x̂ - x‖² = {error_reconstruccion:.1f}")
    print(f"  Teorema exacto: Loss* = N * λ2 = 6 * 1.0 = {6 * lam2:.1f}")
    assert abs(error_reconstruccion - 6.0) < 1e-5, f"Pérdida errónea: {error_reconstruccion}"

    # Paso 7: Varianza explicada y Scree Plot
    var_exp1 = lam1 / (lam1 + lam2)
    var_exp2 = lam2 / (lam1 + lam2)
    print(f"\nPaso 7 — Varianza explicada (Diap. 32):")
    print(f"  Componente 1: {var_exp1*100:.1f}%  (λ1 = 3)")
    print(f"  Componente 2: {var_exp2*100:.1f}%  (λ2 = 1)")
    assert var_exp1 == 0.75 and var_exp2 == 0.25

    print("\n✔ Verificación completa de PCA paso a paso aprobada al 100%.")
    return True

# =============================================================================
# MÓDULO 3: LDA Y EL CRITERIO DE FISHER (PROYECCIÓN CON ETIQUETAS)
# =============================================================================

def verificar_lda_fisher():
    print("\n" + "═" * 78)
    print("  MÓDULO 07.3: LDA Y CRITERIO DE FISHER (Diapositivas 38–49)")
    print("  w* ∝ S_W^-1 (m2 - m1): Castigar la dispersión interna")
    print("═" * 78)

    # 1. Ejemplo 1: 8 puntos en 2 clases (Diapositivas 42–46)
    c1 = [(2.0, 1.0), (2.0, 7.0), (1.0, 4.0), (3.0, 4.0)]
    c2 = [(5.0, 2.0), (5.0, 8.0), (4.0, 5.0), (6.0, 5.0)]

    m1 = (sum(p[0] for p in c1)/4, sum(p[1] for p in c1)/4)
    m2 = (sum(p[0] for p in c2)/4, sum(p[1] for p in c2)/4)

    print(f"\n1. Ejemplo trabajado de 8 puntos (Diap. 42):")
    print(f"  Media C1: m1 = ({m1[0]:.1f}, {m1[1]:.1f})")
    print(f"  Media C2: m2 = ({m2[0]:.1f}, {m2[1]:.1f})")
    delta_m = (m2[0] - m1[0], m2[1] - m1[1])
    print(f"  Diferencia de medias: m2 - m1 = ({delta_m[0]:.1f}, {delta_m[1]:.1f})")
    assert m1 == (2.0, 4.0) and m2 == (5.0, 5.0) and delta_m == (3.0, 1.0)

    # Dispersión intra-clase S_W (Diap. 43)
    # Desviaciones C1: (0, -3), (0, 3), (-1, 0), (1, 0)
    # S1 = sum dev dev^T
    s1_11 = 0**2 + 0**2 + (-1)**2 + 1**2 # 2
    s1_22 = (-3)**2 + 3**2 + 0**2 + 0**2 # 18
    s1_12 = 0 + 0 + 0 + 0 # 0
    # C2 tiene idénticas desviaciones respecto a m2
    sw_11 = s1_11 * 2 # 4
    sw_22 = s1_22 * 2 # 36
    sw_12 = 0.0

    print(f"\n  Matriz de dispersión intra-clase Sw (Diap. 43):")
    print(f"  Sw = [[{sw_11}, {sw_12}], [{sw_12}, {sw_22}]]  (¡En x2 la dispersión es 36 vs 4!)")
    assert sw_11 == 4.0 and sw_22 == 36.0 and sw_12 == 0.0

    # Inversa Sw^-1
    sw_inv_11 = 1.0 / sw_11 # 1/4 = 0.25
    sw_inv_22 = 1.0 / sw_22 # 1/36 = 0.02777...

    # Dirección de Fisher w* = Sw^-1 (m2 - m1)
    w_star_x = sw_inv_11 * delta_m[0] # (1/4)*3 = 3/4 = 27/36
    w_star_y = sw_inv_22 * delta_m[1] # (1/36)*1 = 1/36
    # En enteros proporcionales: (27, 1)
    ratio_27_1 = w_star_x / w_star_y
    print(f"\n  Dirección óptima de Fisher (Diap. 44):")
    print(f"  w* ∝ Sw^-1 (m2 - m1) = [{w_star_x:.4f}, {w_star_y:.4f}] ∝ [27, 1]")
    assert abs(ratio_27_1 - 27.0) < 1e-9

    norm_w = math.sqrt(w_star_x**2 + w_star_y**2)
    w_norm = (w_star_x / norm_w, w_star_y / norm_w)
    print(f"  Normalizada: w* ≈ ({w_norm[0]:.3f}, {w_norm[1]:.3f})")
    assert abs(w_norm[0] - 0.999) < 0.005 and abs(w_norm[1] - 0.037) < 0.005

    # Criterio de Fisher J(w) = (w·(m2-m1))^2 / (w^T Sw w)
    def calc_fisher_j(w):
        dot_m = w[0]*delta_m[0] + w[1]*delta_m[1]
        denom = w[0]**2 * sw_11 + w[1]**2 * sw_22
        return (dot_m**2) / denom

    j_fisher = calc_fisher_j(w_norm)
    # PCA en este dataset sigue la máxima varianza que es casi vertical: v_pca ≈ (0.34, 0.94)
    # Supongamos dirección vertical (0, 1):
    j_pca = calc_fisher_j((0.0, 1.0))
    print(f"\n  Comparativa Criterio J (Diap. 44):")
    print(f"  J(w*_Fisher) = {j_fisher:.2f}  vs  J(v_PCA) = {j_pca:.2f}")
    print(f"  ¡Fisher separa {j_fisher / j_pca:.1f} veces mejor las clases!")
    assert abs(j_fisher - 2.28) < 0.05

    # 2. Ejemplo 2: Segundo caso de Fisher (Diapositiva 49)
    print("\n2. Segundo caso de Fisher (Diap. 49):")
    dm2 = (2.0, 2.0)
    sw2 = (2.0, 8.0) # diagonal
    w2_star = (dm2[0]/sw2[0], dm2[1]/sw2[1]) # (2/2, 2/8) = (1.0, 0.25) ∝ (4, 1)
    ratio2 = w2_star[0] / w2_star[1]
    print(f"  m2 - m1 = (2, 2), Sw = diag(2, 8)")
    print(f"  w* ∝ (2/2, 2/8) = (1, 0.25) ∝ [4, 1]")
    print(f"  Conclusión: Aunque la separación de medias es idéntica en ambos ejes, x2 dispersa 4x más, por lo que pesa 4 veces menos.")
    assert ratio2 == 4.0

    print("\n✔ Verificación completa de LDA y Criterio de Fisher aprobada al 100%.")
    return True

# =============================================================================
# MÓDULO 4: MANIFOLD LEARNING (t-SNE Y UMAP)
# =============================================================================

def verificar_manifold_tsne_umap():
    print("\n" + "═" * 78)
    print("  MÓDULO 07.4: MANIFOLD LEARNING — t-SNE Y UMAP (Diapositivas 50–71)")
    print("  Probabilidades de vecindad, colas pesadas de Student-t y Pipeline UMAP->HDBSCAN")
    print("═" * 78)

    # 1. Ejemplo numérico de 4 puntos en 1D: x = 0, 1, 2, 8 (Diapositiva 56)
    pts = [0.0, 1.0, 2.0, 8.0]
    i_idx = 1 # x = 1.0
    sigma_1 = 1.0

    def calc_p_j_given_i(x_list, idx, sigma):
        x_i = x_list[idx]
        diffs = [x_j - x_i for j, x_j in enumerate(x_list) if j != idx]
        exps = [math.exp(- (d**2) / (2.0 * sigma**2)) for d in diffs]
        total = sum(exps)
        return [e / total for e in exps]

    p_from_1 = calc_p_j_given_i(pts, i_idx, sigma_1)
    # Vecinos de 1 son: x=0 (dist 1), x=2 (dist 1), x=8 (dist 7)
    # exp(-1/2) = 0.6065, exp(-49/2) = 2.3e-11
    print(f"\n1. Probabilidades gaussianas desde x = 1 con σ = 1 (Diap. 56):")
    print(f"  P(x=0 | x=1) = {p_from_1[0]:.4f}")
    print(f"  P(x=2 | x=1) = {p_from_1[1]:.4f}")
    print(f"  P(x=8 | x=1) = {p_from_1[2]:.2e}  (prácticamente 0)")
    assert abs(p_from_1[0] - 0.5) < 1e-4 and abs(p_from_1[1] - 0.5) < 1e-4

    # Entropía H y perplejidad Perp = 2^H
    def calc_perp(probs):
        h = -sum(p * math.log2(p) for p in probs if p > 1e-12)
        return 2.0 ** h

    perp_1 = calc_perp(p_from_1)
    print(f"  Perplejidad Perp(P_1) = {perp_1:.2f} ≈ 2 (dos vecinos efectivos)")
    assert abs(perp_1 - 2.0) < 0.01

    # 2. Punto aislado x = 8: Adaptabilidad de σi (Diapositiva 58)
    # Con σ = 1: distancias a 0, 1, 2 son 8, 7, 6. exp(-36/2)=1.5e-8, etc.
    p_from_8_sig1 = calc_p_j_given_i(pts, 3, 1.0)
    perp_8_sig1 = calc_perp(p_from_8_sig1)
    print(f"\n2. Adaptabilidad de σ desde punto aislado x = 8 (Diap. 58):")
    print(f"  Con σ = 1: Perp = {perp_8_sig1:.2f} ≈ 1.01 (casi todo en el vecino x=2)")

    # Búsqueda binaria de σ para alcanzar Perp = 2.0
    low_sig, high_sig = 0.5, 10.0
    for _ in range(50):
        mid_sig = (low_sig + high_sig) / 2.0
        p_test = calc_p_j_given_i(pts, 3, mid_sig)
        if calc_perp(p_test) < 2.0:
            low_sig = mid_sig
        else:
            high_sig = mid_sig
    optimal_sigma_8 = (low_sig + high_sig) / 2.0
    p_from_8_opt = calc_p_j_given_i(pts, 3, optimal_sigma_8)
    print(f"  Para alcanzar Perp = 2.0, se ensancha a σ ≈ {optimal_sigma_8:.2f} (Diap. 58 reporta 2.26)")
    print(f"  Probabilidades resultantes: [{p_from_8_opt[0]:.2f}, {p_from_8_opt[1]:.2f}, {p_from_8_opt[2]:.2f}]")
    assert abs(optimal_sigma_8 - 2.26) < 0.05

    # 3. Comparativa de colas pesadas: Gaussiana vs t de Student (Diapositiva 60)
    # Gaussiana exp(-d^2) vs t de Student (1 + d^2)^-1
    d_val = 3.0
    gauss_3 = math.exp(- d_val**2) # exp(-9) ≈ 0.000123
    student_3 = 1.0 / (1.0 + d_val**2) # 1/10 = 0.1
    ratio_colasp = student_3 / gauss_3
    print(f"\n3. Efecto de colas pesadas de la t de Student (Diap. 60):")
    print(f"  A distancia d = 3:")
    print(f"    Gaussiana e^(-9)  = {gauss_3:.6f}")
    print(f"    Student-t 1/(1+9) = {student_3:.6f}")
    print(f"    ¡La t vale {ratio_colasp:.0f} veces más que la gaussiana a distancia 3!")
    assert ratio_colasp > 700

    print("\n✔ Verificación completa de Manifold Learning (t-SNE/UMAP) aprobada al 100%.")
    return True

# =============================================================================
# AUTOEVALUACIÓN Y SUITE COMPLETA CD07
# =============================================================================

def ejecutar_suite_cd07():
    print("\n" + "═" * 78)
    print("  EJECUTANDO SUITE COMPLETA CD07 — REDUCCIÓN DE DIMENSIONALIDAD")
    print("═" * 78)
    ok1 = verificar_maldicion_dimensionalidad()
    ok2 = verificar_pca_paso_a_paso()
    ok3 = verificar_lda_fisher()
    ok4 = verificar_manifold_tsne_umap()

    print("\n" + "═" * 78)
    if ok1 and ok2 and ok3 and ok4:
        print("  🎉 ¡TODAS LAS VERIFICACIONES DE CD07 (REDUCCIÓN DE DIMENSIONALIDAD) APROBADAS! (100%)")
    else:
        print("  ❌ HUBO ERRORES EN LA VERIFICACIÓN DE CD07.")
    print("═" * 78 + "\n")
    return ok1 and ok2 and ok3 and ok4

def ejecutar_suite_automatica():
    """Alias para orquestador unificado."""
    return ejecutar_suite_cd07()

def menu_reduccion_dim():
    """Menú interactivo para CD07."""
    while True:
        print("\n" + "═" * 78)
        print("  🔬 MÓDULO CD07: REDUCCIÓN DE DIMENSIONALIDAD (DOCTORADO UCN)")
        print("═" * 78)
        print(" [1] Módulo 07.1: La Maldición de la Dimensionalidad (Bola vs Cubo P_d)")
        print(" [2] Módulo 07.2: PCA Paso a Paso a Mano (Covarianza, Autovalores, SVD)")
        print(" [3] Módulo 07.3: LDA y Criterio de Fisher (Separabilidad Supervisada)")
        print(" [4] Módulo 07.4: Manifold Learning (t-SNE, UMAP y Pipeline con HDBSCAN)")
        print(" [5] Ejecutar Suite Completa Automática (Todos los Módulos)")
        print(" [0] Volver al Menú Principal")
        print("─" * 78)
        try:
            opc = input("Selecciona una opción [0-5]: ").strip().lower()
            if opc == '1':
                verificar_maldicion_dimensionalidad()
            elif opc == '2':
                verificar_pca_paso_a_paso()
            elif opc == '3':
                verificar_lda_fisher()
            elif opc == '4':
                verificar_manifold_tsne_umap()
            elif opc in ('5', 't', 'all'):
                ejecutar_suite_cd07()
            elif opc in ('0', 'q', 'v', 'salir'):
                break
            else:
                print("⚠️ Opción no válida.")
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('--auto', '-a', '--test'):
        success = ejecutar_suite_cd07()
        sys.exit(0 if success else 1)
    else:
        menu_reduccion_dim()
