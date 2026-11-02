#!/usr/bin/env python3
"""
Módulo Educativo: Descenso por Gradiente (Unidad III)
=====================================================
Ciencia de Datos — Doctorado UCN
Basado en el material de B. Raj et al. (CMU 11-785) y P. Liang et al. (Stanford CS221).

Este módulo implementa de manera demostrativa, interactiva y libre de dependencias
externas obligatorias las 5 secciones de la cátedra "Descenso por Gradiente" (47 diapositivas):

  Sección 1 (Diaps.  1-16): El problema de minimizar una función, aproximación lineal local,
                            vector gradiente, producto punto y ortogonalidad a las curvas de nivel.
  Sección 2 (Diaps. 17-25): Condiciones de mínimo en 1D y varias variables: segunda derivada,
                            matriz Hessiana, valores propios y el ejemplo 3D de las Diaps. 24-25.
  Sección 3 (Diaps. 26-35): Soluciones iterativas: descenso guiado por el signo vs. descenso por
                            gradiente, criterios de convergencia y el ejemplo de las 200 épocas
                            en regresión lineal por mínimos cuadrados (Diaps. 34-35).
  Sección 4 (Diaps. 36-45): Análisis del tamaño de paso en cuadráticas (eta_opt = 1/curvatura),
                            los cuatro regímenes (convergencia monótona, un paso, oscilación,
                            divergencia), curvaturas por dirección en 2D y verificación de Diap. 45.
  Sección 5 (Diaps. 39-40, 46-47): Convexidad vs. no convexidad (sensibilidad a la inicialización
                            x^(0) hacia mínimos locales o globales) y síntesis hacia backpropagation/SGD.
"""

import math
import sys
from typing import List, Tuple, Dict, Any


# =============================================================================
# SECCIÓN 1: DERIVADAS, GRADIENTE Y CURVAS DE NIVEL (DIAPS. 1-16)
# =============================================================================

def verify_gradient_geometry(x1: float = 1.5, x2: float = 1.0, a1: float = 1.0, a2: float = 2.0) -> Dict[str, Any]:
    """
    Verifica en f(x1, x2) = 0.5 * (a1 * x1^2 + a2 * x2^2) las dos propiedades
    geométricas del gradiente (Diapositivas 12-15):
      - Propiedad 1 (Diap. 14): Para un paso Delta_x de norma fija delta, el cambio
        lineal Delta_y ≈ grad_f · Delta_x es máximo cuando theta = 0° (alineado con +grad_f)
        y mínimo cuando theta = 180° (alineado con -grad_f).
      - Propiedad 2 (Diap. 15): El vector tangente a la curva de nivel f(x) = c es
        t = [-df/dx2, df/dx1], cuyo producto punto con grad_f es exactamente 0 (perpendicular).
    """
    grad = [a1 * x1, a2 * x2]
    grad_norm = math.hypot(grad[0], grad[1])
    tangent = [-grad[1], grad[0]]
    dot_tangent = grad[0] * tangent[0] + grad[1] * tangent[1]

    delta = 0.01
    # Direcciones unitarias a distintos ángulos respecto al gradiente
    base_angle = math.atan2(grad[1], grad[0])
    directional_changes = {}
    for deg in (0, 60, 90, 120, 180):
        ang = base_angle + math.radians(deg)
        dx = [delta * math.cos(ang), delta * math.sin(ang)]
        dy_linear = grad[0] * dx[0] + grad[1] * dx[1]
        directional_changes[deg] = dy_linear

    return {
        "point": [x1, x2],
        "grad": grad,
        "grad_norm": grad_norm,
        "tangent_level_curve": tangent,
        "dot_grad_tangent": dot_tangent,
        "directional_changes": directional_changes,
    }


def demo_seccion_1() -> bool:
    """
    Ejecuta la demostración de la Sección 1: Derivadas, Gradiente, Producto Punto
    y Perpendicularidad a las Curvas de Nivel (Diapositivas 1-16).
    """
    print("\n" + "=" * 84)
    print(" UNIDAD III — SECCIÓN 1: DERIVADAS, GRADIENTE Y CURVAS DE NIVEL (DIAPS. 1-16)")
    print("=" * 84)

    geom = verify_gradient_geometry(x1=1.5, x2=1.0, a1=1.0, a2=2.0)
    print(f"\nEvaluación sobre f(x1, x2) = 0.5*(x1^2 + 2*x2^2) en el punto x = {geom['point']}:")
    print(f"   • Gradiente ∇f(x)                     = {geom['grad']} (||∇f|| = {geom['grad_norm']:.4f})")
    print(f"   • Tangente a la curva de nivel t      = {geom['tangent_level_curve']}")
    print(f"   • Producto punto ∇f(x) · t            = {geom['dot_grad_tangent']:.6f}  => Perpendicular (Propiedad 2, Diap. 15)")

    print("\nCambio lineal Δy ≈ ∇f · Δx para un paso de largo fijo ||Δx|| = 0.01 según el ángulo θ con ∇f (Diaps. 13-14):")
    print("-" * 84)
    print(f" {'Ángulo θ con ∇f':^20} | {'cos(θ)':^12} | {'Δy ≈ ||∇f||·||Δx||·cos(θ)':^28} | {'Interpretación':^18}")
    print("-" * 84)
    labels = {
        0: "Máximo crecimiento",
        60: "Crece (semiespacio)",
        90: "Curva de nivel (0)",
        120: "Decrece",
        180: "Máximo descenso",
    }
    for deg, dy in geom["directional_changes"].items():
        print(f" {f'{deg}°':^20} | {math.cos(math.radians(deg)):^+12.3f} | {dy:^+28.6f} | {labels[deg]:^18}")
    print("-" * 84)

    ok_perp = abs(geom["dot_grad_tangent"]) < 1e-12
    ok_max = geom["directional_changes"][0] > geom["directional_changes"][60] > 0
    ok_min = geom["directional_changes"][180] < geom["directional_changes"][120] < 0
    return ok_perp and ok_max and ok_min


# =============================================================================
# SECCIÓN 2: SEGUNDA DERIVADA Y MATRIZ HESSIANA (DIAPS. 17-25)
# =============================================================================

def evaluate_slide_20_1d(x: float = 2.0) -> Dict[str, float]:
    """
    Ejemplo en una variable (Diapositiva 20):
      f(x) = x^2 - 4x + 5
      f'(x) = 2x - 4 = 0  ==>  x = 2
      f''(x) = 2 > 0      ==>  x = 2 es mínimo global con f(2) = 1.
    """
    fx = x * x - 4.0 * x + 5.0
    df = 2.0 * x - 4.0
    d2f = 2.0
    return {"x": x, "f": fx, "df": df, "d2f": d2f}


def evaluate_slide_24_25_3d() -> Dict[str, Any]:
    """
    Ejemplo en varias variables (Diapositivas 24-25):
      f(x1, x2, x3) = x1^2 + x1*(1 - x2) + x2^2 - x2*x3 + x3^2 + x3
    Gradiente:
      df/dx1 =  2*x1 - x2 + 1 = 0
      df/dx2 = -x1 + 2*x2 - x3 = 0
      df/dx3 = -x2 + 2*x3 + 1 = 0
    Solución crítica:
      x* = (-1, -1, -1), con f(-1, -1, -1) = -1.
    Hessiano constante:
      H = [[ 2, -1,  0],
           [-1,  2, -1],
           [ 0, -1,  2]]
    Valores propios de esta matriz tridiagonal de Toeplitz 3x3:
      lambda_1 = 2 + sqrt(2) ≈ 3.414214
      lambda_2 = 2 - sqrt(2) ≈ 0.585786
      lambda_3 = 2.0
    Como todos son > 0, H es definido positivo y x* = (-1, -1, -1) es mínimo global.
    """
    x_star = [-1.0, -1.0, -1.0]
    x1, x2, x3 = x_star
    f_val = (x1 ** 2) + x1 * (1.0 - x2) + (x2 ** 2) - x2 * x3 + (x3 ** 2) + x3
    grad_at_star = [
        2.0 * x1 + 1.0 - x2,
        -x1 + 2.0 * x2 - x3,
        -x2 + 2.0 * x3 + 1.0,
    ]
    hessian = [
        [2.0, -1.0, 0.0],
        [-1.0, 2.0, -1.0],
        [0.0, -1.0, 2.0],
    ]
    eigenvalues = [2.0 + math.sqrt(2.0), 2.0 - math.sqrt(2.0), 2.0]
    # Verificar traza y determinante: traza = 6, det = 2*(4-1) - (-1)*(-2) = 6 - 2 = 4
    trace_h = sum(hessian[i][i] for i in range(3))
    prod_eigs = eigenvalues[0] * eigenvalues[1] * eigenvalues[2]

    return {
        "x_star": x_star,
        "f_min": f_val,
        "grad_at_star": grad_at_star,
        "hessian": hessian,
        "eigenvalues": eigenvalues,
        "trace": trace_h,
        "det": prod_eigs,
        "positive_definite": all(lmb > 0 for lmb in eigenvalues),
    }


def demo_seccion_2() -> bool:
    """
    Ejecuta la demostración de la Sección 2: Condición de Mínimo con Segunda Derivada
    y Matriz Hessiana (Diapositivas 17-25).
    """
    print("\n" + "=" * 84)
    print(" UNIDAD III — SECCIÓN 2: SEGUNDA DERIVADA Y MATRIZ HESSIANA (DIAPS. 17-25)")
    print("=" * 84)

    r1d = evaluate_slide_20_1d(2.0)
    print("\n1. Ejemplo en 1D (Diapositiva 20): f(x) = x^2 - 4x + 5")
    print(f"   • Punto crítico: f'({r1d['x']:.0f}) = {r1d['df']:.1f}  ==>  x* = {r1d['x']:.0f}")
    print(f"   • Curvatura:     f''({r1d['x']:.0f}) = {r1d['d2f']:.1f} > 0  ==>  Mínimo global con f(2) = {r1d['f']:.1f}")

    r3d = evaluate_slide_24_25_3d()
    print("\n2. Ejemplo en 3 Variables con Matriz Hessiana (Diapositivas 24-25):")
    print("   Minimizar f(x1, x2, x3) = x1^2 + x1(1 - x2) + x2^2 - x2*x3 + x3^2 + x3")
    print(f"   • Solución de ∇f(x) = 0:  x* = {r3d['x_star']}  (∇f(x*) = {r3d['grad_at_star']})")
    print(f"   • Valor en el punto crítico: f(-1, -1, -1) = {r3d['f_min']:.1f}")
    print("   • Matriz Hessiana ∇²f (simétrica 3x3):")
    for row in r3d["hessian"]:
        print(f"       [{row[0]:+5.1f}, {row[1]:+5.1f}, {row[2]:+5.1f}]")
    eigs_str = ", ".join(f"λ_{i+1} = {v:.3f}" for i, v in enumerate(r3d["eigenvalues"]))
    print(f"   • Valores propios (Diap. 25): {eigs_str}")
    print(f"   • ¿Definido positivo (todos λ_i > 0)?: {'SÍ => Mínimo global confirmado' if r3d['positive_definite'] else 'NO'}")

    ok1 = abs(r1d["df"]) < 1e-9 and r1d["d2f"] > 0 and abs(r1d["f"] - 1.0) < 1e-9
    ok2 = all(abs(g) < 1e-9 for g in r3d["grad_at_star"]) and abs(r3d["f_min"] - (-1.0)) < 1e-9 and r3d["positive_definite"]
    return ok1 and ok2


# =============================================================================
# SECCIÓN 3: DESCENSO POR GRADIENTE Y LAS 200 ÉPOCAS EN REGRESIÓN (DIAPS. 26-35)
# =============================================================================

SLIDE_35_DATA: List[Tuple[float, float]] = [(1.0, 1.0), (2.0, 3.0), (4.0, 3.0)]


def train_linear_regression_slide35(
    data: List[Tuple[float, float]] = SLIDE_35_DATA,
    eta: float = 0.1,
    epochs: int = 200,
) -> Dict[str, Any]:
    """
    Reproduce exactamente el ejemplo de las 200 épocas en Regresión Lineal por
    Mínimos Cuadrados de las Diapositivas 34 y 35:
      Datos D_train = {(1, 1), (2, 3), (4, 3)}, phi(x) = [1, x], eta = 0.1, w^(0) = [0, 0]
      TrainLoss(w) = (1 / |D_train|) * sum_{(x,y)} (w · phi(x) - y)^2
      grad_w TrainLoss(w) = (1 / |D_train|) * sum_{(x,y)} 2 * (w · phi(x) - y) * phi(x)
    """
    w = [0.0, 0.0]
    n = float(len(data))
    snapshots: Dict[int, Dict[str, Any]] = {}

    for t in range(epochs + 1):
        # Evaluar pérdida y gradiente en el paso t
        loss = 0.0
        grad = [0.0, 0.0]
        for x, y in data:
            phi = [1.0, x]
            pred = w[0] * phi[0] + w[1] * phi[1]
            res = pred - y
            loss += res * res
            grad[0] += 2.0 * res * phi[0]
            grad[1] += 2.0 * res * phi[1]
        loss /= n
        grad[0] /= n
        grad[1] /= n

        if t in (0, 1, 2, 5, 10, 50, 100, 200):
            snapshots[t] = {
                "t": t,
                "w": [w[0], w[1]],
                "grad": [grad[0], grad[1]],
                "loss": loss,
            }

        if t < epochs:
            w = [w[0] - eta * grad[0], w[1] - eta * grad[1]]

    return {
        "w_final": w,
        "snapshots": snapshots,
    }


def demo_seccion_3() -> bool:
    """
    Ejecuta la demostración de la Sección 3: Descenso por Gradiente y el ejemplo
    de las 200 épocas en Regresión Lineal (Diapositivas 26-35).
    """
    print("\n" + "=" * 84)
    print(" UNIDAD III — SECCIÓN 3: DESCENSO POR GRADIENTE EN REGRESIÓN LINEAL (DIAPS. 26-35)")
    print("=" * 84)

    res = train_linear_regression_slide35(SLIDE_35_DATA, eta=0.1, epochs=200)
    snaps = res["snapshots"]

    print("\nReproducción exacta de la tabla de la Diapositiva 35:")
    print("Datos {(1, 1), (2, 3), (4, 3)}, φ(x) = [1, x], η = 0.1, w^(0) = [0, 0]:")
    print("-" * 84)
    print(f" {'Época t':^8} | {'∇_w TrainLoss(w) (antes del paso t)':^34} | {'Vector de pesos w^(t)':^24} | {'TrainLoss(w)':^12}")
    print("-" * 84)
    print(f" {0:^8d} | {'—':^34} | {'[0.00, 0.00]':^24} | {snaps[0]['loss']:^12.4f}")
    # En la lámina 35, la fila t muestra el gradiente usado para ir de t-1 a t y el nuevo w^(t)
    g1 = snaps[0]["grad"]
    w1 = snaps[1]["w"]
    print(f" {1:^8d} | {f'[{g1[0]:+.2f}, {g1[1]:+.2f}]':^34} | {f'[{w1[0]:.2f}, {w1[1]:.2f}]':^24} | {snaps[1]['loss']:^12.4f}")
    g2 = snaps[1]["grad"]
    w2 = snaps[2]["w"]
    print(f" {2:^8d} | {f'[{g2[0]:+.2f}, {g2[1]:+.2f}]':^34} | {f'[{w2[0]:.2f}, {w2[1]:.2f}]':^24} | {snaps[2]['loss']:^12.4f}")
    g10 = snaps[10]["grad"]
    w10 = snaps[10]["w"]
    print(f" {10:^8d} | {f'[{g10[0]:+.2f}, {g10[1]:+.2f}]':^34} | {f'[{w10[0]:.2f}, {w10[1]:.2f}]':^24} | {snaps[10]['loss']:^12.4f}")
    g200 = snaps[200]["grad"]
    w200 = snaps[200]["w"]
    print(f" {200:^8d} | {f'[{g200[0]:.2f}, {g200[1]:.2f}]':^34} | {f'[{w200[0]:.2f}, {w200[1]:.2f}]':^24} | {snaps[200]['loss']:^12.4f}")
    print("-" * 84)

    # Verificar concordancia exacta con la Diapositiva 35:
    # t=1: grad=[-4.67, -12.67], w=[0.47, 1.27]
    # t=2: grad=[2.18, 7.24], w=[0.25, 0.54]
    # t=200: grad=[0, 0], w=[1, 0.57]
    ok_t1 = abs(round(g1[0], 2) - (-4.67)) < 1e-6 and abs(round(g1[1], 2) - (-12.67)) < 1e-6
    ok_w1 = abs(round(w1[0], 2) - 0.47) < 1e-6 and abs(round(w1[1], 2) - 1.27) < 1e-6
    ok_t2 = abs(round(g2[0], 2) - 2.18) < 1e-6 and abs(round(g2[1], 2) - 7.24) < 1e-6
    ok_w2 = abs(round(w2[0], 2) - 0.25) < 1e-6 and abs(round(w2[1], 2) - 0.54) < 1e-6
    ok_w200 = abs(round(w200[0], 2) - 1.00) < 1e-6 and abs(round(w200[1], 2) - 0.57) < 1e-6

    return ok_t1 and ok_w1 and ok_t2 and ok_w2 and ok_w200


# =============================================================================
# SECCIÓN 4: LOS 4 REGÍMENES DEL TAMAÑO DE PASO Y CURVATURA 2D (DIAPS. 36-45)
# =============================================================================

def verify_slide_45_python_script() -> List[Dict[str, Any]]:
    """
    Reproduce exactamente el código de la Diapositiva 45 ("Verificación en Python"):
      E(w) = w^2,  dE(w) = 2*w,  E''(w) = 2  ==>  eta_opt = 1 / 2 = 0.5
      Partiendo de w = 1.0 durante 20 iteraciones para eta in [0.1, 0.5, 0.75, 1.1]:
        - eta = 0.1  -> 0.0115   (converge monótono, lento)
        - eta = 0.5  -> 0.0      (un paso y listo)
        - eta = 0.75 -> 9.5e-07  (oscila y converge)
        - eta = 1.1  -> 38.34    (diverge)
    """
    def dE(w: float) -> float:
        return 2.0 * w

    regimes = {
        0.1: "converge monótono, lento (η < η_opt)",
        0.5: "un paso y listo (η = η_opt)",
        0.75: "oscila y converge (η_opt < η < 2η_opt)",
        1.1: "diverge (η > 2η_opt)",
    }
    results = []
    for eta in [0.1, 0.5, 0.75, 1.1]:
        w = 1.0
        trajectory = [w]
        for _ in range(20):
            w = w - eta * dE(w)
            trajectory.append(w)
        results.append({
            "eta": eta,
            "factor_1_minus_2eta": 1.0 - 2.0 * eta,
            "w_after_20": w,
            "w_step_1": trajectory[1],
            "regime": regimes[eta],
            "trajectory": trajectory,
        })
    return results


def simulate_2d_anisotropic_quad(
    eta: float,
    a1: float = 1.0,
    a2: float = 3.0,
    w0: Tuple[float, float] = (-2.0, 1.2),
    steps: int = 18,
) -> Dict[str, Any]:
    """
    Simula las trayectorias 2D sobre la cuadrática desacoplada de las Diapositivas 41-43:
      E(w) = 0.5 * (a1 * w1^2 + a2 * w2^2)
    Con a1 = 1, a2 = 3 (Diap. 42):
      eta_{1,opt} = 1 / a1 = 1.0
      eta_{2,opt} = 1 / a2 = 1/3 ≈ 0.3333
      Cota de convergencia global (Diap. 43): eta < 2 * min_i eta_{i,opt} = 2/3 ≈ 0.6667.
    """
    w1, w2 = w0
    path = [(w1, w2)]
    for _ in range(steps):
        g1 = a1 * w1
        g2 = a2 * w2
        w1 = w1 - eta * g1
        w2 = w2 - eta * g2
        path.append((w1, w2))

    eta1_opt = 1.0 / a1
    eta2_opt = 1.0 / a2
    bound = 2.0 * min(eta1_opt, eta2_opt)

    return {
        "eta": eta,
        "eta1_opt": eta1_opt,
        "eta2_opt": eta2_opt,
        "max_stable_eta": bound,
        "start": w0,
        "end": path[-1],
        "converges": eta < bound,
        "oscillates_in_w2": eta > eta2_opt,
        "path": path,
    }


def demo_seccion_4_y_5() -> bool:
    """
    Ejecuta la demostración de las Secciones 4 y 5: Los 4 regímenes del tamaño de paso,
    Verificación en Python (Diap. 45) y Curvaturas por Dirección en 2D (Diaps. 36-47).
    """
    print("\n" + "=" * 84)
    print(" UNIDAD III — SECCIONES 4 Y 5: TAMAÑO DE PASO, 4 REGÍMENES Y CURVATURA 2D (DIAPS. 36-47)")
    print("=" * 84)

    res_45 = verify_slide_45_python_script()
    print("\n1. Verificación en Python sobre E(w) = w^2 (E'' = 2 => η_opt = 0.5, 2η_opt = 1.0) — Diapositiva 45:")
    print("-" * 84)
    print(f" {'Tasa η':^8} | {'Factor (1 - 2η)':^16} | {'w^(1) (1 paso)':^15} | {'w^(20) (20 pasos)':^18} | {'Régimen Teórico':^22}")
    print("-" * 84)
    for r in res_45:
        print(f" {r['eta']:^8.2f} | {r['factor_1_minus_2eta']:^+16.2f} | {r['w_step_1']:^+15.4f} | {r['w_after_20']:^+18.6g} | {r['regime']}")
    print("-" * 84)

    print("\n2. Trayectorias en 2D sobre E(w) = 1/2*(w1^2 + 3*w2^2) (η_{1,opt}=1, η_{2,opt}=1/3) — Diapositiva 42:")
    cases_42 = [
        ("0.75 * η_{2,opt} (Converge monótono)", 0.75 / 3.0),
        ("1.50 * η_{2,opt} (Zigzag y converge)", 1.50 / 3.0),
        ("2.10 * η_{2,opt} (Diverge / explota)", 2.10 / 3.0),
    ]
    for label, eta_val in cases_42:
        sim = simulate_2d_anisotropic_quad(eta_val)
        w_end = sim["end"]
        print(f"   • {label:<36} (η = {eta_val:.3f}): w^(0)={sim['start']} --> w^(18)=[{w_end[0]:+.4f}, {w_end[1]:+.4f}]")

    # Validar los 4 valores numéricos exactos de la Diapositiva 45
    ok_01 = abs(res_45[0]["w_after_20"] - 0.011529) < 1e-4
    ok_05 = abs(res_45[1]["w_after_20"] - 0.0) < 1e-12
    ok_075 = abs(res_45[2]["w_after_20"] - 9.5367e-07) < 1e-8
    ok_11 = abs(res_45[3]["w_after_20"] - 38.3376) < 1e-2

    return ok_01 and ok_05 and ok_075 and ok_11


def run_all_gradient_demos() -> bool:
    """Ejecuta todas las demostraciones de la Unidad III (Descenso por Gradiente)."""
    ok1 = demo_seccion_1()
    ok2 = demo_seccion_2()
    ok3 = demo_seccion_3()
    ok45 = demo_seccion_4_y_5()
    all_ok = ok1 and ok2 and ok3 and ok45
    print("\n" + "=" * 84)
    print(f" RESUMEN UNIDAD III (DESCENSO POR GRADIENTE): {'✅ TODAS LAS DEMOSTRACIONES VERIFICADAS' if all_ok else '❌ ERROR'}")
    print("=" * 84)
    return all_ok


def interactive_gradient_menu():
    """Menú interactivo en consola para la Unidad III: Descenso por Gradiente."""
    while True:
        print("\n" + "=" * 78)
        print(" UNIDAD III: LABORATORIO DE DESCENSO POR GRADIENTE — DOCTORADO UCN")
        print("=" * 78)
        print("   [1] Sección 1: Derivadas, Gradiente, Producto Punto y Curvas de Nivel (Diaps. 1-16)")
        print("   [2] Sección 2: Condición de Mínimo, Segunda Derivada y Hessiano 3x3 (Diaps. 17-25)")
        print("   [3] Sección 3: Algoritmo Iterativo y las 200 Épocas en Regresión Lineal (Diaps. 26-35)")
        print("   [4] Secciones 4 y 5: Los 4 Regímenes de Paso (Diap. 45) y Trayectorias 2D (Diaps. 36-47)")
        print("   [5] Ejecutar todas las demostraciones de Descenso por Gradiente")
        print("   [6] Volver / Salir")
        print("-" * 78)
        try:
            op = input("Seleccione una opción [1-6]: ").strip().lower()
            if op == "1":
                demo_seccion_1()
            elif op == "2":
                demo_seccion_2()
            elif op == "3":
                demo_seccion_3()
            elif op == "4":
                demo_seccion_4_y_5()
            elif op == "5":
                run_all_gradient_demos()
            elif op in ("6", "q", "salir", "exit"):
                break
            else:
                print("⚠️ Opción no válida.")
        except (KeyboardInterrupt, EOFError):
            print("\nVolviendo...")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("--auto", "-a", "--all"):
        sys.exit(0 if run_all_gradient_demos() else 1)
    interactive_gradient_menu()
