#!/usr/bin/env python3
"""
Módulo Educativo: Aprendizaje de Redes Neuronales
=================================================
Ciencia de Datos — Doctorado UCN
Basado en el material de B. Raj et al. (CMU 11-785) y P. Liang et al. (Stanford CS221).

Este módulo implementa de manera demostrativa, interactiva y libre de dependencias
externas obligatorias las 5 secciones de la cátedra "Aprendizaje de Redes Neuronales":

  Sección 1 (Diaps.  5-14): ¿Qué significa aprender una red? Riesgo Empírico vs. Error Real.
  Sección 2 (Diaps. 15-22): La regla de aprendizaje del perceptrón (Rosenblatt), absorción
                            del umbral, teorema de convergencia (R/gamma)^2 y código de Diap. 22.
  Sección 3 (Diaps. 23-29): ¿Por qué no escala a redes multicapa? El problema combinatorio
                            del doble pentágono, ADALINE (Widrow-Hoff LMS / Regla Delta) y MADALINE.
  Sección 4 (Diaps. 30-39): Activaciones diferenciables, señal de aprendizaje (conteo de errores
                            plano vs. distancia continua con sigmoide) y regresión logística.
  Sección 5 (Diaps. 40-56): Planteamiento formal: representaciones (lineal, sigmoide, softmax)
                            y divergencias (L2, Entropía Cruzada Binaria, Multiclase y KL).
"""

import math
import sys
from typing import List, Tuple, Dict, Any


# =============================================================================
# SECCIÓN 1 Y 2: REGLA DE APRENDIZAJE DEL PERCEPTRÓN (ROSENBLATT, DIAPS. 15-22)
# =============================================================================

SLIDE_22_X: List[List[float]] = [
    [-2.8, 1.6], [-2.2, 0.6], [-1.3, 0.7], [-1.5, -0.9],
    [0.9, 1.9],  [2.3, 2.5],  [1.3, -0.9], [-0.45, 1.6],
]
SLIDE_22_Y: List[int] = [-1, -1, -1, -1, 1, 1, 1, 1]


def dot_product(w: List[float], x: List[float]) -> float:
    """Producto punto w · x."""
    return sum(wi * xi for wi, xi in zip(w, x))


def vec_norm(v: List[float]) -> float:
    """Norma euclidiana ||v||_2."""
    return math.sqrt(sum(vi * vi for vi in v))


def absorb_threshold(weights: List[float], threshold: float, inputs: List[float]) -> Tuple[List[float], List[float]]:
    """
    Truco algebraico para absorber el umbral T (Diapositiva 16):
      sum_{i=1}^N w_i x_i >= T  <==>  sum_{i=1}^{N+1} w_i x_i >= 0
    agregando una entrada constante x_{N+1} = 1 con peso w_{N+1} = -T.
    """
    w_aug = list(weights) + [-threshold]
    x_aug = list(inputs) + [1.0]
    return w_aug, x_aug


def train_rosenblatt_perceptron(
    X: List[List[float]],
    y: List[int],
    max_epochs: int = 50,
) -> Dict[str, Any]:
    """
    Algoritmo del Perceptrón en línea de Rosenblatt (1958) — Diapositivas 18 y 22:
      1. Inicializar w = 0
      2. Repetir hasta una pasada completa sin errores (o max_epochs):
         Para cada par (x_i, y_i):
           Si y_i * (w · x_i) <= 0:
               w <- w + y_i * x_i
    """
    dim = len(X[0])
    w = [0.0] * dim
    history: List[Dict[str, Any]] = []
    converged = False
    epochs_run = 0

    for epoch in range(1, max_epochs + 1):
        epochs_run = epoch
        errors_in_epoch = 0
        for idx, (xi, yi) in enumerate(zip(X, y)):
            margin_val = yi * dot_product(w, xi)
            if margin_val <= 0:
                w_before = list(w)
                w = [ round(wj + yi * xj, 6) for wj, xj in zip(w, xi) ]
                errors_in_epoch += 1
                history.append({
                    "step": len(history) + 1,
                    "epoch": epoch,
                    "sample_idx": idx,
                    "x": xi,
                    "y": yi,
                    "margin_before": margin_val,
                    "w_before": w_before,
                    "w_after": list(w),
                })
        if errors_in_epoch == 0:
            converged = True
            break

    # Calcular radio R y margen geométrico gamma si convergió y ||w|| > 0
    R = max(vec_norm(xi) for xi in X)
    w_norm = vec_norm(w)
    if w_norm > 0 and converged:
        gamma = min((yi * dot_product(w, xi)) / w_norm for xi, yi in zip(X, y))
        theoretical_bound = (R / gamma) ** 2 if gamma > 0 else float("inf")
    else:
        gamma = 0.0
        theoretical_bound = float("inf")

    return {
        "w": w,
        "converged": converged,
        "epochs": epochs_run,
        "total_updates": len(history),
        "history": history,
        "R": R,
        "gamma": gamma,
        "theoretical_bound": theoretical_bound,
    }


def demo_seccion_1_y_2() -> bool:
    """
    Ejecuta la demostración de las Secciones 1 y 2:
    Riesgo Empírico, Absorción del Umbral y el Algoritmo de Rosenblatt en Diapositiva 22.
    """
    print("\n" + "=" * 82)
    print(" UNIDAD II — SECCIONES 1 Y 2: RIESGO EMPÍRICO Y REGLA DEL PERCEPTRÓN (DIAPS. 5-22)")
    print("=" * 82)

    print("\n1. ¿Qué significa aprender una red? (Diapositivas 5-14):")
    print("   • Error Real (inaccesible): TotalErr(W) = ∫ error(f(X; W), g(X)) dX")
    print("   • Riesgo Empírico (entrenamiento): TrainLoss(W) = (1/N) * sum_{i=1}^N error(f(X_i; W), d_i)")

    # Demostración de absorción del umbral (Diap. 16)
    w_orig, T_orig, x_orig = [1.5, -2.0], 0.8, [2.0, 1.0]
    w_aug, x_aug = absorb_threshold(w_orig, T_orig, x_orig)
    lhs = dot_product(w_orig, x_orig) - T_orig
    rhs = dot_product(w_aug, x_aug)
    print("\n2. Truco para absorber el umbral T (Diapositiva 16):")
    print(f"   • Original: w={w_orig}, x={x_orig}, T={T_orig}  =>  w·x - T = {lhs:.4f}")
    print(f"   • Aumentado (x_{{N+1}}=1, w_{{N+1}}=-T): w'={w_aug}, x'={x_aug}  =>  w'·x' = {rhs:.4f}")

    # Demostración exacta de la Diapositiva 22
    res = train_rosenblatt_perceptron(SLIDE_22_X, SLIDE_22_Y)
    print("\n3. Simulación Exacta de la Diapositiva 22 (8 puntos en R^2):")
    print("-" * 82)
    print(f" {'Paso':^4} | {'Época':^5} | {'Muestra i':^9} | {'x_i':^14} | {'y_i':^4} | {'w antes':^14} | {'w después (w + y*x)':^20}")
    print("-" * 82)
    for h in res["history"]:
        x_str = f"[{h['x'][0]:.2f}, {h['x'][1]:.2f}]"
        wb_str = f"[{h['w_before'][0]:.2f}, {h['w_before'][1]:.2f}]"
        wa_str = f"[{h['w_after'][0]:.2f}, {h['w_after'][1]:.2f}]"
        print(f" {h['step']:^4} | {h['epoch']:^5} | {h['sample_idx']:^9} | {x_str:^14} | {h['y']:^+4d} | {wb_str:^14} | {wa_str:^20}")
    print("-" * 82)

    expected_w = [3.25, 1.9]
    ok_w = (abs(res["w"][0] - expected_w[0]) < 1e-6 and abs(res["w"][1] - expected_w[1]) < 1e-6)
    print(f"   • Vector de pesos final aprendido: w = {res['w']} (Esperado en Diap. 22: {expected_w})")
    print(f"   • Número total de actualizaciones: {res['total_updates']} (Convergió en {res['epochs']} épocas)")
    print(f"   • Teorema de Convergencia (Diap. 20):")
    print(f"       Radio máximo R = max ||x_i|| = {res['R']:.4f}")
    print(f"       Margen alcanzado gamma       = {res['gamma']:.4f}")
    print(f"       Cota teórica (R / gamma)^2   = {res['theoretical_bound']:.2f} >= {res['total_updates']} actualizaciones [OK]")

    # Caso no linealmente separable (XOR en {-1, +1})
    xor_X = [[-1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0]]
    xor_y = [-1, 1, 1, -1]
    res_xor = train_rosenblatt_perceptron(xor_X, xor_y, max_epochs=10)
    print(f"\n4. Caso NO linealmente separable — XOR (Diapositiva 21):")
    print(f"   • Tras 10 épocas ({res_xor['total_updates']} actualizaciones), ¿convergió?: {'SÍ' if res_xor['converged'] else 'NO (Cicla indefinidamente)'}")

    return ok_w and res["converged"] and not res_xor["converged"]


# =============================================================================
# SECCIÓN 3: ¿POR QUÉ NO ESCALA A MULTICAPA? Y ADALINE / LMS (DIAPS. 23-29)
# =============================================================================

def train_adaline_lms(
    X: List[List[float]],
    d: List[float],
    lr: float = 0.02,
    epochs: int = 30,
) -> Dict[str, Any]:
    """
    ADALINE (Widrow & Hoff, 1960) — Regla Delta o LMS (Diapositiva 28):
    A diferencia del perceptrón de Rosenblatt, el aprendizaje ocurre ANTES del umbral,
    minimizando el error cuadrático medio (1/2)*(d - z)^2 respecto a la pre-activación
    continua z = w · x:
        w_i <- w_i + eta * (d - z) * x_i
    """
    dim = len(X[0])
    w = [0.0] * dim
    mse_history: List[float] = []

    for _ in range(epochs):
        for xi, di in zip(X, d):
            z = dot_product(w, xi)
            err = di - z
            w = [wj + lr * err * xj for wj, xj in zip(w, xi)]
        # Evaluar MSE al final de la época
        mse = sum(0.5 * (di - dot_product(w, xi)) ** 2 for xi, di in zip(X, d)) / len(X)
        mse_history.append(mse)

    # Evaluar precisión de clasificación con signo(w · x)
    correct = sum(1 for xi, di in zip(X, d) if (1 if dot_product(w, xi) >= 0 else -1) == di)
    return {
        "w": [round(wi, 4) for wi in w],
        "mse_initial": mse_history[0],
        "mse_final": mse_history[-1],
        "mse_history": mse_history,
        "accuracy": correct / len(X),
    }


def demo_seccion_3() -> bool:
    """
    Demuestra la diferencia entre la regla discreta de Rosenblatt y la regla continua
    Delta/LMS de ADALINE (Diapositivas 23-29).
    """
    print("\n" + "=" * 82)
    print(" UNIDAD II — SECCIÓN 3: LÍMITE MULTICAPA Y ADALINE / REGLA DELTA (DIAPS. 23-29)")
    print("=" * 82)
    print("\n1. ¿Por qué la regla del perceptrón no escala a redes multicapa? (Diaps. 23-27):")
    print("   • En el ejemplo del doble pentágono, incluso si fijamos todos los pesos de la red")
    print("     excepto una sola neurona de primera capa, los puntos positivos globales NO son")
    print("     linealmente separables para esa recta individual.")
    print("   • Asignar etiquetas intermedias ocultas es un problema combinatorio exponencial.")

    adaline_res = train_adaline_lms(SLIDE_22_X, [float(yi) for yi in SLIDE_22_Y], lr=0.02, epochs=40)
    print("\n2. Paréntesis Histórico: ADALINE y la Regla Delta / LMS (Widrow & Hoff, 1960 — Diap. 28):")
    print("   • Regla Delta: w_i <- w_i + eta * (d - z) * x_i  (minimiza 1/2*(d - z)^2 antes del escalón)")
    print(f"   • Pesos ADALINE tras 40 épocas (eta=0.02): w = {adaline_res['w']}")
    print(f"   • MSE época 1: {adaline_res['mse_initial']:.4f}  -->  MSE época 40: {adaline_res['mse_final']:.4f}")
    print(f"   • Exactitud de clasificación en los 8 puntos: {adaline_res['accuracy']*100:.1f}%")

    return adaline_res["accuracy"] == 1.0 and adaline_res["mse_final"] < adaline_res["mse_initial"]


# =============================================================================
# SECCIÓN 4: ACTIVACIONES DIFERENCIABLES Y SEÑAL DE APRENDIZAJE (DIAPS. 30-39)
# =============================================================================

def sigmoid(z: float) -> float:
    """Función logística sigmoide sigma(z) = 1 / (1 + e^{-z})."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def evaluate_threshold_signals(threshold_T: float, stiffness_k: float = 2.0) -> Dict[str, float]:
    """
    Compara la señal de aprendizaje al mover un umbral T en 1D sobre un conjunto
    con ligero traslape (Diapositivas 31-36):
      - Con activación de escalón: cuenta errores discretos (derivada 0 casi en todas partes).
      - Con activación sigmoide: mide la suma de distancias continuas sum |sigma(k*(x_i - T)) - y_i|,
        cuya pendiente guía hacia el umbral óptimo incluso sin cruzar ninguna muestra.
    """
    # Conjunto 1D con ligero traslape alrededor de x = 0 (Diap. 31)
    samples_1d = [
        (-3.2, 0), (-2.5, 0), (-1.8, 0), (-1.1, 0), (-0.4, 0), (0.35, 0),
        (-0.35, 1), (0.4, 1), (1.1, 1), (1.8, 1), (2.5, 1), (3.2, 1),
    ]
    discrete_errors = 0
    continuous_l1_dist = 0.0
    continuous_bce = 0.0

    for x_i, y_i in samples_1d:
        pred_step = 1 if x_i >= threshold_T else 0
        if pred_step != y_i:
            discrete_errors += 1

        pred_sig = sigmoid(stiffness_k * (x_i - threshold_T))
        continuous_l1_dist += abs(pred_sig - y_i)
        eps = 1e-12
        continuous_bce += -(y_i * math.log(pred_sig + eps) + (1 - y_i) * math.log(1 - pred_sig + eps))

    return {
        "T": threshold_T,
        "discrete_errors": discrete_errors,
        "continuous_l1_dist": continuous_l1_dist,
        "continuous_bce": continuous_bce / len(samples_1d),
    }


def demo_seccion_4() -> bool:
    """
    Demuestra por qué el conteo de errores tiene derivada 0 entre muestras mientras que
    la sigmoide entrega una señal continua y diferenciable (Diapositivas 30-39).
    """
    print("\n" + "=" * 82)
    print(" UNIDAD II — SECCIÓN 4: ACTIVACIONES DIFERENCIABLES Y SEÑAL DE APRENDIZAJE (DIAPS. 30-39)")
    print("=" * 82)

    print("\nComparación al perturbar ligeramente el umbral T entre dos muestras (ej. de T=0.50 a T=0.90):")
    print("-" * 82)
    print(f" {'Umbral T':^10} | {'N° Errores (Escalón)':^22} | {'Δ Errores':^12} | {'Distancia Continua (Sigmoide)':^30}")
    print("-" * 82)

    t_values = [-0.8, 0.0, 0.50, 0.70, 0.90, 1.50]
    prev_err = None
    results = []
    for t in t_values:
        r = evaluate_threshold_signals(t)
        results.append(r)
        delta_str = "---" if prev_err is None else f"{r['discrete_errors'] - prev_err:+d}"
        prev_err = r["discrete_errors"]
        print(f" {t:^10.2f} | {r['discrete_errors']:^22d} | {delta_str:^12} | {r['continuous_l1_dist']:^30.4f}")
    print("-" * 82)
    print(" • Observación clave (Diap. 34): Entre T=0.50, T=0.70 y T=0.90 el conteo de errores")
    print("   permanece clavado en 2 (derivada = 0, sin señal), mientras que la distancia con")
    print("   sigmoide disminuye continuamente hacia el óptimo en T=0.00.")

    # Verificar que entre T=0.50 y T=0.90 el número de errores discretos no cambia pero la distancia continua sí cambia
    r_05 = evaluate_threshold_signals(0.50)
    r_09 = evaluate_threshold_signals(0.90)
    ok = (r_05["discrete_errors"] == r_09["discrete_errors"]) and (r_05["continuous_l1_dist"] < r_09["continuous_l1_dist"])
    return ok


# =============================================================================
# SECCIÓN 5: REPRESENTACIONES DE SALIDA Y DIVERGENCIAS (DIAPS. 40-56)
# =============================================================================

def softmax(logits: List[float]) -> List[float]:
    """
    Capa Softmax (Diapositiva 51):
      y_i = exp(z_i) / sum_k exp(z_k)
    Resta max(z) por estabilidad numérica.
    """
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def l2_divergence(y_pred: List[float], d_target: List[float]) -> Tuple[float, List[float]]:
    """
    Divergencia L2 (Diapositiva 45):
      div(Y, d) = (1/2) * sum_i (y_i - d_i)^2
      grad_Y div = Y - d
    """
    loss = 0.5 * sum((yi - di) ** 2 for yi, di in zip(y_pred, d_target))
    grad_y = [yi - di for yi, di in zip(y_pred, d_target)]
    return loss, grad_y


def binary_cross_entropy(y_prob: float, d_label: int) -> Tuple[float, float]:
    """
    Entropía Cruzada Binaria (Diapositiva 48):
      div(Y, d) = -d * log(Y) - (1 - d) * log(1 - Y)
      d div / dY = -1/Y (si d=1)  ó  1/(1-Y) (si d=0)
    """
    eps = 1e-12
    y_clamped = min(max(y_prob, eps), 1.0 - eps)
    loss = -(d_label * math.log(y_clamped) + (1 - d_label) * math.log(1.0 - y_clamped))
    grad_y = -1.0 / y_clamped if d_label == 1 else 1.0 / (1.0 - y_clamped)
    return loss, grad_y


def multiclass_cross_entropy_and_kl(
    probs_q: List[float],
    target_p: List[float],
) -> Dict[str, Any]:
    """
    Entropía Cruzada Multiclase y Divergencia de Kullback-Leibler (Diapositivas 52-53):
      H(p, q) = -sum_i p_i log(q_i) = H(p) + KL(p || q)
    Cuando p = d es un vector one-hot, H(d) = 0 y H(d, Y) = KL(d || Y) = -log(y_c).
    Además, al combinar Softmax + Entropía Cruzada, el gradiente respecto al logit z_i
    es simplemente (y_i - d_i).
    """
    eps = 1e-12
    entropy_p = -sum(pi * math.log(pi + eps) for pi in target_p if pi > 0)
    cross_entropy = -sum(pi * math.log(max(qi, eps)) for pi, qi in zip(target_p, probs_q))
    kl_div = sum(pi * math.log((pi + eps) / max(qi, eps)) for pi, qi in zip(target_p, probs_q) if pi > 0)
    grad_logits = [qi - pi for qi, pi in zip(probs_q, target_p)]
    return {
        "H_p": entropy_p,
        "cross_entropy": cross_entropy,
        "kl_div": kl_div,
        "grad_logits": grad_logits,
    }


def demo_seccion_5() -> bool:
    """
    Ejecuta la demostración de la Sección 5 (Diapositivas 40-56):
    Tabla resumen de representaciones y divergencias (L2, Entropía Cruzada Binaria, Softmax + KL).
    """
    print("\n" + "=" * 82)
    print(" UNIDAD II — SECCIÓN 5: REPRESENTACIONES DE SALIDA Y DIVERGENCIAS (DIAPS. 40-56)")
    print("=" * 82)

    print("\n1. ¿Por qué NO usar Divergencia L2 con salida Sigmoide en clasificación? (Diaps. 47-48):")
    print("   Supongamos etiqueta real d = 1 y salida sigmoide muy equivocada Y = sigma(z) = 0.01 (z ≈ -4.6):")
    y_wrong = 0.01
    d_true = 1
    sig_prime = y_wrong * (1.0 - y_wrong)
    _, grad_l2_y = l2_divergence([y_wrong], [float(d_true)])
    grad_l2_z = grad_l2_y[0] * sig_prime  # (Y - d) * Y*(1 - Y)
    bce_loss, grad_bce_y = binary_cross_entropy(y_wrong, d_true)
    grad_bce_z = grad_bce_y * sig_prime   # (-1/Y) * Y*(1 - Y) = Y - 1
    print(f"   • Con L2:  ∂div/∂Y = {grad_l2_y[0]:+.4f}  ==>  ∂div/∂z = (Y-d)*σ'(z) = {grad_l2_z:+.5f} (¡Casi plano por saturación!)")
    print(f"   • Con BCE: ∂div/∂Y = {grad_bce_y:+.2f}  ==>  ∂div/∂z = (-1/Y)*σ'(z) = {grad_bce_z:+.5f} (¡Señal 100x más fuerte!)")

    print("\n2. Clasificación Multiclase: Softmax + Entropía Cruzada / Divergencia KL (Diaps. 50-54):")
    logits = [2.2, 0.5, -1.1]
    target_onehot = [1.0, 0.0, 0.0]  # Clase correcta c = 0
    probs = softmax(logits)
    mc = multiclass_cross_entropy_and_kl(probs, target_onehot)
    print(f"   • Logits de entrada z        = {logits}")
    print(f"   • Probabilidades Softmax Y   = {[round(p, 4) for p in probs]} (Suma = {sum(probs):.4f})")
    print(f"   • Vector objetivo one-hot d  = {target_onehot}")
    print(f"   • Entropía del objetivo H(d) = {mc['H_p']:.6f}")
    print(f"   • Entropía Cruzada H(d, Y)   = {mc['cross_entropy']:.6f}  (-log y_0 = {-math.log(probs[0]):.6f})")
    print(f"   • Divergencia KL(d || Y)     = {mc['kl_div']:.6f}  (Coincide con H(d,Y) pues H(d)=0)")
    print(f"   • Gradiente respecto a z     = {[round(g, 4) for g in mc['grad_logits']]} (Y - d)")

    print("\n3. Tabla Resumen Oficial (Diapositiva 54):")
    print("-" * 82)
    print(f" {'Problema':<22} | {'Objetivo d':<14} | {'Salida Red':<12} | {'Divergencia div(Y, d)':<24}")
    print("-" * 82)
    print(f" {'Regresión':<22} | {'Real / Vector':<14} | {'Lineal':<12} | {'L2: 1/2 ||Y - d||^2':<24}")
    print(f" {'Clasificación binaria':<22} | {'{0, 1}':<14} | {'Sigmoide':<12} | {'-d log Y - (1-d)log(1-Y)':<24}")
    print(f" {'Multiclase':<22} | {'One-hot':<14} | {'Softmax':<12} | {'Entropía Cruzada: -log y_c':<24}")
    print("-" * 82)

    return abs(mc["cross_entropy"] - mc["kl_div"]) < 1e-6 and abs(sum(probs) - 1.0) < 1e-9


def run_all_learning_demos() -> bool:
    """Ejecuta todas las demostraciones de la Unidad II (Aprendizaje de Redes Neuronales)."""
    ok12 = demo_seccion_1_y_2()
    ok3 = demo_seccion_3()
    ok4 = demo_seccion_4()
    ok5 = demo_seccion_5()
    all_ok = ok12 and ok3 and ok4 and ok5
    print("\n" + "=" * 82)
    print(f" RESUMEN UNIDAD II (APRENDIZAJE DE REDES): {'✅ TODAS LAS DEMOSTRACIONES VERIFICADAS' if all_ok else '❌ ERROR'}")
    print("=" * 82)
    return all_ok


def interactive_learning_menu():
    """Menú interactivo en consola para la Unidad II: Aprendizaje de Redes Neuronales."""
    while True:
        print("\n" + "=" * 75)
        print(" UNIDAD II: LABORATORIO DE APRENDIZAJE DE REDES NEURONALES — UCN")
        print("=" * 75)
        print("   [1] Secciones 1 y 2: Riesgo Empírico y Perceptrón de Rosenblatt (Diap. 22)")
        print("   [2] Sección 3: Por qué no escala a Multicapa y ADALINE (Regla Delta/LMS)")
        print("   [3] Sección 4: Señal de Aprendizaje (Escalón vs. Sigmoide Diferenciable)")
        print("   [4] Sección 5: Softmax y Divergencias (L2 vs. Entropía Cruzada / KL)")
        print("   [5] Ejecutar todas las demostraciones de Aprendizaje")
        print("   [6] Volver / Salir")
        print("-" * 75)
        try:
            op = input("Seleccione una opción [1-6]: ").strip().lower()
            if op == "1":
                demo_seccion_1_y_2()
            elif op == "2":
                demo_seccion_3()
            elif op == "3":
                demo_seccion_4()
            elif op == "4":
                demo_seccion_5()
            elif op == "5":
                run_all_learning_demos()
            elif op in ("6", "q", "salir", "exit"):
                break
            else:
                print("⚠️ Opción no válida.")
        except (KeyboardInterrupt, EOFError):
            print("\nVolviendo...")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("--auto", "-a", "--all"):
        sys.exit(0 if run_all_learning_demos() else 1)
    interactive_learning_menu()
