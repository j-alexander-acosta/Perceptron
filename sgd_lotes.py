#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
DESCENSO POR GRADIENTE ESTOCÁSTICO Y LOTES — UNIDAD VI (41 DIAPOSITIVAS)
===============================================================================
Material de estudio y verificación computacional para el curso de Ciencia de
Datos (Doctorado UCN), basado en la cátedra «Descenso por Gradiente Estocástico
y Lotes» (CMU 11-785, B. Raj et al. y Stanford CS221, P. Liang et al.,
41 diapositivas).

Cubre e implementa de forma verificable los 5 bloques de la clase:
  1. El costo del lote y actualizaciones incrementales: SGD y barajado (Diaps. 5–14):
     - Costo por iteración del lote (N gradientes para 1 actualización) frente a
       SGD (N actualizaciones por época con las mismas TN evaluaciones de gradiente).
     - Reproducción exacta de la traza manual de SGD (Diapositiva 14) sobre
       {(1,1), (2,3), (4,3)} con phi(x)=[1,x], eta=0.1, w=[0,0], Loss=(w.phi(x)-y)^2:
         * (1,1): pred=0,    r=-1,    grad=[-2, -2],       w_nuevo=[0.2, 0.2]
         * (2,3): pred=0.6,  r=-2.4,  grad=[-4.8, -9.6],   w_nuevo=[0.68, 1.16]
         * (4,3): pred=5.32, r=+2.32, grad=[4.64, 18.56],  w_nuevo=[0.22, -0.70]
       comparado con la actualización única de lote w=[0.47, 1.27] y el óptimo [1, 0.57].
     - Efecto del orden de presentación (Diaps. 12–13): oscilación cíclica con
       orden fijo por grupos (A luego B) vs. convergencia con barajado (shuffling)
       en cada época.
  2. Tasa decreciente, Robbins-Monro (1951) y velocidad de convergencia (Diaps. 15–20):
     - Perseguir al último ejemplo con paso grande constante vs. paso decreciente.
     - Condiciones suficientes de Robbins y Monro (1951): sum(eta_k) = inf y
       sum(eta_k^2) < inf (eta_k = 1/sqrt(k) y eta_k ~ 1/k).
     - Complejidad en pérdidas fuertemente convexas: Lote O(log(1/eps)) iteraciones
       (N gradientes c/u) vs. SGD O(1/eps) iteraciones (1 gradiente c/u).
     - Experimento en Python (Diapositiva 19): regresión lineal en R^10 donde SGD
       con eta = 1/sqrt(k) se acerca a w_real viendo solo 1/10 de los datos.
  3. ¿Por qué funciona? La varianza del estimador y los mini-lotes (Diaps. 21–29, 39):
     - Estimadores insesgados y ley de varianza:
       Var(Lote) = Var(Loss)/N, Var(Mini-lote) = Var(Loss)/b, Var(SGD) = Var(Loss).
     - Continuo entre b=1 (SGD) y b=N (lote), paralelismo en GPU y trampa de la
       Diapositiva 29 (con b=1 el mini-lote ES SGD).
  4. Optimizadores con tendencia: Momentum, Nesterov, RMSprop y Adam (Diaps. 30–38):
     - Momentum (beta=0.9) y Gradiente Acelerado de Nesterov (gradiente evaluado
       en la posición anticipada w^(k-1) + beta*Delta_w^(k-1)).
     - Verificación exacta de la tabla de la Diapositiva 34 y del RMS de la
       Diapositiva 35:
         * Delta_x = [1, 1, 2, 1, 1.5]      -> total=6.5,  neto=6.5, RMS=1.36
         * Delta_y = [+2.5, -3, +2.5, -2, +1.5] -> total=11.5, neto=1.5, RMS=2.36
     - RMSprop y Adam (momentos m^(k), v^(k) y corrección 1/(1 - beta^k)).
     - Trayectorias en movimiento (Diapositiva 38): SGD vs. Momentum vs. Nesterov
       vs. RMSprop vs. Adam en un valle largo y angosto y en un punto de silla.
===============================================================================
"""

from __future__ import annotations

import math
import random
import sys
from typing import List, Tuple, Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE COLORES ANSI PARA TERMINAL
# ─────────────────────────────────────────────────────────────────────────────
USE_COLOR = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def cyan(t: str) -> str:
    return _c(t, "1;36")


def green(t: str) -> str:
    return _c(t, "1;32")


def yellow(t: str) -> str:
    return _c(t, "1;33")


def red(t: str) -> str:
    return _c(t, "1;31")


def magenta(t: str) -> str:
    return _c(t, "1;35")


def bold(t: str) -> str:
    return _c(t, "1")


def banner(title: str, subtitle: str = "") -> None:
    line = "═" * 79
    print("\n" + cyan(line))
    print(cyan("  " + title))
    if subtitle:
        print("  " + yellow(subtitle))
    print(cyan(line))


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1: EL COSTO DEL LOTE, SGD A MANO (DIAP. 14) Y BARAJADO (DIAPS. 5–14)
# ─────────────────────────────────────────────────────────────────────────────
def sgd_one_epoch_slide14() -> List[Dict[str, Any]]:
    """
    Reproduce exactamente la tabla de la Diapositiva 14 («SGD a mano: una época»):
      Datos: {(1, 1), (2, 3), (4, 3)}, phi(x) = [1, x], eta = 0.1, w = [0, 0]
      Loss(x, y, w) = (w . phi(x) - y)^2
      grad_w Loss   = 2 * r * phi(x), donde r = w . phi(x) - y
    """
    data = [(1.0, 1.0), (2.0, 3.0), (4.0, 3.0)]
    eta = 0.1
    w = [0.0, 0.0]
    rows: List[Dict[str, Any]] = []

    for x, y in data:
        phi = [1.0, x]
        pred = w[0] * phi[0] + w[1] * phi[1]
        r = pred - y
        grad = [2.0 * r * phi[0], 2.0 * r * phi[1]]
        w_new = [w[0] - eta * grad[0], w[1] - eta * grad[1]]
        rows.append({
            "example": (int(x), int(y)),
            "pred": pred,
            "r": r,
            "grad": grad,
            "w_new": w_new,
        })
        w = w_new

    return rows


def demo_lote_vs_sgd_a_mano() -> None:
    """
    Demostración 1 (Diapositivas 5–14):
    - Compara el costo del descenso por gradiente de lote vs. SGD (Diap. 10).
    - Verifica paso a paso la tabla de «SGD a mano: una época» (Diap. 14).
    - Demuestra por qué el orden fijo por grupos cicla y el barajado (shuffling)
      por época converge (Diaps. 12–13).
    """
    banner(
        "SECCIÓN 1: EL COSTO DEL LOTE, SGD A MANO Y BARAJADO (DIAPS. 5–14)",
        "1 época de SGD = N actualizaciones vs. 1 sola actualización de lote",
    )

    print(
        "\n  • " + bold("Comparación algorítmica tras T épocas con N ejemplos (Diap. 10):")
    )
    print(f"    {'Método':<10} | {'Mueve w':<16} | {'Evalúa el gradiente':<22} | {'Con N=1,000,000 y T=1'}")
    print("    " + "─" * 73)
    print(f"    {'Lote':<10} | {'T veces':<16} | {'T * N veces':<22} | {'1 paso  (1,000,000 grads)'}")
    print(f"    {'SGD':<10} | {'T * N veces':<16} | {'T * N veces':<22} | {'1,000,000 pasos (mismo costo)'}")

    # 1) Verificación exacta de la Diapositiva 14
    rows = sgd_one_epoch_slide14()
    print(
        "\n  • "
        + bold("Verificación exacta de «SGD a mano: una época» (Diapositiva 14):")
        + "\n    Datos {(1, 1), (2, 3), (4, 3)}, ϕ(x) = [1, x], η = 0.1, w^(0) = [0, 0]:"
    )
    print(
        f"    {'Ejemplo':<10} | {'Predicción w·ϕ(x)':<18} | {'Residuo r':<11} | "
        f"{'∇_w Loss = 2rϕ(x)':<21} | {'w nuevo (Diap. 14)'}"
    )
    print("    " + "─" * 78)

    expected_w_round = [[0.20, 0.20], [0.68, 1.16], [0.22, -0.70]]
    for idx, row in enumerate(rows):
        ex_str = f"({row['example'][0]}, {row['example'][1]})"
        grad_str = f"[{row['grad'][0]:>5.2f}, {row['grad'][1]:>6.2f}]"
        w_exact = row["w_new"]
        w_disp = f"[{w_exact[0]:>5.2f}, {w_exact[1]:>6.2f}]"
        exp_w = expected_w_round[idx]
        assert abs(round(w_exact[0], 2) - exp_w[0]) < 1e-9
        assert abs(round(w_exact[1], 2) - exp_w[1]) < 1e-9
        print(
            f"    {ex_str:<10} | {row['pred']:>18.2f} | {row['r']:>+11.2f} | "
            f"{grad_str:<21} | {green(w_disp)}"
        )

    # Contraste con 1 paso de lote sobre los mismos 3 puntos
    data = [(1.0, 1.0), (2.0, 3.0), (4.0, 3.0)]
    g0 = sum(2.0 * (0.0 - y) * 1.0 for _, y in data) / 3.0
    g1 = sum(2.0 * (0.0 - y) * x for x, y in data) / 3.0
    w_batch_1 = [-0.1 * g0, -0.1 * g1]
    assert abs(round(w_batch_1[0], 2) - 0.47) < 1e-9
    assert abs(round(w_batch_1[1], 2) - 1.27) < 1e-9

    # Estabilización con más épocas y eta decreciente hacia [1.00, 0.57]
    rng = random.Random(42)
    w_conv = [0.0, 0.0]
    k_step = 0
    for _ in range(400):
        order = list(data)
        rng.shuffle(order)
        for x, y in order:
            k_step += 1
            eta_k = 0.15 / math.sqrt(k_step)
            r = (w_conv[0] + w_conv[1] * x) - y
            w_conv[0] -= eta_k * 2.0 * r * 1.0
            w_conv[1] -= eta_k * 2.0 * r * x

    print(
        f"\n    - Una época de lote da 1 sola actualización: w_lote^(1) = [{w_batch_1[0]:.2f}, {w_batch_1[1]:.2f}]."
    )
    print(
        "    - En el 3.er paso de SGD, (4, 3) tiene ϕ(x)=[1, 4] grande y arrastra fuerte los pesos."
    )
    print(
        f"    - Con más épocas y η decreciente, SGD se estabiliza en w ≈ [{w_conv[0]:.2f}, {w_conv[1]:.2f}] "
        f"(óptimo exacto [1.00, 0.57])."
    )

    # 2) Orden fijo por grupos vs. Barajado en cada época (Diaps. 12–13)
    # Grupo A: pendiente positiva y = +x; Grupo B: pendiente negativa y = -x;
    # El equilibrio global sobre A∪B es la recta horizontal w* = [0, 0].
    group_a = [(0.5, 0.5), (1.0, 1.0), (1.5, 1.5), (2.0, 2.0)]
    group_b = [(0.5, -0.5), (1.0, -1.0), (1.5, -1.5), (2.0, -2.0)]
    ordered_dataset = group_a + group_b  # Todos los A primero, luego todos los B

    def train_order_demo(shuffle_each_epoch: bool, epochs: int = 25) -> Tuple[List[float], float]:
        rng_local = random.Random(7)
        w_loc = [0.0, 0.0]
        eta_loc = 0.06
        for _ in range(epochs):
            pts = list(ordered_dataset)
            if shuffle_each_epoch:
                rng_local.shuffle(pts)
            for x, y in pts:
                r = (w_loc[0] + w_loc[1] * x) - y
                w_loc[0] -= eta_loc * 2.0 * r
                w_loc[1] -= eta_loc * 2.0 * r * x
        loss = sum(((w_loc[0] + w_loc[1] * x) - y) ** 2 for x, y in ordered_dataset) / len(ordered_dataset)
        return w_loc, loss

    w_no_shuf, loss_no_shuf = train_order_demo(shuffle_each_epoch=False)
    w_shuf, loss_shuf = train_order_demo(shuffle_each_epoch=True)

    print(
        "\n  • "
        + bold("Cuidado con el orden de presentación: Grupo A luego Grupo B (Diaps. 12–13):")
    )
    print(
        f"    - Sin barajar (siempre A luego B): termina sesgado hacia el último grupo visto "
        f"w = [{w_no_shuf[0]:+.3f}, {w_no_shuf[1]:+.3f}], TrainLoss = {red(f'{loss_no_shuf:.4f}')}"
    )
    print(
        f"    - Barajando en cada época (shuffling): converge mucho mejor al equilibrio "
        f"w = [{w_shuf[0]:+.3f}, {w_shuf[1]:+.3f}], TrainLoss = {green(f'{loss_shuf:.4f}')}"
    )
    assert loss_shuf < loss_no_shuf
    print(green("  ✓ Verificación completada: traza de la Diap. 14 y efecto del barajado exactos."))


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 2: ROBBINS-MONRO (1951), VELOCIDAD Y PYTHON (DIAPS. 15–20)
# ─────────────────────────────────────────────────────────────────────────────
def demo_robbins_monro_y_experimento_python() -> None:
    """
    Demostración 2 (Diapositivas 15–20):
    - Verifica por qué perseguir al último ejemplo con paso fijo oscila sin fin
      y cómo reducir eta_k amortigua los vaivenes (Diaps. 15–16).
    - Comprueba numéricamente las condiciones de Robbins y Monro (1951) (Diap. 17).
    - Reproduce el experimento de regresión en R^10 de la Diapositiva 19.
    """
    banner(
        "SECCIÓN 2: TASA DECRECIENTE, ROBBINS-MONRO Y SGD EN PYTHON (DIAPS. 15–20)",
        "Condiciones suficientes: ∑ η_k = ∞  y  ∑ η_k² < ∞",
    )

    # 1) Recta por el origen y = w*x: paso constante vs. paso decreciente (Diaps. 15-16)
    pts_origin = [(1.0, 0.6), (2.0, 1.8), (1.5, 0.9), (2.5, 2.2), (1.2, 1.1), (2.2, 1.5)]
    # Pendiente óptima de mínimos cuadrados por el origen: w* = sum(x_i y_i) / sum(x_i^2)
    w_opt = sum(x * y for x, y in pts_origin) / sum(x * x for x, _ in pts_origin)

    w_chase = 0.0
    w_decay = 0.0
    chase_history = []
    decay_history = []
    k = 0
    rng = random.Random(12)
    for _ in range(30):
        batch = list(pts_origin)
        rng.shuffle(batch)
        for x, y in batch:
            k += 1
            # Versión extrema de la Diap. 15: saltar a pasar exactamente por el último punto (w = y/x)
            w_chase = y / x
            chase_history.append(abs(w_chase - w_opt))
            # Versión con tasa decreciente de la Diap. 16: eta_k = 0.15 / sqrt(k)
            eta_k = 0.15 / math.sqrt(k)
            grad = 2.0 * (w_decay * x - y) * x
            w_decay -= eta_k * grad
            decay_history.append(abs(w_decay - w_opt))

    print(
        "\n  • "
        + bold("Recta por el origen (Diaps. 15–16): perseguir al último ejemplo vs. η decreciente:")
    )
    print(f"    - Pendiente óptima global de mínimos cuadrados: w* = {w_opt:.4f}")
    print(
        f"    - Saltando al último punto visto (Diap. 15): sigue saltando siempre, "
        f"|w - w*| final = {red(f'{chase_history[-1]:.4f}')}"
    )
    print(
        f"    - Con tasa decreciente η_k ∝ 1/√k (Diap. 16): se estabiliza cerca del óptimo, "
        f"|w - w*| final = {green(f'{decay_history[-1]:.4f}')}"
    )
    assert decay_history[-1] < chase_history[-1]

    # 2) Condiciones suficientes de Robbins y Monro (1951) (Diap. 17)
    K_MAX = 100_000
    sum_const = sum(0.1 for _ in range(1, K_MAX + 1))
    sum_const_sq = sum(0.01 for _ in range(1, K_MAX + 1))

    sum_inv_k = sum(1.0 / k_i for k_i in range(1, K_MAX + 1))
    sum_inv_k_sq = sum(1.0 / (k_i * k_i) for k_i in range(1, K_MAX + 1))

    sum_inv_k2 = sum(0.1 / (k_i * k_i) for k_i in range(1, K_MAX + 1))
    sum_inv_k2_sq = sum((0.1 / (k_i * k_i)) ** 2 for k_i in range(1, K_MAX + 1))

    print(
        "\n  • "
        + bold("Condiciones de Robbins y Monro (1951) evaluadas hasta K = 100,000 (Diap. 17):")
    )
    print(
        f"    {'Esquema η_k':<18} | {'∑ η_k (debe → ∞)':<20} | {'∑ η_k² (debe < ∞)':<20} | {'Comportamiento'}"
    )
    print("    " + "─" * 78)
    print(
        f"    {'Constante η=0.1':<18} | {sum_const:>18.2f} ✓ | {sum_const_sq:>18.2f} ✗ | "
        f"{red('No deja de oscilar alrededor del mínimo')}"
    )
    print(
        f"    {'Óptima η_k = 1/k':<18} | {sum_inv_k:>18.4f} ✓ | {sum_inv_k_sq:>18.4f} ✓ | "
        f"{green('Cumple ambas (∑ 1/k² = π²/6 ≈ 1.6449)')}"
    )
    print(
        f"    {'Rápida η_k=0.1/k²':<18} | {sum_inv_k2:>18.4f} ✗ | {sum_inv_k2_sq:>18.4f} ✓ | "
        f"{yellow('Se congela antes si empieza lejos (∑ η_k < 0.17)')}"
    )

    # 3) Reproducción del experimento en Python de la Diapositiva 19
    # Usamos N = 10,000 ejemplos en R^10 y miramos solo 1/10 de los datos (1,000 pasos de SGD)
    rng_py = random.Random(0)
    N_exp = 10_000
    D_exp = 10
    w_real = [rng_py.gauss(0.0, 1.0) for _ in range(D_exp)]
    X_exp = [[rng_py.gauss(0.0, 1.0) for _ in range(D_exp)] for _ in range(N_exp)]
    y_exp = [
        sum(x_j * w_j for x_j, w_j in zip(X_exp[i], w_real)) + 0.2 * rng_py.gauss(0.0, 1.0)
        for i in range(N_exp)
    ]

    w_sgd = [0.0] * D_exp
    indices = list(range(N_exp))
    rng_py.shuffle(indices)
    tenth = N_exp // 10
    for k_step, idx in enumerate(indices[:tenth], start=1):
        eta_k = 0.15 / math.sqrt(k_step)
        x_row = X_exp[idx]
        pred = sum(w_j * x_j for w_j, x_j in zip(w_sgd, x_row))
        err = pred - y_exp[idx]
        for j in range(D_exp):
            w_sgd[j] -= eta_k * 2.0 * err * x_row[j]

    init_dist = math.sqrt(sum(w_j ** 2 for w_j in w_real))
    final_dist = math.sqrt(sum((w_j - wr_j) ** 2 for w_j, wr_j in zip(w_sgd, w_real)))

    print(
        "\n  • "
        + bold("Experimento de la Diapositiva 19 (SGD viendo solo 1/10 de los datos):")
    )
    print(f"    - Distancia inicial ||w^(0) - w_real||₂          = {init_dist:.4f}")
    print(
        f"    - Distancia tras ver solo 1/10 de los datos (SGD) = {green(f'{final_dist:.4f}')} "
        "(¡el lote aún lleva 0 actualizaciones!)"
    )
    assert final_dist < 0.25 * init_dist
    print(green("  ✓ Verificación completada: Robbins-Monro y experimento de la Diap. 19 comprobados."))


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 3: VARIANZA DEL ESTIMADOR Y MINI-LOTES (DIAPS. 21–29 Y 39)
# ─────────────────────────────────────────────────────────────────────────────
def demo_varianza_estimador_y_minilotes() -> None:
    """
    Demostración 3 (Diapositivas 21–29 y 39):
    - Comprueba empíricamente que el promedio sobre un mini-lote de tamaño b es
      un estimador insesgado cuya varianza cae exactamente como Var(Loss) / b.
    - Muestra la tabla resumen de las tres formas de dar un paso (Diap. 39).
    """
    banner(
        "SECCIÓN 3: LA VARIANZA DEL ESTIMADOR Y LOS MINI-LOTES (DIAPS. 21–29, 39)",
        "E[L̂_b(w)] = E[Loss(x,y,w)] (insesgado)  y  Var(L̂_b) = (1/b) Var(Loss)",
    )

    rng = random.Random(2026)
    N_pop = 4096
    # Pérdidas individuales Loss(x_i, y_i, w) en un punto w fijo
    losses = [
        (math.sin(2.0 * math.pi * i / N_pop) - 0.4 * math.cos(4.0 * math.pi * i / N_pop) + rng.gauss(0, 0.5)) ** 2
        for i in range(N_pop)
    ]
    mean_true = sum(losses) / N_pop
    var_single = sum((lv - mean_true) ** 2 for lv in losses) / N_pop

    print(
        f"\n  • Población de N = {N_pop} ejemplos: E[Loss] = {mean_true:.4f}, Var(Loss) = {var_single:.4f}"
    )
    print(
        f"\n    {'Tamaño b':<14} | {'Media empírica E[L̂_b]':<22} | {'Var teórica Var/b':<19} | "
        f"{'Var empírica':<15} | {'Actualiz./época (N/b)'}"
    )
    print("    " + "─" * 88)

    batch_sizes = [1, 2, 4, 16, 64, 256, N_pop]
    num_trials = 1500
    for b in batch_sizes:
        if b == N_pop:
            emp_mean = mean_true
            emp_var = 0.0  # sobre el conjunto fijo D_train completo
        else:
            estimates = []
            for _ in range(num_trials):
                sample = rng.sample(losses, b)
                estimates.append(sum(sample) / b)
            emp_mean = sum(estimates) / len(estimates)
            emp_var = sum((e - emp_mean) ** 2 for e in estimates) / len(estimates)
        theo_var = var_single / b
        label = f"b={b} (SGD)" if b == 1 else (f"b={b} (Lote)" if b == N_pop else f"b={b}")
        updates_per_epoch = N_pop // b
        print(
            f"    {label:<14} | {emp_mean:>22.4f} | {theo_var:>19.4f} | "
            f"{emp_var:>15.4f} | {updates_per_epoch:>20d}"
        )

    print(
        "\n  • " + bold("Resumen de las tres formas de dar un paso (Diapositiva 39):")
    )
    print(
        f"    {'Criterio':<28} | {'Lote':<20} | {'Mini-lote (b)':<22} | {'SGD'}"
    )
    print("    " + "─" * 86)
    print(f"    {'Ejemplos por actualización':<28} | {'N':<20} | {'b':<22} | {'1'}")
    print(f"    {'Actualizaciones por época':<28} | {'1':<20} | {'N / b':<22} | {'N'}")
    print(f"    {'Varianza del paso':<28} | {'(1/N) Var(Loss)':<20} | {'(1/b) Var(Loss)':<22} | {'Var(Loss)'}")
    print(f"    {'Quién la usa':<28} | {'Problemas pequeños':<20} | {'Casi todo el mundo':<22} | {'Flujos de datos'}")
    print(
        "\n  • "
        + yellow("Trampa conceptual (Diap. 29): ")
        + "Decir que «con b = 1 el mini-lote es más rápido que SGD» es FALSO: "
        + "con b = 1 el mini-lote ES exactamente SGD."
    )
    print(green("  ✓ Verificación completada: ley 1/b de la varianza del mini-lote demostrada."))


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 4: OPTIMIZADORES CON TENDENCIA: MOMENTUM, NESTEROV, RMSPROP Y ADAM
#            (DIAPS. 30–38)
# ─────────────────────────────────────────────────────────────────────────────
def verify_slide34_35_rmsprop_table() -> Dict[str, float]:
    """
    Verifica exactamente la tabla de 5 pasos de la Diapositiva 34 y los valores
    RMS vertical ≈ 2.36 y horizontal ≈ 1.36 de la Diapositiva 35.
    """
    dx = [1.0, 1.0, 2.0, 1.0, 1.5]
    dy = [2.5, -3.0, 2.5, -2.0, 1.5]

    total_x = sum(abs(v) for v in dx)
    total_y = sum(abs(v) for v in dy)
    net_x = sum(dx)
    net_y = sum(dy)

    rms_x = math.sqrt(sum(v * v for v in dx) / len(dx))
    rms_y = math.sqrt(sum(v * v for v in dy) / len(dy))

    assert abs(total_x - 6.5) < 1e-9
    assert abs(total_y - 11.5) < 1e-9
    assert abs(net_x - 6.5) < 1e-9
    assert abs(net_y - 1.5) < 1e-9
    assert abs(round(rms_x, 2) - 1.36) < 1e-9
    assert abs(round(rms_y, 2) - 2.36) < 1e-9

    return {
        "total_x": total_x,
        "total_y": total_y,
        "net_x": net_x,
        "net_y": net_y,
        "rms_x": rms_x,
        "rms_y": rms_y,
    }


def simulate_optimizers_valley_and_saddle(steps: int = 35) -> Dict[str, Any]:
    """
    Simula los optimizadores de las Diapositivas 31–38:
      - SGD puro
      - Momentum (beta = 0.85)
      - Nesterov (beta = 0.85)
      - RMSprop (gamma = 0.9)
      - Adam (beta1 = 0.9, beta2 = 0.999, con corrección de sesgo)
    en dos escenarios (Diapositiva 38):
      1. Valle largo y angosto: E(x, y) = 0.5 * (x^2 + 25 * y^2) desde (-6.0, 2.0)
      2. Punto de silla:        E(x, y) = 0.5 * (x^2 - y^2)    desde (-4.0, 0.001)
    """
    # Escenario 1: Valle largo y angosto E(x, y) = 0.5*(x^2 + 25*y^2)
    def grad_valley(x: float, y: float) -> Tuple[float, float]:
        return (x, 25.0 * y)

    def loss_valley(x: float, y: float) -> float:
        return 0.5 * (x * x + 25.0 * y * y)

    eta_v = 0.045
    beta_v = 0.75

    # SGD
    sgd_v = [(-6.0, 2.0)]
    for _ in range(steps):
        x, y = sgd_v[-1]
        gx, gy = grad_valley(x, y)
        sgd_v.append((x - eta_v * gx, y - eta_v * gy))

    # Momentum
    mom_v = [(-6.0, 2.0)]
    vx, vy = 0.0, 0.0
    for _ in range(steps):
        x, y = mom_v[-1]
        gx, gy = grad_valley(x, y)
        vx = beta_v * vx - eta_v * gx
        vy = beta_v * vy - eta_v * gy
        mom_v.append((x + vx, y + vy))

    # Nesterov
    nag_v = [(-6.0, 2.0)]
    vx, vy = 0.0, 0.0
    for _ in range(steps):
        x, y = nag_v[-1]
        gx, gy = grad_valley(x + beta_v * vx, y + beta_v * vy)
        vx = beta_v * vx - eta_v * gx
        vy = beta_v * vy - eta_v * gy
        nag_v.append((x + vx, y + vy))

    # RMSprop
    rms_v = [(-6.0, 2.0)]
    ex2, ey2 = 0.0, 0.0
    gamma = 0.9
    eps = 1e-8
    eta_rms = 0.35
    for _ in range(steps):
        x, y = rms_v[-1]
        gx, gy = grad_valley(x, y)
        ex2 = gamma * ex2 + (1.0 - gamma) * (gx * gx)
        ey2 = gamma * ey2 + (1.0 - gamma) * (gy * gy)
        rms_v.append((x - (eta_rms / math.sqrt(ex2 + eps)) * gx, y - (eta_rms / math.sqrt(ey2 + eps)) * gy))

    # Adam
    adam_v = [(-6.0, 2.0)]
    mx, my = 0.0, 0.0
    vx2, vy2 = 0.0, 0.0
    b1, b2 = 0.85, 0.99
    eta_adam = 0.42
    for k in range(1, steps + 1):
        x, y = adam_v[-1]
        gx, gy = grad_valley(x, y)
        mx = b1 * mx + (1.0 - b1) * gx
        my = b1 * my + (1.0 - b1) * gy
        vx2 = b2 * vx2 + (1.0 - b2) * (gx * gx)
        vy2 = b2 * vy2 + (1.0 - b2) * (gy * gy)
        mhat_x = mx / (1.0 - b1 ** k)
        mhat_y = my / (1.0 - b1 ** k)
        vhat_x = vx2 / (1.0 - b2 ** k)
        vhat_y = vy2 / (1.0 - b2 ** k)
        adam_v.append(
            (
                x - (eta_adam / (math.sqrt(vhat_x) + eps)) * mhat_x,
                y - (eta_adam / (math.sqrt(vhat_y) + eps)) * mhat_y,
            )
        )

    # Escenario 2: Punto de silla E(x, y) = 0.5*(x^2 - y^2) desde (-4.0, 0.001)
    def grad_saddle(x: float, y: float) -> Tuple[float, float]:
        return (x, -y)

    sgd_s = [(-4.0, 0.001)]
    mom_s = [(-4.0, 0.001)]
    adam_s = [(-4.0, 0.001)]
    vx_s, vy_s = 0.0, 0.0
    mx_s, my_s, vx2_s, vy2_s = 0.0, 0.0, 0.0, 0.0
    for k in range(1, steps + 1):
        # SGD
        xs, ys = sgd_s[-1]
        gx, gy = grad_saddle(xs, ys)
        sgd_s.append((xs - 0.08 * gx, ys - 0.08 * gy))
        # Momentum
        xm, ym = mom_s[-1]
        gx, gy = grad_saddle(xm, ym)
        vx_s = 0.9 * vx_s - 0.08 * gx
        vy_s = 0.9 * vy_s - 0.08 * gy
        mom_s.append((xm + vx_s, ym + vy_s))
        # Adam (normaliza la pequeñísima derivada en y y escapa de inmediato)
        xa, ya = adam_s[-1]
        gx, gy = grad_saddle(xa, ya)
        mx_s = 0.9 * mx_s + 0.1 * gx
        my_s = 0.9 * my_s + 0.1 * gy
        vx2_s = 0.999 * vx2_s + 0.001 * (gx * gx)
        vy2_s = 0.999 * vy2_s + 0.001 * (gy * gy)
        mh_x = mx_s / (1.0 - 0.9 ** k)
        mh_y = my_s / (1.0 - 0.9 ** k)
        vh_x = vx2_s / (1.0 - 0.999 ** k)
        vh_y = vy2_s / (1.0 - 0.999 ** k)
        adam_s.append(
            (
                xa - (0.18 / (math.sqrt(vh_x) + eps)) * mh_x,
                ya - (0.18 / (math.sqrt(vh_y) + eps)) * mh_y,
            )
        )

    return {
        "valley": {
            "SGD": (sgd_v[-1], loss_valley(*sgd_v[-1])),
            "Momentum": (mom_v[-1], loss_valley(*mom_v[-1])),
            "Nesterov": (nag_v[-1], loss_valley(*nag_v[-1])),
            "RMSprop": (rms_v[-1], loss_valley(*rms_v[-1])),
            "Adam": (adam_v[-1], loss_valley(*adam_v[-1])),
        },
        "saddle": {
            "SGD": sgd_s[-1],
            "Momentum": mom_s[-1],
            "Adam": adam_s[-1],
        },
    }


def demo_optimizadores_con_tendencia() -> None:
    """
    Demostración 4 (Diapositivas 30–38):
    - Verifica la tabla direccional de la Diap. 34 y el RMS de la Diap. 35.
    - Compara qué información usa cada optimizador (Diap. 37).
    - Simula las trayectorias en el valle largo y angosto y en el punto de silla (Diap. 38).
    """
    banner(
        "SECCIÓN 4: OPTIMIZADORES CON TENDENCIA: MOMENTUM, NESTEROV, RMSPROP Y ADAM",
        "Suavizar el gradiente (1.er momento) y adaptar la tasa por dirección (2.º momento)",
    )

    stats = verify_slide34_35_rmsprop_table()
    print(
        "\n  • "
        + bold("Verificación exacta de las Diapositivas 34 y 35 (Tasa por dirección y RMSprop):")
    )
    print("    Pasos: Δx = [1, 1, 2, 1, 1.5] (avanza parejo) | Δy = [+2.5, -3, +2.5, -2, +1.5] (oscila)")
    print(
        f"    - Movimiento total ∑|Δ|:  Δx = {stats['total_x']:>4.1f}  |  Δy = {stats['total_y']:>4.1f}  "
        "(la vertical se mueve casi el doble)"
    )
    print(
        f"    - Avance neto ∑Δ:         Δx = {stats['net_x']:>4.1f}  |  Δy = {stats['net_y']:>4.1f}  "
        "(la que más se mueve es la que menos avanza)"
    )
    rms_x_str = green(f"{stats['rms_x']:.2f}")
    rms_y_str = yellow(f"{stats['rms_y']:.2f}")
    print(
        f"    - Raíz cuadrática media:  RMS horizontal ≈ {rms_x_str}  |  "
        f"RMS vertical ≈ {rms_y_str} (frena la dirección oscilante)"
    )

    print(
        "\n  • " + bold("¿Qué información usa cada optimizador? (Diapositiva 37):")
    )
    print(
        f"    {'Optimizador':<24} | {'Tendencia media (¿hacia dónde?)':<33} | {'Segundo momento (¿cuánto se agita?)'}"
    )
    print("    " + "─" * 92)
    print(f"    {'SGD puro':<24} | {red('no'):<42} | {red('no')}")
    print(f"    {'Momentum / Nesterov':<24} | {green('sí (promedio móvil de ∂_w)'):<42} | {red('no')}")
    print(f"    {'RMSprop':<24} | {red('no'):<42} | {green('sí (promedio móvil de ∂_w²)')}")
    print(f"    {'Adam':<24} | {green('sí (m̂ corregido)'):<42} | {green('sí (v̂ corregido)')}")

    sim = simulate_optimizers_valley_and_saddle(steps=35)
    print(
        "\n  • "
        + bold("Las trayectorias en movimiento tras 35 pasos (Diapositiva 38):")
    )
    print("    1) Valle largo y angosto E(x, y) = 0.5*(x² + 25y²) desde (-6.0, 2.0):")
    for name, (pos, loss_val) in sim["valley"].items():
        print(
            f"       - {name:<10}: posición final = ({pos[0]:>+6.3f}, {pos[1]:>+6.3f}) | "
            f"Pérdida E = {loss_val:.5f}"
        )

    assert sim["valley"]["Nesterov"][1] < sim["valley"]["Momentum"][1] < sim["valley"]["SGD"][1]
    assert sim["valley"]["Adam"][1] < sim["valley"]["SGD"][1]

    print("\n    2) Punto de silla E(x, y) = 0.5*(x² - y²) desde (-4.0, 0.001):")
    for name, pos in sim["saddle"].items():
        escaped = "¡Escapó rápido por el eje inestable y!" if abs(pos[1]) > 1.0 else "Casi pegado en la silla (y ≈ 0)"
        col = green if abs(pos[1]) > 1.0 else (yellow if abs(pos[1]) > 0.05 else red)
        print(
            f"       - {name:<10}: posición = ({pos[0]:>+6.3f}, {pos[1]:>+7.3f}) -> {col(escaped)}"
        )
    assert abs(sim["saddle"]["Adam"][1]) > abs(sim["saddle"]["Momentum"][1]) > abs(sim["saddle"]["SGD"][1])
    print(green("  ✓ Verificación completada: RMSprop (Diap. 35) y trayectorias (Diap. 38) comprobados."))


# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN INTEGRAL Y MENÚ INTERACTIVO DE CONSOLA
# ─────────────────────────────────────────────────────────────────────────────
def run_all_sgd_demos() -> None:
    """Ejecuta secuencialmente todas las verificaciones de la Unidad VI."""
    demo_lote_vs_sgd_a_mano()
    demo_robbins_monro_y_experimento_python()
    demo_varianza_estimador_y_minilotes()
    demo_optimizadores_con_tendencia()


def interactive_sgd_menu() -> None:
    """Menú interactivo en consola para explorar la Unidad VI."""
    while True:
        banner(
            "LABORATORIO UNIDAD VI — DESCENSO POR GRADIENTE ESTOCÁSTICO Y LOTES",
            "Ciencia de Datos • Doctorado UCN (CMU 11-785 & Stanford CS221 — 41 Diapositivas)",
        )
        print("  [1] Costo del lote, «SGD a mano: una época» (Diap. 14) y barajado (Diaps. 5–14)")
        print("  [2] Tasa decreciente, Robbins-Monro (1951) y experimento Python (Diaps. 15–20)")
        print("  [3] La varianza del estimador y actualización por mini-lotes (Diaps. 21–29, 39)")
        print("  [4] Optimizadores con tendencia: Momentum, Nesterov, RMSprop y Adam (Diaps. 30–38)")
        print("  [5] Ejecutar TODAS las verificaciones de la Unidad VI")
        print("  [0] Volver / Salir")
        print("─" * 79)

        try:
            opc = input("Seleccione una opción [0-5]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nSaliendo del laboratorio de la Unidad VI...")
            break

        if opc == "1":
            demo_lote_vs_sgd_a_mano()
        elif opc == "2":
            demo_robbins_monro_y_experimento_python()
        elif opc == "3":
            demo_varianza_estimador_y_minilotes()
        elif opc == "4":
            demo_optimizadores_con_tendencia()
        elif opc == "5":
            run_all_sgd_demos()
        elif opc in ("0", "q", "salir", "exit"):
            break
        else:
            print(yellow("  ⚠️ Opción no válida. Ingrese un número entre 0 y 5."))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("--auto", "-a", "--all"):
        run_all_sgd_demos()
    else:
        interactive_sgd_menu()
