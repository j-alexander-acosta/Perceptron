#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
PROBLEMAS DE CONVERGENCIA EN EL ENTRENAMIENTO — UNIDAD V (58 DIAPOSITIVAS)
===============================================================================
Material de estudio y verificación computacional para el curso de Ciencia de
Datos (Doctorado UCN), basado en la cátedra de Problemas de Convergencia en el
Entrenamiento (CMU 11-785, B. Raj et al., 58 diapositivas).

Cubre e implementa de forma verificable los 5 bloques de la clase:
  1. ¿Encuentra Backprop la solución correcta? (Diaps. 7–16):
     - La divergencia diferenciable como sustituto (proxy) del error 0-1.
     - El experimento clásico de Brady, Raghavan y Slawny (1989):
       * 3 puntos linealmente separables: (1,0)->+1, (0,1)->+1, (-1,0)->-1.
         Con margen epsilon = 0.01, u = sigma^-1(1 - epsilon) = ln(99),
         mínimo único en w1 = w2 = u, b = 0 (frontera x1 + x2 = 0).
       * Adición del cuarto punto «saboteador» (spoiler) lejano (0, -t)->+1.
       * Saturación exponencial de la derivada logística sigma'(z4) ~ e^(-u*t)
         que apaga d(div4)/dw2 = 2*(1 - eps - sigma(z4))*sigma'(z4)*t -> 0,
         creando un mínimo local ancho que deja (0, -t) mal clasificado (3/4),
         mientras el Perceptrón separa los 4 puntos perfectamente (4/4).
  2. Superficie de pérdida, tasa de aprendizaje y número de condición (Diaps. 17–33):
     - Punto de silla canónico E(w1, w2) = 0.5*(w1^2 - w2^2) y literatura
       sobre paisajes de pérdida (Baldi-Hornik, Dauphin, Choromanska, Swirszcz).
     - Las 6 trayectorias exactas de la Diapositiva 27 sobre
       E(w1, w2) = 0.5*(w1^2 + 3*w2^2) desde w^(0) = (-8, 4) con
       eta in {0.25, 1/3, 0.5, 2/3, 0.75, 1.0}.
     - Número de condición kappa = a_max / a_min, factor geométrico óptimo
       (kappa - 1)/(kappa + 1) en eta* = 2/(a_max + a_min), y comparación
       de tasas: fuertemente convexa O(log(1/delta)) vs. Lipschitz O(1/delta).
  3. Métodos de segundo orden: normalizar la curvatura (Diaps. 34–44):
     - Cambio de coordenadas w_hat = A^(1/2) w y regla normalizada A^(-1) grad E.
     - Matriz Hessiana H_E(w) y método de Newton iterado: convergencia en 1 solo
       paso sobre cuadráticas con cualquier número de condición o rotación.
     - Los 2 problemas de Newton: explosión O(N^2) en memoria / O(N^3) en inversión,
       y atracción hacia puntos de silla por curvatura negativa (corregida por
       Levenberg-Marquardt H + lambda*I).
  4. Tasas decrecientes y pasos adaptativos: RProp y QuickProp (Diaps. 45–58):
     - Escape de mínimos locales mediocres con paso grande + decaimiento
       (lineal eta0/k, cuadrático eta0/k^2, exponencial eta0*e^(-beta*(k-1))).
     - RProp (Resilient Propagation, Diaps. 48–53) usando solo el signo de la
       derivada parcial con factores alpha > 1, beta < 1 y retroceso de paso.
     - QuickProp (Fahlman 1988, Diaps. 54–56): ajuste parabólico local por
       diferencia finita de derivadas sucesivas.
===============================================================================
"""

from __future__ import annotations

import math
import sys

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
# FUNCIONES MATEMÁTICAS AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────
def sigmoid(z: float) -> float:
    """Función logística sigma(z) = 1 / (1 + e^-z) numéricamente estable."""
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def dsigmoid_from_z(z: float) -> float:
    """Derivada sigma'(z) = sigma(z)*(1 - sigma(z)) estable ante saturación."""
    if abs(z) > 500:
        return 0.0
    s = sigmoid(z)
    return s * (1.0 - s)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1: EXPERIMENTO DE BRADY, RAGHAVAN Y SLAWNY (1989) — DIAPS. 7 A 16
# ─────────────────────────────────────────────────────────────────────────────
def demo_brady_saboteadores() -> dict:
    """
    Reproduce el experimento de Brady, Raghavan y Slawny (1989) de las
    Diapositivas 8 a 16:
      - 3 puntos base: (1,0)->+1, (0,1)->+1, (-1,0)->-1.
      - Solución analítica con margen epsilon: w1 = w2 = u = ln((1-eps)/eps), b = 0.
      - Adición del 4.º punto saboteador (0, -t)->+1 con t grande.
      - Demuestra que d(div4)/dw2 = 2*(1 - eps - sigma(z4))*sigma'(z4)*t -> 0
        por saturación exponencial de la sigmoide, atrapando a Backprop en un
        mínimo local ancho (3/4 = 75% de aciertos), mientras el Perceptrón
        separa perfectamente los 4 puntos (4/4 = 100%).
    """
    banner(
        "SECCIÓN 1: ¿ENCUENTRA BACKPROP LA SOLUCIÓN CORRECTA? (DIAPS. 7–16)",
        "Experimento de Brady, Raghavan y Slawny (1989) y puntos «saboteadores»",
    )

    eps = 0.01  # Margen epsilon = 0.01 (99% de confianza, Diap. 9)
    u = math.log((1.0 - eps) / eps)  # sigma^-1(1 - eps) = ln(99) ≈ 4.59511985

    pts_3 = [
        (1.0, 0.0, +1, 1.0 - eps),
        (0.0, 1.0, +1, 1.0 - eps),
        (-1.0, 0.0, -1, eps),
    ]

    print(
        f"\n  {bold('1. Problema de 3 puntos con activación logística (Diaps. 8–9):')}"
    )
    print(
        f"     Margen ε = {eps} ⇒ objetivo clase +1 = {1.0 - eps:.2f}, "
        f"objetivo clase -1 = {eps:.2f}"
    )
    print(f"     u = σ⁻¹(1 - ε) = ln((1 - ε)/ε) = ln(99) = {cyan(f'{u:.6f}')}")
    print(
        "     Sistema de 3 ecuaciones (Diap. 9): "
        "w₁ + b = u,  w₂ + b = u,  -w₁ + b = -u"
    )
    print(
        f"     Solución única: w₁ = {green(f'{u:.4f}')}, "
        f"w₂ = {green(f'{u:.4f}')}, b = {green('0.0000')} "
        "(frontera x₁ + x₂ = 0)"
    )

    # Verificación exacta en los 3 puntos
    loss_3 = 0.0
    for x1, x2, label, target in pts_3:
        z = u * x1 + u * x2 + 0.0
        y = sigmoid(z)
        loss_3 += (target - y) ** 2
    assert loss_3 < 1e-14

    # Análisis del 4.º punto saboteador (0, -t) con etiqueta +1 (Diaps. 10–13)
    print(
        f"\n  {bold('2. Agregamos el 4.º punto lejano (0, -t) con clase +1 (Diaps. 10–12):')}"
    )
    print(
        "     Evaluado en la solución de 3 puntos (w₁ = u, w₂ = u, b = 0):\n"
        "     z₄ = -w₂·t + b = -u·t  ⇒  d(div₄)/dw₂ = 2(1 - ε - σ(z₄))·σ'(z₄)·t"
    )
    print(
        "\n     ┌────────┬───────────┬─────────────┬──────────────┬─────────────────┬────────────────┐"
    )
    print(
        "     │   t    │ z₄ = -u·t │   σ(z₄)     │   div₄ (L₂)  │     σ'(z₄)      │  d(div₄)/dw₂   │"
    )
    print(
        "     ├────────┼───────────┼─────────────┼──────────────┼─────────────────┼────────────────┤"
    )

    grad_at_t50 = 0.0
    loss4_at_t50 = 0.0
    for t_val in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        z4 = -u * t_val
        s4 = sigmoid(z4)
        ds4 = dsigmoid_from_z(z4)
        div4 = (1.0 - eps - s4) ** 2
        d_div4_dw2 = 2.0 * (1.0 - eps - s4) * ds4 * t_val
        if t_val == 50.0:
            grad_at_t50 = d_div4_dw2
            loss4_at_t50 = div4
        print(
            f"     │ {t_val:6.1f} │ {z4:9.2f} │ {s4:11.3e} │ {div4:12.6f} │ {ds4:15.3e} │ {d_div4_dw2:14.3e} │"
        )
    print(
        "     └────────┴───────────┴─────────────┴──────────────┴─────────────────┴────────────────┘"
    )
    assert loss4_at_t50 > 0.98 and abs(grad_at_t50) < 1e-80

    # Entrenamiento real con Backprop vs. Perceptrón sobre los 4 puntos (t = 20)
    t_exp = 20.0
    pts_4 = pts_3 + [(0.0, -t_exp, +1, 1.0 - eps)]

    # Backprop desde pesos moderados cercanos al origen (w1=0.5, w2=0.5, b=0.0)
    w1_bp, w2_bp, b_bp = 0.5, 0.5, 0.0
    eta_bp = 0.5
    for _ in range(2500):
        gw1 = gw2 = gb = 0.0
        for x1, x2, _lbl, target in pts_4:
            z = w1_bp * x1 + w2_bp * x2 + b_bp
            y = sigmoid(z)
            ds = dsigmoid_from_z(z)
            # div = (target - y)^2 => d(div)/dz = -2*(target - y)*sigma'(z)
            d_dz = -2.0 * (target - y) * ds
            gw1 += d_dz * x1
            gw2 += d_dz * x2
            gb += d_dz
        w1_bp -= eta_bp * gw1
        w2_bp -= eta_bp * gw2
        b_bp -= eta_bp * gb

    bp_correct = 0
    for x1, x2, lbl, _t in pts_4:
        pred = +1 if (w1_bp * x1 + w2_bp * x2 + b_bp) >= 0 else -1
        if pred == lbl:
            bp_correct += 1

    # Regla del Perceptrón sobre los mismos 4 puntos
    w1_p, w2_p, b_p = 0.5, 0.5, 0.0
    eta_p = 1.0
    perc_epochs = 0
    for ep in range(1, 5000):
        errors = 0
        for x1, x2, lbl, _t in pts_4:
            pred = +1 if (w1_p * x1 + w2_p * x2 + b_p) > 0 else -1
            if pred != lbl:
                w1_p += eta_p * lbl * x1
                w2_p += eta_p * lbl * x2
                b_p += eta_p * lbl
                errors += 1
        if errors == 0:
            perc_epochs = ep
            break

    perc_correct = sum(
        1
        for x1, x2, lbl, _t in pts_4
        if (+1 if (w1_p * x1 + w2_p * x2 + b_p) > 0 else -1) == lbl
    )

    assert bp_correct == 3, "Backprop debe quedar atrapado acertando 3/4 puntos"
    assert perc_correct == 4, "El Perceptrón debe separar los 4/4 puntos"

    print(
        f"\n  {bold(f'3. Comparación empírica en los 4 puntos con t = {t_exp:.0f} (Diap. 13):')}"
    )
    print(
        f"     • {red('Backprop (divergencia L₂ + logística):')} "
        f"w = ({w1_bp:.3f}, {w2_bp:.3f}), b = {b_bp:.3f} "
        f"→ Aciertos: {red(f'{bp_correct}/4 (75%)')} [falla en (0, -{t_exp:.0f})]"
    )
    print(
        f"     • {green('Regla del Perceptrón (convergencia en ' + str(perc_epochs) + ' épocas):')} "
        f"w = ({w1_p:.1f}, {w2_p:.1f}), b = {b_p:.1f} "
        f"→ Aciertos: {green(f'{perc_correct}/4 (100%)')}"
    )
    print(
        f"     {green('✓ Verificado:')} Minimizar la pérdida diferenciable (sustituto) "
        "NO garantiza minimizar el error de clasificación (Diap. 16)."
    )

    return {
        "u": u,
        "loss4_at_t50": loss4_at_t50,
        "grad_at_t50": grad_at_t50,
        "bp_correct": bp_correct,
        "perc_correct": perc_correct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 2: SUPERFICIE DE PÉRDIDA, LAS 6 TRAYECTORIAS Y NÚMERO DE CONDICIÓN
# ─────────────────────────────────────────────────────────────────────────────
def demo_superficie_y_numero_condicion() -> dict:
    """
    Verifica matemáticamente las Diapositivas 17 a 33:
      - Punto de silla E(w1, w2) = 0.5*(w1^2 - w2^2) con autovalores +1 y -1.
      - Las 6 trayectorias exactas de la Diapositiva 27 sobre
        E(w1, w2) = 0.5*(w1^2 + 3*w2^2) desde w^(0) = (-8, 4) con
        eta in {0.25, 1/3, 0.5, 2/3, 0.75, 1.0}.
      - Número de condición kappa = a_max/a_min = 3, paso óptimo global
        eta* = 2/(a1 + a2) = 0.5 con factor geométrico (kappa - 1)/(kappa + 1) = 0.5.
    """
    banner(
        "SECCIÓN 2: TASA DE APRENDIZAJE, 6 TRAYECTORIAS Y NÚMERO DE CONDICIÓN (DIAPS. 17–33)",
        "E(w₁, w₂) = ½(w₁² + 3w₂²) desde w⁽⁰⁾ = (-8, 4) y κ = a_max / a_min",
    )

    a1, a2 = 1.0, 3.0
    eta1_opt = 1.0 / a1  # 1.0
    eta2_opt = 1.0 / a2  # 1/3
    kappa = max(a1, a2) / min(a1, a2)  # 3.0
    eta_star = 2.0 / (a1 + a2)  # 0.5
    rho_star = (kappa - 1.0) / (kappa + 1.0)  # 0.5

    print(
        f"\n  {bold('1. Parámetros de curvatura de E(w₁, w₂) = ½(w₁² + 3w₂²) (Diaps. 26–29):')}"
    )
    print(
        f"     a₁ = {a1:.1f} ⇒ η₁,opt = 1/a₁ = {cyan(f'{eta1_opt:.4f}')}  |  "
        f"a₂ = {a2:.1f} ⇒ η₂,opt = 1/a₂ = {cyan(f'{eta2_opt:.4f}')}"
    )
    print(
        f"     Umbral de convergencia global: η < 2·min(η₁,opt, η₂,opt) = "
        f"{yellow(f'{2.0 * eta2_opt:.4f} (2/3)')}"
    )
    print(
        f"     Número de condición κ = a_max / a_min = {magenta(f'{kappa:.1f}')}  |  "
        f"Factor óptimo (κ-1)/(κ+1) = {green(f'{rho_star:.2f}')} en η* = {eta_star:.2f}"
    )

    etas = [
        (0.25, "0.25 (< η₂,opt)", "Converge monótonamente en ambos ejes"),
        (1.0 / 3.0, "1/3 (= η₂,opt)", "w₂ llega a 0 en 1 paso; w₁ baja lento"),
        (0.50, "0.50 (η₂,opt < η < 2η₂,opt)", "Oscila en w₂ y converge (η* óptimo)"),
        (2.0 / 3.0, "2/3 (= 2η₂,opt)", "Rebota sin fin en w₂ entre +4 y -4"),
        (0.75, "0.75 (> 2η₂,opt)", "Diverge oscilando en w₂"),
        (1.00, "1.00 (= η₁,opt)", "Resuelve w₁ en 1 paso, pero w₂ explota"),
    ]

    print(
        f"\n  {bold('2. Las 6 trayectorias exactas desde w⁽⁰⁾ = (-8, 4) (Diapositiva 27):')}"
    )
    print(
        "     ┌────────────────────────────┬────────────────────┬────────────────────┬────────────────────────────────────────┐"
    )
    print(
        "     │ Tasa de aprendizaje η      │ Paso 1: w⁽¹⁾       │ Paso 4: w⁽⁴⁾       │ Régimen observado (Diap. 27)           │"
    )
    print(
        "     ├────────────────────────────┼────────────────────┼────────────────────┼────────────────────────────────────────┤"
    )

    trajectories = {}
    for eta_val, eta_label, desc in etas:
        w1, w2 = -8.0, 4.0
        pts = [(w1, w2)]
        for _ in range(10):
            w1 = (1.0 - eta_val * a1) * w1
            w2 = (1.0 - eta_val * a2) * w2
            pts.append((w1, w2))
        trajectories[eta_label] = pts
        w1_1, w2_1 = pts[1]
        w1_4, w2_4 = pts[4]
        print(
            f"     │ {eta_label:<26} │ ({w1_1:6.2f}, {w2_1:6.2f})   │ ({w1_4:6.2f}, {w2_4:6.2f})   │ {desc:<38} │"
        )
    print(
        "     └────────────────────────────┴────────────────────┴────────────────────┴────────────────────────────────────────┘"
    )

    # Verificaciones exactas de los 6 casos de la Diapositiva 27
    # Caso 2: eta = 1/3 -> en 1 paso w2 = 0 exacto
    assert abs(trajectories["1/3 (= η₂,opt)"][1][1]) < 1e-12
    # Caso 4: eta = 2/3 -> w2 alterna exactamente +4, -4, +4, -4
    assert abs(trajectories["2/3 (= 2η₂,opt)"][1][1] - (-4.0)) < 1e-12
    assert abs(trajectories["2/3 (= 2η₂,opt)"][2][1] - (+4.0)) < 1e-12
    # Caso 6: eta = 1.0 -> en 1 paso w1 = 0 exacto, pero w2 = -8
    assert abs(trajectories["1.00 (= η₁,opt)"][1][0]) < 1e-12
    assert abs(trajectories["1.00 (= η₁,opt)"][1][1] - (-8.0)) < 1e-12

    print(
        f"     {green('✓ Verificado:')} Las 6 trayectorias reproducen exactamente los "
        "regímenes de la Diapositiva 27."
    )

    return {
        "kappa": kappa,
        "eta_star": eta_star,
        "rho_star": rho_star,
        "trajectories": trajectories,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 3: MÉTODO DE NEWTON, HESSIANO Y TRAMPA DEL PUNTO DE SILLA
# ─────────────────────────────────────────────────────────────────────────────
def demo_newton_y_hessiano() -> dict:
    """
    Verifica las Diapositivas 34 a 44:
      1. Sobre una cuadrática anisotrópica rotada E(w) = 0.5*w^T A w + b^T w + c,
         el método de Newton w^(k+1) = w^(k) - H^-1 grad E converge al mínimo
         exacto en UN SOLO PASO con eta = 1, sin importar el número de condición.
      2. En el punto de silla no convexo E(w1, w2) = 0.5*(w1^2 - w2^2) desde
         w^(0) = (2, 2), el Descenso por Gradiente baja alejándose en w2, mientras
         que Newton es atraído directamente hacia el punto de silla (0, 0)
         (aumentando la pérdida en la dirección de curvatura negativa, Diap. 42),
         y Levenberg-Marquardt (H + lambda*I)^-1 restaura el descenso.
    """
    banner(
        "SECCIÓN 3: MÉTODO DE NEWTON, HESSIANO Y PUNTO DE SILLA (DIAPS. 34–44)",
        "Normalización de curvatura H⁻¹∇E, convergencia en 1 paso y trampa de silla",
    )

    # 1. Cuadrática 2D rotada con alto número de condición (kappa = 10)
    # A = [[5.5, 4.5], [4.5, 5.5]] -> autovalores lambda1 = 10, lambda2 = 1
    a11, a12, a22 = 5.5, 4.5, 5.5
    b1, b2 = -10.0, -2.0  # mínimo exacto en A*w* = -b => w* = (2, -0.222...)

    def eval_quad(w1: float, w2: float) -> float:
        return (
            0.5 * (a11 * w1 * w1 + 2.0 * a12 * w1 * w2 + a22 * w2 * w2)
            + b1 * w1
            + b2 * w2
        )

    def grad_quad(w1: float, w2: float) -> tuple[float, float]:
        return (a11 * w1 + a12 * w2 + b1, a12 * w1 + a22 * w2 + b2)

    det_a = a11 * a22 - a12 * a12  # 30.25 - 20.25 = 10.0
    inv_a11, inv_a12, inv_a22 = a22 / det_a, -a12 / det_a, a11 / det_a

    # Paso único de Newton desde w^(0) = (-6, 5)
    w1_0, w2_0 = -6.0, 5.0
    g1_0, g2_0 = grad_quad(w1_0, w2_0)
    w1_newton = w1_0 - (inv_a11 * g1_0 + inv_a12 * g2_0)
    w2_newton = w2_0 - (inv_a12 * g1_0 + inv_a22 * g2_0)
    gn1, gn2 = grad_quad(w1_newton, w2_newton)
    grad_norm_after_1_newton = math.hypot(gn1, gn2)

    assert grad_norm_after_1_newton < 1e-12

    print(
        f"\n  {bold('1. Cuadrática rotada A = [[5.5, 4.5], [4.5, 5.5]] (λ₁=10, λ₂=1, κ=10):')}"
    )
    print(
        f"     Inicio w⁽⁰⁾ = ({w1_0:.1f}, {w2_0:.1f})  |  "
        f"‖∇E(w⁽⁰⁾)‖ = {math.hypot(g1_0, g2_0):.4f}"
    )
    print(
        f"     Tras 1 solo paso de Newton (η = 1): "
        f"w⁽¹⁾ = ({green(f'{w1_newton:.4f}')}, {green(f'{w2_newton:.4f}')})  |  "
        f"‖∇E(w⁽¹⁾)‖ = {green(f'{grad_norm_after_1_newton:.2e}')}"
    )

    # 2. Problema de la curvatura negativa en un punto de silla (Diaps. 18 y 42)
    # E(w1, w2) = 0.5*(w1^2 - w2^2), grad E = (w1, -w2), H = diag(1, -1)
    s_w1_0, s_w2_0 = 2.0, 1.5
    e_s0 = 0.5 * (s_w1_0**2 - s_w2_0**2)  # 0.5*(4 - 2.25) = +0.875

    # GD con eta = 0.4: w1 -> 2 - 0.4(2) = 1.2, w2 -> 1.5 - 0.4(-1.5) = 2.1
    eta_s = 0.4
    gd_w1 = s_w1_0 - eta_s * s_w1_0
    gd_w2 = s_w2_0 - eta_s * (-s_w2_0)
    e_gd = 0.5 * (gd_w1**2 - gd_w2**2)  # 0.5*(1.44 - 4.41) = -1.485 (baja!)

    # Newton puro (eta = 1): w^(1) = w^(0) - H^-1 grad E = (0, 0) (va directo a la silla!)
    nw_w1 = s_w1_0 - (1.0 / 1.0) * s_w1_0
    nw_w2 = s_w2_0 - (1.0 / (-1.0)) * (-s_w2_0)

    # Levenberg-Marquardt con lambda = 2.0 => H + lambda*I = diag(3, +1) definida positiva
    lam_lm = 2.0
    lm_w1 = s_w1_0 - (1.0 / (1.0 + lam_lm)) * s_w1_0
    lm_w2 = s_w2_0 - (1.0 / (-1.0 + lam_lm)) * (-s_w2_0)
    e_lm = 0.5 * (lm_w1**2 - lm_w2**2)

    assert abs(nw_w1) < 1e-12 and abs(nw_w2) < 1e-12
    assert e_gd < e_s0 and e_lm < e_s0

    print(
        f"\n  {bold('2. Trampa del punto de silla E(w₁, w₂) = ½(w₁² - w₂²) desde (2.0, 1.5) (Diap. 42):')}"
    )
    print(
        f"     • Pérdida inicial E(2.0, 1.5) = {e_s0:+.4f}  (silla en (0, 0) con E = 0)"
    )
    print(
        f"     • {green('Descenso por Gradiente (η=0.4):')} salta a ({gd_w1:.2f}, {gd_w2:.2f}) "
        f"⇒ E = {green(f'{e_gd:+.4f}')} (escapa bajando por w₂)"
    )
    print(
        f"     • {red('Newton puro (H⁻¹∇E):')}          salta a ({nw_w1:.2f}, {nw_w2:.2f}) "
        f"⇒ {red('¡Atrapado en el punto de silla (0, 0)!')}"
    )
    print(
        f"     • {cyan('Levenberg-Marquardt (λ=2.0):')}   salta a ({lm_w1:.2f}, {lm_w2:.2f}) "
        f"⇒ E = {cyan(f'{e_lm:+.4f}')} (corrige el signo negativo)"
    )
    print(
        f"     {green('✓ Verificado:')} Newton converge en 1 paso en cuencos cuadráticos "
        "pero es atraído por puntos de silla cuando H tiene autovalores negativos."
    )

    return {
        "w_newton": (w1_newton, w2_newton),
        "grad_norm_after_1_newton": grad_norm_after_1_newton,
        "e_gd": e_gd,
        "e_lm": e_lm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 4: TASAS DECRECIENTES, RPROP Y QUICKPROP (DIAPS. 45 A 58)
# ─────────────────────────────────────────────────────────────────────────────
def demo_tasas_rprop_quickprop() -> dict:
    """
    Verifica las Diapositivas 45 a 58:
      1. Escape de un mínimo local mediocre usando una tasa inicial grande
         con decaimiento frente a una tasa pequeña fija (Diaps. 45–46).
      2. Comparación entre Descenso por Gradiente estándar, RProp (Diaps. 48–53)
         y QuickProp (Diaps. 54–56) sobre una superficie con alto número de
         condición E(w1, w2) = 0.5*(w1^2 + 25*w2^2) desde (-10, 4).
    """
    banner(
        "SECCIÓN 4: TASAS DECRECIENTES, RPROP Y QUICKPROP (DIAPS. 45–58)",
        "Escape de mínimos locales y pasos adaptativos por coordenada",
    )

    # 1. Superficie 1D de doble valle: E(w) = 0.1*w^4 - 0.15*w^3 - 0.8*w^2 + 1.2*w + 2.0
    # Tiene un mínimo local mediocre en w ≈ +2.368 (E ≈ +1.50), una barrera en
    # w ≈ +0.625 (E ≈ +2.42) y el mínimo global profundo en w ≈ -1.868 (E ≈ -0.84).
    def e_double_well(w: float) -> float:
        return 0.1 * (w**4) - 0.15 * (w**3) - 0.8 * (w**2) + 1.2 * w + 2.0

    def de_double_well(w: float) -> float:
        return 0.4 * (w**3) - 0.45 * (w**2) - 1.6 * w + 1.2

    # Desde w0 = +2.80 (pared derecha del valle mediocre):
    # Con eta pequeño fijo (0.15) queda atrapado en el mínimo local mediocre w ≈ +2.368
    w_small = 2.80
    for _ in range(80):
        w_small -= 0.15 * de_double_well(w_small)
    e_small = e_double_well(w_small)

    # Con eta0 grande (1.25) y decaimiento exponencial salta la barrera hacia el mínimo global w ≈ -1.868
    w_decay = 2.80
    eta0_decay = 1.25
    beta_decay = 0.08
    for k in range(1, 81):
        eta_k = eta0_decay * math.exp(-beta_decay * (k - 1))
        w_decay -= eta_k * de_double_well(w_decay)
    e_decay = e_double_well(w_decay)

    assert e_decay < e_small - 2.0

    print(
        f"\n  {bold('1. Escape de mínimo local mediocre con tasa inicial grande + decaimiento (Diaps. 45–46):')}"
    )
    print(
        f"     • Con η = 0.15 fijo desde w⁽⁰⁾ = +2.80:     "
        f"termina en w = {w_small:+.3f}, E(w) = {yellow(f'{e_small:+.4f}')} (mínimo local mediocre)"
    )
    print(
        f"     • Con η_k = 1.25·e^(-0.08(k-1)) decreciente: "
        f"termina en w = {w_decay:+.3f}, E(w) = {green(f'{e_decay:+.4f}')} (mínimo global)"
    )

    # 2. Comparación GD vs. RProp vs. QuickProp en cañón anisotrópico kappa = 25
    # E(w1, w2) = 0.5*(w1^2 + 25*w2^2) desde w^(0) = (-10, 4)
    a1, a2 = 1.0, 25.0
    w0 = (-10.0, 4.0)
    n_steps = 25

    # (a) GD con eta estable (eta = 0.06 < 2/25 = 0.08)
    gd_w1, gd_w2 = w0
    for _ in range(n_steps):
        gd_w1 -= 0.06 * (a1 * gd_w1)
        gd_w2 -= 0.06 * (a2 * gd_w2)
    e_gd_final = 0.5 * (a1 * gd_w1**2 + a2 * gd_w2**2)

    # (b) RProp exacto según pseudocódigo de la Diapositiva 51
    rp_w = [w0[0], w0[1]]
    rp_delta = [0.5, 0.5]
    rp_dprev = [a1 * rp_w[0], a2 * rp_w[1]]
    alpha_rp, beta_rp = 1.2, 0.5
    delta_min, delta_max = 1e-6, 50.0
    # Primer paso inicial en dirección del signo
    for i in range(2):
        s = 1.0 if rp_dprev[i] > 0 else (-1.0 if rp_dprev[i] < 0 else 0.0)
        rp_w[i] -= s * rp_delta[i]

    for _ in range(n_steps - 1):
        grads = [a1 * rp_w[0], a2 * rp_w[1]]
        for i in range(2):
            d = grads[i]
            prod = d * rp_dprev[i]
            if prod > 0:
                rp_delta[i] = min(alpha_rp * rp_delta[i], delta_max)
                rp_dprev[i] = d
                s = 1.0 if rp_dprev[i] > 0 else -1.0
                rp_w[i] -= s * rp_delta[i]
            elif prod < 0:
                # Cambió de signo: deshacer el último paso y encoger Delta (Diap. 51)
                s_prev = 1.0 if rp_dprev[i] > 0 else -1.0
                rp_w[i] += s_prev * rp_delta[i]
                rp_delta[i] = max(beta_rp * rp_delta[i], delta_min)
                rp_w[i] -= s_prev * rp_delta[i]
                rp_dprev[i] = d
            else:
                rp_dprev[i] = d
    e_rprop_final = 0.5 * (a1 * rp_w[0] ** 2 + a2 * rp_w[1] ** 2)

    # (c) QuickProp (Diaps. 54–55):
    # Paso 0 con descenso inicial pequeño para tener w^(0) y w^(1), luego regla
    # Delta w^(k) = Delta w^(k-1) * E'(w^(k)) / (E'(w^(k-1)) - E'(w^(k)))
    qp_w_prev = [w0[0], w0[1]]
    qp_g_prev = [a1 * qp_w_prev[0], a2 * qp_w_prev[1]]
    # Pequeño paso semilla eta_seed = 0.02
    qp_w = [qp_w_prev[0] - 0.02 * qp_g_prev[0], qp_w_prev[1] - 0.02 * qp_g_prev[1]]
    for _ in range(n_steps - 1):
        qp_g = [a1 * qp_w[0], a2 * qp_w[1]]
        for i in range(2):
            dw_prev = qp_w[i] - qp_w_prev[i]
            denom = qp_g_prev[i] - qp_g[i]
            if abs(denom) > 1e-14:
                dw = dw_prev * (qp_g[i] / denom)
            else:
                dw = 0.0
            qp_w_prev[i] = qp_w[i]
            qp_g_prev[i] = qp_g[i]
            qp_w[i] += dw
    e_quickprop_final = 0.5 * (a1 * qp_w[0] ** 2 + a2 * qp_w[1] ** 2)

    assert e_rprop_final < 0.05 and e_rprop_final < e_gd_final / 50.0
    assert e_quickprop_final < 1e-20

    print(
        f"\n  {bold('2. Comparación en valle mal condicionado E = ½(w₁² + 25w₂²), κ = 25, tras 25 pasos:')}"
    )
    print(
        f"     • {yellow('Descenso por Gradiente (η = 0.06):')} "
        f"w = ({gd_w1:+.4f}, {gd_w2:+.4f})  ⇒  E = {yellow(f'{e_gd_final:.6f}')}"
    )
    print(
        f"     • {cyan('RProp (α = 1.2, β = 0.5, Diap. 51):')} "
        f"w = ({rp_w[0]:+.4f}, {rp_w[1]:+.4f})  ⇒  E = {cyan(f'{e_rprop_final:.6f}')}"
    )
    print(
        f"     • {green('QuickProp (diferencia finita, Diap. 55):')} "
        f"w = ({qp_w[0]:+.4f}, {qp_w[1]:+.4f})  ⇒  E = {green(f'{e_quickprop_final:.2e}')}"
    )
    print(
        f"     {green('✓ Verificado:')} RProp desacopla la escala por coordenada usando "
        "solo el signo, y QuickProp salta al vértice parabólico en 1 paso tras la semilla."
    )

    return {
        "e_small": e_small,
        "e_decay": e_decay,
        "e_gd_final": e_gd_final,
        "e_rprop_final": e_rprop_final,
        "e_quickprop_final": e_quickprop_final,
    }


def run_all_convergence_demos() -> None:
    """Ejecuta secuencialmente todas las demostraciones de la Unidad V."""
    demo_brady_saboteadores()
    demo_superficie_y_numero_condicion()
    demo_newton_y_hessiano()
    demo_tasas_rprop_quickprop()


def interactive_menu() -> None:
    while True:
        banner(
            "UNIDAD V: PROBLEMAS DE CONVERGENCIA EN EL ENTRENAMIENTO (58 DIAPS.)",
            "Selector interactivo de laboratorios matemáticos",
        )
        print("  [1] Experimento de Brady et al. (1989) y puntos «saboteadores» (Diaps. 7–16)")
        print("  [2] Las 6 trayectorias desde (-8, 4) y Número de Condición κ (Diaps. 17–33)")
        print("  [3] Método de Newton, Hessiano y Trampa del Punto de Silla (Diaps. 34–44)")
        print("  [4] Tasas decrecientes, RProp y QuickProp (Diaps. 45–58)")
        print("  [5] Ejecutar TODAS las demostraciones de la Unidad V")
        print("  [0] Salir / Volver")
        choice = input("\n  Seleccione una opción [0-5]: ").strip()
        if choice == "1":
            demo_brady_saboteadores()
        elif choice == "2":
            demo_superficie_y_numero_condicion()
        elif choice == "3":
            demo_newton_y_hessiano()
        elif choice == "4":
            demo_tasas_rprop_quickprop()
        elif choice == "5":
            run_all_convergence_demos()
        elif choice == "0":
            break


if __name__ == "__main__":
    if "--auto" in sys.argv:
        run_all_convergence_demos()
    else:
        interactive_menu()
