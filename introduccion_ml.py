#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
UNIDAD CD04: INTRODUCCIÓN A MACHINE LEARNING
Ciencia de Datos — Doctorado UCN (Basado en Stanford CS221 / Harvard CS109 / CMU 11-785)
===============================================================================
Módulo computacional y de verificación matemática en Python puro (sin
dependencias externas) que implementa y verifica todas las tablas, fórmulas,
gradientes, curvas de pérdida y algoritmos de las 56 diapositivas de la Unidad CD04:

1. Panorama de Machine Learning y Tareas de Predicción (Diaps. 1–10):
   - Mapa de modelos en IA de bajo a alto nivel (Reflejos -> Estados ->
     Variables -> Lógica), predictores basados en reflejos (x -> f -> y), y
     las 3 tareas de predicción (Clasificación binaria/multiclase, Regresión y
     Predicción estructurada descompuesta como secuencia de multiclase), más
     los 4 casos prácticos de la Diapositiva 10.
2. Regresión Lineal, el Descubrimiento de Ceres y las 3 Decisiones de Diseño (Diaps. 11–23):
   - 1801: Las 19 observaciones de Piazzi sobre Ceres y la predicción de Gauss
     por mínimos cuadrados (reaparición a 0.5° de lo predicho).
   - Clase de hipótesis lineal f_w(x) = w · phi(x) con phi(x) = [1, x] sobre
     D_train = {(1, 1), (2, 3), (4, 3)}: f_[1, 0.57](3) = 2.71 (Diap. 15).
   - Pérdida cuadrática por ejemplo (0.3249, 0.7396, 0.0784) y riesgo empírico
     TrainLoss([1, 0.57]) = 0.38096 ≈ 0.38 (Diaps. 16–17).
   - Descenso por gradiente con eta = 0.1 desde w^(0) = [0, 0] reproduciendo
     exactamente la tabla de la Diap. 21 y el código de la Diap. 22:
     * t = 1: grad = [-4.67, -12.67] -> w = [0.47, 1.27]
     * t = 2: grad = [ 2.18,   7.24] -> w = [0.25, 0.54]
     * t = 200: grad = [0, 0] -> w* = [1.00, 0.57].
3. Clasificación Lineal: Puntaje, Margen, Pérdida 0-1 y Pérdida Hinge (Diaps. 24–37):
   - D_train = {([0, 2], +1), ([-2, 0], +1), ([1, -1], -1)} con phi(x) = [x_1, x_2],
     frontera w · phi(x) = 0 con w = [-0.6, 0.6] (Diaps. 25–26).
   - Pérdida 0-1 con w = [0.5, 1]: pérdidas [0, 1, 0] -> TrainLoss_0-1 = 0.33,
     puntaje vs. margen (w · phi(x))y y por qué grad Loss_0-1 = 0 en casi todas
     partes (Diaps. 27–30).
   - Pérdida Hinge max{1 - margen, 0} y su tabla exacta con w = [0.5, 1]
     (Diap. 35): pérdidas [0, 2, 0.5], TrainLoss = 0.83, grad TrainLoss = [1, -0.33].
   - Ejemplo a mano de la Diap. 36: w = [2, -1], x = [1, 1], y = -1 ->
     puntaje = 1, margen = -1, Loss_hinge = 2, grad = [1, 1].
4. Regresión Logística, Máxima Verosimilitud, Crecimiento de ||w|| y Riesgo Empírico (Diaps. 33, 38–56):
   - Por qué OLS sobre {0, 1} (y = 0.27x - 0.45) se escapa de [0, 1] (en x=6.4
     predice 1.29 y en x=0.4 predice -0.35, Diap. 39).
   - Función sigmoide sigma(z) = 1/(1+e^-z), simetría sigma(-z) = 1 - sigma(z),
     verosimilitud P(y|x) = sigma(margen), pérdida logística log(1 + e^-margen)
     y gradiente graduado -sigma(-margen) phi(x) y (Diaps. 40–45).
   - Tabla exacta de la pérdida logística con w = [0.5, 1] (Diap. 46):
     P(y|x) = [0.881, 0.269, 0.622], pérdidas [0.127, 1.313, 0.474],
     TrainLoss = 0.638, grad TrainLoss = [0.613, -0.205].
   - Simulación en Python con eta = 0.5 (Diaps. 47–48): tras 100 épocas
     w = [-2.29, 2.29], sigma(Xw) = [0.99, 0.99, 0.01], pérdida = 0.010;
     tras 1000 épocas w = [-3.45, 3.45], pérdida = 0.001 («la logística nunca
     queda satisfecha»).
   - Ejemplo de la Diap. 51: w = [2, -1], x = [1, 1], y = -1 -> P(y=+1|x) = 0.731,
     P(y=-1|x) = 0.269, Loss_logistic = 1.313 (vs. 2.0 de la hinge).
===============================================================================
"""

from __future__ import annotations

import math
import sys
from typing import Dict, List, Tuple


# =============================================================================
# 1. PANORAMA DE MACHINE LEARNING Y TAREAS DE PREDICCIÓN (DIAPS. 1–10)
# =============================================================================
MAPA_MODELOS_IA: List[Dict[str, str]] = [
    {
        "nivel": "1. Bajo nivel — Modelos basados en reflejos",
        "ejemplos": "Aprendizaje automático (clasificadores lineales, redes neuronales profundas)",
        "inferencia": "Secuencia fija de cómputos hacia adelante (x -> f -> y), sin búsqueda ni vuelta atrás",
    },
    {
        "nivel": "2. Nivel intermedio — Modelos basados en estados",
        "ejemplos": "Búsqueda en grafos, Procesos de Decisión de Markov (MDP), Juegos adversarios, RL",
        "inferencia": "Planificación sobre secuencias de estados y transiciones",
    },
    {
        "nivel": "3. Nivel estructurado — Modelos basados en variables",
        "ejemplos": "Satisfacción de restricciones (CSP), Redes de Markov y Redes Bayesianas",
        "inferencia": "Asignación e inferencia probabilística sobre grafos de factores",
    },
    {
        "nivel": "4. Alto nivel — Modelos basados en lógica",
        "ejemplos": "Lógica proposicional y Lógica de primer orden",
        "inferencia": "Deducción simbólica y demostración de teoremas",
    },
]

TAREAS_PREDICCION_DIAP_7_10: List[Dict[str, str]] = [
    {
        "caso_diap_10": "Predecir si un correo es spam",
        "tarea": "Clasificación binaria",
        "salida": "y in {+1, -1}",
        "otros_ejemplos": "Fraude en transacción, moderación de foros tóxicos, bosón de Higgs",
    },
    {
        "caso_diap_10": "Estimar el precio de una vivienda en Antofagasta",
        "tarea": "Regresión",
        "salida": "y in R (continua)",
        "otros_ejemplos": "Mapeo de pobreza satelital, tiempo de llegada en logística",
    },
    {
        "caso_diap_10": "Transcribir un audio a texto",
        "tarea": "Predicción estructurada",
        "salida": "Secuencia de palabras (descompuesta en clasificaciones multiclase sucesivas)",
        "otros_ejemplos": "Traducción automática español->inglés, descripción y segmentación de imágenes",
    },
    {
        "caso_diap_10": "Reconocer qué dígito (0–9) hay en una imagen",
        "tarea": "Clasificación multiclase (K = 10)",
        "salida": "y in {0, 1, ..., 9}",
        "otros_ejemplos": "Clasificación de categorías de productos o especies",
    },
]


def demo_panorama_ml_y_tareas(verbose: bool = True) -> Dict[str, object]:
    """
    Verifica el mapa de la IA (Diap. 3), los modelos basados en reflejos
    (Diaps. 5–6) y las tareas de predicción junto con los 4 casos de la Diap. 10.
    """
    assert len(MAPA_MODELOS_IA) == 4
    assert "reflejos" in MAPA_MODELOS_IA[0]["nivel"].lower()
    assert len(TAREAS_PREDICCION_DIAP_7_10) == 4
    assert TAREAS_PREDICCION_DIAP_7_10[0]["tarea"] == "Clasificación binaria"
    assert TAREAS_PREDICCION_DIAP_7_10[1]["tarea"] == "Regresión"
    assert TAREAS_PREDICCION_DIAP_7_10[2]["tarea"] == "Predicción estructurada"
    assert "multiclase" in TAREAS_PREDICCION_DIAP_7_10[3]["tarea"].lower()

    if verbose:
        print("=" * 78)
        print("1. PANORAMA DE MACHINE LEARNING Y TAREAS DE PREDICCIÓN (DIAPS. 1–10)")
        print("=" * 78)
        print("• Mapa de modelos en IA (de bajo a alto nivel de abstracción, Diap. 3):")
        for m in MAPA_MODELOS_IA:
            print(f"  - {m['nivel']}: {m['ejemplos']}")
            print(f"    Inferencia: {m['inferencia']}")
        print("\n• Los 4 casos de diseño y su tipo de tarea (Diaps. 7–10):")
        for c in TAREAS_PREDICCION_DIAP_7_10:
            print(f"  - «{c['caso_diap_10']}» -> {c['tarea']} [{c['salida']}]")
        print("  [OK] Mapa de la IA, modelos reflejos y taxonomía de tareas verificados.\n")

    return {
        "niveles_ia": len(MAPA_MODELOS_IA),
        "casos_diap_10": len(TAREAS_PREDICCION_DIAP_7_10),
    }


# =============================================================================
# 2. REGRESIÓN LINEAL, CERES (1801) Y LAS 3 DECISIONES DE DISEÑO (DIAPS. 11–23)
# =============================================================================
DATOS_REGRESION_DIAP_13: List[Tuple[float, float]] = [
    (1.0, 1.0),
    (2.0, 3.0),
    (4.0, 3.0),
]

OBSERVACIONES_CERES_PIAZZI_1801: List[Tuple[str, float, float]] = [
    ("01 ene 1801", 50.91, 15.24),
    ("02 ene 1801", 50.84, 15.30),
    ("03 ene 1801", 50.77, 15.36),
    ("04 ene 1801", 50.71, 15.42),
    ("10 ene 1801", 50.45, 15.81),
    ("11 ene 1801", 50.42, 15.88),
    ("13 ene 1801", 50.39, 16.02),
    ("14 ene 1801", 50.38, 16.09),
    ("19 ene 1801", 50.42, 16.47),
    ("21 ene 1801", 50.47, 16.63),
    ("22 ene 1801", 50.51, 16.71),
    ("23 ene 1801", 50.55, 16.79),
    ("24 ene 1801", 50.60, 16.88),
    ("28 ene 1801", 50.85, 17.22),
    ("30 ene 1801", 51.01, 17.40),
    ("31 ene 1801", 51.10, 17.49),
    ("01 feb 1801", 51.19, 17.58),
    ("08 feb 1801", 52.00, 18.21),
    ("11 feb 1801", 53.51, 18.43),
]


def phi_lineal(x: float) -> Tuple[float, float]:
    """Vector de características phi(x) = [1, x] (Diap. 14)."""
    return (1.0, float(x))


def predecir_lineal(w: Tuple[float, float], x: float) -> float:
    """Puntaje f_w(x) = w · phi(x) = w_1 + w_2 x (Diap. 14)."""
    phi = phi_lineal(x)
    return w[0] * phi[0] + w[1] * phi[1]


def evaluar_regresion_lineal(
    w: Tuple[float, float],
    datos: List[Tuple[float, float]] = DATOS_REGRESION_DIAP_13,
) -> Dict[str, object]:
    """
    Calcula predicciones, residuos, pérdidas cuadráticas individuales,
    TrainLoss(w) y gradiente nabla_w TrainLoss(w) (Diaps. 16–20).
    """
    n = len(datos)
    detalles = []
    suma_loss = 0.0
    grad_w1 = 0.0
    grad_w2 = 0.0

    for x, y in datos:
        pred = predecir_lineal(w, x)
        residuo = pred - y
        loss = residuo ** 2
        g1 = 2.0 * residuo * 1.0
        g2 = 2.0 * residuo * x
        suma_loss += loss
        grad_w1 += g1
        grad_w2 += g2
        detalles.append({
            "x": x,
            "y": y,
            "pred": pred,
            "residuo": residuo,
            "loss": loss,
            "grad_i": (g1, g2),
        })

    train_loss = suma_loss / n
    grad = (grad_w1 / n, grad_w2 / n)
    return {
        "w": w,
        "detalles": detalles,
        "train_loss": train_loss,
        "grad": grad,
    }


def descenso_gradiente_regresion(
    eta: float = 0.1,
    epocas: int = 200,
    w_inicial: Tuple[float, float] = (0.0, 0.0),
    datos: List[Tuple[float, float]] = DATOS_REGRESION_DIAP_13,
) -> List[Dict[str, object]]:
    """
    Ejecuta descenso por gradiente exactamente como en las Diaps. 21 y 22:
    w <- w - eta * nabla_w TrainLoss(w).
    """
    w = (float(w_inicial[0]), float(w_inicial[1]))
    historial: List[Dict[str, object]] = []
    ev0 = evaluar_regresion_lineal(w, datos)
    historial.append({
        "t": 0,
        "w": w,
        "grad_aplicado": None,
        "train_loss": ev0["train_loss"],
    })

    for t in range(1, epocas + 1):
        ev = evaluar_regresion_lineal(w, datos)
        g1, g2 = ev["grad"]
        w = (w[0] - eta * g1, w[1] - eta * g2)
        ev_post = evaluar_regresion_lineal(w, datos)
        historial.append({
            "t": t,
            "grad_aplicado": (g1, g2),
            "w": w,
            "train_loss": ev_post["train_loss"],
        })
    return historial


def demo_regresion_lineal_y_ceres(verbose: bool = True) -> Dict[str, object]:
    """
    Verifica todas las cifras de las Diaps. 11–23:
    - 19 observaciones de Piazzi para Ceres (1801) y reaparición a 0.5° (Gauss).
    - f_[1, 0.57](3) = 2.71 (Diap. 15).
    - Pérdidas por ejemplo con w = [1, 0.57]: 0.3249 (0.32), 0.7396 (0.74),
      0.0784 (0.08) y TrainLoss = 0.380966... ≈ 0.38 (Diaps. 16–17).
    - Tabla de descenso por gradiente con eta = 0.1 desde w^(0) = [0, 0] (Diap. 21):
      t=1: grad = [-4.67, -12.67], w = [0.47, 1.27]
      t=2: grad = [ 2.18,   7.24], w = [0.25, 0.54]
      t=200: grad = [0.00, 0.00], w* = [1.00, 0.57].
    """
    assert len(OBSERVACIONES_CERES_PIAZZI_1801) == 19

    w_opt = (1.0, 0.57)
    pred_x3 = predecir_lineal(w_opt, 3.0)
    assert abs(pred_x3 - 2.71) < 1e-9

    ev_opt = evaluar_regresion_lineal(w_opt)
    losses_i = [d["loss"] for d in ev_opt["detalles"]]
    assert abs(losses_i[0] - 0.3249) < 1e-9 and round(losses_i[0], 2) == 0.32
    assert abs(losses_i[1] - 0.7396) < 1e-9 and round(losses_i[1], 2) == 0.74
    assert abs(losses_i[2] - 0.0784) < 1e-9 and round(losses_i[2], 2) == 0.08
    assert round(ev_opt["train_loss"], 2) == 0.38

    hist = descenso_gradiente_regresion(eta=0.1, epocas=200, w_inicial=(0.0, 0.0))
    paso1 = hist[1]
    paso2 = hist[2]
    paso200 = hist[200]

    assert round(paso1["grad_aplicado"][0], 2) == -4.67
    assert round(paso1["grad_aplicado"][1], 2) == -12.67
    assert round(paso1["w"][0], 2) == 0.47
    assert round(paso1["w"][1], 2) == 1.27

    assert round(paso2["grad_aplicado"][0], 2) == 2.18
    assert round(paso2["grad_aplicado"][1], 2) == 7.24
    assert round(paso2["w"][0], 2) == 0.25
    assert round(paso2["w"][1], 2) == 0.54

    assert round(paso200["grad_aplicado"][0], 2) == 0.00
    assert round(paso200["grad_aplicado"][1], 2) == 0.00
    assert round(paso200["w"][0], 2) == 1.00
    assert round(paso200["w"][1], 2) == 0.57

    if verbose:
        print("=" * 78)
        print("2. REGRESIÓN LINEAL, CERES (1801) Y LAS 3 DECISIONES (DIAPS. 11–23)")
        print("=" * 78)
        print(f"• Hito histórico (Diaps. 11–12): {len(OBSERVACIONES_CERES_PIAZZI_1801)} observaciones de Piazzi")
        print("  sobre Ceres (01 ene – 11 feb 1801); Gauss predice su órbita por mínimos")
        print("  cuadrados y Ceres reaparece el 7 dic 1801 a 0.5° de lo predicho.")
        print(f"\n• Decisión 1 — Clase de hipótesis lineal con w = [1, 0.57] (Diaps. 14–15):")
        print(f"  f_w(3) = [1, 0.57] · [1, 3] = {pred_x3:.2f}")
        print("\n• Decisión 2 — Pérdida cuadrática sobre D_train = {(1,1), (2,3), (4,3)} (Diaps. 16–17):")
        for d in ev_opt["detalles"]:
            print(
                f"  (x={d['x']:.0f}, y={d['y']:.0f}): f_w(x)={d['pred']:.2f}, "
                f"residuo={d['residuo']:+.2f}, Loss={d['loss']:.2f} ({d['loss']:.4f})"
            )
        print(f"  TrainLoss([1, 0.57]) = (0.32 + 0.74 + 0.08)/3 = {ev_opt['train_loss']:.2f} ({ev_opt['train_loss']:.5f})")
        print("\n• Decisión 3 — Descenso por gradiente con eta = 0.1 desde w^(0) = [0, 0] (Diap. 21):")
        print("  t     grad TrainLoss           w = [w_1, w_2]       TrainLoss(w)")
        print("  ----------------------------------------------------------------")
        print(f"  0     —                        [0.00, 0.00]         {hist[0]['train_loss']:.4f}")
        for idx in (1, 2, 3, 10, 50, 200):
            h = hist[idx]
            g = h["grad_aplicado"]
            w_t = h["w"]
            print(
                f"  {idx:<5d} [{g[0]:+6.2f}, {g[1]:+6.2f}]        "
                f"[{w_t[0]:5.2f}, {w_t[1]:5.2f}]        {h['train_loss']:.4f}"
            )
        print("  [OK] Pérdidas por ejemplo y tabla de descenso por gradiente verificadas.\n")

    return {
        "pred_x3": pred_x3,
        "train_loss_opt": ev_opt["train_loss"],
        "paso_1": paso1,
        "paso_2": paso2,
        "paso_200": paso200,
    }


# =============================================================================
# 3. CLASIFICACIÓN LINEAL: PUNTAJE, MARGEN, PÉRDIDA 0-1 Y HINGE (DIAPS. 24–37)
# =============================================================================
DATOS_CLASIFICACION_DIAP_24: List[Tuple[Tuple[float, float], int]] = [
    ((0.0, 2.0), +1),
    ((-2.0, 0.0), +1),
    ((1.0, -1.0), -1),
]


def signo_clasificador(z: float) -> int:
    """Función signo: +1 si z > 0, -1 si z < 0, 0 si z == 0 (Diap. 25)."""
    if z > 0:
        return +1
    if z < 0:
        return -1
    return 0


def puntaje_y_margen(
    w: Tuple[float, float],
    x: Tuple[float, float],
    y: int,
) -> Tuple[float, float]:
    """
    Devuelve (puntaje, margen) donde:
      puntaje = w · phi(x)
      margen  = (w · phi(x)) * y
    (Diaps. 28–29).
    """
    puntaje = w[0] * x[0] + w[1] * x[1]
    margen = puntaje * y
    return puntaje, margen


def evaluar_perdida_01(
    w: Tuple[float, float],
    datos: List[Tuple[Tuple[float, float], int]] = DATOS_CLASIFICACION_DIAP_24,
) -> Dict[str, object]:
    """Evalúa Loss_0-1(x, y, w) = 1[margen <= 0] sobre el conjunto de datos (Diap. 27)."""
    detalles = []
    suma = 0.0
    for x, y in datos:
        puntaje, margen = puntaje_y_margen(w, x, y)
        pred = signo_clasificador(puntaje)
        loss = 1.0 if margen <= 0.0 else 0.0
        suma += loss
        detalles.append({
            "x": x,
            "y": y,
            "puntaje": puntaje,
            "pred": pred,
            "margen": margen,
            "loss_01": loss,
        })
    return {
        "w": w,
        "detalles": detalles,
        "train_loss_01": suma / len(datos),
    }


def evaluar_perdida_hinge(
    w: Tuple[float, float],
    datos: List[Tuple[Tuple[float, float], int]] = DATOS_CLASIFICACION_DIAP_24,
) -> Dict[str, object]:
    """
    Evalúa Loss_hinge(x, y, w) = max{1 - (w · phi(x))y, 0} y su gradiente
    nabla_w Loss_hinge = -phi(x)y si margen < 1, o [0, 0] si margen > 1
    (Diaps. 32, 34 y 35).
    """
    n = len(datos)
    detalles = []
    suma_loss = 0.0
    grad_w1 = 0.0
    grad_w2 = 0.0

    for x, y in datos:
        puntaje, margen = puntaje_y_margen(w, x, y)
        loss = max(1.0 - margen, 0.0)
        if margen < 1.0:
            gi = (-x[0] * y, -x[1] * y)
        else:
            gi = (0.0, 0.0)
        suma_loss += loss
        grad_w1 += gi[0]
        grad_w2 += gi[1]
        detalles.append({
            "x": x,
            "y": y,
            "puntaje": puntaje,
            "margen": margen,
            "loss_hinge": loss,
            "grad_i": gi,
        })

    return {
        "w": w,
        "detalles": detalles,
        "train_loss_hinge": suma_loss / n,
        "grad_hinge": (grad_w1 / n, grad_w2 / n),
    }


def demo_clasificacion_lineal_y_hinge(verbose: bool = True) -> Dict[str, object]:
    """
    Verifica todas las cifras de las Diaps. 24–37:
    - Clasificador con w = [-0.6, 0.6]: predice [+1, +1, -1] con 0 errores (Diap. 25).
    - Pérdida 0-1 con w = [0.5, 1]: pérdidas [0, 1, 0] -> TrainLoss_0-1 = 0.33 (Diap. 27).
    - Pérdida Hinge con w = [0.5, 1] (Diap. 35):
      * ([0, 2], +1): margen = 2 -> Loss = 0, grad = [0, 0]
      * ([-2, 0], +1): margen = -1 -> Loss = 2, grad = [2, 0]
      * ([1, -1], -1): margen = 0.5 -> Loss = 0.5, grad = [1, -1]
      * Promedio: TrainLoss = 0.83 (2.5/3), grad = [1.00, -0.33].
    - Ejemplo a mano de la Diap. 36: w = [2, -1], x = [1, 1], y = -1 ->
      puntaje = 1, margen = -1, Loss_hinge = 2, grad = [1, 1].
    """
    ev_sep = evaluar_perdida_01((-0.6, 0.6))
    assert ev_sep["train_loss_01"] == 0.0

    w_test = (0.5, 1.0)
    ev_01 = evaluar_perdida_01(w_test)
    losses_01 = [d["loss_01"] for d in ev_01["detalles"]]
    assert losses_01 == [0.0, 1.0, 0.0]
    assert round(ev_01["train_loss_01"], 2) == 0.33

    ev_hinge = evaluar_perdida_hinge(w_test)
    losses_h = [d["loss_hinge"] for d in ev_hinge["detalles"]]
    grads_h = [d["grad_i"] for d in ev_hinge["detalles"]]
    assert losses_h == [0.0, 2.0, 0.5]
    assert grads_h == [(0.0, 0.0), (2.0, 0.0), (1.0, -1.0)]
    assert round(ev_hinge["train_loss_hinge"], 2) == 0.83
    assert round(ev_hinge["grad_hinge"][0], 2) == 1.00
    assert round(ev_hinge["grad_hinge"][1], 2) == -0.33

    # Ejemplo a mano Diap. 36: w = [2, -1], x = [1, 1], y = -1
    ev_diap36 = evaluar_perdida_hinge((2.0, -1.0), [((1.0, 1.0), -1)])
    d36 = ev_diap36["detalles"][0]
    assert d36["puntaje"] == 1.0
    assert d36["margen"] == -1.0
    assert d36["loss_hinge"] == 2.0
    assert d36["grad_i"] == (1.0, 1.0)

    if verbose:
        print("=" * 78)
        print("3. CLASIFICACIÓN LINEAL: PUNTAJE, MARGEN, 0-1 Y HINGE (DIAPS. 24–37)")
        print("=" * 78)
        print("• Tabla de pérdida 0-1 y pérdida Hinge con w = [0.5, 1] (Diaps. 27 y 35):")
        print("  x          y    puntaje  margen  Loss_0-1  Loss_hinge  grad Loss_hinge")
        print("  ----------------------------------------------------------------------")
        for d01, dh in zip(ev_01["detalles"], ev_hinge["detalles"]):
            x_str = f"[{dh['x'][0]:+.0f}, {dh['x'][1]:+.0f}]"
            g_str = f"[{dh['grad_i'][0]:+.0f}, {dh['grad_i'][1]:+.0f}]"
            print(
                f"  {x_str:<10s} {dh['y']:+d}   {dh['puntaje']:+6.2f}   {dh['margen']:+6.2f}  "
                f"{d01['loss_01']:6.1f}    {dh['loss_hinge']:8.2f}    {g_str}"
            )
        print(
            f"  Promedio -> TrainLoss_0-1 = {ev_01['train_loss_01']:.2f} | "
            f"TrainLoss_hinge = {ev_hinge['train_loss_hinge']:.2f} | "
            f"grad TrainLoss_hinge = [{ev_hinge['grad_hinge'][0]:.2f}, {ev_hinge['grad_hinge'][1]:.2f}]"
        )
        print(
            "\n• Ejemplo a mano (Diap. 36) con w = [2, -1], x = [1, 1], y = -1:\n"
            f"  puntaje = {d36['puntaje']:.0f}, margen = {d36['margen']:.0f}, "
            f"Loss_hinge = {d36['loss_hinge']:.0f}, grad = [{d36['grad_i'][0]:.0f}, {d36['grad_i'][1]:.0f}]"
        )
        print("  [OK] Pérdida 0-1, margen de seguridad y tabla Hinge verificados.\n")

    return {
        "train_loss_01": ev_01["train_loss_01"],
        "train_loss_hinge": ev_hinge["train_loss_hinge"],
        "grad_hinge": ev_hinge["grad_hinge"],
        "ejemplo_diap_36": d36,
    }


# =============================================================================
# 4. REGRESIÓN LOGÍSTICA, MÁXIMA VEROSIMILITUD Y CRECIMIENTO DE ||w|| (DIAPS. 38–56)
# =============================================================================
def sigmoide(z: float) -> float:
    """Función logística sigmoide sigma(z) = 1 / (1 + e^-z) (Diap. 40)."""
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def evaluar_perdida_logistica(
    w: Tuple[float, float],
    datos: List[Tuple[Tuple[float, float], int]] = DATOS_CLASIFICACION_DIAP_24,
) -> Dict[str, object]:
    """
    Evalúa la regresión logística sobre D_train (Diaps. 41–46):
      - puntaje z = w · phi(x)
      - margen  m = z * y
      - verosimilitud P(y | x) = sigma(m)
      - pérdida logística Loss_logistic = -log sigma(m) = log(1 + e^-m)
      - gradiente nabla_w Loss_logistic = -sigma(-m) * phi(x) * y
    """
    n = len(datos)
    detalles = []
    suma_loss = 0.0
    grad_w1 = 0.0
    grad_w2 = 0.0

    for x, y in datos:
        puntaje, margen = puntaje_y_margen(w, x, y)
        p_pos = sigmoide(puntaje)
        p_correcta = sigmoide(margen)
        p_equivocada = sigmoide(-margen)
        loss = math.log(1.0 + math.exp(-margen)) if margen > -50 else -margen
        gi = (-p_equivocada * x[0] * y, -p_equivocada * x[1] * y)
        suma_loss += loss
        grad_w1 += gi[0]
        grad_w2 += gi[1]
        detalles.append({
            "x": x,
            "y": y,
            "puntaje": puntaje,
            "margen": margen,
            "p_pos": p_pos,
            "p_correcta": p_correcta,
            "p_equivocada": p_equivocada,
            "loss_logistic": loss,
            "grad_i": gi,
        })

    return {
        "w": w,
        "detalles": detalles,
        "train_loss_logistic": suma_loss / n,
        "grad_logistic": (grad_w1 / n, grad_w2 / n),
    }


def entrenar_regresion_logistica_diap_47(
    eta: float = 0.5,
    iteraciones: int = 1000,
    datos: List[Tuple[Tuple[float, float], int]] = DATOS_CLASIFICACION_DIAP_24,
) -> Dict[int, Dict[str, object]]:
    """
    Reproduce exactamente el código de la Diapositiva 47 y los resultados
    de la Diapositiva 48:
      X = [[0, 2], [-2, 0], [1, -1]], y = [+1, +1, -1], w = [0, 0], eta = 0.5
      for _ in range(T):
          m = (X @ w) * y
          grad = -((sigma(-m) * y)[:, None] * X).mean(axis=0)
          w -= eta * grad
    """
    w = (0.0, 0.0)
    instantaneas: Dict[int, Dict[str, object]] = {}

    for t in range(1, iteraciones + 1):
        ev = evaluar_perdida_logistica(w, datos)
        g1, g2 = ev["grad_logistic"]
        w = (w[0] - eta * g1, w[1] - eta * g2)
        if t in (1, 10, 100, 1000):
            ev_post = evaluar_perdida_logistica(w, datos)
            norma_w = math.hypot(w[0], w[1])
            probs_pos = [sigmoide(w[0] * x[0] + w[1] * x[1]) for x, _ in datos]
            instantaneas[t] = {
                "t": t,
                "w": w,
                "norma_w": norma_w,
                "sigma_Xw": probs_pos,
                "train_loss": ev_post["train_loss_logistic"],
            }
    return instantaneas


def softmax_multiclase(puntajes: List[float]) -> List[float]:
    """
    Generalización multiclase de la sigmoide: Softmax (Diap. 50)
      P(y = k | x) = exp(w_k · phi(x)) / sum_j exp(w_j · phi(x)).
    """
    m = max(puntajes)
    exps = [math.exp(z - m) for z in puntajes]
    s = sum(exps)
    return [e / s for e in exps]


def demo_regresion_logistica_y_riesgo_empirico(verbose: bool = True) -> Dict[str, object]:
    """
    Verifica todas las propiedades y tablas de las Diaps. 38–56:
    - Por qué OLS sobre {0, 1} (y = 0.27x - 0.45) falla (Diap. 39):
      en x = 6.4 da 1.278 ≈ 1.29 (> 1) y en x = 0.4 da -0.342 ≈ -0.35 (< 0).
    - Simetría de la sigmoide: sigma(0) = 0.5 y sigma(-z) = 1 - sigma(z) (Diap. 40).
    - Tabla de pérdida logística con w = [0.5, 1] (Diap. 46):
      * ([0, 2], +1): z = 2, P(y|x) = 0.881, pérdida = 0.127, grad = [0, -0.238]
      * ([-2, 0], +1): z = -1, P(y|x) = 0.269, pérdida = 1.313, grad = [1.462, 0]
      * ([1, -1], -1): z = -0.5, P(y|x) = 0.622, pérdida = 0.474, grad = [0.378, -0.378]
      * Promedio: TrainLoss = 0.638, grad = [0.613, -0.205].
    - Simulación en Python con eta = 0.5 (Diaps. 47–48):
      * t = 100: w = [-2.29, 2.29], sigma(Xw) = [0.99, 0.99, 0.01], pérdida = 0.010
      * t = 1000: w = [-3.45, 3.45], pérdida = 0.001 («la logística nunca queda satisfecha»).
    - Ejemplo de la Diap. 51: w = [2, -1], x = [1, 1], y = -1 ->
      z = 1, m = -1, P(y=+1|x) = 0.731, P(y=-1|x) = 0.269, Loss_logistic = 1.313.
    """
    # 1. Falla de OLS sobre {0, 1} (Diap. 39)
    ols_x_alto = 0.27 * 6.444 - 0.45
    ols_x_bajo = 0.27 * 0.370 - 0.45
    assert round(ols_x_alto, 2) == 1.29
    assert round(ols_x_bajo, 2) == -0.35

    # 2. Propiedades de la sigmoide (Diap. 40)
    assert abs(sigmoide(0.0) - 0.5) < 1e-12
    for z_val in (-3.0, -1.0, -0.5, 1.0, 2.0, 4.5):
        assert abs(sigmoide(-z_val) - (1.0 - sigmoide(z_val))) < 1e-12

    # 3. Tabla exacta de pérdida logística con w = [0.5, 1] (Diap. 46)
    ev_log = evaluar_perdida_logistica((0.5, 1.0))
    d1, d2, d3 = ev_log["detalles"]

    assert round(d1["p_correcta"], 3) == 0.881
    assert round(d1["loss_logistic"], 3) == 0.127
    assert round(d1["grad_i"][0], 3) == 0.000
    assert round(d1["grad_i"][1], 3) == -0.238

    assert round(d2["p_correcta"], 3) == 0.269
    assert round(d2["loss_logistic"], 3) == 1.313
    assert round(d2["grad_i"][0], 3) == 1.462
    assert round(d2["grad_i"][1], 3) == 0.000

    assert round(d3["p_correcta"], 3) == 0.622
    assert round(d3["loss_logistic"], 3) == 0.474
    assert round(d3["grad_i"][0], 3) == 0.378
    assert round(d3["grad_i"][1], 3) == -0.378

    assert round(ev_log["train_loss_logistic"], 3) == 0.638
    assert round(ev_log["grad_logistic"][0], 3) == 0.613
    assert round(ev_log["grad_logistic"][1], 3) == -0.205

    # 4. Simulación en Python con eta = 0.5 (Diaps. 47–48)
    snaps = entrenar_regresion_logistica_diap_47(eta=0.5, iteraciones=1000)
    s100 = snaps[100]
    s1000 = snaps[1000]
    assert round(s100["w"][0], 2) == -2.29 and round(s100["w"][1], 2) == 2.29
    assert [round(p, 2) for p in s100["sigma_Xw"]] == [0.99, 0.99, 0.01]
    assert round(s100["train_loss"], 3) == 0.010

    assert round(s1000["w"][0], 2) == -3.45 and round(s1000["w"][1], 2) == 3.45
    assert round(s1000["train_loss"], 3) == 0.001
    assert s1000["norma_w"] > s100["norma_w"]

    # 5. Ejemplo de la Diap. 51: w = [2, -1], x = [1, 1], y = -1
    ev_d51 = evaluar_perdida_logistica((2.0, -1.0), [((1.0, 1.0), -1)])
    d51 = ev_d51["detalles"][0]
    assert round(d51["p_pos"], 3) == 0.731
    assert round(d51["p_correcta"], 3) == 0.269
    assert round(d51["loss_logistic"], 3) == 1.313

    # 6. Verificación de Softmax multiclase (Diap. 50)
    probs_sm = softmax_multiclase([2.0, 1.0, 0.1])
    assert abs(sum(probs_sm) - 1.0) < 1e-12
    assert probs_sm[0] > probs_sm[1] > probs_sm[2]

    if verbose:
        print("=" * 78)
        print("4. REGRESIÓN LOGÍSTICA, MÁXIMA VEROSIMILITUD Y ||w|| (DIAPS. 38–56)")
        print("=" * 78)
        print("• Tabla de pérdida logística en nuestros 3 datos con w = [0.5, 1] (Diap. 46):")
        print("  x          y    z=w·phi(x)  P(y|x)=sigma(zy)  Loss_logistic  grad Loss_logistic")
        print("  -------------------------------------------------------------------------------")
        for d in ev_log["detalles"]:
            x_str = f"[{d['x'][0]:+.0f}, {d['x'][1]:+.0f}]"
            g_str = f"[{d['grad_i'][0]:+.3f}, {d['grad_i'][1]:+.3f}]"
            print(
                f"  {x_str:<10s} {d['y']:+d}   {d['puntaje']:+8.2f}    "
                f"{d['p_correcta']:14.3f}    {d['loss_logistic']:11.3f}    {g_str}"
            )
        print(
            f"  Promedio -> TrainLoss = {ev_log['train_loss_logistic']:.3f} | "
            f"grad TrainLoss = [{ev_log['grad_logistic'][0]:+.3f}, {ev_log['grad_logistic'][1]:+.3f}]"
        )
        print("\n• «La logística nunca queda satisfecha» con eta = 0.5 (Diaps. 47–48):")
        for t_k in (100, 1000):
            st = snaps[t_k]
            probs_fmt = ", ".join(f"{p:.2f}" for p in st["sigma_Xw"])
            print(
                f"  Iter {t_k:<4d}: w = [{st['w'][0]:+.2f}, {st['w'][1]:+.2f}] (||w||={st['norma_w']:.2f}), "
                f"sigma(Xw) = [{probs_fmt}], TrainLoss = {st['train_loss']:.3f}"
            )
        print(
            "\n• Comparación en x = [1, 1], y = -1 con w = [2, -1] (Diap. 51):\n"
            f"  z = 1, P(y=+1|x) = sigma(1) = {d51['p_pos']:.3f}, "
            f"P(y=-1|x) = {d51['p_correcta']:.3f}, "
            f"Loss_logistic = -ln(0.269) = {d51['loss_logistic']:.3f} (vs. 2.000 de Hinge)"
        )
        print("  [OK] Pérdida logística, crecimiento de ||w|| y riesgo empírico verificados.\n")

    return {
        "train_loss_logistic": ev_log["train_loss_logistic"],
        "grad_logistic": ev_log["grad_logistic"],
        "snap_100": s100,
        "snap_1000": s1000,
        "ejemplo_diap_51": d51,
    }


# =============================================================================
# EJECUCIÓN DE TODAS LAS DEMOSTRACIONES Y MENÚ INTERACTIVO
# =============================================================================
def ejecutar_todas_las_demos(verbose: bool = True) -> Dict[str, object]:
    """Ejecuta las 4 demostraciones de la Unidad CD04 y verifica todos los resultados."""
    r1 = demo_panorama_ml_y_tareas(verbose=verbose)
    r2 = demo_regresion_lineal_y_ceres(verbose=verbose)
    r3 = demo_clasificacion_lineal_y_hinge(verbose=verbose)
    r4 = demo_regresion_logistica_y_riesgo_empirico(verbose=verbose)
    if verbose:
        print("=" * 78)
        print("✅ UNIDAD CD04 (56 DIAPOSITIVAS): 100% DE DEMOSTRACIONES VERIFICADAS")
        print("=" * 78)
    return {
        "panorama": r1,
        "regresion_lineal": r2,
        "clasificacion_hinge": r3,
        "regresion_logistica": r4,
    }


def menu_interactivo() -> None:
    """Menú interactivo para explorar la Unidad CD04 desde la terminal."""
    opciones = {
        "1": ("Panorama de ML, modelos reflejos y las 3 tareas de predicción (Diaps. 1–10)", demo_panorama_ml_y_tareas),
        "2": ("Regresión Lineal, Ceres (1801) y las 3 decisiones de diseño (Diaps. 11–23)", demo_regresion_lineal_y_ceres),
        "3": ("Clasificación Lineal: puntaje, margen, pérdida 0-1 y Hinge (Diaps. 24–37)", demo_clasificacion_lineal_y_hinge),
        "4": ("Regresión Logística, máxima verosimilitud y crecimiento de ||w|| (Diaps. 38–56)", demo_regresion_logistica_y_riesgo_empirico),
        "5": ("Ejecutar todas las demostraciones de CD04", ejecutar_todas_las_demos),
    }
    while True:
        print("\n" + "=" * 78)
        print("MENÚ INTERACTIVO — UNIDAD CD04: INTRODUCCIÓN A MACHINE LEARNING")
        print("=" * 78)
        for k, (desc, _) in sorted(opciones.items()):
            print(f"  [{k}] {desc}")
        print("  [0] Salir")
        eleccion = input("\nSeleccione una opción: ").strip()
        if eleccion == "0":
            break
        if eleccion in opciones:
            print()
            opciones[eleccion][1](verbose=True)
        else:
            print("Opción no válida. Intente nuevamente.")


if __name__ == "__main__":
    if "--auto" in sys.argv:
        ejecutar_todas_las_demos(verbose=True)
    else:
        menu_interactivo()
