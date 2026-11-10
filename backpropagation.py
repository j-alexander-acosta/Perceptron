#!/usr/bin/env python3
"""
================================================================================
ALGORITMO DE BACKPROPAGATION EN REDES NEURONALES MULTICAPA
Ciencia de Datos — Basado en material de CMU 11-785 (B. Raj et al.)
y Stanford CS221 (P. Liang et al.)
================================================================================
Implementación computacional en Python puro de las 4 secciones de la cátedra:
  1. El problema y las dos reglas de la cadena (Simple vs. Distribuida, Diaps. 5-13)
  2. Grafos de computación, bloques básicos y ejemplos paso a paso (Diaps. 14-23)
  3. Forward pass y Backward pass en una red por capas: ejemplo ReLU 2 capas y max (Diaps. 24-41)
  4. Ciclo de entrenamiento completo y vista vectorial con Jacobianos (Diaps. 42-60)
================================================================================
"""

import math
import sys

AUTO_MODE = "--auto" in sys.argv


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def pause(msg="Presiona [ENTER] para continuar..."):
    if not AUTO_MODE:
        try:
            input(f"\n{Colors.DIM}{msg}{Colors.RESET}")
        except EOFError:
            pass


def print_banner(title, subtitle=""):
    print("\n" + f"{Colors.CYAN}{'=' * 84}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.HEADER}  {title}{Colors.RESET}")
    if subtitle:
        print(f"{Colors.DIM}  {subtitle}{Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 84}{Colors.RESET}")


# ==============================================================================
# SECCIÓN 1: EL PROBLEMA Y LAS DOS REGLAS DE LA CADENA (Diapositivas 5-13)
# ==============================================================================
def demo_reglas_cadena():
    """
    Verifica numéricamente:
      1. Regla de la cadena simple (composición): dl/dz = (dl/dy) * (dy/dz)
         Ejemplo (Diap. 18): b = a^2, c = b^2 en a = 2 => dc/da = (2b)*(2a) = 8*4 = 32 = 4*a^3
      2. Regla de la cadena distribuida (múltiples caminos, Diaps. 11-12):
         dl/dx = sum_i (dl/dz_i) * (dz_i/dx)
    """
    print_banner(
        "SECCIÓN 1: EL PROBLEMA Y LAS DOS REGLAS DE LA CADENA (Diaps. 5-13)",
        "Cadena Simple (multiplicar por el camino) vs. Cadena Distribuida (sumar sobre caminos)",
    )

    # 1. Cadena simple (Diaps. 10, 12 y 18): a -> b = a^2 -> c = b^2
    a = 2.0
    b = a ** 2          # b = 4
    c = b ** 2          # c = 16
    db_da = 2.0 * a     # 4
    dc_db = 2.0 * b     # 8
    dc_da_chain = dc_db * db_da
    dc_da_analytic = 4.0 * (a ** 3)
    eps = 1e-6
    dc_da_fd = (((a + eps) ** 2) ** 2 - ((a - eps) ** 2) ** 2) / (2 * eps)

    print(f"\n  {Colors.BOLD}1. Regla de la Cadena Simple — Composición (Diaps. 10, 12 y 18):{Colors.RESET}")
    print(f"     Grafo en serie: a -> b = a^2 -> c = b^2  evaluado en a = {a:.1f}")
    print(f"     Forward pass:   b = {b:.1f},  c = {c:.1f}")
    print(f"     Derivadas locales en aristas:  dc/db = 2b = {dc_db:.1f},   db/da = 2a = {db_da:.1f}")
    print(f"     Backward pass (multiplicar por el camino): dc/da = (dc/db)*(db/da) = {dc_db:.1f} * {db_da:.1f} = {Colors.GREEN}{dc_da_chain:.1f}{Colors.RESET}")
    print(f"     Verificación analítica (4a^3 = {dc_da_analytic:.1f}) y diferencias finitas ({dc_da_fd:.6f})")

    assert abs(dc_da_chain - 32.0) < 1e-9

    # 2. Cadena distribuida (Diaps. 11-12): x influye en l a través de M = 3 caminos z_1, z_2, z_3
    # Sea z1 = 2x, z2 = x^2, z3 = 3x - 1, y l = z1^2 + 2*z2 + z3 evaluado en x = 2
    x = 2.0
    z1 = 2.0 * x
    z2 = x ** 2
    z3 = 3.0 * x - 1.0
    l_val = (z1 ** 2) + 2.0 * z2 + z3

    # Derivadas locales de cada camino
    dl_dz1, dz1_dx = 2.0 * z1, 2.0
    dl_dz2, dz2_dx = 2.0, 2.0 * x
    dl_dz3, dz3_dx = 1.0, 3.0

    path1 = dl_dz1 * dz1_dx  # 8 * 2 = 16
    path2 = dl_dz2 * dz2_dx  # 2 * 4 = 8
    path3 = dl_dz3 * dz3_dx  # 1 * 3 = 3
    dl_dx_dist = path1 + path2 + path3

    def eval_l(xv):
        return (2.0 * xv) ** 2 + 2.0 * (xv ** 2) + (3.0 * xv - 1.0)

    dl_dx_fd = (eval_l(x + eps) - eval_l(x - eps)) / (2 * eps)

    print(f"\n  {Colors.BOLD}2. Regla de la Cadena Distribuida — Varios Caminos (Diaps. 11-12):{Colors.RESET}")
    print(f"     x alimenta M = 3 nodos intermedios: z1 = 2x, z2 = x^2, z3 = 3x - 1")
    print(f"     Función objetivo: l(z1, z2, z3) = z1^2 + 2*z2 + z3  en x = {x:.1f}  (l = {l_val:.1f})")
    print(f"       Camino 1 (x -> z1 -> l): (dl/dz1)*(dz1/dx) = {dl_dz1:.1f} * {dz1_dx:.1f} = {path1:.1f}")
    print(f"       Camino 2 (x -> z2 -> l): (dl/dz2)*(dz2/dx) = {dl_dz2:.1f} * {dz2_dx:.1f} = {path2:.1f}")
    print(f"       Camino 3 (x -> z3 -> l): (dl/dz3)*(dz3/dx) = {dl_dz3:.1f} * {dz3_dx:.1f} = {path3:.1f}")
    print(f"     Suma sobre los caminos: dl/dx = {path1:.1f} + {path2:.1f} + {path3:.1f} = {Colors.GREEN}{dl_dx_dist:.1f}{Colors.RESET} (Dif. finitas: {dl_dx_fd:.6f})")

    assert abs(dl_dx_dist - dl_dx_fd) < 1e-6
    pause()
    return {"dc_da_chain": dc_da_chain, "dl_dx_dist": dl_dx_dist}


# ==============================================================================
# SECCIÓN 2: GRAFOS DE COMPUTACIÓN Y EJEMPLOS NUMÉRICOS (Diapositivas 14-23)
# ==============================================================================
def forward_backward_squared_loss_graph(w, phi_x, y):
    """
    Evalúa el grafo de computación de Loss(x, y, w) = (w . phi(x) - y)^2
    (Diapositivas 15, 21, 22 y 23):
      - puntaje = w . phi(x)
      - residuo = puntaje - y
      - loss = residuo^2
    Retorna diccionario con valores hacia adelante (f_i) y gradientes hacia atrás (g_i).
    """
    puntaje = sum(wi * xi for wi, xi in zip(w, phi_x))
    residuo = puntaje - y
    loss = residuo ** 2

    # Backward pass: g_i = dLoss / df_i
    g_loss = 1.0
    dloss_dresiduo = 2.0 * residuo
    g_residuo = g_loss * dloss_dresiduo

    dresiduo_dpuntaje = 1.0
    g_puntaje = g_residuo * dresiduo_dpuntaje

    dpuntaje_dw = list(phi_x)
    g_w = [g_puntaje * xi for xi in dpuntaje_dw]

    return {
        "w": list(w),
        "phi_x": list(phi_x),
        "y": y,
        "puntaje": puntaje,
        "residuo": residuo,
        "loss": loss,
        "g_loss": g_loss,
        "g_residuo": g_residuo,
        "g_puntaje": g_puntaje,
        "g_w": g_w,
    }


def forward_backward_hinge_graph(w, phi_x, y):
    """
    Evalúa el grafo de la Pérdida Hinge (Diapositiva 19):
      Loss(x, y, w) = max{1 - (w . phi(x)) * y, 0}
      grad_w Loss = -1[margen < 1] * phi(x) * y
    """
    puntaje = sum(wi * xi for wi, xi in zip(w, phi_x))
    margen = puntaje * y
    holgura = 1.0 - margen
    loss = max(holgura, 0.0)

    activo = 1.0 if margen < 1.0 else 0.0
    g_w = [-activo * xi * y for xi in phi_x]
    return {
        "puntaje": puntaje,
        "margen": margen,
        "holgura": holgura,
        "loss": loss,
        "grad_w": g_w,
    }


def demo_grafos_computacion():
    """
    Verifica los ejemplos exactos de las Diapositivas 16-23:
      - Ejemplo 1 (Diap. 21): w = [3, 1], phi(x) = [1, 2], y = 2
        => puntaje = 5, residuo = 3, pérdida = 9, grad_w = [6, 12]
      - Ejemplo 2 (Diap. 23): w = [1, 2], phi(x) = [2, 1], y = 3
        => puntaje = 4, residuo = 1, pérdida = 1, grad_w = [4, 2]
      - Pérdida Hinge (Diap. 19)
    """
    print_banner(
        "SECCIÓN 2: GRAFOS DE COMPUTACIÓN Y BACKPROPAGATION PASO A PASO (Diaps. 14-23)",
        "Propagación de valores hacia adelante (f_i) y gradientes hacia atrás (g_i)",
    )

    ej1 = forward_backward_squared_loss_graph([3.0, 1.0], [1.0, 2.0], 2.0)
    print(f"\n  {Colors.BOLD}Ejemplo 1 (Diapositiva 21): Loss(x, y, w) = (w · phi(x) - y)^2{Colors.RESET}")
    print(f"    Entradas: w = {ej1['w']},  phi(x) = {ej1['phi_x']},  y = {ej1['y']}")
    print(f"    [Azul — Forward f_i]:   puntaje = {ej1['puntaje']:.0f}  ->  residuo = {ej1['residuo']:.0f}  ->  pérdida = {Colors.CYAN}{ej1['loss']:.0f}{Colors.RESET}")
    print(f"    [Rojo — Backward g_i]:  g_raíz = {ej1['g_loss']:.0f}  ->  g_residuo = 1*6 = {ej1['g_residuo']:.0f}  ->  g_puntaje = 6*1 = {ej1['g_puntaje']:.0f}")
    print(f"                            grad_w Loss = 6 * [1, 2] = {Colors.GREEN}{ej1['g_w']}{Colors.RESET}")

    assert ej1["puntaje"] == 5.0 and ej1["residuo"] == 3.0 and ej1["loss"] == 9.0
    assert ej1["g_w"] == [6.0, 12.0]

    ej2 = forward_backward_squared_loss_graph([1.0, 2.0], [2.0, 1.0], 3.0)
    print(f"\n  {Colors.BOLD}Ejemplo 2 (Diapositiva 23): Tu Turno — Propagación Completa{Colors.RESET}")
    print(f"    Entradas: w = {ej2['w']},  phi(x) = {ej2['phi_x']},  y = {ej2['y']}")
    print(f"    [Azul — Forward f_i]:   puntaje = {ej2['puntaje']:.0f}  ->  residuo = {ej2['residuo']:.0f}  ->  pérdida = {Colors.CYAN}{ej2['loss']:.0f}{Colors.RESET}")
    print(f"    [Rojo — Backward g_i]:  g_raíz = {ej2['g_loss']:.0f}  ->  g_residuo = 1*2 = {ej2['g_residuo']:.0f}  ->  g_puntaje = 2*1 = {ej2['g_puntaje']:.0f}")
    print(f"                            grad_w Loss = 2 * [2, 1] = {Colors.GREEN}{ej2['g_w']}{Colors.RESET}")

    assert ej2["puntaje"] == 4.0 and ej2["residuo"] == 1.0 and ej2["loss"] == 1.0
    assert ej2["g_w"] == [4.0, 2.0]

    hinge = forward_backward_hinge_graph([0.2, 0.1], [1.0, 2.0], 1.0)
    print(f"\n  {Colors.BOLD}Pérdida Hinge como Grafo (Diapositiva 19):{Colors.RESET}")
    print(f"    w = [0.2, 0.1], phi(x) = [1, 2], y = +1  =>  puntaje = {hinge['puntaje']:.2f}, margen = {hinge['margen']:.2f} < 1")
    print(f"    Loss = max(1 - margen, 0) = {hinge['loss']:.2f},  grad_w Loss = -1[margen < 1]*phi(x)*y = {Colors.YELLOW}{hinge['grad_w']}{Colors.RESET}")

    pause()
    return {"ej1": ej1, "ej2": ej2, "hinge": hinge}


# ==============================================================================
# SECCIÓN 3: FORWARD Y BACKWARD PASS EN RED RELU DE 2 CAPAS (Diapositivas 24-41)
# ==============================================================================
def relu_2layer_network_pass(x, d, W1=None, b1=None, W2=None, b2=None):
    """
    Ejecuta el Forward Pass y Backward Pass exacto del ejemplo numérico de las
    Diapositivas 38, 39 y 57:
      - Capa 1 (2 neuronas ReLU):
          W1 = [[1, 1], [1, -1]],  b1 = [-1, 0]
          z1 = W1 @ x + b1,        y1 = ReLU(z1)
      - Capa 2 (1 neurona lineal):
          W2 = [[2, 1]],           b2 = [1]
          z2 = W2 @ y1 + b2,       Y = z2[0]
      - Divergencia:
          div = (Y - d)^2
    """
    if W1 is None:
        W1 = [[1.0, 1.0], [1.0, -1.0]]
    if b1 is None:
        b1 = [-1.0, 0.0]
    if W2 is None:
        W2 = [[2.0, 1.0]]
    if b2 is None:
        b2 = [1.0]

    # Forward pass
    z1 = [
        W1[0][0] * x[0] + W1[0][1] * x[1] + b1[0],
        W1[1][0] * x[0] + W1[1][1] * x[1] + b1[1],
    ]
    y1 = [max(0.0, z1[0]), max(0.0, z1[1])]
    z2 = [W2[0][0] * y1[0] + W2[0][1] * y1[1] + b2[0]]
    Y = z2[0]
    div = (Y - d) ** 2

    # Backward pass
    gY = 2.0 * (Y - d)
    gz2 = [gY]  # neurona de salida lineal: f2'(z2) = 1

    # Gradientes de capa 2: grad_W2 = gz2 @ y1^T, grad_b2 = gz2
    gW2 = [[gz2[0] * y1[0], gz2[0] * y1[1]]]
    gb2 = [gz2[0]]

    # Bajada a la capa oculta 1: gy1 = W2^T @ gz2
    gy1 = [W2[0][0] * gz2[0], W2[0][1] * gz2[0]]

    # Cruzar la activación ReLU: gz1 = gy1 * 1[z1 > 0]
    relu_mask = [1.0 if z1[0] > 0.0 else 0.0, 1.0 if z1[1] > 0.0 else 0.0]
    gz1 = [gy1[0] * relu_mask[0], gy1[1] * relu_mask[1]]

    # Gradientes de capa 1: grad_W1 = gz1 @ x^T, grad_b1 = gz1
    gW1 = [
        [gz1[0] * x[0], gz1[0] * x[1]],
        [gz1[1] * x[0], gz1[1] * x[1]],
    ]
    gb1 = [gz1[0], gz1[1]]

    return {
        "x": list(x),
        "d": d,
        "z1": z1,
        "y1": y1,
        "z2": z2,
        "Y": Y,
        "div": div,
        "gY": gY,
        "gz2": gz2,
        "gW2": gW2,
        "gb2": gb2,
        "gy1": gy1,
        "relu_mask": relu_mask,
        "gz1": gz1,
        "gW1": gW1,
        "gb1": gb1,
    }


def demo_red_relu_2capas():
    """
    Verifica paso a paso el ejemplo numérico completo de las Diapositivas 38, 39 y 57,
    la verificación por perturbación de la Diapositiva 39 y el enrutador max(z1, z2, z3)
    de la Diapositiva 41.
    """
    print_banner(
        "SECCIÓN 3: FORWARD Y BACKWARD PASS EN UNA RED POR CAPAS (Diaps. 24-41)",
        "Ejemplo numérico completo en Red ReLU de 2 capas (Diaps. 38-39 y 57)",
    )

    res = relu_2layer_network_pass([1.0, 2.0], 3.0)

    print(f"\n  {Colors.BOLD}1. Forward Pass (Diapositiva 38 — Valores en Azul):{Colors.RESET}")
    print(f"     Entrada x = {res['x']},  Objetivo d = {res['d']:.0f}")
    print(f"     Capa 1 (ReLU):")
    print(f"       z1^(1) = 1(1) + 1(2) - 1 = {res['z1'][0]:.0f}   =>  y1^(1) = ReLU({res['z1'][0]:.0f}) = {Colors.CYAN}{res['y1'][0]:.0f}{Colors.RESET} (ACTIVA)")
    print(f"       z2^(1) = 1(1) - 1(2) + 0 = {res['z1'][1]:.0f}  =>  y2^(1) = ReLU({res['z1'][1]:.0f}) = {Colors.CYAN}{res['y1'][1]:.0f}{Colors.RESET} (INACTIVA)")
    print(f"     Capa 2 (Lineal):")
    print(f"       z1^(2) = 2(2) + 1(0) + 1 = {res['z2'][0]:.0f}   =>  Y = {Colors.CYAN}{res['Y']:.0f}{Colors.RESET}")
    print(f"     Divergencia:")
    print(f"       div = (Y - d)^2 = (5 - 3)^2 = {Colors.CYAN}{res['div']:.0f}{Colors.RESET}")

    print(f"\n  {Colors.BOLD}2. Backward Pass (Diapositiva 39 — Gradientes en Rojo):{Colors.RESET}")
    print(f"     En la salida:      d(div)/dY = d(div)/dz^(2) = 2(5 - 3) = {Colors.RED}{res['gY']:.0f}{Colors.RESET}")
    print(f"     Gradientes Capa 2: grad_W2 = {Colors.GREEN}{res['gW2']}{Colors.RESET},  grad_b2 = {Colors.GREEN}{res['gb2']}{Colors.RESET}")
    print(f"     Bajada a Capa 1:")
    print(f"       Neurona h1 (activa, z1=2>0):    y_dot_1 = 2*4 = {res['gy1'][0]:.0f}  ->  z_dot_1 = 8 * f'(2) = {Colors.RED}{res['gz1'][0]:.0f}{Colors.RESET}")
    print(f"       Neurona h2 (inactiva, z2=-1<0): y_dot_2 = 1*4 = {res['gy1'][1]:.0f}  ->  z_dot_2 = 4 * f'(-1) = {Colors.RED}{res['gz1'][1]:.0f}{Colors.RESET} (¡corta el gradiente!)")
    print(f"     Gradientes Capa 1: grad_W1 = {Colors.GREEN}{res['gW1']}{Colors.RESET},  grad_b1 = {Colors.GREEN}{res['gb1']}{Colors.RESET}")

    # Verificación exacta contra las diapositivas 38, 39 y 57
    assert res["z1"] == [2.0, -1.0]
    assert res["y1"] == [2.0, 0.0]
    assert res["Y"] == 5.0 and res["div"] == 4.0
    assert res["gW2"] == [[8.0, 0.0]] and res["gb2"] == [4.0]
    assert res["gy1"] == [8.0, 4.0] and res["gz1"] == [8.0, 0.0]
    assert res["gW1"] == [[8.0, 16.0], [0.0, 0.0]] and res["gb1"] == [8.0, 0.0]

    # Verificación por perturbación finita sugerida en la Diapositiva 39:
    # "perturba un peso en 0.001, repite el forward pass y comprueba que Delta_div / 0.001 coincide"
    delta = 1e-5
    res_pert = relu_2layer_network_pass(
        [1.0, 2.0],
        3.0,
        W1=[[1.0, 1.0 + delta], [1.0, -1.0]],
        b1=[-1.0, 0.0],
        W2=[[2.0, 1.0]],
        b2=[1.0],
    )
    fd_w1_01 = (res_pert["div"] - res["div"]) / delta
    print(f"\n  {Colors.BOLD}3. Verificación por Perturbación (Diapositiva 39):{Colors.RESET}")
    print(f"     Perturbando W1[0,1] (peso de x2 -> h1) en delta = {delta}:")
    print(f"     (div_nuevo - div_original) / delta = {fd_w1_01:.4f}  vs.  grad_W1[0,1] = {res['gW1'][0][1]:.4f}")
    assert abs(fd_w1_01 - 16.0) < 1e-3

    # Función max como enrutador del gradiente (Diapositiva 41)
    z_vec = [1.0, 2.0, 3.0]
    y_max = max(z_vec)
    winner_idx = z_vec.index(y_max)
    dy_dz = [1.0 if i == winner_idx else 0.0 for i in range(len(z_vec))]
    print(f"\n  {Colors.BOLD}4. Función max como Enrutador del Gradiente (Diapositiva 41):{Colors.RESET}")
    print(f"     Para y = max(z1, z2, z3) evaluado en (z1, z2, z3) = (1, 2, 3):")
    print(f"     Ganador: z3 => y = {y_max:.0f},  [dy/dz1, dy/dz2, dy/dz3] = {Colors.GREEN}{dy_dz}{Colors.RESET}")
    assert y_max == 3.0 and dy_dz == [0.0, 0.0, 1.0]

    pause()
    return res


# ==============================================================================
# SECCIÓN 4: VISTA VECTORIAL, JACOBIANOS Y CICLO DE ENTRENAMIENTO (Diaps. 42-60)
# ==============================================================================
def softmax_and_jacobian(z):
    """
    Calcula el vector Softmax y = softmax(z) y su matriz Jacobiana llena J_y(z)
    (Diapositiva 51):
      J_{ij} = y_i * (1[i == j] - y_j)
    """
    m = max(z)
    exps = [math.exp(zi - m) for zi in z]
    s = sum(exps)
    y = [e / s for e in exps]
    n = len(z)
    J = [[y[i] * ((1.0 if i == j else 0.0) - y[j]) for j in range(n)] for i in range(n)]
    return y, J


def demo_vista_vectorial_y_entrenamiento():
    """
    Demuestra:
      1. Diferencia entre Jacobiano Diagonal (activación componente a componente, ej. ReLU/Sigmoide)
         y Jacobiano Lleno (activación vectorial Softmax, Diapositiva 51).
      2. Ciclo completo de entrenamiento por Descenso por Gradiente + Backpropagation (Diaps. 42-45)
         sobre la red ReLU de 2 capas reduciendo la pérdida div = (Y - d)^2 de 4.0 hasta ~0.0.
    """
    print_banner(
        "SECCIÓN 4: VISTA VECTORIAL, JACOBIANOS Y CICLO DE ENTRENAMIENTO (Diaps. 42-60)",
        "Jacobiano transpuesto J_y(z)^T y convergencia del entrenamiento con Backpropagation",
    )

    z_demo = [1.0, 2.0, 0.5]
    y_soft, J_soft = softmax_and_jacobian(z_demo)
    print(f"\n  {Colors.BOLD}1. Jacobiano Diagonal vs. Jacobiano Lleno en Softmax (Diapositiva 51):{Colors.RESET}")
    print(f"     Para z = {z_demo} => softmax(z) = [{', '.join(f'{v:.4f}' for v in y_soft)}]")
    print(f"     Matriz Jacobiana llena J_y(z) (cada salida depende de todas las entradas):")
    for row in J_soft:
        print(f"       [{', '.join(f'{val:+.4f}' for val in row)}]")

    # Ciclo de entrenamiento sobre la red de la Diapositiva 38 con eta = 0.02
    W1 = [[1.0, 1.0], [1.0, -1.0]]
    b1 = [-1.0, 0.0]
    W2 = [[2.0, 1.0]]
    b2 = [1.0]
    x = [1.0, 2.0]
    d = 3.0
    eta = 0.02

    print(f"\n  {Colors.BOLD}2. Ciclo de Entrenamiento (Forward + Backward + Paso de Gradiente, Diaps. 42-45):{Colors.RESET}")
    print(f"     Entrenando la red de la Diapositiva 38 con tasa de aprendizaje eta = {eta}:")
    print(f"     {'Iteración':>10} | {'Predicción Y':>14} | {'Objetivo d':>12} | {'Divergencia (Y-d)^2':>20}")
    print(f"     {'-' * 64}")

    history = []
    for step in range(15):
        out = relu_2layer_network_pass(x, d, W1=W1, b1=b1, W2=W2, b2=b2)
        history.append((step, out["Y"], out["div"]))
        if step in (0, 1, 2, 3, 5, 10, 14):
            print(f"     {step:>10d} | {out['Y']:>14.6f} | {d:>12.2f} | {out['div']:>20.8f}")

        # Actualización de parámetros (Descenso por Gradiente usando los gradientes de Backprop)
        for i in range(2):
            b1[i] -= eta * out["gb1"][i]
            for j in range(2):
                W1[i][j] -= eta * out["gW1"][i][j]
        b2[0] -= eta * out["gb2"][0]
        for j in range(2):
            W2[0][j] -= eta * out["gW2"][0][j]

    assert history[0][2] == 4.0
    assert history[-1][2] < 1e-4
    print(f"\n  {Colors.GREEN}✓ La divergencia cayó de {history[0][2]:.4f} a {history[-1][2]:.8f} en 14 pasos de Backprop + GD.{Colors.RESET}")
    pause()
    return {"initial_loss": history[0][2], "final_loss": history[-1][2], "softmax_J": J_soft}


def run_all_backprop_demos():
    """Ejecuta secuencialmente todas las demostraciones de la Unidad IV."""
    r1 = demo_reglas_cadena()
    r2 = demo_grafos_computacion()
    r3 = demo_red_relu_2capas()
    r4 = demo_vista_vectorial_y_entrenamiento()
    return {"s1": r1, "s2": r2, "s3": r3, "s4": r4}


def interactive_backprop_menu():
    """Menú interactivo en consola para la Unidad IV: Algoritmo de Backpropagation."""
    while True:
        print("\n" + "=" * 84)
        print(" LABORATORIO INTERACTIVO — UNIDAD IV: ALGORITMO DE BACKPROPAGATION (60 Diaps.)")
        print("=" * 84)
        print("  [1] Sección 1: Regla de la Cadena Simple vs. Distribuida (Diaps. 5-13 y 18)")
        print("  [2] Sección 2: Grafos de Computación y Ejemplos Paso a Paso (Diaps. 14-23)")
        print("  [3] Sección 3: Red ReLU de 2 Capas y Enrutador max (Diaps. 24-41)")
        print("  [4] Sección 4: Vista Vectorial, Jacobianos y Ciclo de Entrenamiento (Diaps. 42-60)")
        print("  [5] Ejecutar TODAS las demostraciones de la Unidad IV en secuencia")
        print("  [0] Volver al menú principal")
        print("-" * 84)
        try:
            opc = input("Seleccione opción [0-5]: ").strip()
        except EOFError:
            break
        if opc == "1":
            demo_reglas_cadena()
        elif opc == "2":
            demo_grafos_computacion()
        elif opc == "3":
            demo_red_relu_2capas()
        elif opc == "4":
            demo_vista_vectorial_y_entrenamiento()
        elif opc == "5":
            run_all_backprop_demos()
        elif opc in ("0", "q", "salir", "exit"):
            break
        else:
            print("⚠️ Opción no válida.")


if __name__ == "__main__":
    if AUTO_MODE:
        run_all_backprop_demos()
    else:
        interactive_backprop_menu()
