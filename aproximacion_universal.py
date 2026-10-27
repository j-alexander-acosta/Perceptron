#!/usr/bin/env python3
"""
Módulo Educativo Interactivo: Aproximación Universal con Redes Neuronales
Ciencia de Datos — UCN (Basado en material de CMU 11-785, B. Raj et al.)

Este módulo implementa de forma ejecutable y didáctica los 4 ejes temáticos de la clase:
1. El MLP como función booleana universal (Compuertas universales, DNF, Karnaugh, XOR de 2 neuronas y Costo Exponencial vs. Red Profunda).
2. El MLP como clasificador universal (Polígonos convexos, el límite del círculo, el piso N/2, sesgo -N/2 y el tablero de rombos).
3. El MLP como aproximador universal de funciones continuas (Pulsos cuadrados, suma de escalones y error de aproximación epsilon).
4. Profundidad, ancho y activaciones (Franjas vs. celdas, cuello de botella de información y activaciones graduadas: Escalón, Sigmoide, Tanh, ReLU).
"""

import math
import sys
from typing import List, Tuple, Dict, Any, Callable


# =============================================================================
# 1. UNIDADES BÁSICAS Y ACTIVACIONES (Diapositivas 6 y 7)
# =============================================================================

def step_activation(z: float, threshold: float = 0.0) -> int:
    """Activación de umbral (escalón): dispara 1 si z >= threshold, 0 si no."""
    return 1 if z >= threshold - 1e-12 else 0


def sigmoid(z: float) -> float:
    """Activación sigmoide: versión blanda del escalón en (0, 1)."""
    z_clamped = max(-60.0, min(60.0, z))
    return 1.0 / (1.0 + math.exp(-z_clamped))


def tanh_act(z: float) -> float:
    """Activación tangente hiperbólica en (-1, 1)."""
    return math.tanh(z)


def relu(z: float) -> float:
    """Unidad Lineal Rectificada (ReLU): max(0, z)."""
    return max(0.0, z)


class ThresholdGate:
    """
    Compuerta de umbral (Perceptrón):
    Dispara 1 si sum(w_i * x_i) >= threshold, o equivalentemente sum(w_i * x_i) + b >= 0
    donde el sesgo es b = -threshold.
    """
    def __init__(self, weights: List[float], threshold: float, name: str = ""):
        self.weights = weights
        self.threshold = threshold
        self.bias = -threshold
        self.name = name

    def evaluate(self, inputs: List[float]) -> Tuple[int, float]:
        z = sum(w * x for w, x in zip(self.weights, inputs))
        y = step_activation(z, self.threshold)
        return y, z


# =============================================================================
# 2. EL MLP COMO FUNCIÓN BOOLEANA UNIVERSAL (Diapositivas 11 a 29)
# =============================================================================

class UniversalGateN:
    """
    Compuerta Universal de N entradas (Diapositiva 12):
    Asigna peso +1 a las primeras L entradas (X_1 ... X_L) y -1 a las restantes (X_{L+1} ... X_N).
    - AND universal: umbral = L
    - OR universal: umbral = L - N + 1
    - Mayoría generalizada ("al menos K de N"): umbral = L - N + K
    """
    @staticmethod
    def build(n_total: int, positive_indices: List[int], k_required: int) -> ThresholdGate:
        weights = [1.0 if i in positive_indices else -1.0 for i in range(n_total)]
        L = len(positive_indices)
        threshold = L - n_total + k_required
        return ThresholdGate(weights, threshold, name=f"AlMenos_{k_required}_de_{n_total}")


class FriedlandXOR2Neurons:
    """
    XOR con solo 2 neuronas (Construcción de Gerald Friedland, Diapositiva 15):
    - Neurona oculta calcula h = X AND Y (pesos [1, 1], umbral 2).
    - Neurona de salida recibe conexión directa de X, Y y de la oculta h:
      evalúa 1*X + 1*Y - 2*(X AND Y) >= 1.
    Total: 2 neuronas, 5 pesos y 2 umbrales.
    """
    def __init__(self):
        self.hidden_and = ThresholdGate([1.0, 1.0], threshold=2.0, name="h = X ∧ Y")
        # Recibe [X, Y, h]
        self.output_neuron = ThresholdGate([1.0, 1.0, -2.0], threshold=1.0, name="Salida XOR")

    def forward(self, x: int, y: int) -> Dict[str, Any]:
        h, z_h = self.hidden_and.evaluate([x, y])
        out, z_out = self.output_neuron.evaluate([x, y, h])
        return {
            "x": x, "y": y,
            "z_h": z_h, "h_and": h,
            "z_out": z_out, "out": out,
            "expected": x ^ y
        }


class DNFBooleanMLP:
    """
    Constructor automático de MLP de una sola capa oculta para CUALQUIER función booleana
    vía su Forma Normal Disyuntiva (DNF) (Diapositivas 17 y 18).
    - Cada fila de la tabla de verdad con salida 1 genera una neurona AND en la capa oculta:
      peso +1 si X_i = 1, peso -1 si X_i = 0, y umbral = cantidad de unos en esa fila.
    - La capa de salida es una única neurona OR con pesos +1 y umbral = 1.
    """
    def __init__(self, n_vars: int, minterms: List[Tuple[int, ...]]):
        self.n_vars = n_vars
        self.minterms = minterms
        self.hidden_neurons: List[ThresholdGate] = []
        self.dnf_expressions: List[str] = []

        for row in minterms:
            weights = [1.0 if bit == 1 else -1.0 for bit in row]
            num_ones = sum(row)
            term_str = "".join(f"X{i+1}" if bit == 1 else f"X̄{i+1}" for i, bit in enumerate(row))
            self.dnf_expressions.append(term_str)
            self.hidden_neurons.append(
                ThresholdGate(weights, threshold=float(num_ones), name=term_str)
            )

        # Neurona OR de salida (dispara si al menos un término AND dispara)
        self.output_or = ThresholdGate(
            [1.0] * len(self.hidden_neurons),
            threshold=1.0,
            name="OR_Salida"
        )

    def forward(self, inputs: Tuple[int, ...]) -> Tuple[int, List[int]]:
        if not self.hidden_neurons:
            return 0, []
        hidden_activations = [neuron.evaluate(list(inputs))[0] for neuron in self.hidden_neurons]
        out, _ = self.output_or.evaluate(hidden_activations)
        return out, hidden_activations


def compare_parity_architectures(n_list: List[int] = [2, 4, 6, 8, 10, 12, 16, 20]) -> List[Dict[str, Any]]:
    """
    Compara el costo de calcular la función de Paridad (XOR de N variables, "el tablero de ajedrez")
    usando:
    1) Una sola capa oculta (DNF irreducible): 2^(N-1) ocultas + 1 salida = 2^(N-1) + 1 neuronas.
    2) Red profunda en cascada/árbol con bloques XOR de 3 neuronas: 3*(N-1) neuronas en 2*ceil(log2(N)) capas.
    3) Red profunda con bloques XOR de 2 neuronas (Friedland): 2*(N-1) neuronas.
    (Diapositivas 20 a 26)
    """
    rows = []
    for n in n_list:
        shallow_hidden = 2 ** (n - 1)
        shallow_total = shallow_hidden + 1
        shallow_weights = n * shallow_hidden + shallow_hidden

        deep_3_neurons = 3 * (n - 1)
        deep_2_neurons = 2 * (n - 1)
        tree_layers = 2 * math.ceil(math.log2(n)) if n > 1 else 1

        rows.append({
            "N": n,
            "shallow_hidden": shallow_hidden,
            "shallow_total": shallow_total,
            "shallow_weights": shallow_weights,
            "deep_3_total": deep_3_neurons,
            "deep_2_total": deep_2_neurons,
            "tree_layers": tree_layers
        })
    return rows


# =============================================================================
# 3. EL MLP COMO CLASIFICADOR UNIVERSAL EN R^2 (Diapositivas 30 a 48)
# =============================================================================

class PolygonCircleClassifier:
    """
    Construye un clasificador geométrico para un polígono regular de N lados inscrito/circunscrito
    de radio r centrado en (cx, cy) (Diapositivas 33 a 41).
    - Cada neurona oculta i define un semiplano:
      cos(theta_i)*(x - cx) + sin(theta_i)*(y - cy) <= r  <=>  -cos(theta_i)*x - sin(theta_i)*y >= -r - ...
    - Adentro del polígono: las N neuronas disparan => suma = N.
    - Afuera a distancia d > r (cuando N -> infinito):
      la suma tiende a N * (1/2 + (1/pi) * arcsin(r / d)), cayendo hacia el "piso" N/2.
    - Con sesgo -N/2 en la salida: el piso exterior baja a ~0 y el cilindro interior queda en N/2.
    """
    def __init__(self, n_sides: int, radius: float = 1.0, center: Tuple[float, float] = (0.0, 0.0)):
        self.n_sides = n_sides
        self.radius = radius
        self.cx, self.cy = center
        self.hidden_neurons: List[ThresholdGate] = []

        for i in range(n_sides):
            angle = 2.0 * math.pi * i / n_sides
            nx = -math.cos(angle)
            ny = -math.sin(angle)
            # Condición: (x - cx)*cos(a) + (y - cy)*sin(a) <= r
            # Multiplicando por -1: nx*x + ny*y >= -r + nx*cx + ny*cy
            thresh = -radius + nx * self.cx + ny * self.cy
            self.hidden_neurons.append(ThresholdGate([nx, ny], threshold=thresh, name=f"Lado_{i+1}"))

    def sum_halfplanes(self, x: float, y: float, subtract_floor: bool = False) -> float:
        """Retorna la suma de las N neuronas ocultas (con opción de restar el piso N/2)."""
        s = sum(neuron.evaluate([x, y])[0] for neuron in self.hidden_neurons)
        if subtract_floor:
            return s - (self.n_sides / 2.0)
        return float(s)

    def classify(self, x: float, y: float) -> int:
        """Dispara 1 si el punto está dentro del polígono (suma >= N)."""
        s = self.sum_halfplanes(x, y, subtract_floor=False)
        return 1 if s >= self.n_sides else 0

    @staticmethod
    def theoretical_limit_sum(n_sides: int, radius: float, distance: float) -> float:
        """Fórmula teórica en el límite circular (Diapositiva 38)."""
        if distance <= radius:
            return float(n_sides)
        return n_sides * (0.5 + (1.0 / math.pi) * math.asin(radius / distance))


# =============================================================================
# 4. EL MLP COMO APROXIMADOR UNIVERSAL DE FUNCIONES CONTINUAS (Diapositivas 49 a 54)
# =============================================================================

class PulseFunctionApproximator:
    """
    Aproxima cualquier función continua 1D f(x) en un intervalo [a, b] usando N pulsos cuadrados
    (2*N neuronas de umbral en una sola capa oculta + 1 neurona sumadora lineal en la salida).
    (Diapositivas 49 a 53)
    - Cada pulso i en [T_{i}, T_{i+1}] de altura h_i = f(centro_i) se construye con:
      * Neurona oculta A_i: dispara 1 si x >= T_i (peso +h_i en la salida)
      * Neurona oculta B_i: dispara 1 si x >= T_{i+1} (peso -h_i en la salida)
    """
    def __init__(self, func: Callable[[float], float], a: float, b: float, n_pulses: int):
        self.func = func
        self.a = a
        self.b = b
        self.n_pulses = n_pulses
        self.dx = (b - a) / n_pulses
        self.pulses: List[Tuple[float, float, float]] = []  # (t1, t2, height)

        for i in range(n_pulses):
            t1 = a + i * self.dx
            t2 = t1 + self.dx
            mid = 0.5 * (t1 + t2)
            h = func(mid)
            self.pulses.append((t1, t2, h))

    @property
    def num_hidden_neurons(self) -> int:
        return 2 * self.n_pulses

    def evaluate(self, x: float) -> float:
        total = 0.0
        for (t1, t2, h) in self.pulses:
            u1 = step_activation(x, t1)
            u2 = step_activation(x, t2)
            pulse = u1 - u2
            total += h * pulse
        return total

    def compute_metrics(self, num_samples: int = 500) -> Dict[str, float]:
        max_err = 0.0
        mse = 0.0
        for i in range(num_samples):
            # Evitar evaluar justo en el borde final b donde el último pulso cae
            x = self.a + (self.b - self.a) * (i + 0.5) / num_samples
            y_true = self.func(x)
            y_approx = self.evaluate(x)
            err = abs(y_true - y_approx)
            if err > max_err:
                max_err = err
            mse += err * err
        rmse = math.sqrt(mse / num_samples)
        return {"max_error_epsilon": max_err, "rmse": rmse}


# =============================================================================
# 5. DEMOSTRACIONES EN CONSOLA Y GENERACIÓN DE GRÁFICAS EDUCATIVAS
# =============================================================================

def demo_1_booleana_universal():
    print("\n" + "=" * 82)
    print(" MÓDULO 1: EL MLP COMO FUNCIÓN BOOLEANA UNIVERSAL (DNF, KARNAUGH Y XOR 2 NEURONAS)")
    print("=" * 82)

    # 1A. XOR con solo 2 neuronas (Friedland, Diapositiva 15)
    print("\n▶ 1.1 Construcción de XOR con SOLO 2 NEURONAS (Gerald Friedland - Diapositiva 15):")
    print("  • Neurona oculta: h = X ∧ Y (pesos [1, 1], umbral 2)")
    print("  • Neurona salida: evalúa X + Y - 2*(X ∧ Y) >= 1 (conexión directa de entrada a salida)")
    print("-" * 74)
    print(f"  {'X':^3} | {'Y':^3} || {'z_h':^5} | {'h (X∧Y)':^9} || {'z_out = X+Y-2h':^16} | {'Salida':^8} | {'XOR Real':^8}")
    print("-" * 74)
    xor2 = FriedlandXOR2Neurons()
    for x, y in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        r = xor2.forward(x, y)
        print(f"  {x:^3} | {y:^3} || {r['z_h']:>4.1f}  | {r['h_and']:^9} || {r['z_out']:>10.1f}       | {r['out']:^8} | {r['expected']:^8}")

    # 1B. Construcción DNF del Mapa de Karnaugh (Diapositiva 19)
    print("\n▶ 1.2 Síntesis DNF y Reducción con Mapa de Karnaugh (Diapositiva 19):")
    print("  Función de 4 variables (W, X, Y, Z) con 7 celdas activas (salida = 1):")
    minterms_kmap = [
        (0, 0, 0, 0), (0, 0, 1, 0),
        (0, 1, 0, 0), (0, 1, 0, 1),
        (1, 1, 0, 0),
        (1, 0, 0, 0), (1, 0, 1, 0)
    ]
    dnf_mlp = DNFBooleanMLP(4, minterms_kmap)
    print(f"  • DNF canónica (sin reducir): {len(dnf_mlp.hidden_neurons)} términos AND -> 7 neuronas ocultas + 1 OR:")
    print("    Y = " + " + ".join(dnf_mlp.dnf_expressions))
    print("  • DNF reducida por Karnaugh:  3 términos AND -> ¡Solo 3 neuronas ocultas + 1 OR!")
    print("    Y_red = ȲZ̄ + W̄XȲ + X̄Z̄")

    # Construimos también la red reducida de 3 neuronas ocultas para comprobar su equivalencia:
    # 1) ȲZ̄:     pesos [0, 0, -1, -1], umbral 0
    # 2) W̄XȲ:    pesos [-1, 1, -1, 0], umbral 1
    # 3) X̄Z̄:     pesos [0, -1, 0, -1], umbral 0
    red_h1 = ThresholdGate([0.0, 0.0, -1.0, -1.0], threshold=0.0, name="ȲZ̄")
    red_h2 = ThresholdGate([-1.0, 1.0, -1.0, 0.0], threshold=1.0, name="W̄XȲ")
    red_h3 = ThresholdGate([0.0, -1.0, 0.0, -1.0], threshold=0.0, name="X̄Z̄")
    red_or = ThresholdGate([1.0, 1.0, 1.0], threshold=1.0, name="OR")

    all_match = True
    for bits in range(16):
        w, x, y, z = (bits >> 3) & 1, (bits >> 2) & 1, (bits >> 1) & 1, bits & 1
        out_dnf, _ = dnf_mlp.forward((w, x, y, z))
        h_vals = [red_h1.evaluate([w, x, y, z])[0], red_h2.evaluate([w, x, y, z])[0], red_h3.evaluate([w, x, y, z])[0]]
        out_red, _ = red_or.evaluate(h_vals)
        if out_dnf != out_red:
            all_match = False
    print(f"  • Verificación exhaustiva en las 2^4 = 16 entradas (7 neuronas vs 3 neuronas): {'✅ 100% IDÉNTICAS' if all_match else '❌ ERROR'}")

    # 1C. Comparación Exponencial vs Lineal para Paridad de N variables (Diapositivas 20-26)
    print("\n▶ 1.3 El Peor Caso (Tablero de Ajedrez / Paridad XOR de N variables - Diapositivas 20-25):")
    print("  Comparación del número de neuronas: 1 Capa Oculta (2^(N-1) + 1) vs. Red Profunda (3(N-1)):")
    print("-" * 82)
    print(f"  {'N vars':^6} | {'1 Capa Oculta (2^(N-1)+1)':^25} | {'Red Profunda 3(N-1)':^20} | {'XOR 2-neur 2(N-1)':^17} | {'Capas Árbol':^11}")
    print("-" * 82)
    for row in compare_parity_architectures([2, 4, 6, 8, 10, 12, 16, 20]):
        print(f"  {row['N']:^6} | {row['shallow_total']:>20,d}      | {row['deep_3_total']:^20d} | {row['deep_2_total']:^17d} | {row['tree_layers']:^11d}")
    print("-" * 82)
    print("  💡 Conclusión: En N=16, una sola capa oculta exige 32,769 neuronas; ¡la red profunda solo 45!")


def demo_2_clasificador_universal():
    print("\n" + "=" * 82)
    print(" MÓDULO 2: EL MLP COMO CLASIFICADOR UNIVERSAL (POLÍGONOS, CÍRCULOS Y PISO N/2)")
    print("=" * 82)
    print("\n▶ 2.1 Del Polígono al Cilindro Circular (Diapositivas 33 a 40):")
    print("  Al rodear un círculo de radio r=1.0 con N semiplanos (neuronas ocultas):")
    print("  • Adentro del círculo (d < r): todas las N neuronas disparan => Suma = N.")
    print("  • Afuera del círculo (d > r): la suma cae hacia el piso asintótico N/2.")
    print("  • Si agregamos un sesgo b = -N/2 en la salida, el piso exterior baja a ~0.")
    print("-" * 78)
    print(f"  {'Distancia d':^12} | {'Ubicación':^12} || {'Suma (N=4)':^11} | {'Suma (N=16)':^12} | {'Suma (N=64)':^12} | {'Con sesgo -N/2 (N=64)':^21}")
    print("-" * 78)

    c4 = PolygonCircleClassifier(4, radius=1.0)
    c16 = PolygonCircleClassifier(16, radius=1.0)
    c64 = PolygonCircleClassifier(64, radius=1.0)

    for d in [0.0, 0.5, 0.9, 1.2, 2.0, 5.0, 20.0]:
        # Evaluamos en una dirección diagonal (45 grados)
        px = d * math.cos(math.pi / 4)
        py = d * math.sin(math.pi / 4)
        loc = "ADENTRO" if d <= 1.0 else "AFUERA"
        s4 = c4.sum_halfplanes(px, py)
        s16 = c16.sum_halfplanes(px, py)
        s64 = c64.sum_halfplanes(px, py)
        s64_bias = c64.sum_halfplanes(px, py, subtract_floor=True)
        print(f"  {d:^12.2f} | {loc:^12} || {s4:^11.0f} | {s16:^12.0f} | {s64:^12.0f} | {s64_bias:^+21.1f}")
    print("-" * 78)

    print("\n▶ 2.2 El Tablero de Rombos en 2D (Diapositivas 44 a 47):")
    print("  • Patrón formado por L líneas que se cruzan alternando regiones amarillas/grises.")
    print("  • Con L = 16 líneas: 2 capas ocultas requieren 16 + 40 = 56 neuronas ocultas (57 total);")
    print("    Red XOR profunda requiere 16 + 15*2 = 46 neuronas.")
    print("  • Con L = 64 líneas: 2 capas ocultas requieren 64 + 544 = 608 neuronas ocultas (609 total);")
    print("    Red XOR profunda (12 capas) requiere solo 64 + 63*3 = 253 neuronas (190 con XOR de 2 neuronas).")


def demo_3_aproximador_universal():
    print("\n" + "=" * 82)
    print(" MÓDULO 3: EL MLP COMO APROXIMADOR UNIVERSAL DE FUNCIONES CONTINUAS")
    print("=" * 82)
    print("\n▶ Construcción mediante suma de pulsos cuadrados en [0, 2π] (Diapositivas 49 a 53):")
    print("  Función objetivo: f(x) = sin(x) + 0.5*cos(2x) + 1.5")
    print("  Cada pulso usa 2 neuronas de umbral en la capa oculta (x >= T1 y x >= T2) con pesos +h_i y -h_i.")
    print("-" * 78)
    print(f"  {'N Pulsos':^10} | {'Neuronas Ocultas (2N)':^22} | {'Error Máximo (ε)':^18} | {'RMSE':^18}")
    print("-" * 78)

    target_fn = lambda x: math.sin(x) + 0.5 * math.cos(2.0 * x) + 1.5
    for n_p in [4, 8, 16, 32, 64, 128]:
        approx = PulseFunctionApproximator(target_fn, 0.0, 2.0 * math.pi, n_pulses=n_p)
        m = approx.compute_metrics()
        print(f"  {n_p:^10d} | {approx.num_hidden_neurons:^22d} | {m['max_error_epsilon']:^18.4f} | {m['rmse']:^18.4f}")
    print("-" * 78)
    print("  ⚠️ Distinción clave (Diapositivas 52-53):")
    print("     • REPRESENTAR: calcular exactamente f(x). Con unidades de umbral solo ocurre si f ya es escalonada.")
    print("     • APROXIMAR: para cualquier ε > 0, existe un MLP finito con error menor que ε.")
    print("     • MITO FALSO: 'Una red de una capa con infinitas neuronas modela exactamente cualquier función continua'.")
    print("       La suma de pulsos sigue siendo escalonada; la universalidad es de aproximación, no de igualdad.")


def demo_4_suficiencia_y_activaciones():
    print("\n" + "=" * 82)
    print(" MÓDULO 4: SUFICIENCIA DE LA ARQUITECTURA (ANCHO, PROFUNDIDAD Y ACTIVACIONES)")
    print("=" * 82)
    print("\n▶ 4.1 Franjas y Celdas: ¿Por qué una primera capa angosta pierde información? (Diapositivas 55-57)")
    print("  • Con 8 neuronas de umbral en la capa 1:")
    print("    - Si las 8 rectas son paralelas -> dividen el plano en solo 9 franjas.")
    print("    - Si se reparten 4 y 4 en cruz  -> definen 5 x 5 = 25 celdas.")
    print("  • Con activación de umbral (0 o 1), todos los puntos dentro de una misma celda producen")
    print("    EXACTAMENTE el mismo vector binario. ¡La información fina se perdió para siempre!")
    print("\n▶ 4.2 Umbral vs. Activaciones Graduadas (Diapositivas 7 y 58-60):")
    print("  • Una activación graduada (Sigmoide, Tanh, ReLU) conserva CUÁNTO más allá de la frontera")
    print("    está el punto, permitiendo que capas posteriores reconstruyan patrones complejos.")
    print("-" * 78)
    print(f"  {'z (pre-activación)':^18} || {'Escalón (Umbral)':^16} | {'Sigmoide':^12} | {'Tanh':^12} | {'ReLU':^12}")
    print("-" * 78)
    for z in [-2.0, -0.5, 0.0, 0.5, 2.0, 5.0]:
        print(f"  {z:^18.2f} || {step_activation(z):^16d} | {sigmoid(z):^12.4f} | {tanh_act(z):^12.4f} | {relu(z):^12.2f}")
    print("-" * 78)


def generate_educational_summary_plot(output_path: str = "aproximacion_universal_resumen.png") -> str:
    """
    Genera una lámina educativa de 4 paneles de alta resolución que resume visualmente la clase:
    1) Explosión Exponencial (1 capa) vs. Crecimiento Lineal (Red Profunda) en XOR de N variables.
    2) El MLP como Clasificador Universal: Suma de N=64 semiplanos formando el cilindro circular con piso N/2.
    3) El MLP como Aproximador Universal: Aproximación de una función continua mediante suma de pulsos cuadrados.
    4) Funciones de Activación: Escalón (Umbral) vs. Sigmoide, Tanh y ReLU.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[Nota] Matplotlib/NumPy no disponibles para generar el gráfico resumen.")
        return ""

    plt.rcParams.update({
        "font.sans-serif": "DejaVu Sans",
        "axes.edgecolor": "#334155",
        "axes.linewidth": 0.8
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor("#0f172a")

    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1", labelsize=9)
        ax.xaxis.label.set_color("#e2e8f0")
        ax.yaxis.label.set_color("#e2e8f0")
        ax.title.set_color("#f8fafc")
        ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")

    # Panel 1: Costo Exponencial vs Lineal (XOR N variables)
    ax1 = axes[0, 0]
    n_vals = np.arange(2, 13)
    shallow_neurons = 2 ** (n_vals - 1) + 1
    deep_neurons = 3 * (n_vals - 1)
    deep_friedland = 2 * (n_vals - 1)

    ax1.plot(n_vals, shallow_neurons, "o-", color="#ef4444", linewidth=2.5, label="1 Capa Oculta (DNF): $2^{N-1} + 1$")
    ax1.plot(n_vals, deep_neurons, "s-", color="#22c55e", linewidth=2.5, label="Red Profunda (Bloques XOR 3): $3(N-1)$")
    ax1.plot(n_vals, deep_friedland, "^--", color="#38bdf8", linewidth=2.0, label="Red Profunda (Friedland XOR 2): $2(N-1)$")
    ax1.set_yscale("log")
    ax1.set_title("1. Función Booleana Universal: Costo en Neuronas (XOR N-ario)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Número de entradas booleanas ($N$)")
    ax1.set_ylabel("Total de Neuronas (escala log)")
    leg1 = ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8.5)

    # Panel 2: Clasificador Universal (Del Polígono al Cilindro)
    ax2 = axes[0, 1]
    grid_x = np.linspace(-2.2, 2.2, 160)
    grid_y = np.linspace(-2.2, 2.2, 160)
    GX, GY = np.meshgrid(grid_x, grid_y)
    N_sides = 32
    R = 1.0
    Z_sum = np.zeros_like(GX)
    for i in range(N_sides):
        ang = 2.0 * np.pi * i / N_sides
        Z_sum += ((-np.cos(ang) * GX - np.sin(ang) * GY) >= -R).astype(float)

    im = ax2.contourf(GX, GY, Z_sum, levels=16, cmap="plasma")
    circle = plt.Circle((0, 0), R, color="#38bdf8", fill=False, linewidth=2.2, linestyle="--", label="Frontera Umbral $\\sum y_i = N$")
    ax2.add_patch(circle)
    cbar = fig.colorbar(im, ax=ax2)
    cbar.ax.yaxis.set_tick_params(color="#cbd5e1", labelcolor="#cbd5e1", labelsize=8)
    cbar.set_label("Suma de semiplanos $\\sum y_i$", color="#e2e8f0", fontsize=9)
    ax2.set_title(f"2. Clasificador Universal: Cilindro de N={N_sides} Semiplanos (Piso N/2=16)", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    ax2.set_aspect("equal")

    # Panel 3: Aproximador Universal de Funciones Continuas (Suma de Pulsos)
    ax3 = axes[1, 0]
    fn = lambda x: np.sin(x) + 0.5 * np.cos(2.0 * x) + 1.5
    x_fine = np.linspace(0, 2 * np.pi, 500)
    y_true = fn(x_fine)

    approx_10 = PulseFunctionApproximator(lambda v: math.sin(v) + 0.5 * math.cos(2.0 * v) + 1.5, 0.0, 2 * math.pi, 10)
    approx_32 = PulseFunctionApproximator(lambda v: math.sin(v) + 0.5 * math.cos(2.0 * v) + 1.5, 0.0, 2 * math.pi, 32)
    y_p10 = [approx_10.evaluate( float( xv ) ) for xv in x_fine]
    y_p32 = [approx_32.evaluate( float( xv ) ) for xv in x_fine]

    ax3.plot(x_fine, y_true, color="#f8fafc", linewidth=2.5, label="Función continua real $f(x)$")
    ax3.step(x_fine, y_p10, color="#f59e0b", linewidth=1.8, where="mid", label="Suma de 10 pulsos (20 neuronas)")
    ax3.step(x_fine, y_p32, color="#38bdf8", linewidth=1.5, alpha=0.85, where="mid", label="Suma de 32 pulsos (64 neuronas)")
    ax3.fill_between(x_fine, 0, y_p10, color="#f59e0b", alpha=0.12)
    ax3.set_title("3. Aproximador Universal: Suma de Pulsos Cuadrados", fontsize=11, fontweight="bold", pad=10)
    ax3.set_xlabel("Entrada continua $x$")
    ax3.set_ylabel("$f(x)$")
    ax3.set_ylim(0, 3.2)
    ax3.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8.5)

    # Panel 4: Activaciones (Umbral vs Graduadas)
    ax4 = axes[1, 1]
    z_vals = np.linspace(-4, 4, 400)
    y_step = (z_vals >= 0).astype(float)
    y_sig = 1.0 / (1.0 + np.exp(-z_vals))
    y_tanh = np.tanh(z_vals)
    y_relu = np.maximum(0, z_vals)

    ax4.plot(z_vals, y_step, color="#ef4444", linewidth=2.2, label="Escalón / Umbral (0 o 1, pierde gradación)")
    ax4.plot(z_vals, y_sig, color="#22c55e", linewidth=2.2, label="Sigmoide $\\sigma(z) \\in (0,1)$")
    ax4.plot(z_vals, y_tanh, color="#a855f7", linewidth=2.0, linestyle="--", label="Tanh $\\in (-1,1)$")
    ax4.plot(z_vals, y_relu, color="#38bdf8", linewidth=2.2, label="ReLU $\\max(0, z)$ (no satura para $z>0$)")
    ax4.set_ylim(-1.2, 3.0)
    ax4.set_title("4. El Rol de la Activación: Umbral vs. Activaciones Graduadas", fontsize=11, fontweight="bold", pad=10)
    ax4.set_xlabel("Combinación ponderada $z = \\mathbf{w}\\cdot\\mathbf{x} + b$")
    ax4.set_ylabel("Activación $\\phi(z)$")
    ax4.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8.5, loc="upper left")

    fig.suptitle(
        "Aproximación Universal con Redes Neuronales — Ciencia de Datos UCN (CMU 11-785)",
        color="#f8fafc", fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[Info] Lámina educativa de 4 paneles guardada en: {output_path}")
    return output_path


def run_all_demos():
    demo_1_booleana_universal()
    demo_2_clasificador_universal()
    demo_3_aproximador_universal()
    demo_4_suficiencia_y_activaciones()
    generate_educational_summary_plot()


def interactive_menu():
    while True:
        print("\n" + "=" * 75)
        print(" LABORATORIO EDUCATIVO: APROXIMACIÓN UNIVERSAL CON REDES NEURONALES")
        print(" Ciencia de Datos — UCN (Basado en CMU 11-785, B. Raj et al.)")
        print("=" * 75)
        print(" Seleccione un módulo educativo para explorar:")
        print("   [1] 🧮 El MLP como Función Booleana Universal (DNF, Karnaugh, XOR 2-neur y Costo Exponencial)")
        print("   [2] 📐 El MLP como Clasificador Universal (Polígonos, Círculos, Piso N/2 y Rombos)")
        print("   [3] 📈 El MLP como Aproximador Universal de Funciones Continuas (Suma de Pulsos)")
        print("   [4] ⚖️  Profundidad, Ancho y Activaciones (Franjas/Celdas y Triángulo de Compensaciones)")
        print("   [5] 🚀 Ejecutar TODAS las demostraciones + Generar lámina resumen (.png)")
        print("   [6] 🌐 Abrir Suite Web Interactiva en el navegador")
        print("   [7] Salir")
        print("-" * 75)

        try:
            op = input("Ingrese una opción [1-7]: ").strip().lower()
            if op == "1":
                demo_1_booleana_universal()
            elif op == "2":
                demo_2_clasificador_universal()
            elif op == "3":
                demo_3_aproximador_universal()
            elif op == "4":
                demo_4_suficiencia_y_activaciones()
            elif op == "5":
                run_all_demos()
            elif op == "6":
                import os
                import webbrowser
                html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "web", "index.html"))
                print(f"Abriendo suite educativa web: file://{html_path}")
                webbrowser.open(f"file://{html_path}")
            elif op in ("7", "q", "salir", "exit"):
                print("Saliendo del laboratorio educativo. ¡Éxito en el curso!")
                break
            else:
                print("⚠️ Opción no válida. Elija entre 1 y 7.")
        except (KeyboardInterrupt, EOFError):
            print("\nSaliendo del laboratorio educativo.")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("--auto", "-a", "--all"):
        run_all_demos()
    else:
        interactive_menu()
