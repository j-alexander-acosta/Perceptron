#!/usr/bin/env python3
"""
Módulo de Cátedra CD19: Regularización de Redes Neuronales
Ciencia de Datos - Doctorado UCN
Fuentes Académicas: CMU 11-785 (Bhiksha Raj et al.) y Stanford CS221 (Percy Liang et al.)

Contenidos:
1. Repaso de Divergencias, Saturación Sigmoidal y Suavidad de la Red
   - L2 vs KL: KL con Softmax es convexa en z; L2 sobre probabilidades satura (sigma'(6) ≈ 0.0025).
   - Gradiente de divergencia como error de salida: ∇_z div = y - d.
   - Subespecificación de datos: 2^100 ≈ 10^30 entradas en {0,1}^100.
   - Suavidad del perceptrón: σ(wx), pendientes abruptas vs suaves, constante de Lipschitz.
2. Regularización L2 y Decaimiento de Pesos (Weight Decay)
   - Objetivo regularizado: min_W (1/T) ∑ div + (1/2) λ ∑ ||W_k||_F^2.
   - Regla de actualización: W <- (1 - η λ) W - (η/b) ∑ ∇div.
   - Ejemplo numérico: η = 0.1, λ = 0.1 => factor 0.99 (reducción del 1% por paso).
   - Efecto de λ: λ=0 (sobreajuste), λ óptimo (generalización), λ->inf (colapso a la media y).
3. Dropout y Dropout Invertido (Srivastava et al., 2014)
   - Muestreo Bernoulli con probabilidad de retención α (1-α apagadas).
   - Interpretación de ensamble: 2^N subredes compartiendo pesos (Bagging implícito).
   - Eliminación de co-adaptación de detectores de características.
   - Inferencia tradicional (escalar por α) vs Dropout Invertido (dividir por α en entrenamiento).
   - Ejemplo numérico: f(z) = 2.0, α = 0.8 => E[y] = 1.6; con dropout invertido y = 2.5.
   - Variantes: DropConnect, Zoneout, Whiteout.
4. Trucos del Oficio y Normalización por Lotes (Batch Normalization, Ioffe & Szegedy 2015)
   - Detención temprana (Early Stopping) y fenómeno de Grokking (Power et al., 2022).
   - Recorte de Gradientes (Gradient Clipping, Pascanu et al. 2013) con umbral θ = 5.0.
   - Aumento de datos (Data Augmentation).
   - Batch Normalization: μ_B, σ_B^2, estandarización u_i, parámetros aprendibles γ y β.
   - Micro-ejemplos numéricos exactos de cátedra:
     * Lote z = [1, 2, 3, 6] => μ_B = 3.0, σ_B^2 = 3.5 => u ≈ [-1.07, -0.53, 0.00, 1.60].
     * Lote z = [2, 4, 6, 8] => μ_B = 5.0, σ_B^2 = 5.0 => u ≈ [-1.34, -0.45, 0.45, 1.34].
   - Estadísticas de inferencia y factor de Bessel B/(B-1).
"""

import sys
import math
import random
from typing import List, Tuple, Dict, Any, Optional

# ==============================================================================
# 1. FUNCIONES MATEMÁTICAS FUNDAMENTALES
# ==============================================================================

def sigmoid(z: float) -> float:
    """Función de activación logística sigmoide estándar."""
    if z >= 40.0:
        return 1.0
    if z <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))

def sigmoid_derivative(z: float) -> float:
    """Derivada analítica de la sigmoide: σ'(z) = σ(z)(1 - σ(z))."""
    s = sigmoid(z)
    return s * (1.0 - s)

def softmax(logits: List[float]) -> List[float]:
    """Softmax numéricamente estable para un vector de logits z."""
    max_z = max(logits)
    exps = [math.exp(zi - max_z) for zi in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

# ==============================================================================
# 2. TEMA 1: DIVERGENCIAS, SATURACIÓN Y SUAVIDAD DEL PERCEPTRÓN
# ==============================================================================

def compute_l2_saturation_comparison(z: float = 6.0, target: float = 0.0) -> Dict[str, float]:
    """
    Compara el gradiente de L2 vs el error bajo saturación sigmoidal.
    Diapositiva 5: Para z = 6.0, σ(6) ≈ 0.9975, σ'(6) ≈ 0.0025.
    Con target d = 0, el error es masivo (0.9975), pero el gradiente L2
    se aplana a (y - d) * σ'(z) ≈ 0.00246 debido a la saturación.
    """
    y = sigmoid(z)
    diff = y - target
    s_prime = sigmoid_derivative(z)
    grad_l2 = diff * s_prime
    return {
        "z": z,
        "target": target,
        "output": y,
        "error": diff,
        "sigma_prime": s_prime,
        "grad_l2": grad_l2
    }

def compute_softmax_kl_gradient(y_probs: List[float], targets: List[float]) -> List[float]:
    """
    Calcula el gradiente de la divergencia KL respecto a los logits z.
    Diapositivas 6-7: ∇_z div = y - d (el gradiente es directamente el vector de error).
    Ejemplo exacto de cátedra: d = [0, 1, 0], y = [0.2, 0.5, 0.3] => grad = [0.2, -0.5, 0.3].
    """
    if len(y_probs) != len(targets):
        raise ValueError("Longitudes distintas entre probabilidades y objetivos.")
    return [round(y - d, 4) for y, d in zip(y_probs, targets)]

def compute_subspecification_combinatorics(num_bits: int = 100, observed_samples: float = 1e15) -> Dict[str, Any]:
    """
    Demuestra la subespecificación de datos en espacios de alta dimensión.
    Diapositivas 8-10: Espacio {0,1}^100 tiene 2^100 ≈ 1.267e30 puntos posibles.
    Incluso con 10^15 ejemplos, la fracción observada es apenas 7.89e-16 (falta un factor de 10^15).
    """
    total_space = 2 ** num_bits
    fraction = observed_samples / total_space
    missing_ratio = total_space / observed_samples
    return {
        "num_bits": num_bits,
        "total_space": total_space,
        "total_space_approx": f"{total_space:.4e}",
        "observed_samples": observed_samples,
        "fraction_observed": fraction,
        "missing_factor": missing_ratio
    }

def perceptron_smoothness_profile(weights: List[float], x_range: List[float]) -> Dict[float, List[Tuple[float, float, float]]]:
    """
    Evalúa la respuesta y pendiente de un perceptrón individual σ(wx) para distintos pesos w.
    Diapositivas 11-14:
    Para w = 0.5, la pendiente máxima es w * 0.25 = 0.125 (transición suave).
    Para w = 5.0, la pendiente máxima es w * 0.25 = 1.25 (escalón abrupto que sobreajusta).
    """
    results = {}
    for w in weights:
        curve = []
        for x in x_range:
            z = w * x
            y = sigmoid(z)
            dy_dx = w * sigmoid_derivative(z)
            curve.append((x, y, dy_dx))
        results[w] = curve
    return results

# ==============================================================================
# 3. TEMA 2: REGULARIZACIÓN L2 Y DECAIMIENTO DE PESOS (WEIGHT DECAY)
# ==============================================================================

def simulate_weight_decay_step(
    w_current: float,
    grad_div: float,
    eta: float = 0.1,
    lambda_reg: float = 0.1
) -> Dict[str, float]:
    """
    Ejecuta un paso de actualización con decaimiento de pesos L2:
    W_new = (1 - η * λ) * W - η * grad_div.
    Diapositivas 17-18:
    Con η = 0.1 y λ = 0.1, el factor de contracción es (1 - 0.01) = 0.99 (encoge 1%).
    """
    decay_factor = 1.0 - (eta * lambda_reg)
    w_decayed = decay_factor * w_current
    w_new = w_decayed - (eta * grad_div)
    return {
        "w_initial": w_current,
        "eta": eta,
        "lambda_reg": lambda_reg,
        "decay_factor": decay_factor,
        "percent_shrinkage": (1.0 - decay_factor) * 100.0,
        "w_after_decay": w_decayed,
        "w_final": w_new
    }

def simulate_weight_decay_trajectory(
    w_initial: float = 10.0,
    steps: int = 50,
    eta: float = 0.1,
    lambda_reg: float = 0.1,
    zero_grad: bool = True
) -> List[float]:
    """
    Simula la trayectoria geométrica de contracción de pesos cuando solo actúa el decaimiento.
    W_t = W_0 * (1 - η * λ)^t.
    """
    w = w_initial
    trajectory = [w]
    factor = 1.0 - (eta * lambda_reg)
    for _ in range(steps):
        w = factor * w if zero_grad else (factor * w - eta * 0.05 * math.sin(w))
        trajectory.append(round(w, 5))
    return trajectory

def evaluate_lambda_spectrum(lambdas: Optional[List[float]] = None) -> List[Dict[str, Any]]:
    """
    Analiza el efecto teórico del hiperparámetro λ sobre la función aprendida (Diapositivas 19-20).
    λ = 0: Sobreajuste (función errática con saltos pronunciados).
    λ óptimo: Suave, generaliza a puntos no observados.
    λ -> ∞: Subajuste severo; todos los pesos colapsan a 0 y la salida colapsa a la media constante.
    """
    if lambdas is None:
        lambdas = [0.0, 0.001, 0.01, 0.1, 1.0, 100.0]
    results = []
    for lam in lambdas:
        if lam == 0.0:
            regime = "Sin regularización (λ=0)"
            effect = "Pesos libres para crecer arbitrariamente; función oscilatoria / sobreajuste."
            bias_variance = "Bajo sesgo, varianza extrema."
        elif lam < 0.05:
            regime = f"Regularización suave (λ={lam})"
            effect = "Atenúa valores atípicos de pesos manteniendo alta expresividad."
            bias_variance = "Excelente balance para conjuntos de datos grandes."
        elif lam <= 1.0:
            regime = f"Regularización moderada (λ={lam})"
            effect = "Fuerza transiciones suaves; previene pendientes abruptas en límites de decisión."
            bias_variance = "Óptima generalización en presencia de ruido moderado."
        else:
            regime = f"Regularización extrema (λ={lam})"
            effect = "Pesos forzados hacia cero; la red colapsa a la función constante y_bar."
            bias_variance = "Sesgo severo (subajuste), varianza nula."
        results.append({
            "lambda": lam,
            "regime": regime,
            "effect": effect,
            "bias_variance": bias_variance
        })
    return results

# ==============================================================================
# 4. TEMA 3: DROPOUT Y DROPOUT INVERTIDO
# ==============================================================================

class DropoutSimulator:
    """
    Simulador de Dropout clásico y Dropout Invertido (Srivastava et al., 2014).
    Diapositivas 21-30.
    """
    def __init__(self, keep_prob: float = 0.8, seed: Optional[int] = 42):
        self.alpha = keep_prob
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def generate_mask(self, num_units: int) -> List[float]:
        """Genera máscara Bernoulli con probabilidad de retención α."""
        return [1.0 if self.rng.random() < self.alpha else 0.0 for _ in range(num_units)]

    def forward_train_standard(self, activations: List[float], mask: Optional[List[float]] = None) -> Tuple[List[float], List[float]]:
        """Dropout estándar en entrenamiento: propaga f(z) con prob α, 0 con prob 1-α."""
        if mask is None:
            mask = self.generate_mask(len(activations))
        out = [a * m for a, m in zip(activations, mask)]
        return out, mask

    def forward_test_standard(self, activations: List[float]) -> List[float]:
        """
        Inferencia con Dropout estándar: todas las neuronas activas, multiplicadas por α.
        Diapositiva 30: Si f(z) = 2.0 y α = 0.8, la salida es 0.8 * 2.0 = 1.6.
        """
        return [round(a * self.alpha, 4) for a in activations]

    def forward_train_inverted(self, activations: List[float], mask: Optional[List[float]] = None) -> Tuple[List[float], List[float]]:
        """
        Dropout Invertido en entrenamiento (PyTorch / TensorFlow moderno):
        Escala por (1 / α) en entrenamiento: y_j = D_j * f(z_j) / α.
        Diapositiva 30: Si f(z) = 2.0 y α = 0.8, la salida activa es 2.0 / 0.8 = 2.5.
        Esperanza matemática: E[y] = α * (f(z) / α) = f(z) = 2.0.
        """
        if mask is None:
            mask = self.generate_mask(len(activations))
        scale = 1.0 / self.alpha
        out = [round(a * m * scale, 4) for a, m in zip(activations, mask)]
        return out, mask

    def forward_test_inverted(self, activations: List[float]) -> List[float]:
        """Inferencia con Dropout Invertido: identidad exacta, costo de cómputo nulo."""
        return list(activations)

def calculate_subnetwork_ensemble_scale(num_units: int = 50) -> Dict[str, Any]:
    """
    Calcula el tamaño del ensamble implícito de subredes con Dropout.
    Diapositiva 24: Una capa de N neuronas contiene 2^N subredes posibles.
    """
    total_subnets = 2 ** num_units
    return {
        "num_units": num_units,
        "total_subnetworks": total_subnets,
        "formatted_power": f"2^{num_units}",
        "scientific_approx": f"{total_subnets:.3e}"
    }

# ==============================================================================
# 5. TEMA 4: TRUCOS DEL OFICIO Y BATCH NORMALIZATION
# ==============================================================================

def clip_gradients_norm(grads: List[float], max_norm: float = 5.0) -> Tuple[List[float], float, bool]:
    """
    Recorte de gradientes por norma L2 (Pascanu et al., 2013).
    Diapositiva 33: Si ||g|| > θ, g <- (θ / ||g||) * g.
    """
    current_norm = math.sqrt(sum(g * g for g in grads))
    if current_norm > max_norm and current_norm > 1e-12:
        factor = max_norm / current_norm
        clipped = [round(g * factor, 5) for g in grads]
        return clipped, current_norm, True
    return list(grads), current_norm, False

def clip_gradients_value(grads: List[float], theta: float = 5.0) -> Tuple[List[float], bool]:
    """
    Recorte de gradientes elemento a elemento (Diapositiva 33):
    Si |g_i| > θ, se recorta a sign(g_i) * θ.
    """
    clipped = []
    was_clipped = False
    for g in grads:
        if abs(g) > theta:
            clipped.append(math.copysign(theta, g))
            was_clipped = True
        else:
            clipped.append(g)
    return clipped, was_clipped

def compute_batch_normalization(
    batch_z: List[float],
    gamma: float = 1.0,
    beta: float = 0.0,
    epsilon: float = 1e-5
) -> Dict[str, Any]:
    """
    Implementa la normalización por lotes (Ioffe & Szegedy, 2015).
    Diapositivas 35-42:
    1. Media del lote: μ_B = (1/B) ∑ z_i
    2. Varianza del lote: σ_B^2 = (1/B) ∑ (z_i - μ_B)^2
    3. Normalizar: u_i = (z_i - μ_B) / sqrt(σ_B^2 + ε)
    4. Escalamiento y desplazamiento: z_hat_i = γ * u_i + β
    """
    B = len(batch_z)
    if B == 0:
        raise ValueError("Lote vacío.")

    mu_b = sum(batch_z) / B
    var_b = sum((zi - mu_b) ** 2 for zi in batch_z) / B
    std_b = math.sqrt(var_b + epsilon)

    u_norm = [(zi - mu_b) / std_b for zi in batch_z]
    z_hat = [gamma * ui + beta for ui in u_norm]

    # Factor de corrección de Bessel para estadísticas globales de inferencia: B / (B - 1)
    bessel_factor = B / (B - 1) if B > 1 else 1.0
    var_unbiased = var_b * bessel_factor

    return {
        "batch_size": B,
        "input_z": batch_z,
        "mu_B": round(mu_b, 4),
        "var_B": round(var_b, 4),
        "std_B": round(std_b, 4),
        "u_normalized": [round(u, 4) for u in u_norm],
        "gamma": gamma,
        "beta": beta,
        "z_hat": [round(zh, 4) for zh in z_hat],
        "bessel_factor": round(bessel_factor, 4),
        "var_unbiased": round(var_unbiased, 4)
    }

# ==============================================================================
# 6. INTERFAZ INTERACTIVA CLI
# ==============================================================================

def print_banner():
    print("=" * 80)
    print(" CÁTEDRA CD19: REGULARIZACIÓN DE REDES NEURONALES")
    print(" Doctorado en Informática — Universidad Católica del Norte")
    print(" Fuentes: CMU 11-785 (Bhiksha Raj et al.) & Stanford CS221 (Percy Liang)")
    print("=" * 80)

def demo_modulo_1():
    print("\n" + "=" * 80)
    print(" MÓDULO 19.1: DIVERGENCIAS, SATURACIÓN SIGMOIDAL Y SUAVIDAD DEL PERCEPTRÓN")
    print("=" * 80)
    print("1. Comparación L2 vs KL bajo saturación sigmoidal (Diapositiva 5):")
    sat = compute_l2_saturation_comparison(z=6.0, target=0.0)
    print(f"   Logit de entrada z = {sat['z']}")
    print(f"   Salida predicha y = σ(6) = {sat['output']:.6f}")
    print(f"   Objetivo deseado d = {sat['target']}")
    print(f"   Error de predicción (y - d) = {sat['error']:.6f} (¡ERROR MÁXIMO!)")
    print(f"   Derivada de la sigmoide σ'(6) = {sat['sigma_prime']:.6f}")
    print(f"   Gradiente L2: (y - d) * σ'(z) = {sat['grad_l2']:.6f}")
    print("   -> Conclusión: La función L2 aplana el gradiente a ~0.0025 a pesar del error.")

    print("\n2. Gradiente con Softmax y Divergencia KL (Diapositivas 6-7):")
    y_test = [0.2, 0.5, 0.3]
    d_test = [0.0, 1.0, 0.0]
    grad_kl = compute_softmax_kl_gradient(y_test, d_test)
    print(f"   Probabilidades predichas y: {y_test}")
    print(f"   Etiqueta objetivo d:         {d_test}")
    print(f"   Gradiente directo ∇_z div = y - d: {grad_kl}")
    print("   -> La divergencia KL preserva el error directo sin saturación.")

    print("\n3. Subespecificación del Espacio de Entradas (Diapositivas 8-10):")
    comb = compute_subspecification_combinatorics(100, 1e15)
    print(f"   Entradas binarias: {{0,1}}^{comb['num_bits']}")
    print(f"   Puntos posibles en el espacio: 2^100 ≈ {comb['total_space_approx']}")
    print(f"   Ejemplos observados en entrenamiento: {comb['observed_samples']:.1e}")
    print(f"   Fracción del espacio cubierta: {comb['fraction_observed']:.4e}")
    print(f"   Factor faltante de datos: {comb['missing_factor']:.2e}")
    print("   -> Infinitas redes interpolan los puntos; necesitamos regularizar para suavidad.")

    print("\n4. Suavidad del Perceptrón σ(wx) (Diapositivas 11-14):")
    x_samples = [-2.0, -1.0, 0.0, 1.0, 2.0]
    prof = perceptron_smoothness_profile([0.5, 5.0], x_samples)
    print("   Comparativa de pendientes dy/dx:")
    for w in [0.5, 5.0]:
        print(f"   - Peso w = {w}:")
        for x, y, dy in prof[w]:
            print(f"       x = {x:+4.1f} | y = {y:.4f} | dy/dx = {dy:.4f}")
    print("   -> Controlar ||W|| garantiza una cota en la constante de Lipschitz.")

def demo_modulo_2():
    print("\n" + "=" * 80)
    print(" MÓDULO 19.2: REGULARIZACIÓN L2 Y DECAIMIENTO DE PESOS (WEIGHT DECAY)")
    print("=" * 80)
    print("1. Regla de actualización con Mini-lotes y Weight Decay (Diapositivas 17-18):")
    print("   W <- (1 - η * λ) * W - (η / b) * ∑ ∇div")
    step = simulate_weight_decay_step(w_current=5.0, grad_div=0.5, eta=0.1, lambda_reg=0.1)
    print(f"   Tasa de aprendizaje η = {step['eta']}, Parámetro de penalización λ = {step['lambda_reg']}")
    print(f"   Factor de contracción por paso (1 - η*λ): {step['decay_factor']:.4f}")
    print(f"   Encogimiento porcentual: {step['percent_shrinkage']:.2f}% en cada iteración")
    print(f"   Peso inicial: {step['w_initial']:.4f} -> Tras contracción: {step['w_after_decay']:.4f} -> Tras gradiente: {step['w_final']:.4f}")

    print("\n2. Trayectoria de decaimiento en 10 pasos (solo regularización):")
    traj = simulate_weight_decay_trajectory(w_initial=10.0, steps=10, eta=0.1, lambda_reg=0.1)
    for t, val in enumerate(traj):
        print(f"   Paso {t:2d}: W = {val:.4f}")

    print("\n3. Análisis del Espectro de λ (Diapositivas 19-20):")
    spectrum = evaluate_lambda_spectrum()
    for item in spectrum:
        print(f"   [{item['regime']}]")
        print(f"      Efecto: {item['effect']}")
        print(f"      Balance: {item['bias_variance']}")

def demo_modulo_3():
    print("\n" + "=" * 80)
    print(" MÓDULO 19.3: DROPOUT Y DROPOUT INVERTIDO (Srivastava et al., 2014)")
    print("=" * 80)
    print("1. Sorteo de Subredes y Prevención de Co-adaptación (Diapositivas 21-25):")
    ens = calculate_subnetwork_ensemble_scale(30)
    print(f"   Capa con {ens['num_units']} neuronas ocultas:")
    print(f"   Espacio de subredes posibles: {ens['formatted_power']} ≈ {ens['scientific_approx']}")
    print("   -> Dropout entrena un ensamble de 2^N clasificadores compartiendo parámetros.")

    print("\n2. Ejemplo Numérico Exacto de Cátedra (Diapositiva 30):")
    print("   Neurona oculta con activación base f(z) = 2.0 y tasa de retención α = 0.8:")
    sim = DropoutSimulator(keep_prob=0.8, seed=123)

    print("\n   [MÉTODO CLÁSICO (Srivastava 2014)]")
    z_act = [2.0]
    out_train, mask = sim.forward_train_standard(z_act, mask=[1.0])
    out_test = sim.forward_test_standard(z_act)
    print(f"   - Entrenamiento (cuando activa): y_train = {out_train[0]:.2f}")
    print(f"   - Esperanza matemática: E[y] = α * f(z) = 0.8 * 2.0 = 1.60")
    print(f"   - En Inferencia (Prueba): multiplicar toda la red por α:")
    print(f"     y_test = 0.8 * 2.0 = {out_test[0]:.2f}")

    print("\n   [MÉTODO MODERNO: DROPOUT INVERTIDO (PyTorch)]")
    out_inv_train, _ = sim.forward_train_inverted(z_act, mask=[1.0])
    out_inv_test = sim.forward_test_inverted(z_act)
    print(f"   - Entrenamiento: escalar por (1/α) = (1/0.8) = 1.25:")
    print(f"     y_train = (1 / 0.8) * 2.0 = {out_inv_train[0]:.2f}")
    print(f"   - Esperanza matemática: E[y] = 0.8 * 2.50 = 2.00 (¡idéntica a la activación!)")
    print(f"   - En Inferencia (Prueba): ¡Sin ningún cambio ni escalamiento!")
    print(f"     y_test = {out_inv_test[0]:.2f}")

    print("\n3. Variantes de Dropout (Diapositivas 28-29):")
    print("   - DropConnect (Wan et al., 2013): Apaga conexiones individuales (pesos) en vez de nodos.")
    print("   - Zoneout (Krueger et al., 2016): Para RNNs, conserva el valor del estado anterior h_{t-1}.")
    print("   - Whiteout: Inyecta ruido gaussiano aditivo en lugar de anular a cero.")

def demo_modulo_4():
    print("\n" + "=" * 80)
    print(" MÓDULO 19.4: TRUCOS DEL OFICIO Y BATCH NORMALIZATION (Ioffe & Szegedy, 2015)")
    print("=" * 80)
    print("1. Recorte de Gradiente (Gradient Clipping, Diapositiva 33):")
    raw_grads = [3.2, 7.8, -6.1, 1.5]
    clipped_val, was_val = clip_gradients_value(raw_grads, theta=5.0)
    clipped_norm, c_norm, was_norm = clip_gradients_norm(raw_grads, max_norm=5.0)
    print(f"   Gradientes originales: {raw_grads}")
    print(f"   Recorte por valor (θ=5.0): {clipped_val} (¿Se recortó? {was_val})")
    print(f"   Norma actual L2: {c_norm:.4f}")
    print(f"   Recorte por norma (θ=5.0): {clipped_norm} (¿Se recortó? {was_norm})")

    print("\n2. Micro-ejemplo 1 de Cátedra: Lote z = [1, 2, 3, 6] (Diapositiva 39):")
    bn1 = compute_batch_normalization([1.0, 2.0, 3.0, 6.0])
    print(f"   Entradas z:           {bn1['input_z']}")
    print(f"   Media del lote μ_B:   {bn1['mu_B']:.4f} (Exacto = 3.0)")
    print(f"   Varianza del lote σ_B^2: {bn1['var_B']:.4f} (Exacto = 3.5)")
    print(f"   Desviación estándar:  {bn1['std_B']:.4f}")
    print(f"   Salidas u normalizadas: {bn1['u_normalized']}")
    print("   Cátedra: u ≈ [-1.07, -0.53, 0.00, 1.60]")

    print("\n3. Micro-ejemplo 2 de Cátedra: Lote z = [2, 4, 6, 8] (Diapositiva 42):")
    bn2 = compute_batch_normalization([2.0, 4.0, 6.0, 8.0])
    print(f"   Entradas z:           {bn2['input_z']}")
    print(f"   Media del lote μ_B:   {bn2['mu_B']:.4f} (Exacto = 5.0)")
    print(f"   Varianza del lote σ_B^2: {bn2['var_B']:.4f} (Exacto = 5.0)")
    print(f"   Desviación estándar:  {bn2['std_B']:.4f}")
    print(f"   Salidas u normalizadas: {bn2['u_normalized']}")
    print("   Cátedra: u ≈ [-1.34, -0.45, 0.45, 1.34]")

    print("\n4. Inferencia y Corrección de Bessel:")
    print(f"   Factor de corrección de Bessel B/(B-1) para B=4: {bn2['bessel_factor']:.4f}")
    print(f"   Varianza insesgada acumulada: {bn2['var_unbiased']:.4f}")

# ==============================================================================
# 7. SUITE DE VERIFICACIÓN AUTOMÁTICA (--auto)
# ==============================================================================

def run_auto_tests() -> bool:
    """Ejecuta la suite de verificación automatizada de todos los cálculos de la cátedra."""
    print("=" * 80)
    print(" EJECUTANDO SUITE DE VERIFICACIÓN CD19 (REGULARIZACIÓN)")
    print("=" * 80)
    all_passed = True

    try:
        # Test 1: Saturación Sigmoidal
        sat = compute_l2_saturation_comparison(6.0, 0.0)
        assert abs(sat["output"] - 0.9975) < 0.001, "Fallo en valor sigmoide σ(6)"
        assert abs(sat["sigma_prime"] - 0.00247) < 0.0005, "Fallo en σ'(6)"
        assert abs(sat["grad_l2"] - 0.00246) < 0.0005, "Fallo en gradiente L2"
        print(" [PASS] 19.1.1: Saturación sigmoidal y desvanecimiento de gradiente L2")

        # Test 2: Gradiente Softmax KL
        grad_kl = compute_softmax_kl_gradient([0.2, 0.5, 0.3], [0.0, 1.0, 0.0])
        assert grad_kl == [0.2, -0.5, 0.3], f"Gradiente KL incorrecto: {grad_kl}"
        print(" [PASS] 19.1.2: Gradiente de divergencia KL ∇_z div = y - d")

        # Test 3: Weight Decay
        step = simulate_weight_decay_step(10.0, 0.0, eta=0.1, lambda_reg=0.1)
        assert abs(step["decay_factor"] - 0.99) < 1e-9, "Factor de contracción incorrecto"
        assert abs(step["w_final"] - 9.90) < 1e-9, "Cálculo de decaimiento incorrecto"
        print(" [PASS] 19.2.1: Factor de decaimiento de pesos (1 - ηλ) = 0.99")

        # Test 4: Dropout Invertido
        sim = DropoutSimulator(keep_prob=0.8, seed=42)
        train_out, _ = sim.forward_train_inverted([2.0], mask=[1.0])
        test_out = sim.forward_test_inverted([2.0])
        assert abs(train_out[0] - 2.5) < 1e-4, f"Fallo en Dropout Invertido entrenamiento: {train_out[0]}"
        assert abs(test_out[0] - 2.0) < 1e-4, f"Fallo en Dropout Invertido prueba: {test_out[0]}"
        print(" [PASS] 19.3.1: Escalamiento exacto de Dropout Invertido: 2.0 / 0.8 = 2.5")

        # Test 5: Batch Normalization Micro-ejemplo 1 (z = [1, 2, 3, 6])
        bn1 = compute_batch_normalization([1.0, 2.0, 3.0, 6.0])
        assert abs(bn1["mu_B"] - 3.0) < 1e-6, "Fallo en media del lote 1"
        assert abs(bn1["var_B"] - 3.5) < 1e-6, "Fallo en varianza del lote 1"
        assert abs(bn1["u_normalized"][0] - (-1.0690)) < 0.01, f"Fallo en u[0]: {bn1['u_normalized'][0]}"
        assert abs(bn1["u_normalized"][1] - (-0.5345)) < 0.01, f"Fallo en u[1]: {bn1['u_normalized'][1]}"
        assert abs(bn1["u_normalized"][2] - 0.0) < 0.01, f"Fallo en u[2]: {bn1['u_normalized'][2]}"
        assert abs(bn1["u_normalized"][3] - 1.6036) < 0.01, f"Fallo en u[3]: {bn1['u_normalized'][3]}"
        print(" [PASS] 19.4.1: BatchNorm Lote [1, 2, 3, 6] -> μ=3.0, σ^2=3.5, u≈[-1.07,-0.53,0.0,1.60]")

        # Test 6: Batch Normalization Micro-ejemplo 2 (z = [2, 4, 6, 8])
        bn2 = compute_batch_normalization([2.0, 4.0, 6.0, 8.0])
        assert abs(bn2["mu_B"] - 5.0) < 1e-6, "Fallo en media del lote 2"
        assert abs(bn2["var_B"] - 5.0) < 1e-6, "Fallo en varianza del lote 2"
        assert abs(bn2["u_normalized"][0] - (-1.3416)) < 0.01, f"Fallo en u[0]: {bn2['u_normalized'][0]}"
        assert abs(bn2["u_normalized"][1] - (-0.4472)) < 0.01, f"Fallo en u[1]: {bn2['u_normalized'][1]}"
        assert abs(bn2["u_normalized"][2] - 0.4472) < 0.01, f"Fallo en u[2]: {bn2['u_normalized'][2]}"
        assert abs(bn2["u_normalized"][3] - 1.3416) < 0.01, f"Fallo en u[3]: {bn2['u_normalized'][3]}"
        print(" [PASS] 19.4.2: BatchNorm Lote [2, 4, 6, 8] -> μ=5.0, σ^2=5.0, u≈[-1.34,-0.45,0.45,1.34]")

        # Test 7: Gradient Clipping
        clipped, norm_val, was_cl = clip_gradients_norm([3.0, 4.0], max_norm=2.5)
        assert abs(norm_val - 5.0) < 1e-6, "Norma incorrecta"
        assert was_cl is True, "Debió activarse el recorte"
        assert abs(clipped[0] - 1.5) < 1e-4 and abs(clipped[1] - 2.0) < 1e-4, "Valores recortados erróneos"
        print(" [PASS] 19.4.3: Recorte de Gradiente por Norma L2")

    except AssertionError as e:
        print(f" [FAIL] ERROR EN LA ASERSION: {e}")
        all_passed = False
    except Exception as e:
        print(f" [FAIL] EXCEPCION INESPERADA: {e}")
        all_passed = False

    print("=" * 80)
    if all_passed:
        print(" >>> TODOS LOS TESTS DE CD19 PASARON EXITOSAMENTE (100% OK) <<<")
    else:
        print(" >>> HUBO FALLOS EN LA SUITE DE CD19 <<<")
    print("=" * 80)
    return all_passed

# ==============================================================================
# 8. MENÚ PRINCIPAL
# ==============================================================================

def main_menu():
    while True:
        print_banner()
        print(" Seleccione un módulo temático para explorar:")
        print(" [1] Módulo 19.1: Divergencias, Saturación Sigmoidal y Suavidad de la Red")
        print(" [2] Módulo 19.2: Regularización L2 y Decaimiento de Pesos (Weight Decay)")
        print(" [3] Módulo 19.3: Dropout y Dropout Invertido (Srivastava et al. 2014)")
        print(" [4] Módulo 19.4: Trucos del Oficio y Batch Normalization (Ioffe & Szegedy 2015)")
        print(" [A] Ejecutar Suite Completa de Verificación Numérica (--auto)")
        print(" [0] Regresar al Menú Principal")
        print("-" * 80)
        choice = input(" Ingrese una opción [0-4, A]: ").strip().upper()

        if choice == "1":
            demo_modulo_1()
        elif choice == "2":
            demo_modulo_2()
        elif choice == "3":
            demo_modulo_3()
        elif choice == "4":
            demo_modulo_4()
        elif choice == "A":
            run_auto_tests()
        elif choice == "0":
            print("\n Regresando al orquestador principal...")
            break
        else:
            print("\n Opción inválida. Intente de nuevo.")

        input("\n Presione ENTER para continuar...")

if __name__ == "__main__":
    if "--auto" in sys.argv:
        success = run_auto_tests()
        sys.exit(0 if success else 1)
    else:
        main_menu()
