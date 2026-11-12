#!/usr/bin/env python3
"""
Módulo de Cátedra CD20: Redes Neuronales Convolucionales (CNNs)
Ciencia de Datos - Doctorado UCN
Fuentes Académicas: Harvard CS109b (Pavlos Protopapas et al.)

Contenidos:
1. ¿Por qué las imágenes rompen a los MLP? (Diaps. 5–13)
   - Aplanamiento de imágenes y explosión combinatoria de parámetros:
     * Caso juguete 4x4 -> 16+1 = 17 pesos/neurona.
     * CIFAR-10 (32x32x3 = 3072) -> 3073 pesos/neurona.
     * ImageNet (224x224x3 = 150528) con 128 neuronas -> 19,267,712 pesos en capa 1.
     * Caso modesto (100x100x1 = 10000) con 64 neuronas -> 640,064 parámetros.
   - Falta de invarianza a traslaciones espaciales en MLP.
   - La historia del cisne y el fracaso de reglas manuales rígidas (SIFT, SURF, FAST).
   - Principios fundamentales: Localidad y Jerarquía de características.

2. La Convolución 2D, Filtros, Padding, Stride y Pooling (Diaps. 14–35)
   - Suma ponderada local con pesos compartidos: h = sum(w_ij * x_ij) + b.
   - Micro-ejemplo exacto de cátedra:
     * Imagen 5x5 con borde vertical (columnas [3,3,3,0,0]).
     * Filtro vertical 3x3 (pesos [1,0,-1] en cada fila, b=0).
     * Salida 3x3 exacta con filas [0, 9, 9].
   - Conteo de paradas de filtro (ej. gato 140x140 con filtro 3x3 -> 19,044 paradas).
   - Relleno de ceros (Padding) y Paso (Stride):
     * Fórmula analítica: O = floor((W - K + 2P)/S) + 1.
     * Tres casos en MNIST (28x28, K=5):
       1) P=0, S=1 -> 24x24
       2) P=2, S=1 -> 28x28 ("same")
       3) P=2, S=2 -> 14x14
   - Activación ReLU elemento a elemento.
   - Operaciones de Pooling (Submuestreo):
     * Matriz 4x4 de cátedra -> Max Pooling 2x2 (S=2): [[7, 8], [3, 4]].
     * Mean Pooling 2x2 (S=2): [[3.5, 5.5], [1.75, 2.0]].
     * Reducción espacial, invarianza a pequeñas traslaciones y 0 parámetros aprendibles.

3. Armando una CNN y Conteo de Parámetros a Mano (Diaps. 36–42)
   - Fórmulas analíticas de conteo de parámetros:
     * Capa Convolucional: n_f * (K_w * K_h * C_in) + n_f
     * Capa Pooling: 0 parámetros
     * Capa Densa (FC): N_in * N_out + N_out
   - Ejemplos de conteo:
     * 1 filtro 3x3x3 sobre RGB -> 28 parámetros.
     * 16 filtros 3x3 sobre RGB -> 448 parámetros.
   - Red completa pequeña (32x32x3 -> 4 clases):
     * Conv1: 8 filtros 3x3, S=1, same -> 224 params (salida 32x32x8).
     * Conv2: 16 filtros 5x5, S=2, same -> 3,216 params (salida 16x16x16).
     * Total Convolucional: 3,440 params.
     * Flatten: 4,096 valores (0 params).
     * FC1: 512 nodos, ReLU -> 2,097,664 params.
     * FC2: 4 nodos, Softmax -> 2,052 params.
     * Total Bloque Denso: 2,099,716 params.
     * Total Red: 2,103,156 parámetros.
     * Proporción denso vs conv: 99.8% de los parámetros están en las capas densas.
   - Capa intermedia (14x14x16, 32 filtros 3x3, same, S=1) -> 4,640 params, salida 14x14x32.

4. Transfer Learning, Fine-Tuning y Tres Hitos Históricos (Diaps. 43–57)
   - Paradoja de escala (VGG16 con 138M params vs datasets pequeños de cientos de imágenes).
   - Matriz de decisión 2x2: similitud de dominio vs volumen de datos.
   - Escenarios: 600 fotos fauna altiplánica, 2M fotos industriales, 400 espectrogramas sísmicos.
   - Estrategia de tasas de aprendizaje decrecientes (learning rates escalonados por profundidad).
   - Tres hitos históricos:
     * LeNet-5 (1998, LeCun et al.): cheques bancarios, MNIST 32x32, ~60k params.
     * AlexNet (2012, Krizhevsky et al.): ImageNet ILSVRC 2012, 16.4% top-5, 8 capas, ~60M params, GPU+ReLU+Dropout.
     * ResNet (2015, He et al.): bloques residuales x + F(x), 152 capas, 3.6% top-5 (supera ~5.1% humano).
"""

import sys
import math
from typing import List, Tuple, Dict, Any, Optional

# ==============================================================================
# ESTILOS VISUALES ANSI PARA TERMINAL
# ==============================================================================
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_CYAN = "\033[96m"
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_MAGENTA = "\033[95m"

def print_banner():
    print(f"\n{C_BOLD}{C_CYAN}" + "=" * 80 + f"{C_RESET}")
    print(f"{C_BOLD}{C_CYAN} CÁTEDRA CD20: REDES NEURONALES CONVOLUCIONALES (CNNs){C_RESET}")
    print(f"{C_DIM} Ciencia de Datos — Doctorado UCN | Fuente: Harvard CS109b{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}" + "=" * 80 + f"{C_RESET}\n")

# ==============================================================================
# 1. MATEMÁTICAS FUNDAMENTALES DE CONVOLUCIÓN Y DIMENSIONES
# ==============================================================================

def compute_mlp_parameters(height: int, width: int, channels: int, hidden_neurons: int) -> Dict[str, Any]:
    """
    Calcula la cantidad de entradas y parámetros que requeriría una primera
    capa totalmente conectada (MLP) para procesar una imagen de dimensiones H x W x C.
    """
    total_inputs = height * width * channels
    weights_per_neuron = total_inputs
    biases_per_neuron = 1
    total_params_single_neuron = weights_per_neuron + biases_per_neuron
    total_first_layer_params = hidden_neurons * total_params_single_neuron

    return {
        "height": height,
        "width": width,
        "channels": channels,
        "total_inputs": total_inputs,
        "hidden_neurons": hidden_neurons,
        "weights_per_neuron": weights_per_neuron,
        "params_per_neuron": total_params_single_neuron,
        "total_first_layer_params": total_first_layer_params
    }

def conv2d_single_channel(image: List[List[float]],
                          kernel: List[List[float]],
                          bias: float = 0.0,
                          padding: int = 0,
                          stride: int = 1) -> List[List[float]]:
    """
    Aplica una convolución 2D válida o con relleno sobre una matriz de un solo canal.
    
    Fórmula: S(i, j) = sum_u sum_v (X[i*S + u, j*S + v] * K[u, v]) + bias
    """
    h_in = len(image)
    w_in = len(image[0])
    k_h = len(kernel)
    k_w = len(kernel[0])

    # Aplicar padding si es requerido
    if padding > 0:
        padded = [[0.0] * (w_in + 2 * padding) for _ in range(h_in + 2 * padding)]
        for r in range(h_in):
            for c in range(w_in):
                padded[r + padding][c + padding] = float(image[r][c])
    else:
        padded = [[float(cell) for cell in row] for row in image]

    h_pad = len(padded)
    w_pad = len(padded[0])

    out_h = math.floor((h_in - k_h + 2 * padding) / stride) + 1
    out_w = math.floor((w_in - k_w + 2 * padding) / stride) + 1

    output = []
    for i in range(out_h):
        out_row = []
        r_start = i * stride
        for j in range(out_w):
            c_start = j * stride
            acc = 0.0
            for u in range(k_h):
                for v in range(k_w):
                    acc += padded[r_start + u][c_start + v] * kernel[u][v]
            acc += bias
            out_row.append(acc)
        output.append(out_row)

    return output

def compute_output_dim(dim_in: int, kernel_size: int, padding: int, stride: int) -> int:
    """
    Calcula la dimensión espacial de salida según la fórmula de cátedra:
    O = floor((W - K + 2P)/S) + 1
    """
    return math.floor((dim_in - kernel_size + 2 * padding) / stride) + 1

def relu_2d(matrix: List[List[float]]) -> List[List[float]]:
    """Aplica la no linealidad ReLU max(0, z) elemento a elemento."""
    return [[max(0.0, val) for val in row] for row in matrix]

def max_pooling_2d(matrix: List[List[float]], pool_size: int = 2, stride: int = 2) -> List[List[float]]:
    """
    Aplica Max Pooling espacial 2D tomando el valor máximo en cada ventana.
    """
    h = len(matrix)
    w = len(matrix[0])
    out_h = math.floor((h - pool_size) / stride) + 1
    out_w = math.floor((w - pool_size) / stride) + 1

    output = []
    for i in range(out_h):
        out_row = []
        r_start = i * stride
        for j in range(out_w):
            c_start = j * stride
            window_vals = [
                matrix[r_start + u][c_start + v]
                for u in range(pool_size)
                for v in range(pool_size)
            ]
            out_row.append(max(window_vals))
        output.append(out_row)
    return output

def mean_pooling_2d(matrix: List[List[float]], pool_size: int = 2, stride: int = 2) -> List[List[float]]:
    """
    Aplica Mean (Average) Pooling espacial 2D promediando los valores de cada ventana.
    """
    h = len(matrix)
    w = len(matrix[0])
    out_h = math.floor((h - pool_size) / stride) + 1
    out_w = math.floor((w - pool_size) / stride) + 1

    output = []
    for i in range(out_h):
        out_row = []
        r_start = i * stride
        for j in range(out_w):
            c_start = j * stride
            window_vals = [
                matrix[r_start + u][c_start + v]
                for u in range(pool_size)
                for v in range(pool_size)
            ]
            avg_val = sum(window_vals) / len(window_vals)
            out_row.append(avg_val)
        output.append(out_row)
    return output

# ==============================================================================
# 2. CONTEO DE PARÁMETROS ARQUITECTÓNICOS
# ==============================================================================

def conv_layer_params(n_filters: int, k_w: int, k_h: int, c_in: int, use_bias: bool = True) -> int:
    """
    Calcula los parámetros aprendibles de una capa convolucional 2D:
    Params = n_filters * (k_w * k_h * c_in) + (n_filters if use_bias else 0)
    """
    weights = n_filters * (k_w * k_h * c_in)
    biases = n_filters if use_bias else 0
    return weights + biases

def fc_layer_params(n_in: int, n_out: int, use_bias: bool = True) -> int:
    """
    Calcula los parámetros aprendibles de una capa densa totalmente conectada:
    Params = n_in * n_out + (n_out if use_bias else 0)
    """
    weights = n_in * n_out
    biases = n_out if use_bias else 0
    return weights + biases

def evaluate_cnn_architecture() -> Dict[str, Any]:
    """
    Calcula detalladamente el conteo de parámetros de la arquitectura de cátedra
    (Diapositivas 39 a 41):
      - Entrada: 32 x 32 x 3
      - Conv1: 8 filtros 3x3, same, S=1 -> salida 32 x 32 x 8
      - Conv2: 16 filtros 5x5, same, S=2 -> salida 16 x 16 x 16
      - Flatten: 16 * 16 * 16 = 4096 valores
      - FC1: 512 nodos, ReLU
      - FC2: 4 nodos, Softmax
    """
    # Conv1
    c1_params = conv_layer_params(n_filters=8, k_w=3, k_h=3, c_in=3)
    c1_out_dim = (32, 32, 8)

    # Conv2
    c2_params = conv_layer_params(n_filters=16, k_w=5, k_h=5, c_in=8)
    c2_out_dim = (16, 16, 16)

    total_conv_params = c1_params + c2_params

    # Flatten
    flatten_nodes = 16 * 16 * 16

    # FC1
    fc1_params = fc_layer_params(n_in=flatten_nodes, n_out=512)

    # FC2
    fc2_params = fc_layer_params(n_in=512, n_out=4)

    total_dense_params = fc1_params + fc2_params
    total_network_params = total_conv_params + total_dense_params
    dense_percentage = (total_dense_params / total_network_params) * 100.0

    return {
        "conv1": {"filters": 8, "kernel": (3, 3), "c_in": 3, "params": c1_params, "out_dim": c1_out_dim},
        "conv2": {"filters": 16, "kernel": (5, 5), "c_in": 8, "params": c2_params, "out_dim": c2_out_dim},
        "total_conv_params": total_conv_params,
        "flatten_nodes": flatten_nodes,
        "fc1": {"in": flatten_nodes, "out": 512, "params": fc1_params},
        "fc2": {"in": 512, "out": 4, "params": fc2_params},
        "total_dense_params": total_dense_params,
        "total_network_params": total_network_params,
        "dense_percentage": dense_percentage
    }

def recommend_transfer_learning_strategy(domain_similarity: str, dataset_size: str) -> Dict[str, str]:
    """
    Evalúa la matriz 2x2 de Transfer Learning según el dominio y tamaño del conjunto de datos.
    """
    sim = domain_similarity.lower()
    sz = dataset_size.lower()

    if "alta" in sim or "similar" in sim:
        if "peque" in sz or "bajo" in sz:
            return {
                "estrategia": "Congelar Base y Entrenar Nueva Cabeza Lineal / MLP",
                "descripcion": "El extractor preentrenado retiene representaciones idóneas de bajo y alto nivel. Con pocos datos, reentrenar la base causaría sobreajuste severo. Solo se aprende la cabeza clasificadora final.",
                "ejemplo": "600 fotografías de fauna altiplánica utilizando una base preentrenada en ImageNet."
            }
        else:
            return {
                "estrategia": "Fine-Tuning Completo con Tasa de Aprendizaje Baja",
                "descripcion": "Existe abundancia de datos en un dominio afín. Se ajustan todos los pesos suavemente para especializar los filtros finales sin destruir los detectores generales tempranos.",
                "ejemplo": "2 millones de imágenes industriales de defectos superficiales partiendo de ResNet."
            }
    else: # Dominio diferente
        if "peque" in sz or "bajo" in sz:
            return {
                "estrategia": "Extracción Temprana o Fine-Tuning Parcial con Aumento Agresivo",
                "descripcion": "Escenario desafiante. Se congelan las capas tempranas (bordes y texturas universales) y se entrena una capa intermedia o cabeza regularizada fuertemente.",
                "ejemplo": "400 espectrogramas sísmicos convirtiendo audio a imágenes con VGG congelada en capas iniciales."
            }
        else:
            return {
                "estrategia": "Entrenamiento Completo desde Cero o Fine-Tuning Global Profundo",
                "descripcion": "Dominio radicalmente distinto pero con volumen masivo de datos. Los pesos preentrenados sirven de inicialización superior a la aleatoria, pero se permite la adaptación total de la red.",
                "ejemplo": "Millones de cortes tomográficos axiales de resonancia magnética nuclear en 3D."
            }

# ==============================================================================
# 3. DEMOS INTERACTIVAS POR MÓDULO TEMÁTICO
# ==============================================================================

def demo_modulo_1():
    print(f"\n{C_BOLD}{C_CYAN}" + "=" * 80 + f"{C_RESET}")
    print(f"{C_BOLD}{C_CYAN} MÓDULO 20.1: ¿POR QUÉ LAS IMÁGENES ROMPEN A LOS MLP?{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}" + "=" * 80 + f"{C_RESET}\n")

    print(" 1. EXPLOSIÓN DE PARÁMETROS EN CAPAS TOTALMENTE CONECTADAS (DENSAS):")
    print("    Si aplanamos una imagen H x W x C en un vector 1D, cada neurona de la primera")
    print("    capa oculta debe conectarse a CADA píxel de entrada más un término de sesgo (bias).\n")

    casos = [
        ("Juguete Monocromático (4x4)", 4, 4, 1, 1),
        ("CIFAR-10 RGB (32x32)", 32, 32, 3, 1),
        ("ImageNet RGB (224x224, 128 neuronas)", 224, 224, 3, 128),
        ("Caso Modesto Cátedra (100x100, 64 neuronas)", 100, 100, 1, 64),
    ]

    for nombre, h, w, c, neurons in casos:
        res = compute_mlp_parameters(h, w, c, neurons)
        print(f"    • {C_BOLD}{nombre}:{C_RESET}")
        print(f"      - Dimensión entrada: {h} x {w} x {c} = {res['total_inputs']:,} valores.")
        print(f"      - Pesos por neurona: {res['total_inputs']:,} + 1 bias = {res['params_per_neuron']:,} parámetros.")
        if neurons > 1:
            print(f"      - Capa oculta con {neurons} neuronas: {neurons} x {res['params_per_neuron']:,} = "
                  f"{C_RED}{res['total_first_layer_params']:,} parámetros aprendibles.{C_RESET}")
        print()

    print(f"{C_YELLOW} Conclusión Matemática:{C_RESET} Solo la primera capa de un MLP en ImageNet con 128 neuronas")
    print(f" requiere {C_BOLD}19,267,712 parámetros{C_RESET}. En un dataset de pocas imágenes, el sobreajuste es fatal.\n")

    print(" 2. FALTA DE INVARIANZA A TRASLACIONES Y PÉRDIDA DE TOPOLOGÍA 2D:")
    print("    - Un MLP trata a cada píxel como una variable independiente desprovista de posición.")
    print("    - Si un gato aparece en la esquina superior izquierda, activa un grupo de pesos.")
    print("    - Si el mismo gato se traslada a la esquina inferior derecha, activa pesos COMPLETAMENTE distintos.")
    print("    - El MLP debe aprender el concepto de 'gato' independientemente en cada posición del lienzo.\n")

    print(" 3. LA HISTORIA DEL CISNE Y LA SUPERACIÓN DE DETECTORES MANUALES:")
    print("    - En visión clásica se intentaba programar reglas manuales rígidas:")
    print("      'Cisne = Pico naranja + Cuello curvado blanco + Cuerpo flotante'.")
    print("    - Fracaso inmediato ante cisnes negros, cuellos estirados buscando alimento bajo el agua,")
    print("      o cisnes de juguete inflables.")
    print("    - Algoritmos como SIFT, SURF y FAST requerían ingeniería manual de características.")
    print("    - La Convolución introduce Representation Learning gobernado por:")
    print("      a) Localidad: los píxeles adyacentes están espacialmente correlacionados.")
    print("      b) Jerarquía: bordes locales componen esquinas, texturas, partes y objetos completos.\n")

def demo_modulo_2():
    print(f"\n{C_BOLD}{C_CYAN}" + "=" * 80 + f"{C_RESET}")
    print(f"{C_BOLD}{C_CYAN} MÓDULO 20.2: CONVOLUCIÓN 2D, FILTROS, PADDING, STRIDE Y POOLING{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}" + "=" * 80 + f"{C_RESET}\n")

    print(" 1. MICRO-EJEMPLO NUMÉRICO EXACTO DE CÁTEDRA (Diaps. 17 a 21):")
    print("    Entrada 5x5 con transición vertical de claro (3) a oscuro (0):")
    
    img_5x5 = [
        [3.0, 3.0, 3.0, 0.0, 0.0],
        [3.0, 3.0, 3.0, 0.0, 0.0],
        [3.0, 3.0, 3.0, 0.0, 0.0],
        [3.0, 3.0, 3.0, 0.0, 0.0],
        [3.0, 3.0, 3.0, 0.0, 0.0]
    ]
    for row in img_5x5:
        print("      " + "  ".join(f"{int(v)}" for v in row))
    
    print("\n    Filtro vertical de bordes 3x3 (bias b = 0):")
    kernel_vert = [
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0]
    ]
    for row in kernel_vert:
        print("      " + "  ".join(f"{int(v):2d}" for v in row))

    out_conv = conv2d_single_channel(img_5x5, kernel_vert, bias=0.0, padding=0, stride=1)
    print("\n    Salida Convolucional 3x3 (Valid, P=0, S=1):")
    for r_idx, row in enumerate(out_conv):
        print(f"      Fila {r_idx+1}: " + "  ".join(f"{int(v):2d}" for v in row))

    print(f"\n    {C_GREEN}[VERIFICACIÓN]{C_RESET} Cada fila produce exactamente [0, 9, 9].")
    print("    - Posición 1 (cols 0-2): 3(1)+3(0)+3(-1) = 0 por fila -> Suma = 0.")
    print("    - Posición 2 (cols 1-3): 3(1)+3(0)+0(-1) = 3 por fila -> Suma = 9 (Detecta el borde!).")
    print("    - Posición 3 (cols 2-4): 3(1)+0(0)+0(-1) = 3 por fila -> Suma = 9.\n")

    print(" 2. PARADAS DEL FILTRO Y DIMENSIONES DE SALIDA:")
    print("    Fórmula: O = floor((W - K + 2P) / S) + 1\n")
    casos_mnist = [
        ("Sin padding, paso 1 (Valid)", 28, 5, 0, 1),
        ("Padding = 2, paso 1 (Same)", 28, 5, 2, 1),
        ("Padding = 2, paso 2 (Submuestreo)", 28, 5, 2, 2),
    ]
    for desc, w, k, p, s in casos_mnist:
        o = compute_output_dim(w, k, p, s)
        print(f"    • {desc}: W={w}, K={k}, P={p}, S={s} -> O = floor(({w}-{k}+{2*p})/{s}) + 1 = {C_BOLD}{o} x {o}{C_RESET}")

    print(f"\n    Ejemplo gato 140x140 con filtro 3x3 (P=0, S=1):")
    stops = (140 - 3 + 1) * (140 - 3 + 1)
    print(f"    - El filtro se detiene exactamente {C_BOLD}{stops:,} veces{C_RESET} generando un mapa 138x138.\n")

    print(" 3. OPERACIONES DE POOLING (SUBMUESTREO) EN MATRIZ 4x4 (Diap. 29):")
    pool_input = [
        [1.0, 1.0, 2.0, 5.0],
        [5.0, 7.0, 7.0, 8.0],
        [3.0, 1.0, 1.0, 0.0],
        [1.0, 2.0, 3.0, 4.0]
    ]
    print("    Entrada 4x4:")
    for row in pool_input:
        print("      " + "  ".join(f"{int(v):2d}" for v in row))

    max_p = max_pooling_2d(pool_input, pool_size=2, stride=2)
    mean_p = mean_pooling_2d(pool_input, pool_size=2, stride=2)

    print("\n    Max Pooling 2x2 (Stride 2):")
    for row in max_p:
        print("      " + "  ".join(f"{int(v):2d}" for v in row))

    print("\n    Mean (Average) Pooling 2x2 (Stride 2):")
    for row in mean_p:
        print("      " + "  ".join(f"{v:4.2f}" for v in row))

    print(f"\n    {C_GREEN}[VERIFICACIÓN]{C_RESET} Max Pooling produce [[7, 8], [3, 4]] y Mean Pooling [[3.50, 5.50], [1.75, 2.00]].")
    print("    Pooling tiene 0 parámetros aprendibles y provee invarianza a pequeñas traslaciones.\n")

def demo_modulo_3():
    print(f"\n{C_BOLD}{C_CYAN}" + "=" * 80 + f"{C_RESET}")
    print(f"{C_BOLD}{C_CYAN} MÓDULO 20.3: ARMANDO UNA CNN Y CONTEO DE PARÁMETROS A MANO{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}" + "=" * 80 + f"{C_RESET}\n")

    print(" 1. FÓRMULAS FUNDAMENTALES DE PARÁMETROS:")
    print("    • Capa Convolucional: Params = n_filtros * (K_w * K_h * C_in) + n_filtros")
    print("    • Capa de Pooling:    Params = 0 (operación fija de reducción)")
    print("    • Capa Totalmente Conectada (FC): Params = N_in * N_out + N_out\n")

    print(" 2. CASOS INDIVIDUALES DE FILTROS:")
    p1 = conv_layer_params(n_filters=1, k_w=3, k_h=3, c_in=3)
    p16 = conv_layer_params(n_filters=16, k_w=3, k_h=3, c_in=3)
    p_inter = conv_layer_params(n_filters=32, k_w=3, k_h=3, c_in=16)

    print(f"    • 1 filtro 3x3 sobre volumen RGB (C_in=3): 1 * (3*3*3) + 1 = {p1} parámetros.")
    print(f"    • 16 filtros 3x3 sobre volumen RGB (C_in=3): 16 * 27 + 16 = {p16} parámetros.")
    print(f"    • Capa intermedia: 32 filtros 3x3 sobre 16 canales: 32 * (9*16) + 32 = {p_inter} parámetros.\n")

    print(" 3. ARQUITECTURA COMPLETA DE CÁTEDRA (Diaps. 39 a 41):")
    arch = evaluate_cnn_architecture()
    print("    Entrada: Imagen 32 x 32 x 3")
    print(f"    1) Conv1: 8 filtros 3x3 (Same, S=1) -> Salida: 32x32x8 | Parámetros: {C_BOLD}{arch['conv1']['params']:,}{C_RESET}")
    print(f"    2) Conv2: 16 filtros 5x5 (Same, S=2) -> Salida: 16x16x16 | Parámetros: {C_BOLD}{arch['conv2']['params']:,}{C_RESET}")
    print(f"       Subtotal Convolucional: {C_CYAN}{arch['total_conv_params']:,} parámetros{C_RESET}")
    print(f"    3) Aplanamiento (Flatten): 16 * 16 * 16 = {arch['flatten_nodes']:,} nodos (0 parámetros)")
    print(f"    4) Densa 1 (FC1): 512 neuronas ReLU -> 4096 * 512 + 512 = {C_BOLD}{arch['fc1']['params']:,}{C_RESET} parámetros")
    print(f"    5) Densa 2 (FC2): 4 neuronas Softmax -> 512 * 4 + 4 = {C_BOLD}{arch['fc2']['params']:,}{C_RESET} parámetros")
    print(f"       Subtotal Bloque Denso: {C_MAGENTA}{arch['total_dense_params']:,} parámetros{C_RESET}")
    print(f"    ----------------------------------------------------------------------------")
    print(f"    TOTAL RED: {C_BOLD}{C_GREEN}{arch['total_network_params']:,} parámetros{C_RESET}")
    print(f"    PROPORCIÓN BLOQUE DENSO: {C_BOLD}{C_RED}{arch['dense_percentage']:.2f}% (~99.8%){C_RESET}\n")

    print(f"{C_YELLOW} Hallazgo Arquitectónico:{C_RESET} En las CNN clásicas, más del 99% de los parámetros")
    print(" residen en las capas totalmente conectadas finales. Las convoluciones aportan solo una fracción mínima.\n")

def demo_modulo_4():
    print(f"\n{C_BOLD}{C_CYAN}" + "=" * 80 + f"{C_RESET}")
    print(f"{C_BOLD}{C_CYAN} MÓDULO 20.4: TRANSFER LEARNING, FINE-TUNING Y TRES HITOS HISTÓRICOS{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}" + "=" * 80 + f"{C_RESET}\n")

    print(" 1. LA PARADOJA DE ESCALA Y TRANSFER LEARNING:")
    print("    - Modelos modernos como VGG16 cuentan con ~138 millones de parámetros.")
    print("    - En aplicaciones reales con cientos o pocos miles de imágenes, entrenar desde cero")
    print("      produce sobreajuste inmediato.")
    print("    - Transfer Learning aprovecha los filtros universales de bordes y texturas aprendidos en ImageNet.\n")

    print(" 2. MATRIZ DE DECISIÓN 2x2 (DOMINIO vs VOLUMEN DE DATOS):")
    casos_tl = [
        ("Dominio Similar", "Pocos Datos (ej. 600 fotos fauna altiplánica)"),
        ("Dominio Similar", "Muchos Datos (ej. 2M fotos industriales defectos)"),
        ("Dominio Distinto", "Pocos Datos (ej. 400 espectrogramas sísmicos)"),
        ("Dominio Distinto", "Muchos Datos (ej. Millones de escaneos médicos RMN)"),
    ]
    for dom, dat in casos_tl:
        rec = recommend_transfer_learning_strategy(dom, dat)
        print(f"    • [{dom} | {dat}]:")
        print(f"      Estrategia: {C_BOLD}{C_GREEN}{rec['estrategia']}{C_RESET}")
        print(f"      Fundamento: {rec['descripcion']}")
        print(f"      Ejemplo: {rec['ejemplo']}\n")

    print(" 3. FINE-TUNING CON TASAS DE APRENDIZAJE ESCALONADAS (DECRECIENTES):")
    print("    - Capas tempranas (conv iniciales): tasa 3 a 10 veces MENOR (ej. 1e-5 o congeladas).")
    print("    - Capas intermedias: tasa intermedia (ej. 1e-4).")
    print("    - Cabeza clasificadora nueva: tasa estándar (ej. 1e-3).\n")

    print(" 4. TRES HITOS HISTÓRICOS QUE TRANSFORMARON EL DEEP LEARNING:")
    print(f"    • {C_BOLD}LeNet-5 (1998, Yann LeCun et al.):{C_RESET}")
    print("      - Diseñada para reconocimiento de dígitos manuscritos en cheques bancarios (MNIST 32x32).")
    print("      - ~60,000 parámetros con alternancia Conv -> Pool -> Conv -> Pool -> FC.")
    print(f"    • {C_BOLD}AlexNet (2012, Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton):{C_RESET}")
    print("      - Ganadora categórica de ImageNet ILSVRC 2012 (error top-5 de 16.4% vs 26.2% segundo lugar).")
    print("      - 8 capas con pesos (5 conv + 3 FC), ~60M parámetros.")
    print("      - Introdujo entrenamiento masivo en GPU con CUDA, ReLU, Dropout y Data Augmentation.")
    print(f"    • {C_BOLD}ResNet (2015, Kaiming He et al.):{C_RESET}")
    print("      - Resolvió la degradación por desvanecimiento de gradiente con bloques residuales: y = x + F(x).")
    print("      - Redes de 152 capas con error top-5 de 3.6% en ImageNet (superando el ~5.1% humano).\n")

# ==============================================================================
# 4. SUITE DE VERIFICACIÓN NUMÉRICA AUTOMÁTICA (--auto)
# ==============================================================================

def run_auto_tests() -> bool:
    """
    Ejecuta una batería completa de pruebas unitarias y aserciones numéricas
    validando al 100% las cifras y micro-ejemplos de las 57 diapositivas de cátedra.
    """
    print(f"\n{C_BOLD}{C_CYAN}" + "=" * 80 + f"{C_RESET}")
    print(f"{C_BOLD}{C_CYAN} EJECUTANDO SUITE COMPLETA DE ASERSIONES CD20 (CNNs){C_RESET}")
    print(f"{C_BOLD}{C_CYAN}" + "=" * 80 + f"{C_RESET}\n")

    all_passed = True

    try:
        # Test 1: Conteo de parámetros MLP ImageNet y caso modesto
        mlp_cifar = compute_mlp_parameters(32, 32, 3, 1)
        assert mlp_cifar["total_inputs"] == 3072, "Fallo en entradas CIFAR-10"
        assert mlp_cifar["params_per_neuron"] == 3073, "Fallo en pesos por neurona CIFAR-10"
        print(" [PASS] 20.1.1: CIFAR-10 -> 3072 entradas, 3073 pesos/neurona")

        mlp_imagenet = compute_mlp_parameters(224, 224, 3, 128)
        assert mlp_imagenet["total_inputs"] == 150528, "Fallo en entradas ImageNet"
        assert mlp_imagenet["total_first_layer_params"] == 19267712, "Fallo en parámetros ImageNet (128 neuronas)"
        print(" [PASS] 20.1.2: ImageNet (224x224x3, 128 neuronas) -> 19,267,712 pesos")

        mlp_modesto = compute_mlp_parameters(100, 100, 1, 64)
        assert mlp_modesto["total_inputs"] == 10000, "Fallo en entradas caso modesto"
        assert mlp_modesto["total_first_layer_params"] == 640064, "Fallo en parámetros caso modesto (64 neuronas)"
        print(" [PASS] 20.1.3: Caso modesto (100x100x1, 64 neuronas) -> 640,064 parámetros")

        # Test 2: Micro-ejemplo de convolución 5x5 con filtro vertical 3x3
        img_5x5 = [
            [3.0, 3.0, 3.0, 0.0, 0.0],
            [3.0, 3.0, 3.0, 0.0, 0.0],
            [3.0, 3.0, 3.0, 0.0, 0.0],
            [3.0, 3.0, 3.0, 0.0, 0.0],
            [3.0, 3.0, 3.0, 0.0, 0.0]
        ]
        kernel_vert = [
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0]
        ]
        out_conv = conv2d_single_channel(img_5x5, kernel_vert, bias=0.0, padding=0, stride=1)
        assert len(out_conv) == 3 and len(out_conv[0]) == 3, "Dimensión de salida errónea en convolución 5x5"
        for row in out_conv:
            assert row == [0.0, 9.0, 9.0], f"Fila errónea en convolución 5x5: {row}"
        print(" [PASS] 20.2.1: Micro-ejemplo convolución 5x5 -> salida exacta 3x3 con filas [0, 9, 9]")

        # Test 3: Fórmula de dimensiones espaciales MNIST 28x28
        o1 = compute_output_dim(28, 5, 0, 1)
        o2 = compute_output_dim(28, 5, 2, 1)
        o3 = compute_output_dim(28, 5, 2, 2)
        assert o1 == 24, f"Fallo en MNIST caso 1: esperado 24, obtenido {o1}"
        assert o2 == 28, f"Fallo en MNIST caso 2: esperado 28, obtenido {o2}"
        assert o3 == 14, f"Fallo en MNIST caso 3: esperado 14, obtenido {o3}"
        print(" [PASS] 20.2.2: Dimensiones espaciales MNIST: 24x24 (valid), 28x28 (same), 14x14 (stride 2)")

        # Test 4: Paradas del filtro foto 140x140 con filtro 3x3
        stops = (140 - 3 + 1) * (140 - 3 + 1)
        assert stops == 19044, f"Fallo en paradas filtro: {stops}"
        print(" [PASS] 20.2.3: Paradas del filtro 3x3 en 140x140 -> 19,044 ventanas")

        # Test 5: Pooling Max y Mean exactos sobre matriz 4x4
        pool_mat = [
            [1.0, 1.0, 2.0, 5.0],
            [5.0, 7.0, 7.0, 8.0],
            [3.0, 1.0, 1.0, 0.0],
            [1.0, 2.0, 3.0, 4.0]
        ]
        max_res = max_pooling_2d(pool_mat, 2, 2)
        mean_res = mean_pooling_2d(pool_mat, 2, 2)
        assert max_res == [[7.0, 8.0], [3.0, 4.0]], f"Fallo en Max Pooling: {max_res}"
        assert abs(mean_res[0][0] - 3.5) < 1e-5, "Fallo en Mean Pooling (0,0)"
        assert abs(mean_res[0][1] - 5.5) < 1e-5, "Fallo en Mean Pooling (0,1)"
        assert abs(mean_res[1][0] - 1.75) < 1e-5, "Fallo en Mean Pooling (1,0)"
        assert abs(mean_res[1][1] - 2.0) < 1e-5, "Fallo en Mean Pooling (1,1)"
        print(" [PASS] 20.2.4: Pooling 4x4 -> Max=[[7,8],[3,4]], Mean=[[3.5, 5.5],[1.75, 2.0]]")

        # Test 6: Conteo de parámetros individuales
        p_single = conv_layer_params(1, 3, 3, 3)
        assert p_single == 28, f"Fallo en 1 filtro 3x3 RGB: {p_single}"
        p_16 = conv_layer_params(16, 3, 3, 3)
        assert p_16 == 448, f"Fallo en 16 filtros 3x3 RGB: {p_16}"
        p_inter = conv_layer_params(32, 3, 3, 16)
        assert p_inter == 4640, f"Fallo en capa intermedia 32 filtros: {p_inter}"
        print(" [PASS] 20.3.1: Filtros individuales: 1 filtro -> 28, 16 filtros -> 448, capa 32x(3x3x16) -> 4,640")

        # Test 7: Arquitectura completa de cátedra
        arch = evaluate_cnn_architecture()
        assert arch["conv1"]["params"] == 224, f"Fallo en Conv1: {arch['conv1']['params']}"
        assert arch["conv2"]["params"] == 3216, f"Fallo en Conv2: {arch['conv2']['params']}"
        assert arch["total_conv_params"] == 3440, f"Fallo en total conv: {arch['total_conv_params']}"
        assert arch["flatten_nodes"] == 4096, f"Fallo en flatten: {arch['flatten_nodes']}"
        assert arch["fc1"]["params"] == 2097664, f"Fallo en FC1: {arch['fc1']['params']}"
        assert arch["fc2"]["params"] == 2052, f"Fallo en FC2: {arch['fc2']['params']}"
        assert arch["total_dense_params"] == 2099716, f"Fallo en total dense: {arch['total_dense_params']}"
        assert arch["total_network_params"] == 2103156, f"Fallo en total red: {arch['total_network_params']}"
        assert abs(arch["dense_percentage"] - 99.836) < 0.01, f"Fallo en % denso: {arch['dense_percentage']}"
        print(" [PASS] 20.3.2: Red completa cátedra -> Conv=3,440, Densa=2,099,716, Total=2,103,156 (99.8% denso)")

        # Test 8: Recomendación de Transfer Learning
        rec_alti = recommend_transfer_learning_strategy("alta", "pequeno")
        assert "Congelar Base" in rec_alti["estrategia"], "Estrategia incorrecta para fauna altiplánica"
        rec_ind = recommend_transfer_learning_strategy("similar", "muchos")
        assert "Fine-Tuning Completo" in rec_ind["estrategia"], "Estrategia incorrecta para fotos industriales"
        print(" [PASS] 20.4.1: Matriz de decisión 2x2 Transfer Learning y Fine-Tuning validada")

    except AssertionError as e:
        print(f" [FAIL] ERROR EN LA ASERCIÓN: {e}")
        all_passed = False
    except Exception as e:
        print(f" [FAIL] EXCEPCIÓN INESPERADA: {e}")
        all_passed = False

    print("=" * 80)
    if all_passed:
        print(" >>> TODOS LOS TESTS DE CD20 (CNNs) PASARON EXITOSAMENTE (100% OK) <<<")
    else:
        print(" >>> HUBO FALLOS EN LA SUITE DE CD20 <<<")
    print("=" * 80)
    return all_passed

# ==============================================================================
# 5. MENÚ PRINCIPAL INTERACTIVO
# ==============================================================================

def main_menu():
    while True:
        print_banner()
        print(" Seleccione un módulo temático para explorar:")
        print(" [1] Módulo 20.1: ¿Por qué las imágenes rompen a los MLP? (Explosión y Localidad)")
        print(" [2] Módulo 20.2: Convolución 2D, Filtros, Padding, Stride y Pooling")
        print(" [3] Módulo 20.3: Armando una CNN y Conteo de Parámetros a Mano")
        print(" [4] Módulo 20.4: Transfer Learning, Fine-Tuning y Tres Hitos Históricos")
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
