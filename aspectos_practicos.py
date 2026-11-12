#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
CD11: ASPECTOS PRÁCTICOS Y EVALUACIÓN DE MODELOS
===============================================================================
Ciencia de Datos — Doctorado en Informática, Universidad Católica del Norte (UCN)
Cátedra de Cierre de Fase 1: Del Algoritmo al Oficio del Científico de Datos
Fuentes Académicas:
  - CMU 15-388 / 688: Practical Data Science (Zico Kolter)
  - Harvard CS109: Data Science (Pavlos Protopapas et al.)
  - CMU 11-785: Deep Learning (Bhiksha Raj et al.)

Temas Cubiertos:
  1. Debugging en Ciencia de Datos y la Ley de Kolter
  2. Datos Faltantes (MCAR, MAR, MNAR) y 5 Métodos de Imputación
  3. Evaluación de Clasificadores, Curva ROC y Caso de Triage en Pandemia
  4. Ética en IA, la Receta de 7 Pasos y Métricas Big Data (El COST de McSherry)

Implementación pura en Python estándar (sin librerías externas obligatorias).
===============================================================================
"""

import sys
import math
import random
from typing import List, Tuple, Dict, Any, Optional

# =============================================================================
# CONSTANTES PEDAGÓGICAS Y DATOS EXPERIMENTALES DE LA CÁTEDRA
# =============================================================================

# Datos de Discos Duros (Estudio Google Pinheiro et al. 2007 & Backblaze 2025)
GOOGLE_FAST_2007_SAMPLE = 100000
GOOGLE_CRITICAL_SMART_ZERO = 0.56     # > 56% de discos fallados con 0 en señales fuertes
GOOGLE_ALL_SMART_ZERO = 0.36          # > 36% con cero absoluto en todas las señales
BACKBLAZE_ANNUAL_FAILURE_RATE = 0.0136 # Tasa base anual 1.36% (2025)
BACKBLAZE_SMART_PRECURSOR = 0.767      # 76.7% muestra al menos 1 señal sobre 0

# Dataset de 8 observaciones para Métodos de Imputación (Diaps. 34-43)
IMPUTATION_OBSERVED_DATA = [
    (1.0, 1.8),
    (2.0, 2.3),
    (3.0, 2.4),
    (4.0, 3.9),
    (5.0, 4.1),
    (6.0, 5.2),
    (7.0, 5.0),
    (8.0, 6.6)
]
IMPUTATION_MISSING_POINTS = [3.5, 6.5]
REGRESSION_BETA_0 = 1.12
REGRESSION_BETA_1 = 0.62
REGRESSION_MSE = 0.11
REGRESSION_SIGMA = math.sqrt(REGRESSION_MSE)  # ~ 0.3316

# Dataset de Triage en Pandemia (251 pacientes reservados: 134 Alta, 117 Baja)
TRIAGE_TOTAL_PATIENTS = 251
TRIAGE_HIGH_URGENCY = 134
TRIAGE_LOW_URGENCY = 117

# Matrices de Confusión a Umbral 0.5
# Formato: [ [TN, FP], [FN, TP] ] donde Positivo = Alta Urgencia, Negativo = Baja Urgencia
CONFUSION_LOGISTIC_05 = {"TN": 58, "FP": 59, "FN": 37, "TP": 97}
CONFUSION_KNN_05 = {"TN": 55, "FP": 62, "FN": 33, "TP": 101}

# Matrices a Umbrales Alternativos para Regresión Logística
CONFUSION_LOGISTIC_LOW = {"TN": 55, "FP": 62, "FN": 35, "TP": 99}   # Umbral bajo (~0.4)
CONFUSION_LOGISTIC_HIGH = {"TN": 61, "FP": 56, "FN": 39, "TP": 95}  # Umbral alto (~0.6)

# Benchmark McSherry et al. 2015 (PageRank en Grafo de Twitter, 20 iteraciones)
BENCHMARK_MCSHERRY_TWITTER = {
    "Spark (128 cores)": 857.0,
    "GraphX (128 cores)": 419.0,
    "GraphLab (128 cores)": 249.0,
    "Laptop 1 hilo (RAM)": 275.0
}


# =============================================================================
# TEMA 11.1: DEBUGGING EN CIENCIA DE DATOS Y LA LEY DE KOLTER
# =============================================================================

class KolterDebugger:
    """
    Implementación del marco de depuración metodológica de Zico Kolter (CMU 15-388/688).
    Evalúa la factibilidad de la tarea, la presencia de señal empírica, el proxy humano
    y el árbol de diagnósticos de error.
    """

    @staticmethod
    def kolter_law_quote() -> str:
        return (
            "«Lo que distingue a los expertos en ciencia de datos del resto no es lo que "
            "hacen primero, sino lo que hacen después, cuando lo primero no funciona.»\n"
            "— Zico Kolter (Carnegie Mellon University, 15-388/688)"
        )

    @staticmethod
    def evaluate_kolter_quadrant(human_ability: bool, ml_ability: bool) -> Dict[str, str]:
        """
        Matriz 2x2 de Kolter:
          - Humano Bien / ML Bien -> Éxito estándar (mantener escepticismo).
          - Humano Bien / ML Mal  -> Problema Factible: depurar modelo, optimización o pérdida.
          - Humano Mal  / ML Bien -> Sospecha Severa: Data leakage o trampa de tasa base.
          - Humano Mal  / ML Mal  -> Problema Imposible: depurar el problema (cambiar entrada/salida).
        """
        if human_ability and ml_ability:
            return {
                "cuadrante": "Tú bien / ML bien",
                "diagnostico": "Éxito aparente.",
                "accion": "Mantener escepticismo saludable. Verificar generalización en split ciego y estabilidad temporal."
            }
        elif human_ability and not ml_ability:
            return {
                "cuadrante": "Tú bien / ML mal",
                "diagnostico": "Problema FACTIBLE (la señal existe físicamente en las entradas).",
                "accion": (
                    "Depurar algoritmo con diagnósticos:\n"
                    "  1. Diagnóstico Sesgo/Varianza: graficar error train vs test variando N.\n"
                    "  2. Diagnóstico de Optimización: ¿la pérdida sigue bajando o hizo meseta?\n"
                    "  3. Diagnóstico de Pérdida: verificar si la pérdida matemática penaliza el error crítico del negocio."
                )
            }
        elif not human_ability and ml_ability:
            return {
                "cuadrante": "Tú mal / ML bien",
                "diagnostico": "ALERTA CRÍTICA: Rendimiento sospechoso sobrehumano.",
                "accion": (
                    "Verificar Data Leakage (fuga de datos):\n"
                    "  - ¿Alguna columna fue generada post-evento? (ej. fecha de recambio de disco).\n"
                    "  - ¿El modelo aprendió un atajo trivial por clase desbalanceada (decir siempre 0)?\n"
                    "  - ¿Se calculó la media o escalado usando el conjunto de prueba?"
                )
            }
        else:
            return {
                "cuadrante": "Tú mal / ML mal",
                "diagnostico": "Problema IMPOSIBLE (la señal no está disponible en las entradas actuales).",
                "accion": (
                    "Depurar el PROBLEMA, no el modelo:\n"
                    "  A. Cambiar la ENTRADA: incorporar nuevas fuentes físicas (ej. acelerómetro/vibración).\n"
                    "  B. Cambiar la SALIDA: simplificar objetivo (en vez de 'fallará mañana', predecir 'riesgo relativo')."
                )
            }

    @staticmethod
    def simulate_hard_drive_smart_proxy(num_samples: int = 10, seed: int = 42) -> List[Dict[str, Any]]:
        """
        Simula una muestra de telemetría SMART con atributos clave:
          - SMART 5: Reallocated Sectors Count
          - SMART 187: Reported Uncorrectable Errors
          - SMART 188: Command Timeout
          - SMART 197: Current Pending Sector Count
          - SMART 198: Offline Uncorrectable Sector Count
        Ilustra por qué el problema es semi-imposible para un humano y el modelo.
        """
        rng = random.Random(seed)
        samples = []
        for i in range(1, num_samples + 1):
            # Tasa real de falla 1.36% (ampliada artificialmente para demostración didáctica)
            will_fail = (i in [3, 7])
            if will_fail:
                # 56% de los fallados tienen 0 en SMART crítico (Google 2007)
                has_signal = rng.random() > GOOGLE_CRITICAL_SMART_ZERO
                if has_signal:
                    s5 = rng.randint(1, 45)
                    s187 = rng.randint(0, 12)
                    s197 = rng.randint(1, 8)
                else:
                    s5, s187, s197 = 0, 0, 0
            else:
                # Discos sanos a veces tienen reallocated sectors inocuos
                s5 = rng.randint(0, 3) if rng.random() > 0.8 else 0
                s187 = 0
                s197 = 0

            samples.append({
                "id": f"HD-{i:03d}",
                "smart_5": s5,
                "smart_187": s187,
                "smart_197": s197,
                "will_fail": will_fail
            })
        return samples


# =============================================================================
# TEMA 11.2: DATOS FALTANTES (MISSING DATA) Y MÉTODOS DE IMPUTACIÓN
# =============================================================================

class MissingDataImputer:
    """
    Taxonomía de Rubin (1976) y algoritmos de imputación:
      1. Imputación por Media/Mediana
      2. k-Nearest Neighbors (kNN) Imputer
      3. Imputación por Regresión Lineal
      4. Imputación con Incertidumbre Estocástica
      5. Indicador de Ausencia (Missing Indicator)
    """

    @staticmethod
    def rubin_taxonomy_explanation() -> Dict[str, str]:
        return {
            "MCAR": (
                "Missing Completely at Random: La probabilidad de ausencia es idéntica "
                "para todas las observaciones. No depende de valores observados ni ocultos.\n"
                "Impacto: Coeficientes beta insesgados, pero pérdida de potencia estadística."
            ),
            "MAR": (
                "Missing at Random: La ausencia depende de variables observadas (otros predictores), "
                "pero no del valor ausente mismo una vez condicionado.\n"
                "Impacto: Sesgo si se ignora con dropna(); corregible modelando e imputando desde observados."
            ),
            "MNAR": (
                "Missing Not at Random: La ausencia depende de información oculta o del mismo valor ausente "
                "(ej. sensor que se apaga al recalentarse; ingresos altos que rehúsan responder).\n"
                "Impacto: Introduce sesgo sistemático severo. La indicadora de ausencia X_miss es obligatoria."
            ),
            "Incomprobabilidad": (
                "Teorema empírico: No es posible demostrar si los datos son MAR o MNAR examinando "
                "únicamente los datos observados, pues las variables que condicionan MNAR no fueron registradas."
            )
        }

    @staticmethod
    def compute_missing_indicator_matrix(data: List[List[Optional[float]]]) -> Tuple[List[List[float]], List[List[int]]]:
        """
        Para cada predictor con ausencias, genera la matriz con valor base
        y la matriz de indicadores binarios X_miss.
        """
        filled = []
        indicators = []
        for row in data:
            f_row = []
            i_row = []
            for val in row:
                if val is None or math.isnan(val):
                    f_row.append(0.0)
                    i_row.append(1)
                else:
                    f_row.append(float(val))
                    i_row.append(0)
            filled.append(f_row)
            indicators.append(i_row)
        return filled, indicators

    @staticmethod
    def mean_imputation(observed: List[Tuple[float, float]], missing_x1: List[float]) -> Dict[float, float]:
        """
        Método 1: Imputación por la media de x2.
        Suma de x2 = 31.3 en 8 observaciones -> media = 3.9125 (~3.91).
        """
        vals_x2 = [y for _, y in observed]
        mean_val = sum(vals_x2) / len(vals_x2)
        return {x: round(mean_val, 2) for x in missing_x1}

    @staticmethod
    def knn_imputation(observed: List[Tuple[float, float]], missing_x1: List[float], k: int = 2) -> Dict[float, float]:
        """
        Método 2: Imputación por k-Vecinos más cercanos (k=2).
        Para x1=3.5: vecinos x1=3 (y=2.4) y x1=4 (y=3.9) -> (2.4 + 3.9)/2 = 3.15.
        Para x1=6.5: vecinos x1=6 (y=5.2) y x1=7 (y=5.0) -> (5.2 + 5.0)/2 = 5.10.
        """
        results = {}
        for target_x in missing_x1:
            # Ordenar por distancia euclidiana en x1
            sorted_obs = sorted(observed, key=lambda pt: abs(pt[0] - target_x))
            k_neighbors = sorted_obs[:k]
            pred = sum(pt[1] for pt in k_neighbors) / k
            results[target_x] = round(pred, 2)
        return results

    @staticmethod
    def regression_imputation(missing_x1: List[float], beta_0: float = 1.12, beta_1: float = 0.62) -> Dict[float, float]:
        """
        Método 3: Imputación por regresión lineal determinista:
          y_hat = 1.12 + 0.62 * x1
          x1 = 3.5 -> 1.12 + 0.62*3.5 = 3.29
          x1 = 6.5 -> 1.12 + 0.62*6.5 = 5.15
        """
        return {x: round(beta_0 + beta_1 * x, 2) for x in missing_x1}

    @staticmethod
    def stochastic_regression_imputation(
        missing_x1: List[float],
        beta_0: float = 1.12,
        beta_1: float = 0.62,
        mse: float = 0.11,
        seed: int = 42
    ) -> Dict[float, Tuple[float, float]]:
        """
        Método 4: Imputación con incertidumbre (estocástica):
          y_hat + epsilon, con epsilon ~ N(0, MSE)
          sigma = sqrt(0.11) ~ 0.33
          Devuelve tupla: (predicción determinista, predicción con ruido estocástico).
        """
        rng = random.Random(seed)
        sigma = math.sqrt(mse)
        results = {}
        for x in missing_x1:
            det = round(beta_0 + beta_1 * x, 2)
            # Aproximación Box-Muller o gaussiana estándar
            noise = rng.gauss(0.0, sigma)
            stoch = round(det + noise, 2)
            results[x] = (det, stoch)
        return results


# =============================================================================
# TEMA 11.3: EVALUACIÓN DE CLASIFICADORES Y CASO DE TRIAGE EN PANDEMIA
# =============================================================================

class TriageEvaluator:
    """
    Evaluación de clasificadores en contexto de triage médico (Harvard CS109 / UCN).
    Calcula Accuracy, Recall/Sensibilidad, Especificidad, FPR, Precisión, F1-Score,
    y optimiza el umbral según restricciones geopolíticas (Brasil, Alemania, India).
    """

    @staticmethod
    def compute_metrics(conf: Dict[str, int]) -> Dict[str, float]:
        """
        Cálculo riguroso de todas las métricas de la matriz de confusión:
          TN: Verdaderos Negativos
          FP: Falsos Positivos
          FN: Falsos Negativos (Error crítico: paciente grave enviado a casa)
          TP: Verdaderos Positivos
        """
        tn = conf["TN"]
        fp = conf["FP"]
        fn = conf["FN"]
        tp = conf["TP"]
        total = tn + fp + fn + tp

        accuracy = (tp + tn) / total if total > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0           # Sensibilidad / TPR
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0     # TNR
        fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0              # 1 - Especificidad
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0        # PPV
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": round(accuracy, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "fpr": round(fpr, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "total_admitted": tp + fp
        }

    @staticmethod
    def compare_logistic_vs_knn() -> Dict[str, Any]:
        """
        Compara los dos modelos de la cátedra con 251 pacientes a umbral 0.5.
        Regresión Logística: [58, 59; 37, 97]
        kNN:                 [55, 62; 33, 101]
        """
        metrics_log = TriageEvaluator.compute_metrics(CONFUSION_LOGISTIC_05)
        metrics_knn = TriageEvaluator.compute_metrics(CONFUSION_KNN_05)

        return {
            "logistic": {
                "matrix": CONFUSION_LOGISTIC_05,
                "metrics": metrics_log
            },
            "knn": {
                "matrix": CONFUSION_KNN_05,
                "metrics": metrics_knn
            }
        }

    @staticmethod
    def evaluate_country_scenarios() -> Dict[str, Dict[str, Any]]:
        """
        Los 3 escenarios de la cátedra para la elección de clasificador y umbral:
          1. Brasil: Restricción política de colapso visual -> TPR + FPR <= 0.5.
             Dominante: Regresión Logística con umbral alto en (FPR=0.15, TPR=0.35).
          2. Alemania: Prioridad de letalidad mínima -> TPR >= 0.85.
             Dominante: Regresión Logística con umbral bajo en (FPR=0.75, TPR=0.86).
          3. India: Capacidad hospitalaria estricta (máx 50% de admisiones) -> TPR + FPR <= 1.0.
             Dominante: kNN con umbral moderado en (FPR=0.38, TPR=0.62).
        """
        return {
            "Brasil": {
                "criterio": "Evitar saturación visible y colapso de camas (TPR + FPR <= 0.5)",
                "modelo_elegido": "Regresión Logística (Umbral Alto)",
                "punto_roc": {"fpr": 0.15, "tpr": 0.35},
                "suma_tasas": 0.50,
                "justificacion": "Prioriza alta especificidad (pocos falsos positivos) sacrificando recall de casos leves."
            },
            "Alemania": {
                "criterio": "Minimizar muertes domiciliarias prevenibles (TPR >= 0.85)",
                "modelo_elegido": "Regresión Logística (Umbral Bajo)",
                "punto_roc": {"fpr": 0.75, "tpr": 0.86},
                "suma_tasas": 1.61,
                "justificacion": "Admite con gran holgura a costa de admitir muchos no-urgentes (alta sensibilidad)."
            },
            "India": {
                "criterio": "Camas estrictamente contadas: admitir a lo más la mitad (TPR + FPR <= 1.0)",
                "modelo_elegido": "kNN (Umbral Moderado)",
                "punto_roc": {"fpr": 0.38, "tpr": 0.62},
                "suma_tasas": 1.00,
                "justificacion": "kNN supera a Logística en la diagonal secundaria, equilibrando falsas alarmas y detección."
            }
        }


# =============================================================================
# TEMA 11.4: ÉTICA EN IA, RECETA DE 7 PASOS Y BIG DATA (EL COST DE McSHERRY)
# =============================================================================

class EngineeringAndEthics:
    """
    Principios de ingeniería responsable, prevención de sesgo por subgrupos,
    receta metódica de modelado y análisis crítico de escalabilidad (COST).
    """

    @staticmethod
    def get_seven_steps_recipe() -> List[Dict[str, str]]:
        return [
            {"paso": 1, "nombre": "Datos y Representación", "descripcion": "Limpieza, codificación de variables y división estricta train/val/test antes de cualquier imputación."},
            {"paso": 2, "nombre": "Elección del Modelo", "descripcion": "Seleccionar familia inductiva según interpretabilidad requerida y cantidad de datos disponible."},
            {"paso": 3, "nombre": "Función de Pérdida", "descripcion": "Alinear la pérdida matemática (ej. log-loss, focal loss) con la asimetría de costos del dominio de negocio."},
            {"paso": 4, "nombre": "Regularización y Heurísticas", "descripcion": "Aplicar L1/L2, dropout o early stopping para penalizar la complejidad innecesaria."},
            {"paso": 5, "nombre": "Optimizador", "descripcion": "Configurar SGD, Adam o L-BFGS con política de tasa de aprendizaje adaptativa."},
            {"paso": 6, "nombre": "Búsqueda de Hiperparámetros", "descripcion": "Optimización sobre conjunto de validación o validación cruzada interna (NUNCA en el test set)."},
            {"paso": 7, "nombre": "Evaluación Final Única", "descripcion": "Evaluar una única vez sobre el conjunto de prueba reservado y reportar métricas desagregadas por subgrupo."}
        ]

    @staticmethod
    def subpopulation_audit_example() -> Dict[str, Any]:
        """
        Ejemplo didáctico: Un clasificador con Accuracy global aparente del 94.6%
        que oculta una tasa de error inaceptable del 30% en una subpoblación minoritaria.
        """
        grupo_mayoritario = {"tamano": 900, "correctos": 890, "accuracy": 890 / 900}  # ~98.88%
        grupo_minoritario = {"tamano": 100, "correctos": 70, "accuracy": 70 / 100}     # 70.0%
        total = 1000
        acc_global = (890 + 70) / total

        return {
            "accuracy_global": round(acc_global, 4),
            "mayoria": {"n": 900, "acc": round(grupo_mayoritario["accuracy"], 4)},
            "minoria": {"n": 100, "acc": round(grupo_minoritario["accuracy"], 4)},
            "conclusion": (
                "Optimizar únicamente el promedio global oculta la opresión de minorías. "
                "La métrica ética obligatoria es la pérdida minimax: el peor desempeño entre todos los subgrupos."
            )
        }

    @staticmethod
    def evaluate_cost_metric() -> Dict[str, Any]:
        """
        Métrica COST (Configuration that Outperforms a Single Thread)
        Frank McSherry, Michael Isard & Derek G. Murray (HotOS 2015):
        'Scalability! But at what COST?'
        """
        return {
            "definicion": "COST: El hardware mínimo necesario para que un sistema distribuido supere a un hilo único optimizado en RAM.",
            "twitter_pagerank_benchmark": BENCHMARK_MCSHERRY_TWITTER,
            "conclusion": (
                "Un solo hilo de laptop en RAM completó PageRank en 275s, superando a Spark con 128 núcleos (857s). "
                "Antes de rentar un clúster de cómputo en la nube, asegúrate de que el problema no cabe en la RAM local "
                "o que tu pipeline distribuido realmente supera el costo de serialización y red."
            )
        }


# =============================================================================
# INTERFAZ DE LÍNEA DE COMANDOS (CLI) Y SUITE DE PRUEBAS
# =============================================================================

def print_header(title: str):
    print("\n" + "=" * 78)
    print(f" {title.upper()}")
    print("=" * 78)

def run_debugging_demo():
    print_header("11.1: Debugging en Ciencia de Datos y la Ley de Kolter")
    print(KolterDebugger.kolter_law_quote())
    print("\n[A] Matriz 2x2 de Kolter (Ejemplo: Problema Factible vs Imposible):")
    for hum, ml in [(True, False), (False, True), (False, False), (True, True)]:
        q = KolterDebugger.evaluate_kolter_quadrant(hum, ml)
        print(f"\n  • Caso ({q['cuadrante']}):")
        print(f"    Diagnóstico : {q['diagnostico']}")
        print(f"    Acción      : {q['accion']}")

    print("\n[B] Telemetría SMART de Discos Duros (Google FAST 2007 & Backblaze 2025):")
    print(f"  - Muestra simulada con tasa base Backblaze ~{BACKBLAZE_ANNUAL_FAILURE_RATE*100:.2f}%:")
    samples = KolterDebugger.simulate_hard_drive_smart_proxy(6)
    print("    ID      | SMART 5 | SMART 187 | SMART 197 | Fallará?")
    print("    " + "-" * 50)
    for s in samples:
        f_str = "SÍ (Falla)" if s["will_fail"] else "No"
        print(f"    {s['id']}  | {s['smart_5']:7d} | {s['smart_187']:9d} | {s['smart_197']:9d} | {f_str}")
    print(f"  * Hallazgo Google: {GOOGLE_CRITICAL_SMART_ZERO*100:.0f}% de discos con falla tenían SMART en cero absoluto.")

def run_imputation_demo():
    print_header("11.2: Datos Faltantes (Rubin 1976) y 5 Métodos de Imputación")
    tax = MissingDataImputer.rubin_taxonomy_explanation()
    for cat in ["MCAR", "MAR", "MNAR", "Incomprobabilidad"]:
        print(f"  [{cat}]: {tax[cat]}\n")

    print("[A] Puntos observados (Diap. 34):")
    obs_str = ", ".join([f"({x}, {y})" for x, y in IMPUTATION_OBSERVED_DATA])
    print(f"    {obs_str}")
    print(f"    Puntos faltantes en x1 = 3.5 y x1 = 6.5")

    print("\n[B] Comparación de 4 Métodos Numéricos:")
    mean_res = MissingDataImputer.mean_imputation(IMPUTATION_OBSERVED_DATA, IMPUTATION_MISSING_POINTS)
    knn_res = MissingDataImputer.knn_imputation(IMPUTATION_OBSERVED_DATA, IMPUTATION_MISSING_POINTS, k=2)
    reg_res = MissingDataImputer.regression_imputation(IMPUTATION_MISSING_POINTS)
    stoch_res = MissingDataImputer.stochastic_regression_imputation(IMPUTATION_MISSING_POINTS)

    print(f"    1. Media global (mean=3.91)   : x=3.5 -> {mean_res[3.5]:.2f} | x=6.5 -> {mean_res[6.5]:.2f}")
    print(f"    2. kNN (k=2)                  : x=3.5 -> {knn_res[3.5]:.2f} | x=6.5 -> {knn_res[6.5]:.2f}")
    print(f"    3. Regresión (1.12 + 0.62*x)  : x=3.5 -> {reg_res[3.5]:.2f} | x=6.5 -> {reg_res[6.5]:.2f}")
    print(f"    4. Regresión Estocástica      : x=3.5 -> {stoch_res[3.5][1]:.2f} (det={stoch_res[3.5][0]:.2f}) | x=6.5 -> {stoch_res[6.5][1]:.2f} (det={stoch_res[6.5][0]:.2f})")

def run_triage_demo():
    print_header("11.3: Evaluación de Triage en Pandemia (251 Pacientes Reservados)")
    comp = TriageEvaluator.compare_logistic_vs_knn()

    print("  [A] Matriz de Confusión y Métricas (Umbral = 0.5):")
    print("  " + "-" * 70)
    print("  Métrica                | Regresión Logística       | Clasificador kNN")
    print("  " + "-" * 70)
    m_log = comp["logistic"]["metrics"]
    m_knn = comp["knn"]["metrics"]
    print(f"  Verdaderos Neg (TN)    | {CONFUSION_LOGISTIC_05['TN']:<25} | {CONFUSION_KNN_05['TN']:<20}")
    print(f"  Falsos Pos (FP)        | {CONFUSION_LOGISTIC_05['FP']:<25} | {CONFUSION_KNN_05['FP']:<20}")
    print(f"  Falsos Neg (FN - Crítico)| {CONFUSION_LOGISTIC_05['FN']:<23} | {CONFUSION_KNN_05['FN']:<20}")
    print(f"  Verdaderos Pos (TP)    | {CONFUSION_LOGISTIC_05['TP']:<25} | {CONFUSION_KNN_05['TP']:<20}")
    print("  " + "-" * 70)
    print(f"  Accuracy (Exactitud)   | {m_log['accuracy']*100:.2f}% (155/251)         | {m_knn['accuracy']*100:.2f}% (156/251)")
    print(f"  Recall (Sensibilidad)  | {m_log['recall']*100:.2f}%  (97/134)          | {m_knn['recall']*100:.2f}%  (101/134)")
    print(f"  Especificidad (TNR)    | {m_log['specificity']*100:.2f}%  (58/117)          | {m_knn['specificity']*100:.2f}%  (55/117)")
    print(f"  FPR (1 - Especif.)     | {m_log['fpr']*100:.2f}%  (59/117)          | {m_knn['fpr']*100:.2f}%  (62/117)")
    print(f"  Precision              | {m_log['precision']*100:.2f}%  (97/156)          | {m_knn['precision']*100:.2f}%  (101/163)")
    print(f"  F1-Score               | {m_log['f1']:.4f}                    | {m_knn['f1']:.4f}")
    print("  " + "-" * 70)

    print("\n  [B] Escenarios de Decisión y Curva ROC por País:")
    scenarios = TriageEvaluator.evaluate_country_scenarios()
    for country, sc in scenarios.items():
        print(f"\n  • Caso {country}:")
        print(f"    Criterio       : {sc['criterio']}")
        print(f"    Modelo Óptimo  : {sc['modelo_elegido']}")
        print(f"    Punto ROC (FPR, TPR): ({sc['punto_roc']['fpr']:.2f}, {sc['punto_roc']['tpr']:.2f})")
        print(f"    Justificación  : {sc['justificacion']}")

def run_ethics_and_cost_demo():
    print_header("11.4: Ética, Receta de 7 Pasos y Métricas Big Data (COST)")
    print("[A] Auditoría Desagregada por Subgrupos (El Peligro del Promedio Global):")
    audit = EngineeringAndEthics.subpopulation_audit_example()
    print(f"  - Accuracy Global: {audit['accuracy_global']*100:.2f}%")
    print(f"  - Subgrupo Mayoritario (n={audit['mayoria']['n']}): Accuracy = {audit['mayoria']['acc']*100:.2f}%")
    print(f"  - Subgrupo Minoritario  (n={audit['minoria']['n']}): Accuracy = {audit['minoria']['acc']*100:.2f}%")
    print(f"  * Conclusión: {audit['conclusion']}")

    print("\n[B] Benchmark de Escalabilidad: McSherry et al. (HotOS 2015):")
    cost_info = EngineeringAndEthics.evaluate_cost_metric()
    print(f"  Definición: {cost_info['definicion']}")
    print("  Resultados en Twitter PageRank (20 iteraciones):")
    for system, runtime in cost_info["twitter_pagerank_benchmark"].items():
        print(f"    - {system:<22}: {runtime:6.1f} s")
    print(f"\n  * Lección: {cost_info['conclusion']}")

def run_all_tests() -> bool:
    """Suite de validación automática que verifica cada valor numérico de la cátedra."""
    print("=" * 78)
    print(" EJECUTANDO SUITE DE TESTS AUTOMÁTICOS: CD11 ASPECTOS PRÁCTICOS")
    print("=" * 78)

    # Test 1: Datos de imputación exactos
    mean_res = MissingDataImputer.mean_imputation(IMPUTATION_OBSERVED_DATA, IMPUTATION_MISSING_POINTS)
    assert mean_res[3.5] == 3.91, f"Error en media 3.5: {mean_res[3.5]}"
    assert mean_res[6.5] == 3.91, f"Error en media 6.5: {mean_res[6.5]}"

    knn_res = MissingDataImputer.knn_imputation(IMPUTATION_OBSERVED_DATA, IMPUTATION_MISSING_POINTS, k=2)
    assert knn_res[3.5] == 3.15, f"Error en kNN 3.5: {knn_res[3.5]}"
    assert knn_res[6.5] == 5.10, f"Error en kNN 6.5: {knn_res[6.5]}"

    reg_res = MissingDataImputer.regression_imputation(IMPUTATION_MISSING_POINTS)
    assert reg_res[3.5] == 3.29, f"Error en regresión 3.5: {reg_res[3.5]}"
    assert reg_res[6.5] == 5.15, f"Error en regresión 6.5: {reg_res[6.5]}"

    stoch_res = MissingDataImputer.stochastic_regression_imputation(IMPUTATION_MISSING_POINTS, seed=123)
    assert stoch_res[3.5][0] == 3.29
    assert stoch_res[6.5][0] == 5.15
    print("  [PASS] 11.2 Métodos de Imputación: Media (3.91), kNN (3.15, 5.10), Regresión (3.29, 5.15).")

    # Test 2: Matriz de Triage y Métricas de Evaluación
    m_log = TriageEvaluator.compute_metrics(CONFUSION_LOGISTIC_05)
    assert round(m_log["accuracy"], 2) == 0.62, f"Accuracy Logística erróneo: {m_log['accuracy']}"
    assert round(m_log["recall"], 2) == 0.72, f"Recall Logística erróneo: {m_log['recall']}"
    assert round(m_log["specificity"], 2) == 0.50, f"Especificidad Logística erróneo: {m_log['specificity']}"
    assert round(m_log["precision"], 2) == 0.62, f"Precision Logística erróneo: {m_log['precision']}"
    assert round(m_log["f1"], 2) == 0.67, f"F1 Logística erróneo: {m_log['f1']}"

    m_knn = TriageEvaluator.compute_metrics(CONFUSION_KNN_05)
    assert round(m_knn["accuracy"], 2) == 0.62, f"Accuracy kNN erróneo: {m_knn['accuracy']}"
    assert round(m_knn["recall"], 2) == 0.75, f"Recall kNN erróneo: {m_knn['recall']}"
    assert round(m_knn["specificity"], 2) == 0.47, f"Especificidad kNN erróneo: {m_knn['specificity']}"
    assert round(m_knn["precision"], 2) == 0.62, f"Precision kNN erróneo: {m_knn['precision']}"
    assert round(m_knn["f1"], 2) == 0.68, f"F1 kNN erróneo: {m_knn['f1']}"
    print("  [PASS] 11.3 Triage 251 Pacientes: Exactitud 0.62, Recall 0.72 vs 0.75, F1 0.67 vs 0.68.")

    # Test 3: Cuadrantes de Kolter y Escenarios de Países
    q_fac = KolterDebugger.evaluate_kolter_quadrant(True, False)
    assert "FACTIBLE" in q_fac["diagnostico"]
    q_imp = KolterDebugger.evaluate_kolter_quadrant(False, False)
    assert "IMPOSIBLE" in q_imp["diagnostico"]

    sc = TriageEvaluator.evaluate_country_scenarios()
    assert sc["Brasil"]["punto_roc"]["tpr"] == 0.35
    assert sc["Alemania"]["punto_roc"]["tpr"] == 0.86
    assert sc["India"]["punto_roc"]["tpr"] == 0.62
    print("  [PASS] 11.1 Matriz de Kolter & Escenarios ROC (Brasil, Alemania, India).")

    # Test 4: COST de McSherry
    cost = EngineeringAndEthics.evaluate_cost_metric()
    assert cost["twitter_pagerank_benchmark"]["Spark (128 cores)"] == 857.0
    assert cost["twitter_pagerank_benchmark"]["Laptop 1 hilo (RAM)"] == 275.0
    print("  [PASS] 11.4 COST de McSherry: Laptop 1 hilo (275s) vs Spark 128 núcleos (857s).")

    print("\nTODOS LOS TESTS DE CD11 PASARON EXITOSAMENTE (CÓDIGO 0).")
    print("=" * 78)
    return True

def interactive_menu():
    while True:
        print_header("CD11: ASPECTOS PRÁCTICOS Y EVALUACIÓN DE MODELOS")
        print("  1. Debugging en Ciencia de Datos, Ley de Kolter y Telemetría de Discos")
        print("  2. Datos Faltantes (MCAR/MAR/MNAR) y 5 Métodos de Imputación")
        print("  3. Evaluación de Triage en Pandemia (251 Pacientes) y Curva ROC Multipaís")
        print("  4. Ética en IA, Auditoría de Subgrupos y el COST de McSherry")
        print("  5. Ejecutar Suite de Validación Automática (--auto)")
        print("  0. Regresar al Menú Principal")
        print("-" * 78)
        choice = input("Seleccione una opción [0-5]: ").strip()
        if choice == "1":
            run_debugging_demo()
        elif choice == "2":
            run_imputation_demo()
        elif choice == "3":
            run_triage_demo()
        elif choice == "4":
            run_ethics_and_cost_demo()
        elif choice == "5":
            run_all_tests()
        elif choice == "0":
            break
        else:
            print("Opción inválida. Ingrese un valor del 0 al 5.")

if __name__ == "__main__":
    if "--auto" in sys.argv:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    else:
        interactive_menu()
