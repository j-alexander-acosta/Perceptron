#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
MÓDULO CD01: INTRODUCCIÓN AL CURSO DE CIENCIA DE DATOS
================================================================================
Ciencia de Datos — Universidad Católica del Norte (UCN)
Basado en material de:
  • CMU 15-388/688 (Z. Kolter)
  • Harvard CS109 (P. Protopapas et al.)
  • Stanford CS221 (P. Liang et al.)
  • UC Berkeley DS100 (Política de uso de LLMs)

Implementación computacional y verificación matemática de las 63 diapositivas
de la Unidad CD01 (primera unidad del curso en orden cronológico):

  1. Caso Real 1 — Ciencia de Datos en la Pandemia, Sesgo en la Tasa de
     Letalidad y el Caso del Toque de Queda (Diapositivas 5–11, 19–21):
       - Los 5 pasos del proceso de Ciencia de Datos aplicados a COVID-19.
       - Las 4 preguntas clave: modelamiento, predicción, inferencia causal
         y optimización, más la ética de elegir la métrica (Flaxman et al., 2020).
       - Sesgos de cobertura en testeo (PCR/antígenos), movilidad celular
         (Google/Apple) y registros civiles de defunciones.
       - Verificación exacta de la Diapositiva 9: Tasa de letalidad aparente
         (CFR = 100 / 2,000 = 5.0%) vs. tasa de letalidad real
         (IFR = 100 / 20,000 = 0.5%) -> inflación de 10x (un orden de magnitud).
       - Análisis del Toque de Queda Nocturno (Diapositiva 21): inferencia
         causal, sesgo de smartphones y variable de confusión (miedo al virus).

  2. El Modelo Epidemiológico SIR a Mano, R0 y el Efecto de las Cuarentenas
     (Diapositivas 12–18):
       - Dinámica en diferencias diarias sobre N = S + I + R:
           contagios_t      = beta * S_t * I_t / N
           recuperaciones_t = gamma * I_t
           S_{t+1} = S_t - contagios_t
           I_{t+1} = I_t + contagios_t - recuperaciones_t
           R_{t+1} = R_t + recuperaciones_t
       - Verificación exacta de la tabla «La dinámica SIR, a mano» (Diap. 13)
         con N = 1,000, beta = 0.3, gamma = 0.1, S_0 = 990, I_0 = 10, R_0 = 0:
           Día 0 : contagios=2.97,  recup=1.00, S=990.0, I=10.0,  R=0.0
           Día 1 : contagios=3.54,  recup=1.20, S=987.0, I=12.0,  R=1.0
           Día 2 : contagios=4.22,  recup=1.43, S=983.5, I=14.3,  R=2.2
           Día 28: contagios=30.24, recup=31.59, S=319.0, I=315.9, R=365.0
         y ~95% de la población infectada al final de la epidemia (Diap. 14).
       - Efecto de las cuarentenas sobre R_0 = beta / gamma (Diap. 15):
         al bajar beta de 0.3 (R_0 = 3.0) a 0.16 (R_0 = 1.6), el pico de
         infectados simultáneos cae de ~316 (día 28) a ~89 (día 61).

  3. ¿Qué es Ciencia de Datos?, El Caso Netflix Prize (2006–2009) y la
     Paradoja de la Sobre-optimización (Diapositivas 22–40):
       - Espectro Máquina <-> Humano (Daniel Keim, Visual Analytics) y el ciclo
         iterativo de 5 etapas (Recolección -> Procesamiento -> Exploración y
         visualización -> Análisis y ML -> Conocimiento y decisiones).
       - Productos sin ML complejo: visualización de esperanza de vida en
         Our World in Data (Diap. 39) y mapeo de pobreza desde el cielo usando
         el material del techo como variable proxy (Abelson et al., 2014, Diap. 40).
       - Verificación exacta de la matriz del Netflix Prize (Diaps. 27–28):
         480,189 usuarios x 17,770 películas = 8,532,958,530 (~8,533 millones)
         de celdas; 100,480,507 calificaciones observadas (~1.18% ~= 1.2%),
         faltando casi el 99% de la tabla con sesgo MNAR.
       - Progresión del RMSE (Diaps. 30–31): Cinematch (0.951) -> Año 1 (+8.43%)
         -> Año 2 (+9.44%) -> Año 3 (+10.06%, RMSE = 0.856, BellKor's Pragmatic
         Chaos por 20 minutos) y por qué nunca llegó a producción (Diaps. 32–34).

  4. Los Tres Relatos de la IA, el Mapa de 4 Niveles y la Política de Uso de
     LLMs en Ciencia de Datos (Diapositivas 1–4, 41–63):
       - El Test de Turing (1950) y los 3 relatos históricos:
           1) Relato simbólico (1956–1987): Dartmouth 1956, traducción rusa e
              informe ALPAC (1966, 1.er invierno), sistemas expertos (1987, 2.o invierno).
           2) Relato neuronal (1943–hoy): McCulloch-Pitts (1943), Hebb (1949),
              Rosenblatt (1958), Minsky y Papert (1969), Backprop (1986),
              AlexNet (2012, impulsado por datos + cómputo), AlphaGo (2016), LLMs.
           3) Relato estadístico (1801–hoy): Gauss y Legendre (1801, órbita de
              Ceres por mínimos cuadrados), Fisher (1936), Bellman (1957), Vapnik (1995).
       - El Mapa de 4 Niveles de Modelos en IA (Diaps. 3 y 50):
           Reflejos -> Estados -> Variables -> Lógica.
       - Desglose del trabajo ante LLMs (UC Berkeley DS100, Diap. 58) y
         evaluación de los 4 casos prácticos de la política del curso (Diap. 61).
================================================================================
"""

from __future__ import annotations
import math
import sys
from typing import Dict, List, Tuple


# ==============================================================================
# UTILIDADES DE FORMATO Y ESTILO EN TERMINAL
# ==============================================================================

class Colores:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def banner_seccion(titulo: str, subtitulo: str = "") -> None:
    ancho = 78
    print(f"\n{Colores.CYAN}{Colores.BOLD}{'═' * ancho}{Colores.RESET}")
    print(f"{Colores.BOLD}{Colores.HEADER}  {titulo}{Colores.RESET}")
    if subtitulo:
        print(f"{Colores.DIM}  {subtitulo}{Colores.RESET}")
    print(f"{Colores.CYAN}{Colores.BOLD}{'═' * ancho}{Colores.RESET}")


# ==============================================================================
# 1. CASO REAL 1: CIENCIA DE DATOS EN LA PANDEMIA Y SESGO DE LETALIDAD (DIAPS. 5–11, 19–21)
# ==============================================================================

def calcular_sesgo_letalidad(
    muertes: int = 100,
    casos_confirmados: int = 2000,
    infecciones_reales: int = 20000,
) -> Dict[str, float]:
    """
    Reproduce el cálculo exacto de la Diapositiva 9 («El sesgo en acción:
    la tasa de letalidad»):
      - Letalidad aparente (CFR) = muertes / casos_confirmados
      - Letalidad real (IFR)     = muertes / infecciones_reales
      - Factor de inflación      = CFR / IFR = infecciones_reales / casos_confirmados
    """
    cfr = muertes / casos_confirmados
    ifr = muertes / infecciones_reales
    factor_inflacion = cfr / ifr
    return {
        "muertes": float(muertes),
        "casos_confirmados": float(casos_confirmados),
        "infecciones_reales": float(infecciones_reales),
        "cfr_porcentaje": cfr * 100.0,
        "ifr_porcentaje": ifr * 100.0,
        "factor_inflacion": factor_inflacion,
    }


def obtener_preguntas_pandemia() -> List[Dict[str, str]]:
    """
    Clasificación de las 4 preguntas clave del Caso Pandemia (Diapositiva 6).
    """
    return [
        {
            "pregunta": "¿Qué tan rápido se propaga el virus?",
            "tipo_analisis": "modelamiento",
            "herramienta": "Modelo compartimental SIR y estimación de R_0 = beta / gamma",
        },
        {
            "pregunta": "¿Cuándo se saturarán los hospitales?",
            "tipo_analisis": "predicción",
            "herramienta": "Proyección de curvas de hospitalización y camas UCI a corto plazo",
        },
        {
            "pregunta": "¿Funcionan las cuarentenas?",
            "tipo_analisis": "inferencia causal",
            "herramienta": "Modelos contrafactuales de intervenciones no farmacológicas (Flaxman et al., 2020)",
        },
        {
            "pregunta": "¿Cómo distribuir vacunas con equidad?",
            "tipo_analisis": "optimización",
            "herramienta": "Asignación bajo restricciones logísticas y criterios éticos de priorización",
        },
    ]


def demo_caso_pandemia_y_sesgo_letalidad() -> Dict[str, object]:
    """
    Demostración y verificación de las Diapositivas 5–11 y 19–21.
    """
    banner_seccion(
        "1. CASO REAL 1: CIENCIA DE DATOS EN LA PANDEMIA Y SESGO DE LETALIDAD",
        "Diapositivas 5–11 y 19–21 • Formulación, Sesgo de Muestreo (CFR vs. IFR) y Causalidad",
    )

    preguntas = obtener_preguntas_pandemia()
    assert len(preguntas) == 4
    tipos = [p["tipo_analisis"] for p in preguntas]
    assert tipos == ["modelamiento", "predicción", "inferencia causal", "optimización"]

    print(f"\n{Colores.BOLD}Paso 1 — Formular la pregunta y elegir la métrica (Diaps. 6–7):{Colores.RESET}")
    for i, p in enumerate(preguntas, start=1):
        print(
            f"  {i}. {Colores.CYAN}{p['pregunta']:<40}{Colores.RESET} "
            f"-> {Colores.BOLD}{p['tipo_analisis'].upper()}{Colores.RESET}"
        )
    print(
        f"  {Colores.YELLOW}• Nota ética (Flaxman et al., 2020, Nature):{Colores.RESET} "
        "¿Minimizar muertes totales, impacto económico o priorizar equidad regional? "
        "Elegir la métrica no es un detalle técnico: es una decisión ética."
    )

    # Verificación exacta Diapositiva 9: Tasa de letalidad aparente vs. real
    res_sesgo = calcular_sesgo_letalidad(
        muertes=100, casos_confirmados=2000, infecciones_reales=20000
    )
    assert math.isclose(res_sesgo["cfr_porcentaje"], 5.0, abs_tol=1e-9)
    assert math.isclose(res_sesgo["ifr_porcentaje"], 0.5, abs_tol=1e-9)
    assert math.isclose(res_sesgo["factor_inflacion"], 10.0, abs_tol=1e-9)

    print(f"\n{Colores.BOLD}Paso 2 — El sesgo en acción: Tasa de Letalidad (Diapositiva 9):{Colores.RESET}")
    print(
        f"  • Con {int(res_sesgo['muertes'])} muertes y "
        f"{int(res_sesgo['casos_confirmados']):,} casos confirmados (solo sintomáticos):"
    )
    print(
        f"      Letalidad aparente (CFR) = 100 / 2,000  = "
        f"{Colores.RED}{Colores.BOLD}{res_sesgo['cfr_porcentaje']:.1f}%{Colores.RESET}"
    )
    print(
        f"  • Si las infecciones reales eran {int(res_sesgo['infecciones_reales']):,}:"
    )
    print(
        f"      Letalidad real (IFR)     = 100 / 20,000 = "
        f"{Colores.GREEN}{Colores.BOLD}{res_sesgo['ifr_porcentaje']:.1f}%{Colores.RESET}"
    )
    print(
        f"  • Conclusión: Mismo numerador, denominador sesgado -> "
        f"la letalidad aparente queda inflada {Colores.BOLD}{res_sesgo['factor_inflacion']:.0f} veces{Colores.RESET} "
        "(cambia el orden de magnitud)."
    )

    # Ejercicio de la Diapositiva 21: Toque de queda nocturno
    print(f"\n{Colores.BOLD}Ejercicio de Síntesis — Evaluar un toque de queda nocturno (Diapositiva 21):{Colores.RESET}")
    print("  1. Tipo de análisis : Inferencia causal (estimar el efecto de una intervención).")
    print("  2. Sesgo en Mobility: Sub-representa a quienes no tienen smartphone (adultos mayores, zonas vulnerables).")
    print("  3. ¿Basta ver que bajaron movilidad y contagios?: NO sin más supuestos.")
    print("     Existe una variable de confusión: el miedo al virus pudo reducir ambos a la vez.")

    print(f"\n{Colores.GREEN}✔ Verificación Diaps. 5–11 y 19–21 completada con éxito.{Colores.RESET}")
    return {
        "preguntas": preguntas,
        "sesgo_letalidad": res_sesgo,
    }


# ==============================================================================
# 2. EL MODELO SIR A MANO, R_0 Y EL EFECTO DE LAS CUARENTENAS (DIAPS. 12–18)
# ==============================================================================

def simular_sir_discreto(
    N: float = 1000.0,
    beta: float = 0.3,
    gamma: float = 0.1,
    S0: float = 990.0,
    I0: float = 10.0,
    R0_init: float = 0.0,
    dias: int = 160,
) -> Dict[str, object]:
    """
    Simula el modelo epidemiológico SIR en tiempo discreto (paso de 1 día)
    exactamente como en las Diapositivas 12 a 15:
        contagios_t      = beta * S_t * I_t / N
        recuperaciones_t = gamma * I_t
        S_{t+1} = S_t - contagios_t
        I_{t+1} = I_t + contagios_t - recuperaciones_t
        R_{t+1} = R_t + recuperaciones_t
    """
    S, I, R = float(S0), float(I0), float(R0_init)
    historial: List[Dict[str, float]] = []

    dia_pico = 0
    max_I = I0

    for t in range(dias + 1):
        contagios = (beta * S * I) / N
        recuperaciones = gamma * I
        historial.append(
            {
                "dia": float(t),
                "contagios": contagios,
                "recuperaciones": recuperaciones,
                "S": S,
                "I": I,
                "R": R,
            }
        )
        if I > max_I:
            max_I = I
            dia_pico = t

        # Actualización para el día t + 1
        S_next = S - contagios
        I_next = I + contagios - recuperaciones
        R_next = R + recuperaciones
        S, I, R = S_next, I_next, R_next

    r0_basico = beta / gamma
    porcentaje_infectado_final = ((N - historial[-1]["S"]) / N) * 100.0

    return {
        "N": N,
        "beta": beta,
        "gamma": gamma,
        "R0_basico": r0_basico,
        "dia_pico": dia_pico,
        "max_I": max_I,
        "porcentaje_infectado_final": porcentaje_infectado_final,
        "historial": historial,
    }


def demo_modelo_sir_a_mano_y_cuarentenas() -> Dict[str, object]:
    """
    Reproduce y verifica exactamente la tabla de la Diapositiva 13 («La dinámica
    SIR, a mano»), la curva de 160 días de la Diapositiva 14 y el efecto de las
    cuarentenas sobre R_0 de la Diapositiva 15.
    """
    banner_seccion(
        "2. MODELO EPIDEMIOLÓGICO SIR A MANO Y EFECTO DE LAS CUARENTENAS",
        "Diapositivas 12–18 • Tabla SIR (Día 0, 1, 2, Pico Día 28) y Mitigación R_0: 3.0 -> 1.6",
    )

    # Caso base sin mitigación (Diapositivas 13 y 14)
    sir_base = simular_sir_discreto(
        N=1000.0, beta=0.3, gamma=0.1, S0=990.0, I0=10.0, R0_init=0.0, dias=160
    )
    hist = sir_base["historial"]

    # Verificaciones exactas contra la tabla de la Diapositiva 13
    d0, d1, d2, d28 = hist[0], hist[1], hist[2], hist[28]

    # Día 0: contagios = 2.97, recup = 1.00, S = 990.0, I = 10.0, R = 0.0
    assert math.isclose(d0["contagios"], 2.97, abs_tol=1e-2)
    assert math.isclose(d0["recuperaciones"], 1.00, abs_tol=1e-2)
    assert math.isclose(d0["S"], 990.0, abs_tol=1e-1)
    assert math.isclose(d0["I"], 10.0, abs_tol=1e-1)
    assert math.isclose(d0["R"], 0.0, abs_tol=1e-1)

    # Día 1: contagios = 3.54, recup = 1.20, S = 987.0, I = 12.0, R = 1.0
    assert math.isclose(d1["contagios"], 3.54, abs_tol=1e-2)
    assert math.isclose(d1["recuperaciones"], 1.20, abs_tol=1e-2)
    assert math.isclose(d1["S"], 987.03, abs_tol=1e-2)
    assert math.isclose(d1["I"], 11.97, abs_tol=1e-2)
    assert math.isclose(d1["R"], 1.00, abs_tol=1e-2)

    # Día 2: contagios = 4.22, recup = 1.43, S = 983.5, I = 14.3, R = 2.2
    assert math.isclose(d2["contagios"], 4.22, abs_tol=1e-2)
    assert math.isclose(d2["recuperaciones"], 1.43, abs_tol=1e-2)
    assert math.isclose(d2["S"], 983.49, abs_tol=1e-2)
    assert math.isclose(d2["I"], 14.32, abs_tol=1e-2)
    assert math.isclose(d2["R"], 2.20, abs_tol=1e-2)

    # Día 28 (pico): contagios = 30.24, recup = 31.59, S = 319.0, I = 315.9, R = 365.0
    assert sir_base["dia_pico"] == 28
    assert math.isclose(d28["contagios"], 30.24, abs_tol=1e-2)
    assert math.isclose(d28["recuperaciones"], 31.59, abs_tol=1e-2)
    assert math.isclose(d28["S"], 319.0, abs_tol=0.1)
    assert math.isclose(d28["I"], 315.9, abs_tol=0.1)
    assert math.isclose(d28["R"], 365.0, abs_tol=0.1)
    assert 94.0 <= sir_base["porcentaje_infectado_final"] <= 95.5

    print(f"\n{Colores.BOLD}Tabla «La dinámica SIR, a mano» (Diapositiva 13 — N=1000, β=0.3, γ=0.1, R0=3.0):{Colores.RESET}")
    print("  ┌────────┬───────────┬────────────────┬───────────┬───────────┬───────────┐")
    print("  │  Día   │ Contagios │ Recuperaciones │     S     │     I     │     R     │")
    print("  ├────────┼───────────┼────────────────┼───────────┼───────────┼───────────┤")
    for fila in (d0, d1, d2, d28):
        dia_lbl = f"{int(fila['dia'])}" + ("*" if int(fila["dia"]) == 28 else " ")
        print(
            f"  │  {dia_lbl:>4}  │   {fila['contagios']:>6.2f}  │     {fila['recuperaciones']:>6.2f}     │"
            f"  {fila['S']:>7.1f}  │  {fila['I']:>7.1f}  │  {fila['R']:>7.1f}  │"
        )
    print("  └────────┴───────────┴────────────────┴───────────┴───────────┴───────────┘")
    print(
        f"  * Día {sir_base['dia_pico']} (pico): I_max = {Colores.RED}{Colores.BOLD}{sir_base['max_I']:.1f}{Colores.RESET} "
        f"(casi un tercio de la población infectada a la vez)."
    )
    print(
        f"  • Al día 160 (Diapositiva 14): {Colores.BOLD}{sir_base['porcentaje_infectado_final']:.1f}%{Colores.RESET} "
        "de la población habrá pasado por la infección (~95%)."
    )

    # Caso con cuarentena (Diapositiva 15: beta = 0.16, gamma = 0.1 -> R0 = 1.6)
    sir_cuarentena = simular_sir_discreto(
        N=1000.0, beta=0.16, gamma=0.1, S0=990.0, I0=10.0, R0_init=0.0, dias=160
    )
    assert math.isclose(sir_cuarentena["R0_basico"], 1.6, abs_tol=1e-9)
    assert sir_cuarentena["dia_pico"] == 61
    assert math.isclose(round(sir_cuarentena["max_I"]), 89, abs_tol=1)

    print(f"\n{Colores.BOLD}El efecto de las cuarentenas sobre R_0 = β/γ (Diapositiva 15):{Colores.RESET}")
    print(
        f"  • Sin mitigación (β = 0.30, γ = 0.10 -> R_0 = 3.0): "
        f"pico I_max = {Colores.RED}{sir_base['max_I']:.1f} (~316){Colores.RESET} en el {Colores.BOLD}día {sir_base['dia_pico']}{Colores.RESET}."
    )
    print(
        f"  • Con cuarentena (β = 0.16, γ = 0.10 -> R_0 = 1.6): "
        f"pico I_max = {Colores.GREEN}{sir_cuarentena['max_I']:.1f} (~89){Colores.RESET} en el {Colores.BOLD}día {sir_cuarentena['dia_pico']}{Colores.RESET}."
    )
    print(
        "  • Eso es «aplanar la curva»: un cambio en β reduce el pico a casi una cuarta parte "
        "y gana más de un mes de preparación hospitalaria."
    )

    print(f"\n{Colores.GREEN}✔ Verificación Diaps. 12–18 (Modelo SIR y cuarentenas) completada con éxito.{Colores.RESET}")
    return {
        "sir_sin_mitigacion": sir_base,
        "sir_con_cuarentena": sir_cuarentena,
    }


# ==============================================================================
# 3. EL CASO NETFLIX PRIZE Y LA PARADOJA DE LA SOBRE-OPTIMIZACIÓN (DIAPS. 22–40)
# ==============================================================================

def analizar_netflix_prize() -> Dict[str, object]:
    """
    Verifica los números exactos del Caso Netflix Prize (2006–2009) en las
    Diapositivas 26 a 34:
      - Usuarios = 480,189; Películas = 17,770
      - Total de celdas = 8,532,958,530 (~8,533 millones)
      - Calificaciones observadas = 100,480,507 (~1.178% ~= 1.2% observado;
        ~98.82% faltante, MNAR)
      - Evolución del RMSE:
          2006 Cinematch baseline : RMSE = 0.951 (0.00% mejora)
          2007 Año 1              : +8.43% mejora
          2008 Año 2              : +9.44% mejora
          2009 Año 3 (Ganador)    : +10.06% mejora (RMSE = 0.856)
    """
    usuarios = 480_189
    peliculas = 17_770
    total_celdas = usuarios * peliculas
    observadas = 100_480_507
    fraccion_observada = observadas / total_celdas
    porcentaje_observado = fraccion_observada * 100.0
    porcentaje_faltante = 100.0 - porcentaje_observado

    rmse_cinematch = 0.951
    rmse_ganador = 0.856
    mejora_2007 = 8.43
    mejora_2008 = 9.44
    mejora_2009 = 10.06

    return {
        "usuarios": usuarios,
        "peliculas": peliculas,
        "total_celdas": total_celdas,
        "total_celdas_millones": total_celdas / 1e6,
        "observadas": observadas,
        "porcentaje_observado": porcentaje_observado,
        "porcentaje_faltante": porcentaje_faltante,
        "rmse_cinematch": rmse_cinematch,
        "rmse_ganador": rmse_ganador,
        "evolucion_mejora": [
            ("2006 (Cinematch)", 0.00, rmse_cinematch),
            ("2007 (Año 1)", mejora_2007, round(rmse_cinematch * (1.0 - mejora_2007 / 100.0), 3)),
            ("2008 (Año 2)", mejora_2008, round(rmse_cinematch * (1.0 - mejora_2008 / 100.0), 3)),
            ("2009 (BellKor's Pragmatic Chaos)", mejora_2009, rmse_ganador),
        ],
    }


def calcular_rmse(predicciones: List[float], reales: List[float]) -> float:
    """
    Fórmula de la Diapositiva 26:
        RMSE = sqrt( (1/n) * sum_{i=1}^n (prediccion_i - calificacion_i)^2 )
    """
    assert len(predicciones) == len(reales) and len(predicciones) > 0
    mse = sum((p - y) ** 2 for p, y in zip(predicciones, reales)) / len(predicciones)
    return math.sqrt(mse)


def demo_netflix_prize_y_sobreoptimizacion() -> Dict[str, object]:
    """
    Demostración y verificación de las Diapositivas 22 a 40.
    """
    banner_seccion(
        "3. EL CASO NETFLIX PRIZE (2006–2009) Y LA SOBRE-OPTIMIZACIÓN",
        "Diapositivas 22–40 • Matriz Dispersa (98.8% vacío MNAR), RMSE y Ciclo de 5 Etapas",
    )

    datos_nf = analizar_netflix_prize()
    assert datos_nf["total_celdas"] == 8_532_958_530
    assert math.isclose(datos_nf["total_celdas_millones"], 8532.95853, abs_tol=1e-3)
    assert math.isclose(datos_nf["porcentaje_observado"], 1.17756, abs_tol=1e-3)
    assert math.isclose(datos_nf["porcentaje_faltante"], 98.82244, abs_tol=1e-3)

    print(f"\n{Colores.BOLD}El tamaño del vacío en la matriz de Netflix (Diapositivas 27–28):{Colores.RESET}")
    print(
        f"  • Dimensiones      : {datos_nf['usuarios']:,} usuarios × {datos_nf['peliculas']:,} películas "
        f"= {Colores.BOLD}{datos_nf['total_celdas']:,}{Colores.RESET} celdas (~{datos_nf['total_celdas_millones']:,.0f} millones)."
    )
    print(
        f"  • Calificaciones   : {datos_nf['observadas']:,} observadas "
        f"({Colores.GREEN}{Colores.BOLD}{datos_nf['porcentaje_observado']:.2f}% ≈ 1.2%{Colores.RESET} observado)."
    )
    print(
        f"  • Vacío estructural: Falta el {Colores.RED}{Colores.BOLD}{datos_nf['porcentaje_faltante']:.2f}% (casi el 99%){Colores.RESET} "
        "de la tabla, y NO falta al azar (MNAR: la gente califica lo que eligió ver)."
    )

    print(f"\n{Colores.BOLD}Tres años de carrera por el 10% en RMSE (Diapositivas 30–31):{Colores.RESET}")
    for etapa, mejora, rmse_val in datos_nf["evolucion_mejora"]:
        print(f"  • {etapa:<34}: mejora = {mejora:>6.2f}%  |  RMSE ≈ {rmse_val:.3f}")
    print(
        "  • Empate fotográfico en 2009: dos equipos alcanzaron 10.06%; ganó "
        f"{Colores.CYAN}BellKor's Pragmatic Chaos{Colores.RESET} por enviarlo 20 minutos antes."
    )

    print(f"\n{Colores.BOLD}La confesión de 2012 y la Paradoja de la Sobre-optimización (Diaps. 32–34):{Colores.RESET}")
    print("  1. El ensamble ganador (>100 modelos) NUNCA se desplegó en producción.")
    print("  2. El costo de ingeniería y mantenimiento no justificaba la ganancia de precisión offline.")
    print("  3. Netflix pasó del DVD por correo al streaming: el problema dejó de ser predecir estrellas")
    print("     y pasó a ser ranking/descubrimiento con retroalimentación implícita.")

    print(f"\n{Colores.GREEN}✔ Verificación Diaps. 22–40 (Netflix Prize y 5 etapas) completada con éxito.{Colores.RESET}")
    return datos_nf


# ==============================================================================
# 4. LOS TRES RELATOS DE LA IA, EL MAPA DE 4 NIVELES Y POLÍTICA DE LLMS (DIAPS. 1–4, 41–63)
# ==============================================================================

def obtener_mapa_ia_y_relatos() -> Dict[str, object]:
    """
    Estructura los 4 niveles de modelos en IA (Diapositivas 3 y 50), los 3 relatos
    históricos de la IA (Diapositivas 41–51) y los 4 casos de la política de uso
    de LLMs (Diapositivas 58–61).
    """
    niveles_ia = [
        {
            "nivel": 1,
            "nombre": "Reflejos",
            "modelos": "Aprendizaje automático (perceptrón, regresión, redes profundas)",
            "cobertura_curso": "Se recorre entero en el curso",
        },
        {
            "nivel": 2,
            "nombre": "Estados",
            "modelos": "Búsqueda en grafos, Procesos de Decisión de Markov (MDP), Juegos",
            "cobertura_curso": "Cubre búsqueda, decisiones y refuerzo (sin juegos)",
        },
        {
            "nivel": 3,
            "nombre": "Variables",
            "modelos": "Satisfacción de restricciones (CSP), Redes de Markov y Bayesianas",
            "cobertura_curso": "Cubre modelos probabilísticos (sin restricciones CSP)",
        },
        {
            "nivel": 4,
            "nombre": "Lógica",
            "modelos": "Lógica proposicional y de primer orden",
            "cobertura_curso": "Queda fuera del curso",
        },
    ]

    relatos_ia = {
        "simbolico": {
            "periodo": "1956–1987",
            "idea_central": "Manipular símbolos mediante reglas explícitas («Si toda inteligencia es manipulación de símbolos, basta con programarla»).",
            "hitos": [
                "1956: Taller de Dartmouth (McCarthy, Minsky, Newell, Simon) — nace el nombre 'Inteligencia Artificial'.",
                "1966: Informe ALPAC tras el fracaso en traducción automática ruso-inglés («el vodka es bueno, pero la carne está podrida») -> 1.er invierno de la IA.",
                "1970–1980: Sistemas expertos con reglas IF-THEN; frágiles ante la incertidumbre y el sentido común -> 2.o invierno de la IA (1987).",
            ],
        },
        "neuronal": {
            "periodo": "1943–hoy",
            "idea_central": "Aprender representaciones ajustando pesos en redes inspiradas en neuronas biológicas.",
            "hitos": [
                "1943: Neurona formal de McCulloch y Pitts (inspirada en Alan Turing).",
                "1949: Regla de aprendizaje sináptico de Donald Hebb.",
                "1958: Frank Rosenblatt construye el Perceptrón.",
                "1969: Minsky y Papert publican 'Perceptrons' (límite XOR de una capa) -> congela el área por más de una década.",
                "1986: Rumelhart, Hinton y Williams popularizan Backpropagation en redes multicapa.",
                "2012: AlexNet en ImageNet -> el salto entre 1986 y 2012 fue impulsado por DATOS y CÓMPUTO (GPUs).",
            ],
        },
        "estadistico": {
            "periodo": "1801–hoy",
            "idea_central": "Ajustar modelos matemáticos a datos con ruido cuantificando la incertidumbre.",
            "hitos": [
                "1801: Gauss y Legendre inventan mínimos cuadrados para predecir la órbita del asteroide Ceres.",
                "1936: Ronald Fisher introduce el análisis discriminante lineal.",
                "1957: Richard Bellman formula los Procesos de Decisión de Markov (MDP).",
                "1995: Cortes y Vapnik desarrollan las Máquinas de Soporte Vectorial (SVM).",
            ],
        },
    }

    casos_llm = [
        {
            "caso": 1,
            "descripcion": "Pedir al LLM que explique la diferencia entre loc e iloc en pandas.",
            "veredicto": "Apropiado",
            "justificacion": "Es equivalente a consultar documentación sobre un concepto puntual o API.",
        },
        {
            "caso": 2,
            "descripcion": "Pedir al LLM que escriba el análisis exploratorio (EDA) completo del proyecto.",
            "veredicto": "Inapropiado",
            "justificacion": "Externaliza los pasos 1 (contexto) y 2 (planificación), que son el núcleo del aprendizaje.",
        },
        {
            "caso": 3,
            "descripcion": "Pegar una función propia que lanza un error junto con el stack trace y pedir hipótesis.",
            "veredicto": "Apropiado",
            "justificacion": "Depuración asistida sobre código propio que el estudiante comprende y verifica.",
        },
        {
            "caso": 4,
            "descripcion": "Usar código generado por el LLM para una visualización sin entender qué grafica cada línea.",
            "veredicto": "Inapropiado",
            "justificacion": "Viola la regla de oro: eres responsable del 100% del código que entregas y debes poder defenderlo.",
        },
    ]

    return {
        "niveles_ia": niveles_ia,
        "relatos_ia": relatos_ia,
        "casos_llm": casos_llm,
    }


def demo_historia_ia_mapa_y_politica_llms() -> Dict[str, object]:
    """
    Demostración y verificación de las Diapositivas 1–4 y 41–63.
    """
    banner_seccion(
        "4. LOS TRES RELATOS DE LA IA, MAPA DE 4 NIVELES Y POLÍTICA DE LLMs",
        "Diapositivas 1–4 y 41–63 • Simbólico, Neuronal, Estadístico y Uso Responsable de LLMs",
    )

    info = obtener_mapa_ia_y_relatos()
    assert len(info["niveles_ia"]) == 4
    assert [n["nombre"] for n in info["niveles_ia"]] == [
        "Reflejos",
        "Estados",
        "Variables",
        "Lógica",
    ]
    assert len(info["casos_llm"]) == 4
    assert [c["veredicto"] for c in info["casos_llm"]] == [
        "Apropiado",
        "Inapropiado",
        "Apropiado",
        "Inapropiado",
    ]

    print(f"\n{Colores.BOLD}Mapa de 4 Niveles de Modelos en IA (Diapositivas 3 y 50):{Colores.RESET}")
    for niv in info["niveles_ia"]:
        print(
            f"  Nivel {niv['nivel']} — {Colores.CYAN}{niv['nombre']:<10}{Colores.RESET}: "
            f"{niv['modelos']} ({Colores.BOLD}{niv['cobertura_curso']}{Colores.RESET})"
        )

    print(f"\n{Colores.BOLD}Los Tres Relatos de la Historia de la IA (Diapositivas 41–51):{Colores.RESET}")
    for clave, rel in info["relatos_ia"].items():
        print(f"  • {Colores.YELLOW}Relato {clave.capitalize()} ({rel['periodo']}){Colores.RESET}: {rel['idea_central']}")
        for h in rel["hitos"]:
            print(f"      - {h}")

    print(f"\n{Colores.BOLD}Política de Uso de LLMs — Evaluación de los 4 Casos (Diapositivas 58–61):{Colores.RESET}")
    for c in info["casos_llm"]:
        color_v = Colores.GREEN if c["veredicto"] == "Apropiado" else Colores.RED
        print(
            f"  Caso {c['caso']}: {c['descripcion']}\n"
            f"          -> {color_v}{Colores.BOLD}{c['veredicto'].upper()}{Colores.RESET}: {c['justificacion']}"
        )

    print(f"\n{Colores.GREEN}✔ Verificación Diaps. 1–4 y 41–63 completada con éxito.{Colores.RESET}")
    return info


# ==============================================================================
# EJECUCIÓN COMPLETA Y MENÚ INTERACTIVO
# ==============================================================================

def ejecutar_suite_completa() -> Dict[str, object]:
    """
    Ejecuta todas las demostraciones y verificaciones de la Unidad CD01.
    """
    banner_seccion(
        "SUITE COMPLETA CD01: INTRODUCCIÓN AL CURSO DE CIENCIA DE DATOS (63 DIAPS.)",
        "Universidad Católica del Norte • Verificación Numérica y Conceptual",
    )
    r1 = demo_caso_pandemia_y_sesgo_letalidad()
    r2 = demo_modelo_sir_a_mano_y_cuarentenas()
    r3 = demo_netflix_prize_y_sobreoptimizacion()
    r4 = demo_historia_ia_mapa_y_politica_llms()

    print(f"\n{Colores.GREEN}{Colores.BOLD}{'═' * 78}{Colores.RESET}")
    print(
        f"{Colores.GREEN}{Colores.BOLD}  ✔ TODAS LAS VERIFICACIONES DE LA UNIDAD CD01 (63 DIAPOSITIVAS) PASARON AL 100%{Colores.RESET}"
    )
    print(f"{Colores.GREEN}{Colores.BOLD}{'═' * 78}{Colores.RESET}\n")
    return {
        "pandemia": r1,
        "sir": r2,
        "netflix": r3,
        "historia_ia_llms": r4,
    }


def menu_interactivo() -> None:
    while True:
        banner_seccion(
            "MENÚ INTERACTIVO — CD01: INTRODUCCIÓN AL CURSO (63 DIAPOSITIVAS)",
            "Seleccione una demostración o ejecute la suite completa",
        )
        print("  [1] Caso Real 1: Pandemia, 4 Tipos de Preguntas y Sesgo en Letalidad (Diaps. 5–11, 19–21)")
        print("  [2] Modelo SIR a Mano (Tabla Diap. 13), R_0 y Efecto de las Cuarentenas (Diaps. 12–18)")
        print("  [3] Caso Netflix Prize (98.8% vacío MNAR, RMSE) y Sobre-optimización (Diaps. 22–40)")
        print("  [4] Los Tres Relatos de la IA, Mapa de 4 Niveles y Política de LLMs (Diaps. 1–4, 41–63)")
        print("  [5] Ejecutar suite completa de verificaciones CD01")
        print("  [0] Volver / Salir")
        opcion = input(f"\n{Colores.BOLD}Opción > {Colores.RESET}").strip()

        if opcion == "1":
            demo_caso_pandemia_y_sesgo_letalidad()
        elif opcion == "2":
            demo_modelo_sir_a_mano_y_cuarentenas()
        elif opcion == "3":
            demo_netflix_prize_y_sobreoptimizacion()
        elif opcion == "4":
            demo_historia_ia_mapa_y_politica_llms()
        elif opcion == "5":
            ejecutar_suite_completa()
        elif opcion == "0":
            break
        else:
            print(f"{Colores.RED}Opción no válida.{Colores.RESET}")


if __name__ == "__main__":
    if "--auto" in sys.argv:
        ejecutar_suite_completa()
    else:
        menu_interactivo()
