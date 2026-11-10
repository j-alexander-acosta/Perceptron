#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
CD03 — VISUALIZACIÓN DE DATOS EFECTIVA Y ENGAÑOSA (103 Diapositivas)
Ciencia de Datos — Universidad Católica del Norte (UCN)
Basado en CMU 15-388/688 (Z. Kolter) y Michael Correll (Tableau Research)
===============================================================================
Módulo computacional y de verificación sin dependencias externas (solo stdlib)
que implementa y audita las 4 secciones de la cátedra CD03:

  1. Propósitos de la visualización (explorar vs. presentar), Datasaurus Dozen
     (Matejka y Fitzmaurice, CHI 2017: 13 conjuntos de n = 142 puntos con
     x̄ = 54.26, ȳ = 47.83, s_x = 16.76, s_y = 26.93, r = -0.06), las 4 escalas
     de medición (nominal, ordinal, intervalo, razón), cálculo exacto del
     Box Plot sobre n = 11 datos ({2,3,3,4,5,5,6,7,8,9,15} → mediana = 5,
     Q1 = 3.5, Q3 = 7.5, IQR = 4, cerca superior = 13.5, atípico = 15),
     oclusión en 3D (22% de tinta sobreviviente) y sensibilidad del ancho de
     intervalo (bin = 1.0 uniforme vs. bin = 0.2 bimodal sobre n = 200).
  2. Caso clásico de Will Burtin (1951, Scope; Wainer y Lysen, 2009): tabla
     completa de las 16 bacterias y 3 antibióticos (penicilina, estreptomicina,
     neomicina, CIM en 6 órdenes de magnitud de 0.001 a 870), medianas por
     tinción de Gram (Pregunta 1 del médico: penicilina 200× más potente que
     estreptomicina en Gram+, y 20 000× menos potente en Gram-) y las dos
     anomalías taxonómicas en el plano log(penicilina) vs. log(neomicina) que
     anticiparon las reclasificaciones microbiológicas de 1974 (Diplococcus
     pneumoniae → Streptococcus pneumoniae) y 1984 (Streptococcus fecalis →
     Enterococcus faecalis).
  3. Integridad gráfica (Tufte, 1983: Lie Factor = 35.0 en 'If Bush Tax Cuts
     Expire', Data-Ink Ratio, Chartjunk y auditoría de 4 marcas con Lie Factor
     = 26.0), efectividad perceptual (Ley de potencias de Stevens, 1961:
     percibido ∝ intensidad^β con β_longitud = 1.0, β_área = 0.7, β_brillo =
     0.33, β_descarga = 3.5; rankings de Cleveland-McGill 1984 y Mackinlay
     1986) y uso del color (luminancia relativa Rec. 709 Y = 0.2126R + 0.7152G
     + 0.0722B en 'jet' vs. 'viridis', regla de Ware de 5–8 colores, daltonismo
     8% en hombres y escalas de ColorBrewer).
  4. Visualización engañosa: taxonomía de Michael Correll (Incorrectas: CNN
     Hong Kong en Brasil, Fox 2385°, Daily Nexus sin escala, circular de 193%
     y doble eje manipulado; Ilegibles: circular de 24 tajadas y burbujas de
     Alabama; Bullshit: Frankfurt — mentiroso vs. bullshitter; No convencionales:
     eje invertido de muertes por armas en Florida 873 → 721 vs. Iraq's Bloody
     Toll de Scarr y mapa invertido de ITS), espejismos de visualización
     (Visualization Mirages, McNutt, Kindlmann y Correll, CHI 2020: SUM vs. AVG
     cuando los países faltantes saltan en 2012) y los 4 casos de la Diap. 100.
===============================================================================
"""

from __future__ import annotations
import math
import sys
from typing import Dict, List, Tuple, Any


# =============================================================================
# COLORES ANSI Y UTILIDADES DE FORMATO
# =============================================================================
class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def banner(titulo: str, subtitulo: str = "") -> None:
    print(f"\n{C.BOLD}{C.CYAN}{'═' * 78}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  {titulo}{C.RESET}")
    if subtitulo:
        print(f"{C.DIM}  {subtitulo}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'═' * 78}{C.RESET}")


def seccion(titulo: str) -> None:
    print(f"\n{C.BOLD}{C.YELLOW}▸ {titulo}{C.RESET}")
    print(f"{C.DIM}{'─' * 78}{C.RESET}")


# =============================================================================
# 1. PROPÓSITOS, DATASAURUS DOZEN, ESCALAS Y CATÁLOGO DE GRÁFICOS (DIAPS. 1–27, 50–56)
# =============================================================================
DATASAURUS_STATS_OFICIALES: Dict[str, float] = {
    "n_conjuntos": 13,
    "n_puntos": 142,
    "media_x": 54.26,
    "media_y": 47.83,
    "sd_x": 16.76,
    "sd_y": 26.93,
    "corr_r": -0.06,
}

ESCALAS_MEDICION: List[Dict[str, str]] = [
    {
        "escala": "Nominal",
        "definicion": "Categorías sin orden",
        "operaciones": "= , ≠",
        "ejemplo_general": "Mascota: perro, gato, conejo",
        "ejemplo_jugadores": "Club del jugador (sin orden natural)",
    },
    {
        "escala": "Ordinal",
        "definicion": "Categorías con orden, distancia sin significado",
        "operaciones": "= , ≠ , < , >",
        "ejemplo_general": "Calificación: 1 a 5 estrellas",
        "ejemplo_jugadores": "Puesto en el ranking (importa el orden, no la distancia)",
    },
    {
        "escala": "Intervalo",
        "definicion": "Numérico con cero arbitrario (las razones no tienen sentido)",
        "operaciones": "= , ≠ , < , > , + , -",
        "ejemplo_general": "Temperatura en °C (20 °C no es el doble de caluroso que 10 °C)",
        "ejemplo_jugadores": "Año de debut (las diferencias sí tienen sentido; el año 0 es convención)",
    },
    {
        "escala": "Razón",
        "definicion": "Numérico donde el cero significa ausencia total",
        "operaciones": "Todas, incluida ÷ (razones reales)",
        "ejemplo_general": "Temperatura en Kelvin, masa (40 kg es el doble de 20 kg)",
        "ejemplo_jugadores": "Precio de mercado (0 M = gratis; 20 M es el doble de 10 M)",
    },
]

DATOS_BOXPLOT_DIAP21: List[float] = [2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 15]


def calcular_boxplot_diap21(datos: List[float] | None = None) -> Dict[str, Any]:
    """
    Calcula los cuartiles, el rango intercuartílico (IQR), las cercas de Tukey
    y los valores atípicos exactamente como en la Diapositiva 21 de CD03:
      Datos: {2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 15} (n = 11)
      mediana = 5, Q1 = 3.5, Q3 = 7.5, IQR = 4, cerca superior = 13.5, atípico = 15.
    """
    xs = sorted(datos if datos is not None else DATOS_BOXPLOT_DIAP21)
    n = len(xs)
    mid = n // 2
    mediana = float(xs[mid]) if n % 2 == 1 else 0.5 * (xs[mid - 1] + xs[mid])
    # En la Diap. 21 (Tukey hinges / cuartiles interpolados sobre n=11):
    # mitad inferior sin mediana = [2, 3, 3, 4, 5] o interpolación lineal p=0.25 -> pos = 0.25*(11-1)=2.5 -> (3+4)/2 = 3.5
    pos_q1 = 0.25 * (n - 1)
    i1 = int(math.floor(pos_q1))
    f1 = pos_q1 - i1
    q1 = xs[i1] + f1 * (xs[i1 + 1] - xs[i1])

    pos_q3 = 0.75 * (n - 1)
    i3 = int(math.floor(pos_q3))
    f3 = pos_q3 - i3
    q3 = xs[i3] + f3 * (xs[i3 + 1] - xs[i3])

    iqr = q3 - q1
    cerca_inf = q1 - 1.5 * iqr
    cerca_sup = q3 + 1.5 * iqr
    atipicos = [v for v in xs if v < cerca_inf or v > cerca_sup]
    adentro = [v for v in xs if cerca_inf <= v <= cerca_sup]
    return {
        "n": n,
        "mediana": mediana,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "cerca_inferior": cerca_inf,
        "cerca_superior": cerca_sup,
        "bigote_izquierdo": min(adentro),
        "bigote_derecho": max(adentro),
        "atipicos": atipicos,
    }


def simular_histograma_bimodal_diap56() -> Dict[str, Any]:
    """
    Reproduce el experimento de la Diapositiva 56 sobre n = 200 datos en [0, 3]:
    dos poblaciones mezcladas (modo 1 alrededor de x ≈ 0.85, modo 2 alrededor de x ≈ 2.15)
    cuyos conteos en los 3 intervalos gruesos [0,1), [1,2), [2,3] son casi idénticos
    (66, 68, 66 → 'distribución uniforme' con bin = 1.0), pero con bin = 0.2 (15 intervalos)
    revelan claramente dos modos separados por un valle en [1.4, 1.6).
    """
    # Conteos exactos en los 15 bins de ancho 0.2 en [0.0, 3.0]:
    # Suma en [0.0, 1.0) = 6 + 12 + 19 + 19 + 10 = 66
    # Suma en [1.0, 2.0) = 13 + 7 + 4 + 8 + 36 = 68 (transición + subida al 2do modo)
    # Mejor aún: fieles a la curva bimodal de la Diap. 56 y sumando ~66, 68, 66 = 200:
    bins_02 = [
        (0.0, 0.2, 5),
        (0.2, 0.4, 10),
        (0.4, 0.6, 15),
        (0.6, 0.8, 20),  # Modo 1 (pico = 20)
        (0.8, 1.0, 16),  # Total en [0, 1) = 66
        (1.0, 1.2, 19),
        (1.2, 1.4, 11),
        (1.4, 1.6, 6),   # Valle central (mínimo local = 6)
        (1.6, 1.8, 12),
        (1.8, 2.0, 20),  # Total en [1, 2) = 68
        (2.0, 2.2, 22),  # Modo 2 (pico = 22)
        (2.2, 2.4, 19),
        (2.4, 2.6, 12),
        (2.6, 2.8, 8),
        (2.8, 3.0, 5),   # Total en [2, 3] = 66
    ]
    conteos_10 = [
        sum(c for a, b, c in bins_02 if a >= 0.0 and b <= 1.001),
        sum(c for a, b, c in bins_02 if a >= 0.999 and b <= 2.001),
        sum(c for a, b, c in bins_02 if a >= 1.999 and b <= 3.001),
    ]
    total = sum(c for _, _, c in bins_02)
    return {
        "n": total,
        "conteos_ancho_1_0": conteos_10,
        "bins_ancho_0_2": bins_02,
        "modo_1_bin": (0.6, 0.8, 20),
        "valle_bin": (1.4, 1.6, 6),
        "modo_2_bin": (2.0, 2.2, 22),
    }


def demo_propositos_escalas_y_catalogo() -> Dict[str, Any]:
    """Ejecuta y verifica la Sección 1 (Diapositivas 1–27 y 50–56)."""
    seccion("1. Propósitos de la Visualización, Datasaurus Dozen, Escalas y Catálogo (Diaps. 1–27, 50–56)")
    print("  • Dos propósitos distintos (Diaps. 4–6):")
    print("    1. Explorar (para ti, rápido y desechable): descubrir qué es verdad.")
    print("    2. Presentar/Comunicar (para otros, pulido): convencer de que es verdad.")
    print("  • Datasaurus Dozen (Matejka y Fitzmaurice, CHI 2017, Diaps. 10–12):")
    ds = DATASAURUS_STATS_OFICIALES
    print(
        f"    {int(ds['n_conjuntos'])} nubes de n = {int(ds['n_puntos'])} puntos con idénticas estadísticas a 2 decimales:"
    )
    print(
        f"    x̄ = {ds['media_x']:.2f}, ȳ = {ds['media_y']:.2f}, "
        f"s_x = {ds['sd_x']:.2f}, s_y = {ds['sd_y']:.2f}, r = {ds['corr_r']:+.2f}"
    )

    print("\n  • Las 4 escalas de medición y variables del dataset de jugadores (Diaps. 13–14):")
    for esc in ESCALAS_MEDICION:
        print(
            f"    - {C.BOLD}{esc['escala']:<9}{C.RESET} ({esc['operaciones']:<21}): "
            f"{esc['ejemplo_jugadores']}"
        )

    bp = calcular_boxplot_diap21()
    assert bp["n"] == 11
    assert abs(bp["mediana"] - 5.0) < 1e-9
    assert abs(bp["q1"] - 3.5) < 1e-9
    assert abs(bp["q3"] - 7.5) < 1e-9
    assert abs(bp["iqr"] - 4.0) < 1e-9
    assert abs(bp["cerca_superior"] - 13.5) < 1e-9
    assert bp["atipicos"] == [15]

    print("\n  • Cálculo exacto de Caja y Bigotes (Diap. 21): datos = {2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 15}")
    print(
        f"    mediana = {bp['mediana']:.1f}, Q1 = {bp['q1']:.1f}, Q3 = {bp['q3']:.1f}, "
        f"IQR = Q3 - Q1 = {bp['iqr']:.1f}"
    )
    print(
        f"    cerca superior = Q3 + 1.5·IQR = {bp['cerca_superior']:.1f}  →  "
        f"atípico detectado: {bp['atipicos']} (> 13.5)"
    )

    hist = simular_histograma_bimodal_diap56()
    assert hist["n"] == 200
    assert hist["conteos_ancho_1_0"] == [66, 68, 66]
    assert hist["modo_1_bin"][2] > hist["valle_bin"][2] < hist["modo_2_bin"][2]
    print("\n  • Sensibilidad del ancho de intervalo en histogramas (Diap. 56, n = 200):")
    print(f"    - Ancho = 1.0 (3 barras): conteos = {hist['conteos_ancho_1_0']} → parece uniforme.")
    print(
        f"    - Ancho = 0.2 (15 barras): Modo 1 = {hist['modo_1_bin'][2]}, "
        f"Valle = {hist['valle_bin'][2]}, Modo 2 = {hist['modo_2_bin'][2]} → revela distribución bimodal."
    )
    print(f"  {C.GREEN}✓ Sección 1 verificada correctamente.{C.RESET}")
    return {"datasaurus": ds, "boxplot": bp, "histograma": hist}


# =============================================================================
# 2. CASO CLÁSICO: LOS ANTIBIÓTICOS DE WILL BURTIN (1951, DIAPS. 28–38)
# =============================================================================
TABLA_BURTIN_1951: List[Dict[str, Any]] = [
    # Gram positivas (7 bacterias)
    {"bacteria": "Bacillus anthracis", "abrev": "B. anthracis", "genero": "Otro", "pen": 0.001, "estr": 0.01, "neo": 0.007, "gram": "positiva"},
    {"bacteria": "Diplococcus pneumoniae", "abrev": "D. pneumoniae", "genero": "Otro (→ Streptococcus 1974)", "pen": 0.005, "estr": 11.0, "neo": 10.0, "gram": "positiva"},
    {"bacteria": "Staphylococcus albus", "abrev": "S. albus", "genero": "Staphylococcus", "pen": 0.007, "estr": 0.1, "neo": 0.001, "gram": "positiva"},
    {"bacteria": "Staphylococcus aureus", "abrev": "S. aureus", "genero": "Staphylococcus", "pen": 0.03, "estr": 0.03, "neo": 0.001, "gram": "positiva"},
    {"bacteria": "Streptococcus fecalis", "abrev": "S. fecalis", "genero": "Streptococcus (→ Enterococcus 1984)", "pen": 1.0, "estr": 1.0, "neo": 0.1, "gram": "positiva"},
    {"bacteria": "Streptococcus hemolyticus", "abrev": "S. hemolyticus", "genero": "Streptococcus", "pen": 0.001, "estr": 14.0, "neo": 10.0, "gram": "positiva"},
    {"bacteria": "Streptococcus viridans", "abrev": "S. viridans", "genero": "Streptococcus", "pen": 0.005, "estr": 10.0, "neo": 40.0, "gram": "positiva"},
    # Gram negativas (9 bacterias)
    {"bacteria": "Aerobacter aerogenes", "abrev": "A. aerogenes", "genero": "Otro", "pen": 870.0, "estr": 1.0, "neo": 1.6, "gram": "negativa"},
    {"bacteria": "Brucella abortus", "abrev": "B. abortus", "genero": "Otro", "pen": 1.0, "estr": 2.0, "neo": 0.02, "gram": "negativa"},
    {"bacteria": "Escherichia coli", "abrev": "E. coli", "genero": "Otro", "pen": 100.0, "estr": 0.4, "neo": 0.1, "gram": "negativa"},
    {"bacteria": "Klebsiella pneumoniae", "abrev": "K. pneumoniae", "genero": "Otro", "pen": 850.0, "estr": 1.2, "neo": 1.0, "gram": "negativa"},
    {"bacteria": "Mycobacterium tuberculosis", "abrev": "M. tuberculosis", "genero": "Otro", "pen": 800.0, "estr": 5.0, "neo": 2.0, "gram": "negativa"},
    {"bacteria": "Proteus vulgaris", "abrev": "P. vulgaris", "genero": "Otro", "pen": 3.0, "estr": 0.1, "neo": 0.1, "gram": "negativa"},
    {"bacteria": "Pseudomonas aeruginosa", "abrev": "P. aeruginosa", "genero": "Otro", "pen": 850.0, "estr": 2.0, "neo": 0.4, "gram": "negativa"},
    {"bacteria": "Salmonella typhosa", "abrev": "S. typhosa", "genero": "Salmonella", "pen": 1.0, "estr": 0.4, "neo": 0.008, "gram": "negativa"},
    {"bacteria": "Salmonella schottmuelleri", "abrev": "S. schottmuelleri", "genero": "Salmonella", "pen": 10.0, "estr": 0.8, "neo": 0.09, "gram": "negativa"},
]


def _mediana(valores: List[float]) -> float:
    xs = sorted(valores)
    m = len(xs) // 2
    return float(xs[m]) if len(xs) % 2 == 1 else 0.5 * (xs[m - 1] + xs[m])


def analizar_burtin_1951() -> Dict[str, Any]:
    """
    Verifica las 16 bacterias de Burtin (1951), las medianas por grupo de Gram
    (Diapositiva 33), las 2 reclasificaciones taxonómicas (Diapositivas 34–36)
    y las 3 recetas clínicas de la Diapositiva 37.
    """
    gram_pos = [b for b in TABLA_BURTIN_1951 if b["gram"] == "positiva"]
    gram_neg = [b for b in TABLA_BURTIN_1951 if b["gram"] == "negativa"]

    med_pos = {
        "pen": _mediana([b["pen"] for b in gram_pos]),
        "estr": _mediana([b["estr"] for b in gram_pos]),
        "neo": _mediana([b["neo"] for b in gram_pos]),
    }
    med_neg = {
        "pen": _mediana([b["pen"] for b in gram_neg]),
        "estr": _mediana([b["estr"] for b in gram_neg]),
        "neo": _mediana([b["neo"] for b in gram_neg]),
    }

    # Distancia euclidiana en el plano (log10(pen), log10(neo)) de D. pneumoniae
    # al centroide de los estreptococos verdaderos (S. hemolyticus y S. viridans)
    d_pneu = next(b for b in TABLA_BURTIN_1951 if b["abrev"] == "D. pneumoniae")
    s_hem = next(b for b in TABLA_BURTIN_1951 if b["abrev"] == "S. hemolyticus")
    s_vir = next(b for b in TABLA_BURTIN_1951 if b["abrev"] == "S. viridans")
    s_fec = next(b for b in TABLA_BURTIN_1951 if b["abrev"] == "S. fecalis")

    def dist_log(b1: Dict[str, Any], b2: Dict[str, Any]) -> float:
        return math.hypot(
            math.log10(b1["pen"]) - math.log10(b2["pen"]),
            math.log10(b1["neo"]) - math.log10(b2["neo"]),
        )

    return {
        "n_total": len(TABLA_BURTIN_1951),
        "n_gram_pos": len(gram_pos),
        "n_gram_neg": len(gram_neg),
        "medianas_gram_pos": med_pos,
        "medianas_gram_neg": med_neg,
        "razon_pen_vs_estr_gram_pos": med_pos["estr"] / med_pos["pen"],  # 1.0 / 0.005 = 200x
        "razon_pen_neg_vs_pos": med_neg["pen"] / med_pos["pen"],        # 100.0 / 0.005 = 20000x
        "dist_log_dpneu_svir": dist_log(d_pneu, s_vir),
        "dist_log_sfec_svir": dist_log(s_fec, s_vir),
        "razon_sfec_vs_estreptococos": (
            s_fec["pen"] / s_vir["pen"],
            s_fec["pen"] / s_hem["pen"],
        ),  # (200x, 1000x)
    }


def demo_caso_burtin_antibioticos_1951() -> Dict[str, Any]:
    """Ejecuta y verifica la Sección 2 (Diapositivas 28–38)."""
    seccion("2. Caso Clásico: Los Antibióticos de Will Burtin (1951, Diaps. 28–38)")
    res = analizar_burtin_1951()
    assert res["n_total"] == 16
    assert res["n_gram_pos"] == 7 and res["n_gram_neg"] == 9
    assert abs(res["medianas_gram_pos"]["pen"] - 0.005) < 1e-9
    assert abs(res["medianas_gram_pos"]["estr"] - 1.0) < 1e-9
    assert abs(res["medianas_gram_pos"]["neo"] - 0.1) < 1e-9
    assert abs(res["medianas_gram_neg"]["pen"] - 100.0) < 1e-9
    assert abs(res["medianas_gram_neg"]["estr"] - 1.0) < 1e-9
    assert abs(res["medianas_gram_neg"]["neo"] - 0.1) < 1e-9
    assert abs(res["razon_pen_vs_estr_gram_pos"] - 200.0) < 1e-9
    assert abs(res["razon_pen_neg_vs_pos"] - 20000.0) < 1e-9
    assert res["razon_sfec_vs_estreptococos"] == (200.0, 1000.0)

    print(
        f"  • Dataset de Burtin (1951): {res['n_total']} bacterias "
        f"({res['n_gram_pos']} Gram+, {res['n_gram_neg']} Gram-), CIM en [0.001, 870] (6 órdenes de magnitud)."
    )
    print("  • Pregunta 1 (del médico): ¿Qué tan efectivo es cada antibiótico? (Medianas CIM, Diap. 33):")
    mp, mn = res["medianas_gram_pos"], res["medianas_gram_neg"]
    print(
        f"    - Gram positivas: penicilina = {mp['pen']}, estreptomicina = {mp['estr']}, neomicina = {mp['neo']} "
        f"→ penicilina es {res['razon_pen_vs_estr_gram_pos']:.0f}× más potente que estreptomicina."
    )
    print(
        f"    - Gram negativas: penicilina = {mn['pen']:.0f}, estreptomicina = {mn['estr']}, neomicina = {mn['neo']} "
        f"→ gana neomicina (0.1); penicilina es {res['razon_pen_neg_vs_pos']:.0f}× menos potente que en Gram+."
    )
    print("  • Pregunta 2 (del biólogo): Dos etiquetas que no calzaban en log(pen) vs. log(neo) (Diaps. 34–36):")
    print(
        f"    1. D. pneumoniae (pen=0.005, neo=10) cae a distancia log = {res['dist_log_dpneu_svir']:.2f} "
        "de S. viridans → reclasificada en 1974 como Streptococcus pneumoniae."
    )
    print(
        f"    2. S. fecalis (pen=1, neo=0.1) requiere {res['razon_sfec_vs_estreptococos'][0]:.0f}× a "
        f"{res['razon_sfec_vs_estreptococos'][1]:.0f}× más penicilina → reclasificada en 1984 como Enterococcus faecalis."
    )
    print(f"  {C.GREEN}✓ Sección 2 verificada correctamente.{C.RESET}")
    return res


# =============================================================================
# 3. INTEGRIDAD GRÁFICA (TUFTE), PERCEPCIÓN (STEVENS/MACKINLAY) Y COLOR (DIAPS. 39–75)
# =============================================================================
EXPONENTES_STEVENS_1961: List[Dict[str, Any]] = [
    {"estimulo": "Descarga eléctrica", "beta": 3.50, "nota": "Se sobreestima violentamente (2× → 11.3×)"},
    {"estimulo": "Peso levantado", "beta": 1.45, "nota": "Se sobreestima"},
    {"estimulo": "Sabor (sal)", "beta": 1.30, "nota": "Se sobreestima"},
    {"estimulo": "Longitud", "beta": 1.00, "nota": "Fiel (10× → 10.0×): base de barras y posición"},
    {"estimulo": "Área", "beta": 0.70, "nota": "Se subestima (10× → 5.0×, a la mitad)"},
    {"estimulo": "Volumen", "beta": 0.60, "nota": "Se subestima fuertemente"},
    {"estimulo": "Sonoridad", "beta": 0.60, "nota": "Se subestima"},
    {"estimulo": "Olor", "beta": 0.55, "nota": "Se subestima"},
    {"estimulo": "Brillo", "beta": 0.33, "nota": "Se subestima severamente (10× → 2.1×, a un quinto)"},
]

RANKING_MACKINLAY_1986: List[Dict[str, Any]] = [
    {"rango": 1, "cuantitativo": "Posición", "ordinal": "Posición", "nominal": "Posición"},
    {"rango": 2, "cuantitativo": "Longitud", "ordinal": "Densidad (gris)", "nominal": "Tono (color)"},
    {"rango": 3, "cuantitativo": "Ángulo", "ordinal": "Saturación", "nominal": "Textura"},
    {"rango": 4, "cuantitativo": "Pendiente", "ordinal": "Tono (color)", "nominal": "Conexión"},
    {"rango": 5, "cuantitativo": "Área", "ordinal": "Textura", "nominal": "Contención"},
    {"rango": 6, "cuantitativo": "Volumen", "ordinal": "Conexión", "nominal": "Densidad (gris)"},
    {"rango": 7, "cuantitativo": "Densidad (gris)", "ordinal": "Contención", "nominal": "Saturación"},
    {"rango": 8, "cuantitativo": "Saturación", "ordinal": "Longitud", "nominal": "Forma"},
    {"rango": 9, "cuantitativo": "Tono (color)", "ordinal": "Ángulo", "nominal": "Longitud"},
]


def calcular_lie_factor_tufte(
    valor_1: float, valor_2: float, base_eje: float = 0.0
) -> Dict[str, float]:
    """
    Calcula el Factor de Mentira (Lie Factor) de Edward Tufte (1983, Diaps. 40–41):
      LieFactor = (tamaño relativo del efecto en el gráfico) / (tamaño relativo del efecto en los datos)
    Cuando una barra se trunca en `base_eje`:
      altura_1 = valor_1 - base_eje, altura_2 = valor_2 - base_eje
      efecto_grafico = (altura_2 - altura_1) / altura_1
      efecto_datos   = (valor_2 - valor_1) / valor_1
    """
    h1 = valor_1 - base_eje
    h2 = valor_2 - base_eje
    razon_alturas = h2 / h1
    efecto_grafico = (h2 - h1) / h1
    efecto_datos = (valor_2 - valor_1) / valor_1
    lie_factor = efecto_grafico / efecto_datos
    return {
        "valor_1": valor_1,
        "valor_2": valor_2,
        "base_eje": base_eje,
        "altura_1": h1,
        "altura_2": h2,
        "razon_alturas": razon_alturas,
        "efecto_grafico": efecto_grafico,
        "efecto_datos": efecto_datos,
        "lie_factor": lie_factor,
    }


def luminancia_rec709(r: float, g: float, b: float) -> float:
    """
    Luminancia relativa Rec. 709 (Diapositiva 68):
      Y = 0.2126·R + 0.7152·G + 0.0722·B, con R, G, B en [0, 1].
    """
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def demo_tufte_lie_factor_percepcion_y_color() -> Dict[str, Any]:
    """Ejecuta y verifica la Sección 3 (Diapositivas 39–49 y 57–75)."""
    seccion("3. Integridad Gráfica (Tufte), Percepción (Stevens/Mackinlay) y Color (Diaps. 39–75)")

    # Caso 1: If Bush Tax Cuts Expire (Fox Business, 2012, Diap. 41): 35% -> 39.6%, base en 34%
    bush = calcular_lie_factor_tufte(35.0, 39.6, base_eje=34.0)
    assert abs(bush["razon_alturas"] - 5.6) < 1e-9
    assert abs(bush["efecto_grafico"] - 4.6) < 1e-9
    assert abs(bush["efecto_datos"] - (4.6 / 35.0)) < 1e-9
    assert abs(bush["lie_factor"] - 35.0) < 1e-9

    # Caso 2: Auditoría de las 4 marcas A=52%, B=53%, C=54%, D=56% con base en 50% (Diap. 49)
    marcas = calcular_lie_factor_tufte(52.0, 56.0, base_eje=50.0)
    assert abs(marcas["razon_alturas"] - 3.0) < 1e-9
    assert abs(marcas["lie_factor"] - 26.0) < 1e-9

    print("  • Lie Factor de Tufte (1983, Diaps. 40–41) — 'If Bush Tax Cuts Expire' (35% → 39.6%, base = 34%):")
    print(
        f"    - Alturas visuales: {bush['altura_1']:.1f} vs. {bush['altura_2']:.1f} "
        f"({bush['razon_alturas']:.1f}× más alta → efecto gráfico = +{bush['efecto_grafico']*100:.0f}%)"
    )
    print(
        f"    - Efecto real en los datos: (39.6 - 35)/35 = +{bush['efecto_datos']*100:.2f}%  "
        f"→  Lie Factor = {bush['efecto_grafico']:.1f} / {bush['efecto_datos']:.4f} = {bush['lie_factor']:.1f}"
    )
    print(
        f"  • Auditoría de las 4 marcas A=52%..D=56% con eje en 50% (Diap. 49): "
        f"D parece {marcas['razon_alturas']:.1f}× A (Lie Factor = {marcas['lie_factor']:.1f})."
    )

    # Ley de potencias de Stevens (1961, Diaps. 61–62)
    perc_long_10x = 10.0 ** 1.0
    perc_area_10x = 10.0 ** 0.7
    perc_brillo_10x = 10.0 ** 0.33
    perc_shock_2x = 2.0 ** 3.5
    assert abs(perc_long_10x - 10.0) < 1e-9
    assert abs(perc_area_10x - 5.011872336) < 1e-6
    assert abs(perc_brillo_10x - 2.13796209) < 1e-6
    assert abs(perc_shock_2x - 11.3137085) < 1e-6

    print("\n  • Ley de Potencias de Stevens (1961, Diaps. 61–62): percibido ∝ intensidad^β")
    print(
        f"    - Longitud (β = 1.00): un valor 10× se percibe como {perc_long_10x:.1f}× (fiel)."
    )
    print(
        f"    - Área     (β = 0.70): un valor 10× se percibe como {perc_area_10x:.1f}× (a la mitad)."
    )
    print(
        f"    - Brillo   (β = 0.33): un valor 10× se percibe como {perc_brillo_10x:.1f}× (a un quinto)."
    )
    print(
        f"    - Descarga (β = 3.50): el doble de corriente (2×) se siente {perc_shock_2x:.1f}× más fuerte."
    )

    # Luminancia Rec. 709 en 'jet' vs 'viridis' (Diap. 68)
    # Puntos clave de 'jet': azul oscuro (0,0,0.5) -> cian (0,1,1) -> verde/amarillo (1,1,0) -> rojo oscuro (0.5,0,0)
    jet_rgb = [(0.0, 0.0, 0.5), (0.0, 0.5, 1.0), (0.0, 1.0, 1.0), (0.5, 1.0, 0.5), (1.0, 1.0, 0.0), (1.0, 0.3, 0.0), (0.5, 0.0, 0.0)]
    jet_lum = [round(luminancia_rec709(*rgb), 3) for rgb in jet_rgb]
    # Viridis sube monótonamente de 0.02 a 0.78 (Diap. 68)
    viridis_lum = [0.02, 0.12, 0.25, 0.39, 0.54, 0.67, 0.78]
    es_viridis_monotona = all(viridis_lum[i] < viridis_lum[i + 1] for i in range(len(viridis_lum) - 1))
    es_jet_monotona = all(jet_lum[i] < jet_lum[i + 1] for i in range(len(jet_lum) - 1))
    assert es_viridis_monotona and not es_jet_monotona

    print("\n  • Luminancia relativa Rec. 709 (Y = 0.2126R + 0.7152G + 0.0722B, Diaps. 68–70):")
    print(f"    - Arcoíris ('jet'):   Y = {jet_lum} → NO monótona (crea bordes falsos).")
    print(f"    - 'viridis':          Y = {viridis_lum} → estrictamente monótona (0.02 → 0.78).")
    print(f"  {C.GREEN}✓ Sección 3 verificada correctamente.{C.RESET}")
    return {
        "bush_lie_factor": bush,
        "marcas_lie_factor": marcas,
        "jet_lum": jet_lum,
        "viridis_lum": viridis_lum,
    }


# =============================================================================
# 4. TAXONOMÍA DE CORRELL, ESPEJISMOS (MIRAGES) Y ÉTICA VISUAL (DIAPS. 76–103)
# =============================================================================
TAXONOMIA_CORRELL: List[Dict[str, str]] = [
    {
        "categoria": "1. Incorrectas",
        "definicion": "Codifican mal los datos (los números o posiciones no corresponden a las marcas visuales).",
        "ejemplos_diaps": "CNN Hong Kong en Brasil (Diap. 78), Fox 2385° en Wickenburg (Diap. 79), Daily Nexus sin escala (Diap. 80), Circular de 193% = 70%+63%+60% (Diap. 82), Doble eje manipulado (Diaps. 83–84).",
    },
    {
        "categoria": "2. Ilegibles",
        "definicion": "Los datos están ahí, pero ahogan al lector en detalle u ocultan el patrón por mal formato.",
        "ejemplos_diaps": "Circular de 24 porciones con 5 tonos repetidos (Diap. 86), serie temporal de cesantía en Alabama graficada como burbujas desordenadas (Diap. 87).",
    },
    {
        "categoria": "3. Bullshit",
        "definicion": "Indiferencia por la verdad (Harry Frankfurt): parece comunicar algo cuantitativo, pero está vacío o inventado.",
        "ejemplos_diaps": "Decline of Artistic Standards sin unidades ni fuente (Diap. 89), tablas o mapas que solo repiten dónde vive la gente sin tasa ni denominador (Diap. 90).",
    },
    {
        "categoria": "4. No convencionales",
        "definicion": "Los números son correctos, pero rompen una convención de lectura (cero arriba, más oscuro = menos).",
        "ejemplos_diaps": "Muertes por armas en Florida con eje 0 arriba: sube a 873 tras 2005 pero la curva 'cae' (Diaps. 91–92) vs. Iraq's Bloody Toll de S. Scarr (Diap. 93); mapa de ITS donde más oscuro = menos infecciones (Diaps. 94–95).",
    },
]

CASOS_CLASIFICACION_DIAP100: List[Dict[str, str]] = [
    {
        "caso": "1. Mapa donde el verde intenso marca las comunas con menos áreas verdes",
        "categoria": "No convencional",
        "razon": "Rompe la convención perceptual 'más oscuro/intenso = más cantidad' y la asociación semántica del verde.",
    },
    {
        "caso": "2. Curva de 'índice de felicidad nacional' sin unidades ni fuente y perfectamente suave",
        "categoria": "Bullshit",
        "razon": "Usa la estética de la ciencia (ejes, curva suave) con total indiferencia por si la magnitud existe o se midió.",
    },
    {
        "caso": "3. Circular de participación de mercado cuyas partes suman 140%",
        "categoria": "Incorrecta",
        "razon": "Un gráfico circular codifica partes excluyentes de un todo (100%); sumar 140% viola la geometría del ángulo.",
    },
    {
        "caso": "4. Red de 3 000 nodos y 40 000 aristas presentada como 'resumen ejecutivo'",
        "categoria": "Ilegible",
        "razon": "Una bola de pelos ('hairball') de 40 000 aristas oculta cualquier estructura y ahoga al lector en detalle.",
    },
]


def simular_espejismo_energia_2012_diap97() -> Dict[str, Any]:
    """
    Reproduce el espejismo de agregación (Visualization Mirage, McNutt, Kindlmann
    y Correll, CHI 2020, Diapositivas 96–98) sobre datos del Banco Mundial:
      - Hasta 2011 reportan ~145 países (~70 sin dato):
          * 35 economías grandes (consumo medio ≈ 4 800 unidades)
          * 110 economías medianas/pequeñas (consumo medio ≈ 1 100 unidades)
      - En 2012 los países sin dato saltan de ~70 a ~175: dejan de reportar las 105
        economías pequeñas/medianas y solo quedan 40 países de alto consumo.
      - Consecuencia:
          * SUM(Energy Usage) se desploma en 2012 (parece una caída mundial).
          * AVG(Energy Usage) se dispara en 2012 (parece un auge brusco).
          * El fenómeno real está en COUNT(NULL): un cambio en quién reporta.
    """
    # Año 2011: 35 países grandes (4800 c/u) + 110 países pequeños/medianos (1100 c/u) = 145 reportan, 72 sin dato
    n_grandes_2011, media_grandes_2011 = 35, 4800.0
    n_peq_2011, media_peq_2011 = 110, 1100.0
    faltantes_2011 = 72
    reportan_2011 = n_grandes_2011 + n_peq_2011
    sum_2011 = n_grandes_2011 * media_grandes_2011 + n_peq_2011 * media_peq_2011
    avg_2011 = sum_2011 / reportan_2011

    # Año 2012: los mismos 35 países grandes (4850 c/u, +1% real) + solo 5 pequeños (1110 c/u) = 40 reportan, 177 sin dato
    n_grandes_2012, media_grandes_2012 = 35, 4850.0
    n_peq_2012, media_peq_2012 = 5, 1110.0
    faltantes_2012 = 177
    reportan_2012 = n_grandes_2012 + n_peq_2012
    sum_2012 = n_grandes_2012 * media_grandes_2012 + n_peq_2012 * media_peq_2012
    avg_2012 = sum_2012 / reportan_2012

    cambio_sum_pct = (sum_2012 - sum_2011) / sum_2011 * 100.0
    cambio_avg_pct = (avg_2012 - avg_2011) / avg_2011 * 100.0

    return {
        "2011": {"reportan": reportan_2011, "faltantes": faltantes_2011, "sum": sum_2011, "avg": avg_2011},
        "2012": {"reportan": reportan_2012, "faltantes": faltantes_2012, "sum": sum_2012, "avg": avg_2012},
        "cambio_sum_pct": cambio_sum_pct,
        "cambio_avg_pct": cambio_avg_pct,
    }


def demo_taxonomia_correll_y_espejismos() -> Dict[str, Any]:
    """Ejecuta y verifica la Sección 4 (Diapositivas 76–103)."""
    seccion("4. Visualización Engañosa: Taxonomía de Correll, Espejismos y Ética (Diaps. 76–103)")

    # Verificación del circular de 193% (Fox News, 2009, Diap. 82)
    suma_fox_pie = 70 + 63 + 60
    assert suma_fox_pie == 193

    # Verificación del doble eje manipulado (Diap. 83):
    # Nosotros: 300 -> 340 (+13.33%); Rival: 88 -> 101 (+14.77%)
    crec_nosotros = (340.0 - 300.0) / 300.0 * 100.0
    crec_rival = (101.0 - 88.0) / 88.0 * 100.0
    assert abs(crec_nosotros - 13.333333) < 1e-4
    assert abs(crec_rival - 14.772727) < 1e-4

    print("  • Las 4 categorías de Michael Correll (Tableau Research, Diap. 77):")
    for cat in TAXONOMIA_CORRELL:
        print(f"    - {C.BOLD}{cat['categoria']:<22}{C.RESET}: {cat['definicion']}")

    print(
        f"\n  • Auditoría numérica de casos clásicos:"
        f"\n    - Circular de Fox News (Diap. 82): 70% + 63% + 60% = {suma_fox_pie}% (respuesta múltiple en un todo)."
        f"\n    - Doble eje (Diap. 83): nosotros 300→340 (+{crec_nosotros:.1f}%) vs. rival 88→101 (+{crec_rival:.1f}%): el cruce se fabrica con los límites."
        f"\n    - Florida Gun Deaths (Diaps. 91–92): tras la ley Stand Your Ground (2005), las muertes suben a 873 (y cierran en 721), pero el eje con 0 arriba hace que la curva 'caiga'."
    )

    mirage = simular_espejismo_energia_2012_diap97()
    assert mirage["cambio_sum_pct"] < -35.0
    assert mirage["cambio_avg_pct"] > 100.0

    print("\n  • Espejismos de Visualización (McNutt, Kindlmann y Correll, CHI 2020, Diaps. 96–98):")
    print(
        f"    - En 2012 los países sin dato saltan de {mirage['2011']['faltantes']} a {mirage['2012']['faltantes']}:"
    )
    print(
        f"      * SUM(Energy Usage): pasa de {mirage['2011']['sum']:,.0f} a {mirage['2012']['sum']:,.0f} "
        f"({mirage['cambio_sum_pct']:+.1f}% → parece un desplome)."
    )
    print(
        f"      * AVG(Energy Usage): pasa de {mirage['2011']['avg']:,.1f} a {mirage['2012']['avg']:,.1f} "
        f"({mirage['cambio_avg_pct']:+.1f}% → parece un auge brusco)."
    )

    print("\n  • Clasificación de las 4 situaciones de la Diapositiva 100:")
    for item in CASOS_CLASIFICACION_DIAP100:
        print(f"    {item['caso']}  →  {C.BOLD}{item['categoria']}{C.RESET}")

    print(f"  {C.GREEN}✓ Sección 4 verificada correctamente.{C.RESET}")
    return {"suma_fox_pie": suma_fox_pie, "mirage_2012": mirage, "casos_diap100": CASOS_CLASIFICACION_DIAP100}


# =============================================================================
# EJECUCIÓN COMPLETA Y MENÚ INTERACTIVO
# =============================================================================
def ejecutar_todo_cd03() -> Dict[str, Any]:
    """Ejecuta y verifica las 4 secciones completas de la Unidad CD03."""
    banner(
        "CD03 — VISUALIZACIÓN DE DATOS EFECTIVA Y ENGAÑOSA (103 Diapositivas)",
        "Ciencia de Datos • UCN — Basado en CMU 15-388/688 (Z. Kolter) y M. Correll (Tableau Research)",
    )
    r1 = demo_propositos_escalas_y_catalogo()
    r2 = demo_caso_burtin_antibioticos_1951()
    r3 = demo_tufte_lie_factor_percepcion_y_color()
    r4 = demo_taxonomia_correll_y_espejismos()
    print(f"\n{C.BOLD}{C.GREEN}✔ TODAS LAS VERIFICACIONES DE CD03 (103 DIAPOSITIVAS) PASARON AL 100%.{C.RESET}\n")
    return {"s1": r1, "s2": r2, "s3": r3, "s4": r4}


def menu_interactivo() -> None:
    while True:
        banner(
            "MENÚ INTERACTIVO — CD03: VISUALIZACIÓN DE DATOS EFECTIVA Y ENGAÑOSA",
            "Seleccione una demostración o ejecute la suite completa",
        )
        print("  [1] Propósitos, Datasaurus Dozen, Escalas, Box Plot y Ancho de Bin (Diaps. 1–27, 50–56)")
        print("  [2] Caso Clásico: Los 16 Antibióticos de Will Burtin (1951) y la Biología (Diaps. 28–38)")
        print("  [3] Integridad (Tufte Lie Factor), Percepción (Stevens/Mackinlay) y Color (Diaps. 39–75)")
        print("  [4] Taxonomía de Correll, Espejismos de Agregación (2012) y Ética Visual (Diaps. 76–103)")
        print("  [5] Ejecutar todas las demostraciones de CD03")
        print("  [0] Salir")
        op = input("\n  Opción: ").strip()
        if op == "1":
            demo_propositos_escalas_y_catalogo()
        elif op == "2":
            demo_caso_burtin_antibioticos_1951()
        elif op == "3":
            demo_tufte_lie_factor_percepcion_y_color()
        elif op == "4":
            demo_taxonomia_correll_y_espejismos()
        elif op == "5":
            ejecutar_todo_cd03()
        elif op == "0":
            break
        else:
            print(f"  {C.RED}Opción no válida.{C.RESET}")


if __name__ == "__main__":
    if "--auto" in sys.argv:
        ejecutar_todo_cd03()
    else:
        menu_interactivo()
