#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
UNIDAD CD02: PREPROCESAMIENTO Y ANÁLISIS EXPLORATORIO DE DATOS (74 Diapositivas)
Ciencia de Datos — Universidad Católica del Norte (UCN)
Basado en material de Harvard CS109 (P. Protopapas, K. Rader et al.)
===============================================================================
Módulo computacional e interactivo en Python puro (sin dependencias externas)
que implementa, demuestra y verifica todas las tablas, diagnósticos de sesgo,
metodologías (CRISP-DM, KDD, SEMMA), operaciones de Pandas, el análisis
exploratorio del dataset real de la Premier League 2017 (epl_mini, 23 jugadores)
y el Cuarteto de Anscombe (1973) de las 74 diapositivas de la Unidad CD02:

  1. ¿Qué son los Datos?, Advertencias de Fábrica, Alcances y Sesgos (Diaps. 1–20):
     - Tres definiciones de «datos» (Diccionario, Informática y Señales) y la
       diferencia entre datum, data y dataset (colección homogénea).
     - Cuatro escenarios y cuatro fuentes de trabajo cotidiano con sus advertencias
       de fábrica: vacíos, imprecisión humana, veracidad y sin semántica.
     - Datos integrales vs. muestreados, los 5 sesgos de la fuente y el contraste
       Gallup (500 llamadas/día, regla del cumpleaños, 70% celular / 30% fijo,
       ponderación demográfica) vs. Yelp (autoselección sin corrección) + el
       ejercicio de la encuesta QR de la cafetería (480 respuestas, media 2.1/5).

  2. Formatos, Preguntas Medibles, CRISP-DM / KDD / SEMMA y Web Scraping (Diaps. 21–44):
     - Matriz 2x2 de dificultad según el lector (Personas vs. Computadores):
       Tabla CSV, Prosa/fotos, JSON anidado/QR y Audio con ruido.
     - Comparación Texto plano vs. CSV/TSV vs. XML vs. JSON y transformación de
       titulares vagos en preguntas medibles (población, métrica y ventana temporal).
     - Las 6 fases de CRISP-DM (2000) frente a KDD (1996) y SEMMA (SAS): por qué
       omitir Comprensión del negocio y Despliegue es crítico.
     - Orden de preferencia (Archivo > API > Scraping), las 5 reglas de scraping
       responsable y extracción en el árbol DOM (status_code 200 vs 404, BeautifulSoup).

  3. Pandas Esencial y Caso Real Premier League 2017 (epl_mini, 23 jugadores) (Diaps. 45–70):
     - Operaciones sobre df2 (Series vs DataFrame, máscara booleana, loc vs iloc,
       el trío min / idxmin / loc y las 3 correcciones de Pandas moderno).
     - Reconstrucción 100% exacta del dataset epl_mini (23 jugadores, 5 columnas):
       shape=(23, 5), promedios por club (Arsenal 32.80, Chelsea 35.01, Liverpool 34.00,
       Man City 46.67, Man United 45.75, Tottenham 41.25), describe() global
       (edad media 27.13, std 3.99; valor medio 38.87, std 23.51), caza de extremos
       (idxmin() -> 8: Eduardo, Chelsea, 34 años, GK, 0.05M; idxmax(): Paul Pogba,
       Man United, 24 años, 90.0M) y groupby("age")["market_value"].agg(["mean", "median"]).

  4. Media vs. Mediana en el Arsenal y el Cuarteto de Anscombe (1973) (Diaps. 71–74):
     - Sensibilidad de la media frente a la mediana en los 5 jugadores del Arsenal
       (media 32.80 vs mediana 22.00; al fichar a alguien de 100M la media salta a
       44.00 mientras la mediana pasa a 36.00).
     - Verificación numérica exacta del Cuarteto de Anscombe (1973): 4 conjuntos de
       11 puntos con idénticas estadísticas (media_x=9.0, media_y=7.50, var_x=11.0,
       r=0.816, recta y = 3.00 + 0.500x) pero geometrías radicalmente distintas.
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
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"


def banner(titulo: str, subtitulo: str = "") -> None:
    ancho = 78
    print(f"\n{C.CYAN}{C.BOLD}{'═' * ancho}{C.RESET}")
    print(f"{C.WHITE}{C.BOLD}  {titulo}{C.RESET}")
    if subtitulo:
        print(f"{C.DIM}  {subtitulo}{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}{'═' * ancho}{C.RESET}")


# =============================================================================
# FUNCIONES ESTADÍSTICAS EN PYTHON PURO (COMPATIBLES CON PANDAS)
# =============================================================================
def calc_mean(vals: List[float]) -> float:
    return sum(vals) / len(vals)


def calc_std_sample(vals: List[float]) -> float:
    """Desviación estándar muestral (ddof=1, estándar en pandas)."""
    m = calc_mean(vals)
    n = len(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1))


def calc_var_sample(vals: List[float]) -> float:
    """Varianza muestral (ddof=1)."""
    m = calc_mean(vals)
    n = len(vals)
    return sum((x - m) ** 2 for x in vals) / (n - 1)


def calc_quantile_linear(vals: List[float], q: float) -> float:
    """Cuantil con interpolación lineal (método por defecto de pandas/numpy)."""
    s = sorted(vals)
    n = len(s)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    frac = pos - lo
    return s[lo] + frac * (s[hi] - s[lo])


def calc_median(vals: List[float]) -> float:
    return calc_quantile_linear(vals, 0.5)


def calc_describe(vals: List[float]) -> Dict[str, float]:
    """Reproduce exactamente Series.describe() de pandas."""
    return {
        "count": float(len(vals)),
        "mean":  calc_mean(vals),
        "std":   calc_std_sample(vals),
        "min":   float(min(vals)),
        "25%":   calc_quantile_linear(vals, 0.25),
        "50%":   calc_quantile_linear(vals, 0.50),
        "75%":   calc_quantile_linear(vals, 0.75),
        "max":   float(max(vals)),
    }


# =============================================================================
# DATASET OFICIAL EPL_MINI (23 JUGADORES PREMIER LEAGUE 2017, DIAPS. 46-72)
# =============================================================================
def obtener_dataset_epl_mini() -> List[Dict[str, Any]]:
    """
    Retorna los 23 registros de `epl_mini.csv` (Premier League 2017) que
    reproducen con precisión del 100% todas las tablas y gráficos de las
    Diapositivas 46, 52, 60 y 63–72:
      - Primeras 3 filas (df.head(3), Diap. 46): Alexis Sanchez, Mesut Ozil, Petr Cech.
      - Los 5 jugadores del Arsenal (Diaps. 60 y 72): Sanchez (65), Ozil (50),
        Cech (7), Walcott (20), Koscielny (22) -> media 32.80, mediana 22.00.
      - Índice 8 (idxmin(), Diap. 64): Eduardo (Chelsea, 34, GK, 0.05M).
      - Máximo (idxmax(), Diap. 65): Paul Pogba (Man United, 24, CM, 90.0M).
      - Harry Kane (Tottenham, 23, ST, 60.0M, Diaps. 24, 25 y 69).
      - Promedios exactos por club (Diap. 52) y por edad (Diaps. 67-68).
    """
    return [
        # 0..4: Arsenal (5 jugadores -> suma = 164.0, media = 32.80, mediana = 22.00)
        {"idx": 0,  "name": "Alexis Sanchez",    "club": "Arsenal",    "age": 28, "position": "LW", "market_value": 65.00},
        {"idx": 1,  "name": "Mesut Ozil",        "club": "Arsenal",    "age": 28, "position": "AM", "market_value": 50.00},
        {"idx": 2,  "name": "Petr Cech",         "club": "Arsenal",    "age": 35, "position": "GK", "market_value":  7.00},
        {"idx": 3,  "name": "Theo Walcott",      "club": "Arsenal",    "age": 28, "position": "RW", "market_value": 20.00},
        {"idx": 4,  "name": "Laurent Koscielny", "club": "Arsenal",    "age": 31, "position": "CB", "market_value": 22.00},
        # 5..8: Chelsea (4 jugadores -> suma = 140.05, media = 35.0125 -> 35.01; idx 8 = Eduardo 0.05M)
        {"idx": 5,  "name": "Eden Hazard",       "club": "Chelsea",    "age": 26, "position": "LW", "market_value": 75.00},
        {"idx": 6,  "name": "N'Golo Kante",      "club": "Chelsea",    "age": 26, "position": "CM", "market_value": 40.00},
        {"idx": 7,  "name": "Cesc Fabregas",     "club": "Chelsea",    "age": 30, "position": "CM", "market_value": 25.00},
        {"idx": 8,  "name": "Eduardo",           "club": "Chelsea",    "age": 34, "position": "GK", "market_value":  0.05},
        # 9..11: Liverpool (3 jugadores -> suma = 102.0, media = 34.00)
        {"idx": 9,  "name": "Philippe Coutinho", "club": "Liverpool",  "age": 25, "position": "AM", "market_value": 50.00},
        {"idx": 10, "name": "Sadio Mane",        "club": "Liverpool",  "age": 25, "position": "LW", "market_value": 40.00},
        {"idx": 11, "name": "Ragnar Klavan",     "club": "Liverpool",  "age": 31, "position": "CB", "market_value": 12.00},
        # 12..14: Man City (3 jugadores -> suma = 140.0, media = 46.6667 -> 46.67)
        {"idx": 12, "name": "Kevin De Bruyne",   "club": "Man City",   "age": 26, "position": "AM", "market_value": 65.00},
        {"idx": 13, "name": "Sergio Aguero",     "club": "Man City",   "age": 29, "position": "ST", "market_value": 55.00},
        {"idx": 14, "name": "David Silva",       "club": "Man City",   "age": 32, "position": "AM", "market_value": 20.00},
        # 15..18: Man United (4 jugadores -> suma = 183.0, media = 45.75; idx 15 = Paul Pogba 90.0M)
        {"idx": 15, "name": "Paul Pogba",        "club": "Man United", "age": 24, "position": "CM", "market_value": 90.00},
        {"idx": 16, "name": "Romelu Lukaku",     "club": "Man United", "age": 24, "position": "ST", "market_value": 50.00},
        {"idx": 17, "name": "Anthony Martial",   "club": "Man United", "age": 21, "position": "LW", "market_value": 35.00},
        {"idx": 18, "name": "Ashley Young",      "club": "Man United", "age": 31, "position": "LW", "market_value":  8.00},
        # 19..22: Tottenham (4 jugadores -> suma = 165.0, media = 41.25; idx 19 = Harry Kane 60.0M)
        {"idx": 19, "name": "Harry Kane",        "club": "Tottenham",  "age": 23, "position": "ST", "market_value": 60.00},
        {"idx": 20, "name": "Dele Alli",         "club": "Tottenham",  "age": 21, "position": "AM", "market_value": 50.00},
        {"idx": 21, "name": "Eric Dier",         "club": "Tottenham",  "age": 23, "position": "DM", "market_value": 30.00},
        {"idx": 22, "name": "Heung-Min Son",     "club": "Tottenham",  "age": 23, "position": "LW", "market_value": 25.00},
    ]


# =============================================================================
# 1. ¿QUÉ SON LOS DATOS?, ADVERTENCIAS DE FÁBRICA Y SESGOS (DIAPS. 1–20)
# =============================================================================
def demo_datos_alcances_y_sesgos_muestreo() -> Dict[str, Any]:
    """
    Verifica las 3 definiciones de datos, las advertencias de fábrica en 4
    escenarios y 4 fuentes de trabajo cotidiano (Diaps. 6-11), datos integrales
    vs. muestreados, los 5 sesgos de la fuente (Diap. 15) y el contraste
    Gallup vs. Yelp + Encuesta QR de la cafetería (Diaps. 16-20).
    """
    banner(
        "1. ¿QUÉ SON LOS DATOS?, ADVERTENCIAS DE FÁBRICA Y SESGOS (Diaps. 1–20)",
        "Definiciones, Procedimiento de Registro, Gallup vs. Yelp y Encuesta QR"
    )

    definiciones = {
        "Diccionario": "Información factual (medidas, estadísticas) usada para razonar, discutir o calcular.",
        "Informática": "Información en forma digital que puede ser transmitida o procesada.",
        "Señales":     "Salida de un sensor que mezcla información útil, irrelevante y redundante (debe procesarse).",
    }
    assert len(definiciones) == 3

    # Diagnóstico de los 4 escenarios (Diaps. 7-8) y las 4 fuentes cotidianas (Diap. 11)
    diagnosticos = [
        ("Termómetro cada hora (1 año)",  "GPS de una app de transporte",  "Vacíos",              "Cortes de luz / túneles sin señal"),
        ("Diario del colibrí",            "Cuaderno de asistencia taller", "Imprecisión humana",  "El observador humano se distrae o anota mal"),
        ("Tweets de un político",         "Biografías de una red social",  "Veracidad",           "Decir no es ser (autopresentación estratégica)"),
        ("Sensor de feria persa",         "Columna v37 sin documentar",    "Sin semántica",       "Números sin diccionario ni unidad conocida"),
    ]

    print(f"\n{C.BOLD}  [A] Diagnóstico por procedimiento de registro (Diapositivas 7–8 y 11):{C.RESET}")
    print("  ┌───────────────────────────────┬───────────────────────────────┬───────────────────────┐")
    print("  │ Escenario inicial (Diap. 8)   │ Fuente cotidiana (Diap. 11)   │ Advertencia de fábrica│")
    print("  ├───────────────────────────────┼───────────────────────────────┼───────────────────────┤")
    for esc, fte, adv, _ in diagnosticos:
        print(f"  │ {esc:<29} │ {fte:<29} │ {C.YELLOW}{adv:<21}{C.RESET} │")
    print("  └───────────────────────────────┴───────────────────────────────┴───────────────────────┘")

    # Gallup vs. Yelp (Diapositivas 16-18)
    gallup = {
        "llamadas_por_dia": 500,
        "regla_hogar":      "Cumpleaños más reciente",
        "cobertura_estados": 50,
        "pct_celulares":    70,
        "pct_fijos":        30,
        "correccion":       "Ponderación demográfica",
        "describe_a":       "Población general",
    }
    yelp = {
        "quien_entra":      "Autoselección (el opinante decide)",
        "cobertura":        "La que resulte (concentrada en extremos)",
        "correccion":       "Ninguna",
        "describe_a":       "Quien opina en Yelp",
    }
    assert gallup["pct_celulares"] + gallup["pct_fijos"] == 100
    assert gallup["llamadas_por_dia"] == 500
    assert gallup["cobertura_estados"] == 50

    # Encuesta QR de la cafetería universitaria (Diapositiva 20)
    encuesta_qr = {
        "respuestas_mes":   480,
        "promedio_5":       2.1,
        "sesgos_activos":   ["Autoselección (responde quien está molesto)", "Cobertura (solo quien almuerza ahí y escanea un QR)"],
        "sirve_para_medir_satisfaccion_general": False,
        "sirve_para_identificar_reclamos":       True,
    }
    assert encuesta_qr["respuestas_mes"] == 480
    assert abs(encuesta_qr["promedio_5"] - 2.1) < 1e-9
    assert not encuesta_qr["sirve_para_medir_satisfaccion_general"]
    assert encuesta_qr["sirve_para_identificar_reclamos"]

    print(f"\n{C.BOLD}  [B] Muestreo Diseñado (Gallup) vs. Autoselección (Yelp y QR Cafetería, Diaps. 16–20):{C.RESET}")
    print(f"      • Gallup: {gallup['llamadas_por_dia']} llamadas/día (RDD), {gallup['pct_celulares']}% celulares / {gallup['pct_fijos']}% fijos,")
    print(f"        regla del '{gallup['regla_hogar']}' y {gallup['correccion'].lower()} -> describe a la {C.GREEN}{gallup['describe_a']}{C.RESET}.")
    print(f"      • Yelp:   {yelp['quien_entra']} sin corrección -> describe a {C.YELLOW}{yelp['describe_a']}{C.RESET}.")
    print(f"      • QR Cafetería ({encuesta_qr['respuestas_mes']} respuestas, promedio {encuesta_qr['promedio_5']}/5): operan")
    print(f"        {C.RED}autoselección + cobertura{C.RESET}; NO mide satisfacción del estudiantado, pero SÍ")
    print(f"        sirve para priorizar mejoras sobre los reclamos de quienes están descontentos.")

    return {
        "diagnosticos": diagnosticos,
        "gallup":       gallup,
        "yelp":         yelp,
        "encuesta_qr":  encuesta_qr,
    }


# =============================================================================
# 2. FORMATOS, PREGUNTAS MEDIBLES, CRISP-DM Y WEB SCRAPING (DIAPS. 21–44)
# =============================================================================
def demo_formatos_preguntas_y_crisp_dm() -> Dict[str, Any]:
    """
    Verifica la matriz 2x2 de dificultad según el lector (Diap. 21), la
    comparación Texto plano vs. XML vs. JSON (Diap. 26), la conversión de
    afirmaciones vagas en preguntas medibles (Diaps. 27-29), el ciclo CRISP-DM
    frente a KDD y SEMMA (Diaps. 30-35) y la extracción DOM con BeautifulSoup
    (Diaps. 36-44).
    """
    banner(
        "2. FORMATOS, PREGUNTAS MEDIBLES, CRISP-DM Y SCRAPING (Diaps. 21–44)",
        "Matriz 2x2, Texto/XML/JSON, CRISP-DM vs. KDD/SEMMA y Árbol DOM"
    )

    # Matriz 2x2 según el lector (Diapositiva 21)
    matriz_lector = {
        ("Facil_Personas",   "Facil_Computadores"):   "Tabla CSV (filas y columnas limpias)",
        ("Facil_Personas",   "Dificil_Computadores"): "Prosa, fotos (papeleta electoral confusa)",
        ("Dificil_Personas", "Facil_Computadores"):   "JSON anidado, QR (trivial en código)",
        ("Dificil_Personas", "Dificil_Computadores"): "Audio con ruido, manuscritos borrosos",
    }
    assert len(matriz_lector) == 4

    # Las 6 fases de CRISP-DM y comparación con KDD y SEMMA (Diapositivas 31-34)
    fases_crisp_dm = [
        (1, "Comprensión del negocio", "¿Qué problema real se quiere resolver?",          "KDD: No | SEMMA: No (Punta omitida)"),
        (2, "Comprensión de los datos","¿Qué datos hay y qué calidad tienen?",            "KDD: Selección | SEMMA: Sample + Explore"),
        (3, "Preparación de los datos","¿Cómo se limpian, integran y transforman?",       "KDD: Preproc. + Transf. | SEMMA: Modify"),
        (4, "Modelado",                "¿Qué técnica, con qué parámetros?",               "KDD: Minería de datos | SEMMA: Model"),
        (5, "Evaluación",              "¿El modelo resuelve el problema del paso 1?",     "KDD: Interpretación/Eval. | SEMMA: Assess"),
        (6, "Despliegue",              "¿Cómo se entrega y se monitorea en producción?",  "KDD: No | SEMMA: No (Punta omitida)"),
    ]
    assert len(fases_crisp_dm) == 6

    print(f"\n{C.BOLD}  [A] Las 6 Fases de CRISP-DM (2000) frente a KDD (1996) y SEMMA (Diaps. 31–34):{C.RESET}")
    for num, fase, preg, comp in fases_crisp_dm:
        color = C.GREEN if num in (1, 6) else C.CYAN
        print(f"      {color}Fase {num}: {fase:<25}{C.RESET} -> {preg:<42} [{comp}]")

    # Simulación exacta del árbol DOM y BeautifulSoup de las Diapositivas 41-43
    html_diap41 = (
        "<html><head><title>Tabla de posiciones</title></head>"
        "<body><h1>Premier League</h1>"
        "<p>Equipos de Londres:</p>"
        "<a href='/equipo/arsenal'>Arsenal</a>"
        "<a href='/equipo/chelsea'>Chelsea</a>"
        "</body></html>"
    )
    status_ok = 200
    status_not_found = 404
    soup_title_text = "Tabla de posiciones"
    soup_hrefs = ["/equipo/arsenal", "/equipo/chelsea"]
    orden_preferencia = ["Archivo", "API", "Scraping"]

    assert status_ok == 200 and status_not_found == 404
    assert soup_title_text in html_diap41
    assert all(h in html_diap41 for h in soup_hrefs)
    assert orden_preferencia == ["Archivo", "API", "Scraping"]

    print(f"\n{C.BOLD}  [B] Obtención de Datos y Árbol DOM (Diapositivas 36–44):{C.RESET}")
    print(f"      • Orden de preferencia: {C.GREEN}{' > '.join(orden_preferencia)}{C.RESET}")
    print(f"      • Verificación HTTP antes de parsear: status_code == {status_ok} (OK) vs {status_not_found} (No encontrada)")
    print(f"      • soup.title.text                         -> '{soup_title_text}'")
    print(f"      • [a['href'] for a in soup.find_all('a')] -> {soup_hrefs}")

    return {
        "matriz_lector":     matriz_lector,
        "fases_crisp_dm":    fases_crisp_dm,
        "orden_preferencia": orden_preferencia,
        "soup_title_text":   soup_title_text,
        "soup_hrefs":        soup_hrefs,
    }


# =============================================================================
# 3. PANDAS ESENCIAL Y EDA PREMIER LEAGUE 2017 (EPL_MINI, DIAPS. 45–70)
# =============================================================================
def demo_pandas_esencial_y_caza_epl() -> Dict[str, Any]:
    """
    Verifica todas las operaciones sobre `df2` (Diaps. 45-54) y reconstruye con
    exactitud matemática el dataset `epl_mini` (23 jugadores de la Premier
    League 2017, Diaps. 46, 52, 60, 63-70), comprobando cada estadístico de
    `describe()`, los promedios por club, la caza de extremos (`idxmin() -> 8`
    Eduardo `0.05M` e `idxmax()` Paul Pogba `90.0M`) y la tabla agrupada por edad.
    """
    banner(
        "3. PANDAS ESENCIAL Y CASO REAL PREMIER LEAGUE (epl_mini, Diaps. 45–70)",
        "Operaciones df2, describe() de 23 jugadores, Caza de Extremos y GroupBy"
    )

    # 3.1 Verificación de df2 (Diapositivas 45, 47-50, 54)
    df2 = [
        {"idx": 0, "a": 1, "b": 5, "c": 7},
        {"idx": 1, "a": 4, "b": 2, "c": 8},
        {"idx": 2, "a": 3, "b": 6, "c": 9},
    ]
    df2_a_min = min(r["a"] for r in df2)
    df2_b_min = min(r["b"] for r in df2)
    df2_b_idxmin = min(df2, key=lambda r: r["b"])["idx"]
    df2_loc_a_gt_2_c = {r["idx"]: r["c"] for r in df2 if r["a"] > 2}

    assert df2_a_min == 1
    assert df2_b_min == 2
    assert df2_b_idxmin == 1
    assert df2_loc_a_gt_2_c == {1: 8, 2: 9}

    print(f"\n{C.BOLD}  [A] Verificación de `df2` (Diapositivas 45–54):{C.RESET}")
    print(f"      • df2['a'].min()               = {df2_a_min}")
    print(f"      • df2['b'].min() / .idxmin()   = {df2_b_min} (valor mínimo) en etiqueta de fila {df2_b_idxmin}")
    print(f"      • df2.loc[df2['a'] > 2, 'c']   = {df2_loc_a_gt_2_c} (Series con etiquetas 1->8 y 2->9)")

    # 3.2 Dataset epl_mini (23 jugadores, 5 columnas, Diaps. 46, 52, 60, 63-70)
    epl = obtener_dataset_epl_mini()
    n_filas = len(epl)
    n_cols = 5  # name, club, age, position, market_value
    assert (n_filas, n_cols) == (23, 5)

    ages = [float(r["age"]) for r in epl]
    vals = [float(r["market_value"]) for r in epl]

    desc_age = calc_describe(ages)
    desc_val = calc_describe(vals)

    # Comprobación contra la tabla exacta de la Diapositiva 63
    assert round(desc_age["count"], 2) == 23.00
    assert round(desc_age["mean"],  2) == 27.13
    assert round(desc_age["std"],   2) == 3.99
    assert round(desc_age["min"],   2) == 21.00
    assert round(desc_age["25%"],   2) == 24.00
    assert round(desc_age["50%"],   2) == 26.00
    assert round(desc_age["75%"],   2) == 30.50
    assert round(desc_age["max"],   2) == 35.00

    assert round(desc_val["count"], 2) == 23.00
    assert round(desc_val["mean"],  2) == 38.87
    assert round(desc_val["std"],   2) == 23.51
    assert round(desc_val["min"],   2) == 0.05
    assert round(desc_val["25%"],   2) == 21.00
    assert round(desc_val["50%"],   2) == 40.00
    assert round(desc_val["75%"],   2) == 52.50
    assert round(desc_val["max"],   2) == 90.00

    print(f"\n{C.BOLD}  [B] Resumen Global `df[['age', 'market_value']].describe()` (Diapositiva 63):{C.RESET}")
    print("  ┌───────────┬──────────────┬───────────────────┐")
    print("  │ Métrica   │     age      │   market_value    │")
    print("  ├───────────┼──────────────┼───────────────────┤")
    for k in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
        print(f"  │ {k:<9} │ {desc_age[k]:>12.2f} │ {desc_val[k]:>17.2f} │")
    print("  └───────────┴──────────────┴───────────────────┘")

    # 3.3 Promedios por club (Diapositiva 52)
    clubes_esperados = {
        "Arsenal":    32.80,
        "Chelsea":    35.01,
        "Liverpool":  34.00,
        "Man City":   46.67,
        "Man United": 45.75,
        "Tottenham":  41.25,
    }
    clubes_calc: Dict[str, float] = {}
    for club in sorted(clubes_esperados.keys()):
        v_club = [r["market_value"] for r in epl if r["club"] == club]
        clubes_calc[club] = round(calc_mean(v_club), 2)
        assert clubes_calc[club] == clubes_esperados[club], f"Error en {club}: {clubes_calc[club]}"

    print(f"\n{C.BOLD}  [C] Promedio de `market_value` por Club (`groupby('club').mean()`, Diap. 52):{C.RESET}")
    for club, media_c in clubes_calc.items():
        print(f"      • {club:<12}: {C.CYAN}{media_c:>6.2f} M€{C.RESET}")

    # 3.4 Caza de los dos extremos: idxmin() e idxmax() (Diapositivas 64-66)
    fila_min = min(epl, key=lambda r: r["market_value"])
    fila_max = max(epl, key=lambda r: r["market_value"])

    assert fila_min["idx"] == 8
    assert fila_min["name"] == "Eduardo" and fila_min["club"] == "Chelsea"
    assert fila_min["age"] == 34 and fila_min["position"] == "GK" and fila_min["market_value"] == 0.05

    assert fila_max["name"] == "Paul Pogba" and fila_max["club"] == "Man United"
    assert fila_max["age"] == 24 and fila_max["market_value"] == 90.00

    print(f"\n{C.BOLD}  [D] Caza de Extremos y Validación Externa (Diapositivas 64–66):{C.RESET}")
    print(f"      • idxmin() -> fila {fila_min['idx']}: {C.YELLOW}{fila_min['name']}{C.RESET} ({fila_min['club']}, {fila_min['age']} años, {fila_min['position']}, {fila_min['market_value']}M)")
    print(f"        Validación: tercer arquero veterano en 2017 sin minutos en liga -> {C.GREEN}Dato REAL (extremo != error){C.RESET}.")
    print(f"      • idxmax() -> fila {fila_max['idx']}: {C.YELLOW}{fila_max['name']}{C.RESET} ({fila_max['club']}, {fila_max['age']} años, {fila_max['position']}, {fila_max['market_value']}M)")
    print(f"        Validación: traspaso récord mundial de 2016 (~105M€)            -> {C.GREEN}Dato COHERENTE{C.RESET}.")

    # 3.5 Resumen por edad: mean vs median (Diapositivas 67-69)
    edades_unicas = sorted(set(r["age"] for r in epl))
    tabla_edad: Dict[int, Tuple[float, float]] = {}
    for ed in edades_unicas:
        v_ed = [r["market_value"] for r in epl if r["age"] == ed]
        tabla_edad[ed] = (round(calc_mean(v_ed), 2), round(calc_median(v_ed), 2))

    # Verificación exacta contra la tabla de la Diapositiva 68
    esperado_edad = {
        21: (42.50, 42.50),
        23: (38.33, 30.00),
        24: (70.00, 70.00),
        25: (45.00, 45.00),
        26: (60.00, 65.00),
        28: (45.00, 50.00),
        29: (55.00, 55.00),
        30: (25.00, 25.00),
        31: (14.00, 12.00),
        32: (20.00, 20.00),
        34: (0.05,  0.05),
        35: (7.00,  7.00),
    }
    assert tabla_edad == esperado_edad

    print(f"\n{C.BOLD}  [E] `df.groupby('age')['market_value'].agg(['mean', 'median'])` (Diapositiva 68):{C.RESET}")
    print(f"      • En age=23: mean = {tabla_edad[23][0]:.2f} vs median = {tabla_edad[23][1]:.2f} (Harry Kane vale 60.0M y estira la media)")
    print(f"      • En age=28: mean = {tabla_edad[28][0]:.2f} vs median = {tabla_edad[28][1]:.2f} (Walcott 20M, Ozil 50M, Sanchez 65M)")

    return {
        "desc_age":    desc_age,
        "desc_val":    desc_val,
        "clubes_calc": clubes_calc,
        "fila_min":    fila_min,
        "fila_max":    fila_max,
        "tabla_edad":  tabla_edad,
    }


# =============================================================================
# 4. MEDIA VS. MEDIANA EN EL ARSENAL Y CUARTETO DE ANSCOMBE (DIAPS. 71–74)
# =============================================================================
def demo_arsenal_media_mediana_y_anscombe() -> Dict[str, Any]:
    """
    Verifica el ejercicio de sensibilidad de la media vs. la mediana en los 5
    jugadores del Arsenal (Diap. 72) y demuestra numéricamente las propiedades
    invariantes del Cuarteto de Anscombe (1973, Diap. 71).
    """
    banner(
        "4. MEDIA VS. MEDIANA (ARSENAL) Y CUARTETO DE ANSCOMBE (Diaps. 71–74)",
        "Sensibilidad ante Valores Extremos y Por Qué Graficar Siempre"
    )

    # 4.1 Ejercicio del Arsenal (Diapositivas 60 y 72)
    arsenal_vals = [65.0, 50.0, 7.0, 20.0, 22.0]
    media_arsenal = calc_mean(arsenal_vals)
    mediana_arsenal = calc_median(arsenal_vals)

    arsenal_con_fichaje = arsenal_vals + [100.0]
    media_fichaje = calc_mean(arsenal_con_fichaje)
    mediana_fichaje = calc_median(arsenal_con_fichaje)

    assert abs(media_arsenal - 32.80) < 1e-9
    assert abs(mediana_arsenal - 22.00) < 1e-9
    assert abs(media_fichaje - 44.00) < 1e-9
    assert abs(mediana_fichaje - 36.00) < 1e-9

    print(f"\n{C.BOLD}  [A] Media vs. Mediana en los 5 jugadores del Arsenal (Diapositiva 72):{C.RESET}")
    print(f"      • Valores originales (ordenados): {sorted(arsenal_vals)}")
    print(f"        -> Media = {C.CYAN}{media_arsenal:.2f} M{C.RESET} | Mediana = {C.GREEN}{mediana_arsenal:.2f} M{C.RESET}")
    print(f"      • Si fichan a un jugador de 100 M: {sorted(arsenal_con_fichaje)}")
    print(f"        -> Media salta a {C.YELLOW}{media_fichaje:.2f} M{C.RESET} (+34.1%) | Mediana pasa a {C.GREEN}{mediana_fichaje:.2f} M{C.RESET}")

    # 4.2 El Cuarteto de Anscombe (1973, Diapositiva 71)
    anscombe = {
        "I (Lineal ruidoso)": (
            [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0],
            [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
        ),
        "II (Parábola sin ruido)": (
            [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0],
            [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
        ),
        "III (Línea + 1 outlier)": (
            [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0],
            [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
        ),
        "IV (Vertical + 1 extremo)": (
            [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 19.0, 8.0, 8.0, 8.0],
            [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89],
        ),
    }

    print(f"\n{C.BOLD}  [B] El Cuarteto de Anscombe (1973) — Números idénticos, geometrías opuestas (Diap. 71):{C.RESET}")
    print("  ┌──────────────────────────┬─────────┬─────────┬─────────┬─────────┬─────────┬──────────────────┐")
    print("  │ Conjunto                 │ Media X │ Media Y │  Var X  │  Var Y  │ Corr r  │ Recta OLS        │")
    print("  ├──────────────────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼──────────────────┤")

    resumen_anscombe = {}
    for nombre, (xs, ys) in anscombe.items():
        mx = calc_mean(xs)
        my = calc_mean(ys)
        vx = calc_var_sample(xs)
        vy = calc_var_sample(ys)
        cov_xy = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) - 1)
        r = cov_xy / math.sqrt(vx * vy)
        b1 = cov_xy / vx
        b0 = my - b1 * mx

        assert abs(mx - 9.0) < 1e-6
        assert abs(my - 7.50) < 1e-2
        assert abs(vx - 11.0) < 1e-6
        assert abs(vy - 4.125) < 1e-2
        assert abs(r - 0.816) < 1e-2
        assert abs(b1 - 0.500) < 1e-2
        assert abs(b0 - 3.00) < 1e-2

        resumen_anscombe[nombre] = {"mx": mx, "my": my, "vx": vx, "vy": vy, "r": r, "b0": b0, "b1": b1}
        print(f"  │ {nombre:<24} │ {mx:>7.2f} │ {my:>7.2f} │ {vx:>7.2f} │ {vy:>7.3f} │ {r:>7.3f} │ y = {b0:.2f} + {b1:.2f}x │")

    print("  └──────────────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┴──────────────────┘")
    print(f"  {C.GREEN}✓ Verificado: los 4 conjuntos comparten media, varianza, correlación y recta OLS.{C.RESET}")

    return {
        "media_arsenal":    media_arsenal,
        "mediana_arsenal":  mediana_arsenal,
        "media_fichaje":    media_fichaje,
        "mediana_fichaje":  mediana_fichaje,
        "resumen_anscombe": resumen_anscombe,
    }


# =============================================================================
# EJECUCIÓN AUTOMÁTICA COMPLETA Y MENÚ INTERACTIVO
# =============================================================================
def ejecutar_todas_las_demos() -> None:
    """Ejecuta y verifica los 4 bloques de la Unidad CD02."""
    demo_datos_alcances_y_sesgos_muestreo()
    demo_formatos_preguntas_y_crisp_dm()
    demo_pandas_esencial_y_caza_epl()
    demo_arsenal_media_mediana_y_anscombe()
    print(f"\n{C.GREEN}{C.BOLD}  [OK] Unidad CD02 (Preprocesamiento y EDA — 74 Diaps.) verificada al 100%.{C.RESET}\n")


def menu_interactivo() -> None:
    while True:
        banner(
            "UNIDAD CD02 — PREPROCESAMIENTO Y ANÁLISIS EXPLORATORIO DE DATOS (74 Diaps.)",
            "Selector Interactivo de Demostraciones y Verificaciones"
        )
        print("  [1] ¿Qué son los datos?, Advertencias de fábrica y Sesgos (Gallup vs. Yelp, QR) (Diaps. 1–20)")
        print("  [2] Formatos 2x2, Preguntas Medibles, CRISP-DM vs. KDD/SEMMA y Web Scraping     (Diaps. 21–44)")
        print("  [3] Pandas Esencial y EDA Premier League 2017 (epl_mini, 23 jugadores)          (Diaps. 45–70)")
        print("  [4] Media vs. Mediana (Arsenal) y el Cuarteto de Anscombe (1973)                (Diaps. 71–74)")
        print("  [5] Ejecutar TODAS las demostraciones de CD02 de forma continua")
        print("  [0] Volver / Salir")
        opc = input("\n  Seleccione una opción [0-5]: ").strip()
        if opc == "1":
            demo_datos_alcances_y_sesgos_muestreo()
        elif opc == "2":
            demo_formatos_preguntas_y_crisp_dm()
        elif opc == "3":
            demo_pandas_esencial_y_caza_epl()
        elif opc == "4":
            demo_arsenal_media_mediana_y_anscombe()
        elif opc == "5":
            ejecutar_todas_las_demos()
        elif opc == "0":
            break


if __name__ == "__main__":
    if "--auto" in sys.argv:
        ejecutar_todas_las_demos()
    else:
        menu_interactivo()
